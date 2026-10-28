import { useEffect, useMemo, useRef, useState } from "react";
import { type Result, useRequest } from "./api";
import { Notice } from "./Viewer";
import BufferedIntervalViewer from "./BufferedIntervalViewer";
import { type IntervalAnnotation } from "./intervalAnnotations";
import { type ViewingUpdate } from "./playback";

export type HumanPunch = {
  event_id: string;
  start: number;
  stop: number;
  punch_type?: string;
  type_uncertain?: boolean;
  kind?: "activity" | "uncertain";
  partial_overlap?: boolean;
};
type Revision = {
  reviewed_at: string;
  source: string;
  payload: { operation: string; event?: HumanPunch; event_id?: string };
  events: HumanPunch[];
  complete: boolean;
  sha256: string;
};
type State = {
  head: string;
  events: HumanPunch[];
  complete: boolean;
  history: Revision[];
};
type Entry = {
  id: string;
  recording: string;
  start: number;
  stop: number;
  context_start: number;
  context_end: number;
  frame_count: number;
};
const labels = [
  ["jab", "Jab"],
  ["cross", "Cross"],
  ["lead_hook", "Lead Hook"],
  ["rear_hook", "Rear Hook"],
  ["lead_uppercut", "Lead Uppercut"],
  ["rear_uppercut", "Rear Uppercut"],
];
const name = (p: HumanPunch) =>
  p.kind
    ? p.kind === "activity"
      ? "Punching activity"
      : "Uncertain section"
    : p.type_uncertain
      ? "Type uncertain"
      : (labels.find(([v]) => v === p.punch_type)?.[1] ?? p.punch_type);
export default function PunchLabeling({
  entry: e,
  mode = "punch",
  annotations,
  onDirty,
  onSaving,
  onNext,
  onProgress,
}: {
  entry: Entry;
  mode?: "punch" | "activity";
  annotations: IntervalAnnotation[];
  onDirty: (v: boolean) => void;
  onSaving: (v: boolean) => void;
  onNext: () => void;
  onProgress: (s: { complete: boolean; events: HumanPunch[] }) => void;
}) {
  const activity = mode === "activity";
  const endpoint = activity
    ? `/api/activity-evaluation/${e.id}`
    : `/api/intervals/${e.id}/events`;
  const query = useRequest<Result<State>>(endpoint);
  const [state, setState] = useState<State | null>(null);
  const [draft, setDraft] = useState<{
    event_id: string;
    start: number | null;
    end: number | null;
    punch_type: string;
    type_uncertain: boolean;
    kind: "activity" | "uncertain";
  } | null>(null);
  const [seen, setSeen] = useState<Set<number>>(new Set());
  const [viewing, setViewing] = useState<ViewingUpdate | null>(null);
  const [attest, setAttest] = useState(false),
    [saving, setSaving] = useState(false),
    [feedback, setFeedback] = useState("");
  const [error, setError] = useState("");
  const flight = useRef(false),
    request = useRef<{ body: string; id: string } | null>(null);
  useEffect(() => {
    if (query.data?.data) setState(query.data.data);
  }, [query.data]);
  const reviewSeen = useMemo(
    () => new Set([...seen].filter((n) => n >= e.start && n < e.stop)),
    [seen, e.start, e.stop],
  );
  const full =
    !!viewing?.complete ||
    [...seen].filter((n) => n >= e.start && n < e.stop).length ===
      e.stop - e.start;
  function edit(p: HumanPunch) {
    if (
      draft &&
      !window.confirm(
        activity
          ? "Discard unsaved section changes?"
          : "Discard unsaved punch changes?",
      )
    )
      return;
    setDraft({
      event_id: p.event_id,
      start: p.start,
      end: p.stop - 1,
      punch_type: p.punch_type ?? "",
      type_uncertain: p.type_uncertain ?? false,
      kind: p.kind ?? "activity",
    });
    setAttest(false);
    onDirty(true);
    setError("");
    setFeedback("");
  }
  async function save(
    operation: "upsert" | "remove" | "complete",
    eventId?: string,
  ) {
    if (flight.current || !state) return;
    if (
      operation === "upsert" &&
      (!draft ||
        draft.start === null ||
        draft.end === null ||
        draft.end < draft.start ||
        (!activity && !draft.punch_type))
    ) {
      setError(
        activity
          ? "Mark the first and last frames of the section."
          : "Mark the first and last punch frames and choose a type, or select Type uncertain.",
      );
      return;
    }
    if (operation === "complete" && (!full || !attest || draft)) {
      setError(
        activity
          ? "Watch the whole block, save any section changes, then confirm the review is complete."
          : "Watch the whole interval, save any punch changes, and confirm all punches are labeled.",
      );
      return;
    }
    const event =
      operation === "upsert" && draft
        ? {
            event_id: draft.event_id,
            start: draft.start,
            stop: draft.end! + 1,
            ...(activity
              ? { kind: draft.kind }
              : {
                  punch_type: draft.punch_type,
                  type_uncertain: draft.type_uncertain,
                }),
          }
        : undefined;
    const payload = {
      expected_head: state.head,
      operation,
      ...(event ? { event } : {}),
      ...(eventId ? { event_id: eventId } : {}),
      inspected_frames: [...seen]
        .filter((n) => operation !== "complete" || (n >= e.start && n < e.stop))
        .sort((a, b) => a - b),
      ...(operation === "complete"
        ? {
            watched_entire: true,
            ...(activity
              ? {
                  all_activity_labeled: true,
                  no_activity_observed: state.events.length === 0,
                }
              : { all_punches_labeled: true }),
            ...(viewing?.evidence.segments.length
              ? { playback_coverage: viewing.evidence }
              : {}),
          }
        : {}),
    };
    const body = JSON.stringify(payload);
    if (request.current?.body !== body)
      request.current = { body, id: crypto.randomUUID() };
    flight.current = true;
    setSaving(true);
    onSaving(true);
    setError("");
    setFeedback("");
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Review-Client": "dashboard-v0.2",
        },
        body: JSON.stringify({ ...payload, request_id: request.current!.id }),
      });
      const result = await response.json();
      if (!response.ok) {
        const messages: Record<string, string> = {
          combine_activity_bouts_under_quarter_second:
            "Combine consecutive punches separated by less than 0.25 seconds into one activity interval.",
          activity_outside_reviewed_block:
            "Keep activity and uncertain sections inside this review block.",
          event_review_changed_reload:
            "Another change was saved. Reload before saving again.",
          inspect_both_event_boundaries:
            "Inspect each changed boundary with the exact-frame controls before saving.",
          event_outside_available_context:
            "Keep the punch inside the available footage.",
          event_outside_interval: "The punch must overlap this interval.",
          watch_entire_interval_before_completing:
            "Some of the interval still needs viewing.",
        };
        throw new Error(
          messages[result.issues?.[0]] ??
            "Could not save. Check the boundaries and retry; your changes are retained.",
        );
      }
      const record = result.data as Revision;
      const next = {
        head: record.sha256,
        events: record.events,
        complete: record.complete,
        history: [...state.history, record],
      };
      setState(next);
      onProgress(next);
      setDraft(null);
      setAttest(false);
      onDirty(false);
      setFeedback(
        operation === "complete"
          ? "Interval saved as completely labeled."
          : operation === "remove"
            ? activity
              ? "Section removed. Its history is preserved."
              : "Punch removed. Its history is preserved."
            : activity
              ? "Section saved. Continue reviewing the block."
              : "Punch saved. Continue looking for others.",
      );
      if (operation === "complete") onNext();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Save failed; changes retained.",
      );
    } finally {
      flight.current = false;
      setSaving(false);
      onSaving(false);
    }
  }
  if (query.error) return <Notice>{query.error}</Notice>;
  if (!state)
    return (
      <Notice>
        {activity ? "Loading activity review…" : "Loading punch labels…"}
      </Notice>
    );
  const controls = (frame: number, ready: boolean, inspect?: () => void) =>
    draft && (
      <div className="event-boundaries">
        <button
          disabled={!ready || saving}
          onClick={() => {
            inspect?.();
            setDraft({ ...draft, start: frame });
            onDirty(true);
          }}
        >
          Set start here
        </button>
        <span>Start {draft.start ?? "—"}</span>
        <button
          disabled={!ready || saving}
          onClick={() => {
            inspect?.();
            setDraft({ ...draft, end: frame });
            onDirty(true);
          }}
        >
          Set end here
        </button>
        <span>
          {activity ? "Last included frame" : "Last punch frame"}{" "}
          {draft.end ?? "—"}
        </span>
        {!ready && <small>Pause to inspect an exact frame.</small>}
      </div>
    );
  return (
    <>
      <BufferedIntervalViewer
        intervalId={e.id}
        recording={e.recording}
        annotations={activity ? [] : annotations}
        showSourceAnnotations={!activity}
        seenFrames={reviewSeen}
        onViewing={setViewing}
        humanEvents={state.events}
        onHumanSelect={(id) => {
          const p = state.events.find((p) => p.event_id === id);
          if (p) edit(p);
        }}
        inspectionControls={controls}
        interval={{
          start: e.start,
          end: e.stop - 1,
          count: e.frame_count,
          contextStart: e.context_start,
          contextEnd: e.context_end,
          delay: 40,
        }}
        onVisible={(f) =>
          setSeen((old) =>
            old.has(f.frame_index) ? old : new Set([...old, f.frame_index]),
          )
        }
      />
      <section className="panel interval-form">
        <div className="event-heading">
          <h2>{activity ? "Punching activity" : "Punches you added"}</h2>
          <span className="muted">
            {state.complete && !draft
              ? activity
                ? state.events.some((e) => e.kind === "uncertain")
                  ? "Reviewed · uncertainty"
                  : "Reviewed"
                : "Completely labeled"
              : "In progress"}{" "}
            · {state.events.length} {activity ? "sections" : "punches"}
          </span>
        </div>
        {activity && (
          <p className="muted">
            One activity interval can contain several punches. Keep pauses under
            0.25 seconds in the same interval. Mark anything unclear as
            uncertain.
          </p>
        )}
        {(activity
          ? (["activity", "uncertain"] as const)
          : (["activity"] as const)
        ).map((kind) => (
          <button
            key={kind}
            disabled={saving || !!draft}
            onClick={() => {
              setDraft({
                event_id: crypto.randomUUID(),
                start: null,
                end: null,
                punch_type: "",
                type_uncertain: false,
                kind,
              });
              onDirty(true);
              setFeedback("");
            }}
          >
            {activity
              ? kind === "activity"
                ? "Add activity"
                : "Mark uncertain section"
              : "Add punch"}
          </button>
        ))}
        {draft && (
          <div className="event-editor">
            {!activity && (
              <>
                {" "}
                <label>
                  Punch type{" "}
                  <select
                    aria-label="Punch type"
                    disabled={saving || draft.type_uncertain}
                    value={draft.punch_type}
                    onChange={(ev) => {
                      setDraft({ ...draft, punch_type: ev.target.value });
                      onDirty(true);
                    }}
                  >
                    <option value="">Choose a punch</option>
                    {draft.type_uncertain && (
                      <option value="unknown">Type uncertain</option>
                    )}
                    {labels.map(([v, t]) => (
                      <option key={v} value={v}>
                        {t}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={draft.type_uncertain}
                    disabled={saving}
                    onChange={(ev) => {
                      setDraft({
                        ...draft,
                        type_uncertain: ev.target.checked,
                        punch_type: ev.target.checked ? "unknown" : "",
                      });
                      onDirty(true);
                    }}
                  />{" "}
                  Type uncertain
                </label>
              </>
            )}
            {activity && (
              <label>
                Section{" "}
                <select
                  aria-label="Section kind"
                  value={draft.kind}
                  disabled={saving}
                  onChange={(ev) => {
                    setDraft({
                      ...draft,
                      kind: ev.target.value as "activity" | "uncertain",
                    });
                    onDirty(true);
                  }}
                >
                  <option value="activity">Punching activity</option>
                  <option value="uncertain">Uncertain section</option>
                </select>
              </label>
            )}
            <div className="interval-save">
              <button
                className="primary"
                disabled={saving}
                onClick={() => void save("upsert")}
              >
                {activity ? "Save section" : "Save punch"}
              </button>
              <button
                disabled={saving}
                onClick={() => {
                  setDraft(null);
                  onDirty(false);
                  setError("");
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        )}
        {state.events.map((p, i) => (
          <div key={p.event_id} className="human-event-row">
            <span>
              {i + 1}. {name(p)} · {p.start}–{p.stop - 1}
              {p.partial_overlap ? " · Partial overlap" : ""}
            </span>
            <button disabled={saving || !!draft} onClick={() => edit(p)}>
              Edit {activity ? "section" : "punch"} {i + 1}
            </button>
            <button
              disabled={saving || !!draft}
              onClick={() => void save("remove", p.event_id)}
            >
              Remove {activity ? "section" : "punch"} {i + 1}
            </button>
          </div>
        ))}
        {state.events.some((p) => p.type_uncertain) && (
          <p className="muted">
            Uncertain types remain unknown, even when the interval is completely
            labeled.
          </p>
        )}
        <div className="interval-checks">
          <label>
            <input
              type="checkbox"
              checked={attest}
              disabled={!full || saving || !!draft}
              onChange={(ev) => {
                setAttest(ev.target.checked);
                onDirty(ev.target.checked);
              }}
            />{" "}
            {activity
              ? state.events.length
                ? "I watched the entire block and marked all punching activity and uncertain sections."
                : "I watched the entire block and saw no punching activity."
              : "I watched the entire interval and labeled every punch."}
          </label>
        </div>
        <div className="interval-save">
          <button
            className="primary"
            disabled={
              saving ||
              !attest ||
              (!activity && !state.events.length) ||
              !!draft
            }
            onClick={() => void save("complete")}
          >
            {saving ? "Saving…" : "Save complete & Next"}
          </button>
          <span className="muted">
            {full ? "Interval viewed" : `Viewing ${viewing?.percent ?? 0}%`}
          </span>
        </div>
        {feedback && <p role="status">{feedback}</p>}
        {error && <p role="alert">{error}</p>}
      </section>
      <details className="panel interval-references">
        <summary>Labeling history & details</summary>
        <p>
          {activity
            ? "Start = first activity or uncertain frame. End = last included frame. Source annotations and model predictions are hidden during labeling. "
            : "Start = first punch frame. End = last included punch frame. "}{" "}
          Stored intervals use an exclusive stop. Human labels are separate from
          original dataset marks. Unreviewed footage remains unknown.
        </p>
        {state.history.map((r, i) => (
          <p key={r.sha256}>
            {i + 1}. {r.reviewed_at} · {r.source} · {r.payload.operation}
            {r.payload.event
              ? ` · ${name(r.payload.event)} · ${r.payload.event.start}–${r.payload.event.stop - 1}`
              : ""}
            {r.complete ? " · Completely labeled" : ""}
          </p>
        ))}
      </details>
    </>
  );
}
