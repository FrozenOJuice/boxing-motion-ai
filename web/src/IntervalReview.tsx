import { useEffect, useRef, useState } from "react";
import { type Result, useRequest } from "./api";
import { Notice } from "./Viewer";
import PunchLabeling, { type HumanPunch } from "./PunchLabeling";
import BufferedIntervalViewer from "./BufferedIntervalViewer";
import { type IntervalAnnotation } from "./intervalAnnotations";
import { type ViewingUpdate } from "./playback";

type Entry = {
  id: string;
  recording: string;
  role: string;
  purpose: string;
  start: number;
  stop: number;
  context_start: number;
  context_end: number;
  frame_count: number;
  timing: { start: string; stop_exclusive: string };
  discovery_hint?: string;
};
type Payload = {
  decision: string;
  notes: string;
  tags: string[];
  watched_entire: boolean;
  visible_continuous: boolean;
  unannotated_punches: string;
};
type Revision = {
  interval_id: string;
  reviewed_at: string;
  status: string;
  payload: Payload;
  source: string;
};
type Queue = {
  manifest: { entries: Entry[] };
  history: Revision[];
  head: string;
};
type Detail = {
  entry: Entry;
  head: string;
  history: Revision[];
  annotations: IntervalAnnotation[];
  status: string;
};
const decisions = [
  ["no_punch", "No punches"],
  ["punch_present", "Punch present"],
  ["not_sure", "Not sure"],
];
const statusName = (status?: string) =>
  status === "verified_background"
    ? "No punches verified"
    : status === "needs_event_labeling"
      ? "Punches to label"
      : status === "unresolved"
        ? "Needs attention"
        : "To review";
const seconds = (s: string) => {
  const [n, d] = s.split("/").map(Number);
  return n / (d ?? 1);
};

export default function IntervalReview({
  onDirty,
  onSaving,
}: {
  onDirty: (value: boolean) => void;
  onSaving: (value: boolean) => void;
}) {
  const [pilot, setPilot] = useState<
    "original" | "motion-v2" | "events" | "activity" | "localization"
  >("original");
  return (
    <IntervalQueue
      key={pilot}
      pilot={pilot}
      onPilot={setPilot}
      onDirty={onDirty}
      onSaving={onSaving}
    />
  );
}

function IntervalQueue({
  onDirty,
  onSaving,
  pilot,
  onPilot,
}: {
  onDirty: (value: boolean) => void;
  onSaving: (value: boolean) => void;
  pilot: "original" | "motion-v2" | "events" | "activity" | "localization";
  onPilot: (
    pilot: "original" | "motion-v2" | "events" | "activity" | "localization",
  ) => void;
}) {
  const localization = pilot === "localization";
  const query = localization
    ? "?pilot=localization-v1"
    : pilot === "motion-v2"
      ? "?pilot=motion-v2"
      : "";
  const activity = pilot === "activity";
  const labeling = pilot === "events" || activity || localization;
  const statuses = useRequest<
    Result<Record<string, { complete: boolean; events: HumanPunch[] }>>
  >(
    localization
      ? "/api/localization-events/status"
      : activity
        ? "/api/activity-evaluation/status"
        : labeling
          ? "/api/interval-event-status"
          : null,
  );
  const [progress, setProgress] = useState<
    Record<string, { complete: boolean; events: HumanPunch[] }>
  >({});
  const [refresh, setRefresh] = useState(0),
    [selected, setSelected] = useState("");
  const [dirty, setDirty] = useState(false),
    [busy, setBusy] = useState(false),
    [feedback, setFeedback] = useState("");
  const queue = useRequest<Result<Queue>>(`/api/intervals${query}`, refresh);
  const entries = (queue.data?.data?.manifest.entries ?? []).filter((e) =>
    activity
      ? /^(V7|V8|V10)-eval-[12]$/.test(e.id)
      : localization || !labeling || /^V4-bg-[1-5]$/.test(e.id),
  );
  useEffect(() => {
    if (!selected && entries.length) setSelected(entries[0].id);
  }, [selected, entries]);
  const current = useRequest<Result<Detail>>(
    selected ? `/api/intervals/${selected}${query}` : null,
    refresh,
  );
  const data = current.data?.data;
  const change = (value: boolean) => {
    setDirty(value);
    onDirty(value);
  };
  return (
    <div className="review-layout">
      <section className="queue-panel panel">
        <div className="panel-header">
          <h2>
            Intervals <span className="count">{entries.length}</span>
          </h2>
        </div>
        <label className="queue-pilot-selector">
          Review pilot
          <select
            aria-label="Review pilot"
            value={pilot}
            disabled={busy}
            onChange={(event) => {
              if (!dirty || window.confirm("Discard unsaved review changes?")) {
                change(false);
                onPilot(
                  event.target.value as
                    | "original"
                    | "motion-v2"
                    | "events"
                    | "activity"
                    | "localization",
                );
              }
            }}
          >
            <option value="activity">Activity evaluation · V7/V8/V10</option>
            <option value="localization">Individual punches · V6 / V5</option>
            <option value="events">Label missing punches · V4</option>
            <option value="original">Original pilot</option>
            <option value="motion-v2">Motion pilot 2 · V4/V6</option>
          </select>
        </label>
        <div className="case-list">
          {entries.map((e, i) => {
            const history =
              queue.data?.data?.history.filter((r) => r.interval_id === e.id) ??
              [];
            return (
              <button
                key={e.id}
                className={`case-item ${selected === e.id ? "selected" : ""}`}
                disabled={busy}
                onClick={() => {
                  if (
                    selected !== e.id &&
                    (!dirty ||
                      window.confirm("Discard unsaved review changes?"))
                  ) {
                    change(false);
                    setFeedback("");
                    setSelected(e.id);
                  }
                }}
              >
                <strong>
                  {e.recording} ·{" "}
                  {labeling
                    ? activity
                      ? "Activity block"
                      : "Punch interval"
                    : e.purpose === "evaluation_block"
                      ? "Continuous review"
                      : "Background candidate"}{" "}
                  {i + 1}
                </strong>
                {e.discovery_hint && (
                  <span className="queue-discovery-hint">
                    {e.discovery_hint} · Unverified
                  </span>
                )}
                <span className="queue-meta">
                  <span>
                    {e.role === "evaluation_reserved"
                      ? "Evaluation only"
                      : e.role === "development"
                        ? "Development only"
                        : "Training candidate"}
                  </span>
                  <span>
                    {labeling
                      ? (progress[e.id] ?? statuses.data?.data?.[e.id])
                          ?.complete
                        ? (
                            progress[e.id] ?? statuses.data?.data?.[e.id]
                          )?.events.some((e) => e.kind === "uncertain")
                          ? "Reviewed · uncertainty"
                          : activity
                            ? "Reviewed"
                            : "Completely labeled"
                        : activity
                          ? "To review"
                          : "Punches to label"
                      : statusName(history.at(-1)?.status)}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
      </section>
      <div className="review-main">
        {queue.error || current.error ? (
          <Notice>{queue.error || current.error}</Notice>
        ) : !data ? (
          <Notice>Loading interval…</Notice>
        ) : labeling ? (
          <PunchLabeling
            key={selected}
            mode={
              localization ? "localization" : activity ? "activity" : "punch"
            }
            entry={data.entry}
            annotations={data.annotations}
            onDirty={change}
            onSaving={(v) => {
              setBusy(v);
              onSaving(v);
            }}
            onProgress={(state) =>
              setProgress((old) => ({ ...old, [selected]: state }))
            }
            onNext={() => {
              change(false);
              const next =
                entries[entries.findIndex((e) => e.id === selected) + 1];
              if (next) setSelected(next.id);
            }}
          />
        ) : (
          <IntervalForm
            key={`${selected}:${refresh}`}
            data={data}
            pilot={pilot === "motion-v2" ? "motion-v2" : "original"}
            feedback={feedback}
            onDirty={change}
            onSaving={(v) => {
              setBusy(v);
              onSaving(v);
            }}
            onSaved={(next) => {
              change(false);
              const nextEntry =
                entries[entries.findIndex((e) => e.id === selected) + 1];
              setFeedback(
                next && !nextEntry
                  ? "Review saved · End of queue."
                  : "Review saved.",
              );
              if (next && nextEntry) setSelected(nextEntry.id);
              setRefresh((v) => v + 1);
            }}
          />
        )}
      </div>
    </div>
  );
}

function IntervalForm({
  data,
  feedback,
  onDirty,
  onSaving,
  onSaved,
  pilot,
}: {
  data: Detail;
  pilot: "original" | "motion-v2";
  feedback: string;
  onDirty: (v: boolean) => void;
  onSaving: (v: boolean) => void;
  onSaved: (next: boolean) => void;
}) {
  const e = data.entry,
    prior = data.history.at(-1)?.payload;
  const [decision, setDecision] = useState(prior?.decision ?? "");
  const [notes, setNotes] = useState(prior?.notes ?? ""),
    [tags, setTags] = useState<string[]>(prior?.tags ?? []);
  const [watched, setWatched] = useState(false),
    [visible, setVisible] = useState(false);
  const [unannotated, setUnannotated] = useState(
    prior?.unannotated_punches ?? "unknown",
  );
  const [seen, setSeen] = useState<Set<number>>(new Set()),
    [saving, setSaving] = useState(false),
    [error, setError] = useState("");
  const [viewing, setViewing] = useState<ViewingUpdate | null>(null);
  const inFlight = useRef(false);
  const request = useRef<{ body: string; id: string } | null>(null);
  const full = !!viewing?.complete || seen.size === e.stop - e.start;
  const conflict = data.annotations.length > 0;
  const change = () => {
    onDirty(true);
    setError("");
  };
  async function save(next: boolean) {
    if (inFlight.current) return;
    if (!decision || (!seen.size && !viewing?.evidence.segments.length)) {
      setError("Watch the interval and choose a decision.");
      return;
    }
    if (
      decision === "no_punch" &&
      (!full || !watched || !visible || conflict)
    ) {
      setError(
        conflict
          ? "A source punch overlaps this interval. Keep it unresolved for conflict review."
          : "Watch the whole interval, then explicitly confirm viewing and visibility.",
      );
      return;
    }
    const payload = {
      expected_head: data.head,
      decision,
      notes,
      tags,
      watched_entire: watched,
      visible_continuous: visible,
      inspected_frames: [...seen].sort((a, b) => a - b),
      ...(viewing?.evidence.segments.length
        ? { playback_coverage: viewing.evidence }
        : {}),
      unannotated_punches: decision === "no_punch" ? "no" : unannotated,
    };
    const body = JSON.stringify(payload);
    if (!request.current || request.current.body !== body)
      request.current = { body, id: crypto.randomUUID() };
    inFlight.current = true;
    setSaving(true);
    onSaving(true);
    setError("");
    try {
      const r = await fetch(
        `/api/intervals/${e.id}${pilot === "original" ? "" : "?pilot=motion-v2"}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Review-Client": "dashboard-v0.2",
          },
          body: JSON.stringify({ ...payload, request_id: request.current.id }),
        },
      );
      const result = await r.json();
      if (!r.ok) {
        const messages: Record<string, string> = {
          interval_review_changed_reload:
            "Another review was saved. Copy any notes, then reload before saving",
          source_punch_conflict_needs_review:
            "A source punch overlaps this interval and needs conflict review",
          watch_entire_interval_before_certifying:
            "Watch the complete interval before confirming no punches",
          interval_source_changed:
            "The source data changed. This interval needs revalidation",
        };
        const code = result.issues?.[0];
        throw new Error(
          messages[code] ||
            "The review could not be saved. Check the source data or retry",
        );
      }
      onSaved(next);
    } catch (err) {
      setError(
        `${err instanceof Error ? err.message : "Save failed"}. Your changes have not been discarded.`,
      );
    } finally {
      inFlight.current = false;
      setSaving(false);
      onSaving(false);
    }
  }
  return (
    <>
      <BufferedIntervalViewer
        intervalId={e.id}
        pilot={pilot}
        annotations={data.annotations}
        seenFrames={seen}
        onViewing={setViewing}
        recording={e.recording}
        interval={{
          start: e.start,
          end: e.stop - 1,
          count: e.frame_count,
          contextStart: e.context_start,
          contextEnd: e.context_end,
          delay:
            (1000 *
              (seconds(e.timing.stop_exclusive) - seconds(e.timing.start))) /
            (e.stop - e.start),
        }}
        onVisible={(f) => {
          if (f.frame_index >= e.start && f.frame_index < e.stop)
            setSeen((old) =>
              old.has(f.frame_index) ? old : new Set([...old, f.frame_index]),
            );
        }}
      />
      <section className="panel interval-form">
        <h2>Are there any punches?</h2>
        <p className="muted">
          {e.role === "evaluation_reserved"
            ? "Reserved for evaluation. "
            : e.role === "development"
              ? "Development only. "
              : "Training candidate. "}
          {statusName(data.status)} ·{" "}
          {full ? "Interval viewed" : `Viewing ${viewing?.percent ?? 0}%`} ·{" "}
          {seen.size} {seen.size === 1 ? "frame" : "frames"} inspected
          individually
        </p>
        <div className="decision-options">
          {decisions.map(([value, label]) => (
            <button
              key={value}
              disabled={saving}
              aria-pressed={decision === value}
              className={decision === value ? "active" : ""}
              onClick={() => {
                setDecision(value);
                change();
              }}
            >
              {label}
            </button>
          ))}
        </div>
        {decision === "no_punch" && (
          <div className="interval-checks">
            {conflict && (
              <Notice>
                Existing punch annotations overlap this interval. Choose Not
                sure for conflict review; they cannot be silently relabeled.
              </Notice>
            )}
            <label>
              <input
                type="checkbox"
                disabled={!full || saving || conflict}
                checked={watched}
                onChange={(ev) => {
                  setWatched(ev.target.checked);
                  change();
                }}
              />{" "}
              I watched the entire interval and observed no punches.
            </label>
            <label>
              <input
                type="checkbox"
                disabled={saving}
                checked={visible}
                onChange={(ev) => {
                  setVisible(ev.target.checked);
                  change();
                }}
              />{" "}
              The boxer is visible throughout, with no scene changes.
            </label>
            <details>
              <summary>Movement tags (optional)</summary>
              {[
                ["ordinary", "Ordinary movement"],
                ["guard", "Guard / recovery"],
                ["defense", "Defensive movement"],
                ["pause", "Pause"],
                ["feint", "Feint"],
              ].map(([value, label]) => (
                <label key={value}>
                  <input
                    type="checkbox"
                    disabled={saving}
                    checked={tags.includes(value)}
                    onChange={() => {
                      setTags((old) =>
                        old.includes(value)
                          ? old.filter((x) => x !== value)
                          : [...old, value],
                      );
                      change();
                    }}
                  />
                  {label}
                </label>
              ))}
            </details>
          </div>
        )}
        {decision === "punch_present" && (
          <div>
            <p>
              Punches will be flagged for event labeling. This interval will not
              become background.
            </p>
            <label>
              Did you see any punches that weren't marked?{" "}
              <select
                disabled={saving}
                value={unannotated}
                onChange={(ev) => {
                  setUnannotated(ev.target.value);
                  change();
                }}
              >
                <option value="unknown">Not sure</option>
                <option value="yes">Yes</option>
                <option value="no">No</option>
              </select>
            </label>
          </div>
        )}
        {decision && (
          <details>
            <summary>Notes (optional)</summary>
            <textarea
              aria-label="Interval notes"
              value={notes}
              disabled={saving}
              onChange={(ev) => {
                setNotes(ev.target.value);
                change();
              }}
            />
          </details>
        )}
        {error && <p role="alert">{error}</p>}
        {feedback && <p role="status">{feedback}</p>}
        <div className="interval-save">
          <button
            className="primary"
            disabled={saving || !decision}
            onClick={() => void save(true)}
          >
            {saving ? "Saving…" : "Save & Next"}
          </button>
          <button
            disabled={saving || !decision}
            onClick={() => void save(false)}
          >
            Save
          </button>
        </div>
      </section>
      <details className="panel interval-references">
        <summary>Review history & exact timing</summary>
        <p>
          {e.id} · [{e.start}, {e.stop}) · PTS seconds {e.timing.start}–
          {e.timing.stop_exclusive}
        </p>
        {data.history.map((r, i) => (
          <p key={i}>
            {r.reviewed_at} · {r.source} · {statusName(r.status)} ·{" "}
            {r.payload.notes}
          </p>
        ))}
        <p>
          Unknown footage stays unknown. A punch-present decision does not
          certify complete event labels or timing.
        </p>
      </details>
    </>
  );
}
