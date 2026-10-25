import { useEffect, useState } from "react";
import { type Case, frameBounds, saveReview, title } from "./api";
import Viewer, { Notice } from "./Viewer";

export default function ReviewWorkbench({
  data,
  recording,
  onSaved,
  onDirty,
}: {
  data: Case;
  recording: string;
  onSaved: (next: boolean) => void;
  onDirty: (dirty: boolean) => void;
}) {
  const edit = data.edit?.data;
  const original = frameBounds(
    data.annotation,
    data.recording.data?.preparation.source_frame_count ?? 0,
  );
  const prior = edit?.history.at(-1);
  const savedProposal = edit?.decisions.at(-1)?.decision;
  const initialLabel =
    (savedProposal?.proposed_label as string | undefined) ??
    prior?.confirmed_label ??
    data.annotation.punch_type_provisional;
  const initialBounds =
    (savedProposal?.proposed_bounds as
      { start: number; end: number } | undefined) ?? prior?.confirmed_bounds;
  const initialStart = initialBounds?.start ?? original?.[0] ?? 0;
  const initialEnd = initialBounds?.end ?? original?.[1] ?? 0;
  const [jump, setJump] = useState<{ index: number; token: number }>();
  const [action, setAction] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [notes, setNotes] = useState("");
  const [uncertainty, setUncertainty] = useState("");
  const [stance, setStance] = useState("");
  const [label, setLabel] = useState(initialLabel);
  const [start, setStart] = useState(initialStart);
  const [end, setEnd] = useState(initialEnd);
  const [inspected, setInspected] = useState<number[]>([]);
  const [attest, setAttest] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [requestId, setRequestId] = useState(() => crypto.randomUUID());
  const dirty = !!(
    action ||
    notes ||
    uncertainty ||
    stance ||
    attest ||
    label !== initialLabel ||
    start !== initialStart ||
    end !== initialEnd
  );
  useEffect(() => {
    onDirty(dirty);
  }, [dirty, onDirty]);
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [dirty]);
  const valid =
    !!edit &&
    !!action &&
    !!reviewer.trim() &&
    !!notes.trim() &&
    attest &&
    Number.isInteger(start) &&
    Number.isInteger(end) &&
    start >= 0 &&
    end >= start &&
    end < (data.recording.data?.preparation.source_frame_count ?? 0) &&
    (action !== "confirm" ||
      (!!stance.trim() &&
        inspected.includes(start) &&
        inspected.includes(end))) &&
    (!["ambiguous", "hold"].includes(action) || !!uncertainty.trim());
  async function save(next: boolean) {
    if (!valid || !edit) return;
    setBusy(true);
    setError("");
    try {
      await saveReview(recording, data.annotation.source_row, {
        request_id: requestId,
        expected_ledger_sha256: edit.ledger_sha256,
        expected_revision_sha256: edit.revision_sha256,
        source_candidate_sha256: edit.source_candidate_sha256,
        action,
        reviewer,
        notes,
        uncertainty,
        lead_rear_evidence: stance,
        proposed_label: label,
        proposed_bounds: {
          start,
          end,
          convention: "zero_based_decoded_inclusive",
        },
        inspected_frames: inspected,
        attest_human: attest,
      });
      onDirty(false);
      onSaved(next);
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  }
  function changed() {
    setRequestId(crypto.randomUUID());
    setAttest(false);
  }
  return (
    <>
      <Viewer
        data={data}
        jump={jump}
        recording={recording}
        onVisible={(frame) =>
          setInspected((previous) =>
            previous.includes(frame.frame_index)
              ? previous
              : [...previous, frame.frame_index].slice(-512),
          )
        }
        onBoundary={(kind, index) => {
          changed();
          (kind === "start" ? setStart : setEnd)(index);
        }}
      />
      <section className="panel padded review-editor" onChange={changed}>
        <span className="eyebrow">
          HUMAN REVIEW · VERSION {edit?.ledger_version ?? "UNAVAILABLE"}
        </span>
        <h3>Confirm or propose a correction</h3>
        <p>
          Original: {title(data.annotation.punch_type_provisional)} · raw frames{" "}
          {data.annotation.raw_start}–{data.annotation.raw_end}. Model
          predictions and automated observations are not human reviews.
        </p>
        {!edit && (
          <Notice>
            Review source is unavailable or stale. Saving is disabled.
          </Notice>
        )}
        <div className="button-group">
          {["confirm", "ambiguous", "hold", "exclude"].map((a) => (
            <button
              key={a}
              disabled={busy}
              aria-pressed={action === a}
              className={action === a ? "selected" : ""}
              onClick={() => {
                changed();
                setAction(a);
              }}
            >
              {title(a).replace(/^./, (s) => s.toUpperCase())}
            </button>
          ))}
        </div>
        <div className="review-fields">
          <label>
            Proposed punch label
            <select value={label} onChange={(e) => setLabel(e.target.value)}>
              {data.labels.map((l) => (
                <option key={l} value={l}>
                  {title(l)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Proposed start frame
            <input
              type="number"
              min={0}
              value={start}
              onChange={(e) => setStart(Number(e.target.value))}
            />
          </label>
          <label>
            Proposed end frame (inclusive)
            <input
              type="number"
              min={start}
              value={end}
              onChange={(e) => setEnd(Number(e.target.value))}
            />
          </label>
          <label>
            Reviewer
            <input
              value={reviewer}
              onChange={(e) => setReviewer(e.target.value)}
              maxLength={160}
            />
          </label>
          <label>
            Evidence and notes
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              maxLength={4000}
            />
          </label>
          <label>
            Lead/rear or stance evidence (required to confirm)
            <textarea
              value={stance}
              onChange={(e) => setStance(e.target.value)}
              maxLength={2000}
            />
          </label>
          <label>
            Unresolved uncertainty (required for ambiguous/hold)
            <textarea
              value={uncertainty}
              onChange={(e) => setUncertainty(e.target.value)}
              maxLength={4000}
            />
          </label>
        </div>
        <div className="button-group">
          <button onClick={() => setJump({ index: start, token: Date.now() })}>
            Inspect proposed start
          </button>
          <button onClick={() => setJump({ index: end, token: Date.now() })}>
            Inspect proposed end
          </button>
        </div>
        <p>
          Inspect both proposed boundary frames before confirming. Viewed start:{" "}
          {inspected.includes(start) ? "yes" : "no"}; end:{" "}
          {inspected.includes(end) ? "yes" : "no"}. Corrections remain separate
          from original data and require later dataset preparation.
        </p>
        <label className="toggle" onChange={(e) => e.stopPropagation()}>
          <input
            type="checkbox"
            checked={attest}
            onChange={(e) => setAttest(e.target.checked)}
          />{" "}
          I personally reviewed this case; these are my decisions and evidence,
          not automated certification.
        </label>
        {error && (
          <Notice>
            {error} Use browser reload to obtain the latest version; unsaved
            changes will be discarded only after confirmation.
          </Notice>
        )}
        <div className="button-group">
          <button disabled={!valid || busy} onClick={() => void save(false)}>
            {busy ? "Saving…" : "Save review"}
          </button>
          <button disabled={!valid || busy} onClick={() => void save(true)}>
            Save & next annotation
          </button>
        </div>
        <details>
          <summary>
            Review history ({edit?.history.length ?? 0} revisions)
          </summary>
          {edit?.history.map((r, i) => (
            <article key={r.revision_id ?? i}>
              <h4>
                {r.actor_type === "human"
                  ? "Human decision"
                  : "Automated observation"}{" "}
                · {r.disposition}
              </h4>
              <p>
                {r.reviewer} · {r.reviewed_at}
              </p>
              <p>
                {r.confirmed_label ? title(r.confirmed_label) : r.label_status}{" "}
                ·{" "}
                {r.confirmed_bounds
                  ? `${r.confirmed_bounds.start}–${r.confirmed_bounds.end} inclusive`
                  : r.boundary_status}
              </p>
              <p>{r.reasons.join(" · ")}</p>
              <p>{r.unresolved_uncertainty.join(" · ")}</p>
            </article>
          ))}
          {edit?.decisions.map((r, i) => (
            <details key={i}>
              <summary>Saved proposal and audit inputs · {r.saved_at}</summary>
              <pre>{JSON.stringify(r.decision, null, 2)}</pre>
            </details>
          ))}
        </details>
      </section>
    </>
  );
}
