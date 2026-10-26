import { useEffect, useState } from "react";
import {
  type Case,
  type Frame,
  frameBounds,
  saveReview,
  title,
  percent,
} from "./api";
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
  const [choice, setChoice] = useState("");
  const [unresolvedAction, setUnresolvedAction] = useState("");
  const action =
    choice === "yes" || choice === "change" ? "confirm" : unresolvedAction;
  const [visibleFrame, setVisibleFrame] = useState<Frame>();
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
    choice ||
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
  function choose(value: string) {
    changed();
    setChoice(value);
    setUnresolvedAction("");
    if (value === "yes") {
      setLabel(data.annotation.punch_type_provisional);
      setStart(original?.[0] ?? -1);
      setEnd(original?.[1] ?? -1);
    }
    if (value === "change") {
      setLabel(initialLabel);
      setStart(initialStart);
      setEnd(initialEnd);
    }
  }
  const prediction = data.evidence.data?.case.prediction;
  const confidence =
    prediction?.probabilities[data.labels.indexOf(prediction.prediction)];
  return (
    <>
      <Viewer
        data={data}
        recording={recording}
        jump={jump}
        proposedBounds={choice === "change" ? [start, end] : undefined}
        onVisible={(frame) => {
          setVisibleFrame(frame);
          setInspected((previous) =>
            previous.includes(frame.frame_index)
              ? previous
              : [...previous, frame.frame_index].slice(-512),
          );
        }}
        onBoundary={
          choice === "change"
            ? (kind, index) => {
                changed();
                (kind === "start" ? setStart : setEnd)(index);
              }
            : undefined
        }
      />
      <section className="decision-panel" aria-label="Review decision">
        <h3>Is this punch correct?</h3>
        {!edit && (
          <Notice>Review data is unavailable. Saving is disabled.</Notice>
        )}
        <div className="decision-choices">
          {[
            ["yes", "Yes"],
            ["change", "Change it"],
            ["unsure", "Not sure"],
          ].map(([value, text]) => (
            <button
              key={value}
              aria-pressed={choice === value}
              disabled={busy}
              onClick={() => choose(value)}
            >
              {text}
            </button>
          ))}
        </div>
        {choice && (
          <fieldset
            disabled={busy}
            className="decision-form"
            onChange={changed}
          >
            {choice === "change" && (
              <div className="correction-fields review-fields">
                <label>
                  Correct punch label
                  <select
                    value={label}
                    onChange={(e) => setLabel(e.target.value)}
                  >
                    {data.labels.map((l) => (
                      <option key={l} value={l}>
                        {title(l)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Start frame
                  <input
                    type="number"
                    min={0}
                    value={start}
                    onChange={(e) => setStart(Number(e.target.value))}
                  />
                </label>
                <label>
                  End frame (included)
                  <input
                    type="number"
                    min={start}
                    value={end}
                    onChange={(e) => setEnd(Number(e.target.value))}
                  />
                </label>
              </div>
            )}
            {choice === "unsure" && (
              <label className="decision-reason">
                What should happen to this punch?
                <select
                  value={unresolvedAction}
                  onChange={(e) => setUnresolvedAction(e.target.value)}
                >
                  <option value="">Choose an outcome…</option>
                  <option value="ambiguous">
                    Can't tell what the punch is
                  </option>
                  <option value="hold">Leave it for a later review</option>
                  <option value="exclude">Don't use this punch</option>
                </select>
              </label>
            )}
            <div className="review-fields">
              <label>
                Your name
                <input
                  value={reviewer}
                  onChange={(e) => setReviewer(e.target.value)}
                  maxLength={160}
                />
              </label>
              <label>
                What did you observe?
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  maxLength={4000}
                />
              </label>
              {action === "confirm" && (
                <label>
                  How did you identify the lead or rear arm?
                  <textarea
                    value={stance}
                    onChange={(e) => setStance(e.target.value)}
                    maxLength={2000}
                  />
                  <small>
                    Use the boxer's stance, not screen left or right.
                  </small>
                </label>
              )}
              {["hold", "ambiguous"].includes(action) && (
                <label>
                  What is still uncertain?
                  <textarea
                    value={uncertainty}
                    onChange={(e) => setUncertainty(e.target.value)}
                    maxLength={4000}
                  />
                </label>
              )}
            </div>
            {action === "confirm" && (
              <div className="boundary-checks">
                <span>Check the boundaries</span>
                <button
                  onClick={() => setJump({ index: start, token: Date.now() })}
                >
                  View start{inspected.includes(start) ? " ✓" : ""}
                </button>
                <button
                  onClick={() => setJump({ index: end, token: Date.now() })}
                >
                  View end{inspected.includes(end) ? " ✓" : ""}
                </button>
              </div>
            )}
            <label
              className="toggle attestation"
              onChange={(e) => e.stopPropagation()}
            >
              <input
                type="checkbox"
                checked={attest}
                onChange={(e) => setAttest(e.target.checked)}
              />
              I personally reviewed this punch and the evidence above.
            </label>
          </fieldset>
        )}
        {error && <Notice>{error} Reload to review the latest version.</Notice>}
        <div className="review-save">
          <small>
            {choice && !valid
              ? "Complete your evidence, boundary checks and confirmation to save."
              : ""}
          </small>
          {choice && (
            <button disabled={!valid || busy} onClick={() => void save(false)}>
              Save
            </button>
          )}
          <button
            className="primary-action"
            disabled={!valid || busy}
            onClick={() => void save(true)}
          >
            {busy ? "Saving…" : "Save & Next"}
          </button>
        </div>
      </section>
      <details className="advanced-details">
        <summary>Advanced details</summary>
        <section>
          <h3>Original annotation</h3>
          <dl>
            <dt>Recording / source row</dt>
            <dd>
              {recording} / {data.annotation.source_row}
            </dd>
            <dt>Source label</dt>
            <dd>{data.annotation.source_class}</dd>
            <dt>Raw boundaries</dt>
            <dd>
              {data.annotation.raw_start}–{data.annotation.raw_end}
            </dd>
            <dt>Mapping</dt>
            <dd>{data.annotation.annotation_end_assumption}</dd>
            <dt>Detailed review state</dt>
            <dd>
              {prior?.disposition ?? "unreviewed"} · {prior?.label_status} /{" "}
              {prior?.boundary_status}
            </dd>
          </dl>
          <pre>{JSON.stringify(data.annotation, null, 2)}</pre>
        </section>
        <section>
          <h3>Model prediction</h3>
          <p>
            {prediction
              ? `${title(prediction.prediction)} · ${percent(confidence)}`
              : "Unavailable"}
          </p>
          <small>Model confidence is not annotation evidence.</small>
          <pre>{JSON.stringify(data.evidence, null, 2)}</pre>
        </section>
        <details>
          <summary>Frame timing and landmark confidence</summary>
          <p>
            Last displayed frame: {visibleFrame?.frame_index ?? "None"}.
            Coordinates and confidence retain source validity masks.
          </p>
          <pre>{JSON.stringify(visibleFrame ?? {}, null, 2)}</pre>
        </details>
        <details>
          <summary>Recording provenance and quality metadata</summary>
          <pre>{JSON.stringify(data.recording, null, 2)}</pre>
        </details>
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
              <details>
                <summary>Full revision</summary>
                <pre>{JSON.stringify(r, null, 2)}</pre>
              </details>
            </article>
          ))}
          {edit?.decisions.map((r, i) => (
            <details key={i}>
              <summary>Saved proposal · {r.saved_at}</summary>
              <pre>{JSON.stringify(r, null, 2)}</pre>
            </details>
          ))}
        </details>
        <p>
          Ledger version {edit?.ledger_version ?? "unavailable"}. Corrections
          are separately versioned; original data is unchanged.
        </p>
      </details>
    </>
  );
}
