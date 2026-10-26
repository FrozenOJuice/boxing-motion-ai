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
import { useReviewer } from "./reviewer";

const uncertaintyReasons = [
  ["punch", "Punch type is unclear", "ambiguous"],
  ["arm", "Lead or rear arm is unclear", "ambiguous"],
  ["timing", "Start or end is unclear", "ambiguous"],
  ["mixed", "More than one punch in the interval", "ambiguous"],
  ["visibility", "Movement is hard to see", "hold"],
  ["later", "Needs another review", "hold"],
  ["other", "Other reason", "hold"],
] as const;
const confirmationAttestation =
  "I personally verified the punch label, the lead/rear arm from the boxer’s stance, and the start/end frames.";
const uncertaintyAttestation =
  "I personally reviewed this punch and selected the reason above.";

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
  const initialStart = initialBounds?.start ?? original?.[0] ?? -1;
  const initialEnd = initialBounds?.end ?? original?.[1] ?? -1;
  const [jump, setJump] = useState<{ index: number; token: number }>();
  const [choice, setChoice] = useState("");
  const [reason, setReason] = useState("");
  const [exclude, setExclude] = useState(false);
  const selectedReason = uncertaintyReasons.find((r) => r[0] === reason);
  const action =
    choice === "yes" || choice === "change"
      ? "confirm"
      : selectedReason
        ? exclude
          ? "exclude"
          : selectedReason[2]
        : "";
  const [visibleFrame, setVisibleFrame] = useState<Frame>();
  const identity = useReviewer();
  const { reviewer } = identity;
  const [notes, setNotes] = useState("");

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
    reason ||
    exclude ||
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
  const correcting =
    choice === "change" &&
    (label !== data.annotation.punch_type_provisional ||
      start !== original?.[0] ||
      end !== original?.[1]);
  const boundsValid =
    Number.isInteger(start) &&
    Number.isInteger(end) &&
    start >= 0 &&
    end >= start &&
    end < (data.recording.data?.preparation.source_frame_count ?? 0);
  const validation = !edit
    ? "Review data is unavailable. Reload this case."
    : !reviewer.trim()
      ? "Enter your name once to save reviews."
      : reviewer.trim().length > 160
        ? "Your name must be at most 160 characters."
        : !action
          ? "Choose why this punch needs another look."
          : action === "confirm" && !data.labels.includes(label)
            ? "Choose a punch label."
            : action === "confirm" && !boundsValid
              ? "Enter whole-number start/end frames within the recording, with end at or after start."
              : correcting && !notes.trim()
                ? "Briefly explain your correction."
                : choice === "unsure" && reason === "other" && !notes.trim()
                  ? "Add a short explanation for Other reason."
                  : action === "confirm" &&
                      (!inspected.includes(start) || !inspected.includes(end))
                    ? "View the start and end frames before confirming."
                    : !attest
                      ? "Check the statement below only after you have verified it."
                      : "";
  const valid = !!choice && !validation;
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
        reviewer: reviewer.trim(),
        notes: [
          action === "confirm"
            ? `${choice === "yes" ? "Reviewer selected Yes for the original annotation" : "Reviewer proposed this annotation"}: ${label}, frames ${start}–${end} inclusive. Explicit attestation: ${confirmationAttestation}`
            : `Reviewer selected: ${selectedReason?.[1]}.${exclude ? " Explicitly excluded this example from future candidates." : ""}`,
          notes.trim() ? `Reviewer note: ${notes.trim()}` : "",
        ]
          .filter(Boolean)
          .join("\n"),
        uncertainty:
          choice === "unsure"
            ? [selectedReason?.[1], notes.trim()].filter(Boolean).join(" · ")
            : "",
        lead_rear_evidence:
          action === "confirm"
            ? `Reviewer explicitly attested: ${confirmationAttestation}`
            : "",
        proposed_label: action === "confirm" ? label : null,
        proposed_bounds:
          action === "confirm"
            ? { start, end, convention: "zero_based_decoded_inclusive" }
            : null,
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
    setReason("");
    setExclude(false);
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
            {choice === "yes" && (
              <p className="confirmation-scope">
                Confirm <strong>{title(label)}</strong>, frames {start}–{end}{" "}
                (end included).
              </p>
            )}
            <div className="reviewer-identity">
              {identity.editing ? (
                <label>
                  Your name
                  <input
                    value={reviewer}
                    maxLength={160}
                    onChange={(e) => identity.update(e.target.value)}
                    onBlur={() => {
                      if (reviewer.trim()) identity.setEditing(false);
                    }}
                    autoComplete="name"
                  />
                </label>
              ) : (
                <span>
                  Reviewing as <strong>{reviewer}</strong>{" "}
                  <button onClick={() => identity.setEditing(true)}>
                    Change name
                  </button>
                </span>
              )}
              {identity.storageError && (
                <p role="alert">{identity.storageError}</p>
              )}
            </div>
            {choice === "unsure" && (
              <>
                <label className="decision-reason">
                  Why are you not sure?
                  <select
                    value={reason}
                    onChange={(e) => setReason(e.target.value)}
                  >
                    <option value="">Choose a reason…</option>
                    {uncertaintyReasons.map(([value, text]) => (
                      <option key={value} value={value}>
                        {text}
                      </option>
                    ))}
                  </select>
                </label>
                <details className="review-more-options">
                  <summary>More options</summary>
                  <label className="toggle">
                    <input
                      type="checkbox"
                      checked={exclude}
                      onChange={(e) => setExclude(e.target.checked)}
                    />
                    Exclude this example from future candidates
                  </label>
                </details>
              </>
            )}
            {correcting || (choice === "unsure" && reason === "other") ? (
              <label className="review-note">
                {correcting ? "Why change it?" : "Additional explanation"}
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  maxLength={2000}
                />
              </label>
            ) : (
              <details className="optional-notes">
                <summary>Add a note (optional)</summary>
                <label className="review-note">
                  Additional explanation (optional)
                  <textarea
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    maxLength={2000}
                  />
                </label>
              </details>
            )}
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
              {action === "confirm"
                ? confirmationAttestation
                : uncertaintyAttestation}
            </label>
          </fieldset>
        )}
        {error && (
          <div role="alert" className="notice">
            {error}
          </div>
        )}
        <div className="review-save">
          <small>{choice ? validation : ""}</small>
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
