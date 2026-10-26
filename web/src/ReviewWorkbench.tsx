import { useEffect, useState } from "react";
import {
  type Case,
  type Frame,
  type VerificationChecks,
  frameBounds,
  saveReview,
  title,
  percent,
  reviewSource,
} from "./api";
import Viewer, { Notice } from "./Viewer";

const uncertaintyReasons = [
  ["punch", "Punch type is unclear", "ambiguous"],
  ["arm", "Lead or rear arm is unclear", "ambiguous"],
  ["timing", "Start or end is unclear", "ambiguous"],
  ["mixed", "More than one punch in the interval", "ambiguous"],
  ["visibility", "Movement is hard to see", "hold"],
  ["later", "Needs another review", "hold"],
  ["other", "Other reason", "hold"],
] as const;
const noChecks: VerificationChecks = {
  punch_type: false,
  hand: false,
  timing: false,
};
const checkLabels = {
  punch_type: "Punch type is correct.",
  hand: "Lead/rear hand is correct.",
  timing: "Start/end timing is correct.",
} as const;
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
      ? "verify"
      : selectedReason
        ? exclude
          ? "exclude"
          : selectedReason[2]
        : "";
  const [visibleFrame, setVisibleFrame] = useState<Frame>();
  const [notes, setNotes] = useState("");

  const [label, setLabel] = useState(initialLabel);
  const [start, setStart] = useState(initialStart);
  const [end, setEnd] = useState(initialEnd);
  const [inspected, setInspected] = useState<number[]>([]);
  const [attest, setAttest] = useState(false);
  const [checks, setChecks] = useState<VerificationChecks>(noChecks);
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
  function previousChecks(
    targetLabel: string,
    targetStart: number,
    targetEnd: number,
  ): VerificationChecks {
    if (prior?.actor_type !== "human") return noChecks;
    const target =
      targetStart >= 0 && targetEnd >= targetStart
        ? { start: targetStart, end: targetEnd }
        : null;
    const saved = prior.verification;
    const priorLabel =
      saved?.label ??
      prior.confirmed_label ??
      data.annotation.punch_type_provisional;
    const priorBounds = saved ? saved.bounds : prior.confirmed_bounds;
    if (
      priorLabel !== targetLabel ||
      (priorBounds?.start ?? null) !== (target?.start ?? null) ||
      (priorBounds?.end ?? null) !== (target?.end ?? null)
    )
      return noChecks;
    return saved
      ? { punch_type: saved.punch_type, hand: saved.hand, timing: saved.timing }
      : {
          punch_type: prior.label_status === "confirmed",
          hand: prior.label_status === "confirmed",
          timing: prior.boundary_status === "confirmed",
        };
  }
  const inherited = previousChecks(label, start, end);
  const count = Object.values(checks).filter(Boolean).length;
  const priorCount = Object.values(inherited).filter(Boolean).length;
  const newlyCheckingMovement =
    (checks.punch_type && !inherited.punch_type) ||
    (checks.hand && !inherited.hand);
  const validation = !edit
    ? "Review data is unavailable. Reload this case."
    : !action
      ? "Choose why this punch needs another look."
      : action === "verify" && !data.labels.includes(label)
        ? "Choose a punch label."
        : action === "verify" &&
            (checks.timing || choice === "change") &&
            !boundsValid
          ? "Enter whole-number start/end frames within the recording, with end at or after start."
          : correcting && !notes.trim()
            ? "Briefly explain your correction."
            : choice === "unsure" && reason === "other" && !notes.trim()
              ? "Add a short explanation for Other reason."
              : action === "verify" && !count && !priorCount
                ? "Select the aspects you verified. You can save just one."
                : action === "verify" &&
                    checks.timing &&
                    !inherited.timing &&
                    (!inspected.includes(start) || !inspected.includes(end))
                  ? "View the start and end frames before verifying timing."
                  : action === "verify" &&
                      newlyCheckingMovement &&
                      !inspected.some(
                        (i) => !boundsValid || (i >= start && i <= end),
                      )
                    ? "Display a frame within the punch before verifying its type or hand."
                    : action !== "verify" && !attest
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
        verification:
          action === "verify"
            ? { schema_version: "annotation-verification-1.0", ...checks }
            : undefined,
        notes: [
          action === "verify"
            ? `Selected manual verification state for ${label}: ${Object.entries(
                checks,
              )
                .map(([axis, value]) => `${axis}=${value}`)
                .join(", ")}.`
            : `Reviewer selected: ${selectedReason?.[1]}.${exclude ? " Explicitly excluded this example from future candidates." : ""}`,
          notes.trim() ? `Reviewer note: ${notes.trim()}` : "",
        ]
          .filter(Boolean)
          .join("\n"),
        uncertainty:
          choice === "unsure"
            ? [selectedReason?.[1], notes.trim()].filter(Boolean).join(" · ")
            : "",
        lead_rear_evidence: "",
        proposed_label: action === "verify" ? label : null,
        proposed_bounds:
          action === "verify" && boundsValid
            ? { start, end, convention: "zero_based_decoded_inclusive" }
            : null,
        inspected_frames: inspected,
        attest_human: action === "verify" ? true : attest,
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
    setChecks(
      value === "yes"
        ? previousChecks(
            data.annotation.punch_type_provisional,
            original?.[0] ?? -1,
            original?.[1] ?? -1,
          )
        : value === "change"
          ? previousChecks(initialLabel, initialStart, initialEnd)
          : noChecks,
    );
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
                setChecks(noChecks);
                (kind === "start" ? setStart : setEnd)(index);
              }
            : undefined
        }
      />
      <section className="decision-panel" aria-label="Review decision">
        <h3>Is this punch correct?</h3>
        {data.review.data?.verification_status &&
          data.review.data.verification_status !== "unverified" && (
            <p className="saved-verification">
              Saved:{" "}
              {data.review.data.verification_status === "fully_verified"
                ? "Fully verified"
                : "Partially verified"}
            </p>
          )}
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
                    onChange={(e) => {
                      setChecks(noChecks);
                      setLabel(e.target.value);
                    }}
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
                    onChange={(e) => {
                      setChecks(noChecks);
                      setStart(Number(e.target.value));
                    }}
                  />
                </label>
                <label>
                  End frame (included)
                  <input
                    type="number"
                    min={start}
                    value={end}
                    onChange={(e) => {
                      setChecks(noChecks);
                      setEnd(Number(e.target.value));
                    }}
                  />
                </label>
              </div>
            )}
            {choice === "yes" && (
              <p className="confirmation-scope">
                Review <strong>{title(label)}</strong>, frames {start}–{end}{" "}
                (end included).
              </p>
            )}
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
            {action === "verify" ? (
              <div className="independent-checks">
                {(Object.keys(checkLabels) as (keyof VerificationChecks)[]).map(
                  (axis) => {
                    return (
                      <label key={axis} className="toggle">
                        <input
                          type="checkbox"
                          aria-label={checkLabels[axis]}
                          checked={checks[axis]}
                          onChange={(e) =>
                            setChecks({ ...checks, [axis]: e.target.checked })
                          }
                        />
                        <span>
                          {checkLabels[axis]}
                          {inherited[axis] && checks[axis] && (
                            <small>Previously verified manually</small>
                          )}
                        </span>
                      </label>
                    );
                  },
                )}
                <p>
                  {count === 3
                    ? "All three selected"
                    : count
                      ? `Partially verified · ${count}/3 selected`
                      : "No confirmations selected"}
                </p>
                {checks.punch_type && !checks.hand && (
                  <small>
                    The six-class label, including jab/cross, needs lead/rear
                    hand verification too.
                  </small>
                )}
                <div className="boundary-checks">
                  <button
                    onClick={() => setJump({ index: start, token: Date.now() })}
                  >
                    View start
                    {inspected.includes(start) || inherited.timing ? " ✓" : ""}
                  </button>
                  <button
                    onClick={() => setJump({ index: end, token: Date.now() })}
                  >
                    View end
                    {inspected.includes(end) || inherited.timing ? " ✓" : ""}
                  </button>
                </div>
              </div>
            ) : (
              <label
                className="toggle attestation"
                onChange={(e) => e.stopPropagation()}
              >
                <input
                  type="checkbox"
                  checked={attest}
                  onChange={(e) => setAttest(e.target.checked)}
                />
                {uncertaintyAttestation}
              </label>
            )}
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
                {reviewSource(r) === "manual"
                  ? "Manual review"
                  : "Automated observation"}{" "}
                · {r.disposition}
              </h4>
              <p>
                {r.reviewed_at}
                {r.reviewer ? ` · Legacy reviewer: ${r.reviewer}` : ""}
              </p>
              <p>
                {r.confirmed_label ? title(r.confirmed_label) : r.label_status}{" "}
                ·{" "}
                {r.confirmed_bounds
                  ? `${r.confirmed_bounds.start}–${r.confirmed_bounds.end} inclusive`
                  : r.boundary_status}
              </p>
              {r.verification && (
                <p>
                  {Object.entries(checkLabels)
                    .map(
                      ([axis, text]) =>
                        `${text} ${r.verification?.[axis as keyof VerificationChecks] ? "Yes" : "Unverified"}`,
                    )
                    .join(" · ")}
                </p>
              )}
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
