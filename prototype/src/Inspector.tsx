import { useRef, type KeyboardEvent } from "react";
import {
  checkpoint,
  clock,
  punchName,
  type Selection,
  type VideoItem,
} from "./model";
import { Button, Icon, IconButton } from "./ui";
import { checkSelection, type Draft } from "./selection";
import type { InspectorTab } from "./navigation";
export default function Inspector({
  video,
  draft,
  result,
  busy,
  inspectionReady,
  tab,
  selections,
  onTab,
  onClear,
  onReplay,
  onClassify,
  onOpen,
}: {
  video: VideoItem;
  draft: Draft;
  result: Selection | null;
  busy: boolean;
  inspectionReady: boolean;
  tab: InspectorTab;
  selections: Selection[];
  onTab: (tab: InspectorTab) => void;
  onClear: () => void;
  onReplay: () => void;
  onClassify: () => void;
  onOpen: (s: Selection) => void;
}) {
  const checked = checkSelection(video, draft);
  const hasDraft = draft.start !== null || draft.end !== null;
  const tabs = useRef<Partial<Record<InspectorTab, HTMLButtonElement | null>>>(
    {},
  );
  function tabKey(e: KeyboardEvent, current: InspectorTab) {
    let next: InspectorTab;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight")
      next = current === "selection" ? "saved" : "selection";
    else if (e.key === "Home") next = "selection";
    else if (e.key === "End") next = "saved";
    else return;
    e.preventDefault();
    onTab(next);
    tabs.current[next]?.focus();
  }
  return (
    <aside
      data-tab={tab}
      className="analysis-panel"
      aria-label="Punch analysis"
    >
      <div
        className="inspector-tabs"
        role="tablist"
        aria-label="Punch inspector"
      >
        {(["selection", "saved"] as const).map((t) => (
          <button
            key={t}
            ref={(el) => {
              tabs.current[t] = el;
            }}
            id={`tab-${t}`}
            role="tab"
            type="button"
            aria-selected={tab === t}
            aria-controls={`panel-${t}`}
            tabIndex={tab === t ? 0 : -1}
            onClick={() => onTab(t)}
            onKeyDown={(e) => tabKey(e, t)}
          >
            {t === "selection"
              ? "Selection"
              : `Saved punches (${selections.length})`}
          </button>
        ))}
      </div>
      {hasDraft && (
        <>
          <div className="panel-heading">
            <h2>Selected range</h2>
            <IconButton
              icon="close"
              label="Clear selection"
              disabled={busy}
              onClick={onClear}
            />
          </div>
          <div className="selection-summary">
            <div className="selection-times">
              <div>
                <span>Start</span>
                <strong>
                  {draft.start === null ? "Not set" : clock(draft.start, true)}
                </strong>
              </div>
              <div>
                <span>End</span>
                <strong>
                  {draft.end === null ? "Not set" : clock(draft.end, true)}
                </strong>
              </div>
              <div>
                <span>Duration</span>
                <strong>
                  {checked.bounds
                    ? `${(checked.bounds.end - checked.bounds.start).toFixed(3)}s`
                    : "—"}
                </strong>
              </div>
            </div>
            <Button
              icon="play"
              disabled={!checked.bounds || busy}
              onClick={onReplay}
            >
              Replay selection
            </Button>
          </div>
          <div className="panel-action">
            <Button
              variant="primary"
              disabled={!checked.bounds || (!busy && !inspectionReady)}
              aria-disabled={busy || !checked.bounds || !inspectionReady}
              onClick={onClassify}
            >
              {busy ? "Loading example…" : "Classify punch"}
            </Button>
            <p className="selection-status" role="status">
              {checked.state !== "valid"
                ? checked.reason
                : !inspectionReady
                  ? "Loading source frame…"
                  : "Prototype · No model runs"}
            </p>
          </div>
        </>
      )}
      <div
        className="panel-content"
        id={`panel-${tab}`}
        role="tabpanel"
        aria-labelledby={`tab-${tab}`}
        tabIndex={0}
      >
        {tab === "saved" ? (
          <div className="saved-list">
            {selections.length ? (
              selections.map((s, i) => (
                <button
                  key={s.id}
                  disabled={busy}
                  aria-pressed={result?.id === s.id}
                  className={result?.id === s.id ? "is-active" : ""}
                  onClick={() => onOpen(s)}
                >
                  <span>
                    {s.kind === "recorded"
                      ? punchName(s.prediction)
                      : `Selection ${i + 1}`}
                  </span>
                  <small>
                    {clock(s.bounds.start, true)} – {clock(s.bounds.end, true)}
                  </small>
                  <small>
                    {s.kind === "recorded"
                      ? "Recorded model output"
                      : "Prototype example"}
                  </small>
                </button>
              ))
            ) : (
              <p>No saved punches yet.</p>
            )}
          </div>
        ) : (
          <>
            {!hasDraft && (
              <div className="selection-empty">
                <Icon name="bracket-in" size={24} />
                <h2>Select one punch</h2>
                <p>
                  Click the timeline to seek. Drag across a punch, or use Set
                  start and Set end.
                </p>
              </div>
            )}
            {(busy || result) && (
              <section
                className="prediction"
                aria-live="polite"
                aria-busy={busy}
              >
                {busy ? (
                  <p role="status">Loading example output…</p>
                ) : result ? (
                  <>
                    <div className="result-source">
                      {result.kind === "recorded"
                        ? "Saved model output"
                        : "Prototype example"}
                    </div>
                    <h3 className="prediction-title">
                      {punchName(result.prediction)}
                    </h3>
                    <p>
                      {result.kind === "recorded"
                        ? "Recorded output for this sample selection. No model ran."
                        : "No model ran for this selection."}
                    </p>
                    <div className="scores">
                      {Object.entries(result.scores)
                        .sort((a, b) => b[1] - a[1])
                        .map(([name, score]) => (
                          <div
                            className={`score ${name === result.prediction ? "score--top" : ""}`}
                            key={name}
                          >
                            <div>
                              <span>{punchName(name)}</span>
                              <span>{score.toFixed(3)}</span>
                            </div>
                            <div
                              className="score-track"
                              role="meter"
                              aria-label={punchName(name)}
                              aria-valuemin={0}
                              aria-valuemax={1}
                              aria-valuenow={score}
                            >
                              <span style={{ width: `${score * 100}%` }} />
                            </div>
                          </div>
                        ))}
                    </div>
                    <p className="score-note">
                      Uncalibrated model scores, not probabilities of
                      correctness.
                    </p>
                  </>
                ) : null}
              </section>
            )}
            {result && (
              <details className="model-details">
                <summary>Model information</summary>
                <p>Level 1 · Six punch categories</p>
                <p>
                  New selections use example output. Sample outputs are recorded
                  from the existing trained checkpoint.
                </p>
                <p>
                  Checkpoint <code>{checkpoint.slice(0, 12)}</code>
                </p>
                <p>
                  Experimental recognition. New-boxer performance is
                  unvalidated.
                </p>
              </details>
            )}
          </>
        )}
      </div>
    </aside>
  );
}
