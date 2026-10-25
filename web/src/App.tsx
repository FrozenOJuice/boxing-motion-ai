import { useEffect, useMemo, useState } from "react";
import {
  type Case,
  type Experiments,
  type Inventory,
  type Result,
  type Reviews,
  percent,
  title,
  useRequest,
} from "./api";
import { Notice } from "./Viewer";
import ReviewWorkbench from "./ReviewWorkbench";
import { Dataset, ExperimentsView, Models } from "./Overview";
const sections = ["Review Queue", "Dataset", "Experiments", "Models"] as const;
type Section = (typeof sections)[number];
function NavIcon({ index }: { index: number }) {
  return (
    <svg
      width="19"
      height="19"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      aria-hidden="true"
    >
      {index === 0 ? (
        <>
          <rect x="4" y="3" width="16" height="18" rx="3" />
          <path d="M8 8h8M8 12h8M8 16h4" />
        </>
      ) : index === 1 ? (
        <>
          <ellipse cx="12" cy="5" rx="8" ry="3" />
          <path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0" />
        </>
      ) : index === 2 ? (
        <>
          <path d="M9 3h6M10 3v7L4 20h16l-6-10V3M7 16h10" />
        </>
      ) : (
        <>
          <rect x="5" y="5" width="14" height="14" rx="3" />
          <path d="M9 9h6v6H9zM9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m14 0h3M2 15h3m14 0h3" />
        </>
      )}
    </svg>
  );
}
export default function App() {
  const [section, setSection] = useState<Section>("Review Queue");
  const [refresh, setRefresh] = useState(0);
  const [dirty, setDirty] = useState(false);
  const [saved, setSaved] = useState("");
  const queue = useRequest<Result<Reviews>>("/api/reviews", refresh);
  const inventory = useRequest<Result<Inventory>>("/api/inventory");
  const experiments = useRequest<Result<Experiments>>("/api/experiments");
  const [selected, setSelected] = useState("");
  const [recording, setRecording] = useState("all");
  const [status, setStatus] = useState("all");
  const [search, setSearch] = useState("");
  const entries = queue.data?.data?.entries ?? [];
  const filtered = useMemo(
    () =>
      entries.filter((e) => {
        const source = e.review.data?.source;
        return (
          (recording === "all" || e.id.split(":")[0] === recording) &&
          (status === "all" ||
            (status === "dispute"
              ? e.priority === "dispute"
              : status === "anchor"
                ? e.priority === "anchor"
                : status === "ambiguous"
                  ? e.review.data?.latest_action === "ambiguous"
                  : e.review.data?.latest?.disposition === status)) &&
          `${e.id} ${source?.source_class ?? ""} ${source?.punch_type_provisional ?? ""}`
            .toLowerCase()
            .includes(search.toLowerCase())
        );
      }),
    [queue.data, recording, status, search],
  );
  useEffect(() => {
    if (filtered.length && !filtered.some((e) => e.id === selected))
      setSelected(filtered[0].id);
    if (queue.data && !filtered.length) setSelected("");
  }, [filtered, selected]);
  const current = useRequest<Result<Case>>(
    selected ? `/api/cases/${selected.replace(":", "/")}` : null,
    refresh,
  );
  const navigate = (id: string) => {
    if (!dirty || window.confirm("Discard unsaved review changes?")) {
      setDirty(false);
      setSelected(id);
    }
  };
  const data = current.data?.data;
  const position = filtered.findIndex((e) => e.id === selected);
  const selectedEntry = entries.find((e) => e.id === selected);
  const prediction = data?.evidence.data?.case.prediction;
  const confidence = prediction
    ? prediction.probabilities[
        data?.labels.indexOf(prediction.prediction) ?? -1
      ]
    : undefined;
  const review = data?.review.data?.latest;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            B<span>↗</span>
          </div>
          <div>
            BOXING MOTION<span>AI DEVELOPMENT STUDIO</span>
          </div>
        </div>
        <div className="workspace-label">
          WORKSPACE <span>LOCAL</span>
        </div>
        <nav aria-label="Main navigation">
          {sections.map((s, i) => (
            <button
              key={s}
              className={section === s ? "active" : ""}
              aria-current={section === s ? "page" : undefined}
              onClick={() => {
                if (
                  !dirty ||
                  window.confirm("Discard unsaved review changes?")
                ) {
                  setDirty(false);
                  setSection(s);
                }
              }}
            >
              <NavIcon index={i} />
              {s}
              {s === "Review Queue" && queue.data?.data && (
                <small>{queue.data.data.summary.case_count}</small>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="connection">
            <i />
            Local workspace
          </div>
          <p>
            Read-only review
            <br />
            Original data preserved
          </p>
          <div className="version">
            DASHBOARD <b>v0.2</b>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <span>
            BOXINGVI <span className="slash">/</span> {section}
          </span>
          <div>
            <span className="status-dot" />
            LOCAL SESSION{" "}
            <span className="read-only">▣ INTERACTIVE REVIEW</span>
          </div>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                DEVELOPMENT /{" "}
                {section === "Review Queue"
                  ? "ANNOTATION CURATION"
                  : "RESEARCH INVENTORY"}
              </div>
              <h1>
                {section === "Review Queue"
                  ? "Annotation review"
                  : section === "Dataset"
                    ? "Dataset overview"
                    : section === "Experiments"
                      ? "Experiment library"
                      : "Model reference"}
              </h1>
              <p>
                {section === "Review Queue"
                  ? "Inspect the movement. Preserve the evidence."
                  : section === "Dataset"
                    ? "Source coverage, class balance and review readiness."
                    : section === "Experiments"
                      ? "Reproducible studies. Preserved results."
                      : "Frozen model results, with their evaluation context."}
              </p>
            </div>
            <span className="outline-badge">● V9 RESERVED</span>
          </div>
          {(queue.error || inventory.error) && (
            <Notice>
              Could not load project metadata: {queue.error || inventory.error}.
              Start the local backend and check registered artifacts.
            </Notice>
          )}
          {section === "Review Queue" ? (
            <>
              <div className="review-summary">
                <span>
                  <b>{queue.data?.data?.summary.case_count ?? "—"}</b>{" "}
                  diagnostic cases
                </span>
                <span>
                  <b>
                    {entries.filter((e) => e.priority === "dispute").length}
                  </b>{" "}
                  priority disputes
                </span>
                <span>
                  <b>{queue.data?.data?.summary.human_accepted ?? "—"}</b> human
                  accepted
                </span>
                <span>
                  <b>{queue.data?.data?.summary.human_reviewed ?? 0}</b> human
                  reviewed (includes unresolved)
                </span>
                <span className="muted-text">
                  Corrections are separately versioned; source annotations stay
                  unchanged
                </span>
              </div>
              <div className="review-layout">
                <section className="queue-panel panel">
                  <div className="panel-header">
                    <h2>Review queue</h2>
                    <span className="count">{filtered.length}</span>
                  </div>
                  <div className="queue-filters">
                    <input
                      aria-label="Search cases"
                      placeholder="Search class or source row…"
                      value={search}
                      onChange={(e) => {
                        if (
                          !dirty ||
                          window.confirm("Discard unsaved review changes?")
                        ) {
                          setDirty(false);
                          setSearch(e.target.value);
                        }
                      }}
                    />
                    <div>
                      <select
                        aria-label="Filter recording"
                        value={recording}
                        onChange={(e) => {
                          if (
                            !dirty ||
                            window.confirm("Discard unsaved review changes?")
                          ) {
                            setDirty(false);
                            setRecording(e.target.value);
                          }
                        }}
                      >
                        <option value="all">All recordings</option>
                        {Object.keys(inventory.data?.data?.records ?? {}).map(
                          (v) => (
                            <option key={v}>{v}</option>
                          ),
                        )}
                      </select>
                      <select
                        aria-label="Filter review state"
                        value={status}
                        onChange={(e) => {
                          if (
                            !dirty ||
                            window.confirm("Discard unsaved review changes?")
                          ) {
                            setDirty(false);
                            setStatus(e.target.value);
                          }
                        }}
                      >
                        <option value="all">All states</option>
                        <option value="anchor">Proposed anchors</option>
                        <option value="ambiguous">Human ambiguous</option>
                        <option value="dispute">Priority disputes</option>
                        <option value="hold">On hold</option>
                        <option value="provisional">Provisional</option>
                        <option value="accept">Accepted</option>
                        <option value="exclude">Excluded</option>
                      </select>
                    </div>
                  </div>
                  <div className="case-list">
                    {!queue.data && !queue.error && (
                      <Notice>Loading review ledger…</Notice>
                    )}
                    {!filtered.length && queue.data && (
                      <Notice>No cases match these filters.</Notice>
                    )}
                    {filtered.map((e) => (
                      <button
                        className={`case-item ${selected === e.id ? "selected" : ""}`}
                        key={e.id}
                        onClick={() => navigate(e.id)}
                      >
                        <div>
                          <span className="case-recording">
                            {e.id.split(":")[0]}
                          </span>
                          <span
                            className={`case-status ${e.priority === "dispute" ? "amber" : ""}`}
                          >
                            {e.review.data?.latest?.actor_type === "human"
                              ? (
                                  e.review.data.latest_action ??
                                  e.review.data.latest.disposition
                                ).toUpperCase()
                              : e.priority === "dispute"
                                ? "DISPUTED"
                                : (e.review.data?.latest?.disposition?.toUpperCase() ??
                                  e.review.status.toUpperCase())}
                          </span>
                        </div>
                        <strong>
                          {title(
                            e.review.data?.source.punch_type_provisional ??
                              "Unavailable",
                          )}
                        </strong>
                        <div>
                          <span>Source row {e.id.split(":")[1]}</span>
                          <span>↗</span>
                        </div>
                      </button>
                    ))}
                  </div>
                </section>
                <div className="review-main">
                  {saved && <Notice>{saved}</Notice>}
                  {!selected ? (
                    <Notice>
                      Select an annotation case to inspect its source frames.
                    </Notice>
                  ) : current.error ? (
                    <Notice>Case unavailable: {current.error}</Notice>
                  ) : !data ? (
                    <Notice>Loading source annotation…</Notice>
                  ) : (
                    <>
                      <div className="case-heading">
                        <div>
                          <span className="eyebrow">
                            {selected.split(":")[0]}{" "}
                            <span className="slash">/</span> SOURCE ROW{" "}
                            {data.annotation.source_row}
                          </span>
                          <h2>
                            {title(data.annotation.punch_type_provisional)}{" "}
                            <span className="badge amber">
                              {review?.actor_type === "human"
                                ? title(
                                    data.review.data?.latest_action ??
                                      review.disposition,
                                  )
                                : selectedEntry?.priority === "dispute"
                                  ? "Disputed"
                                  : review?.disposition === "hold"
                                    ? "On hold"
                                    : review?.disposition === "exclude"
                                      ? "Excluded"
                                      : data.review.data?.human_accepted
                                        ? "Human accepted"
                                        : "Provisional"}
                            </span>
                          </h2>
                        </div>
                        <div className="button-group">
                          <button
                            aria-label="Previous annotation"
                            disabled={position <= 0}
                            onClick={() => navigate(filtered[position - 1].id)}
                          >
                            ←
                          </button>
                          <span>
                            {position + 1} / {filtered.length}
                          </span>
                          <button
                            aria-label="Next annotation"
                            disabled={
                              position < 0 || position >= filtered.length - 1
                            }
                            onClick={() => navigate(filtered[position + 1].id)}
                          >
                            →
                          </button>
                        </div>
                      </div>
                      <ReviewWorkbench
                        key={`${selected}:${refresh}`}
                        onDirty={setDirty}
                        onSaved={(next) => {
                          setDirty(false);
                          setSaved(
                            `Saved review for ${selected}. Original data unchanged.`,
                          );
                          if (next && filtered[position + 1])
                            setSelected(filtered[position + 1].id);
                          setRefresh((n) => n + 1);
                        }}
                        data={data}
                        recording={selected.split(":")[0]}
                      />
                      <section className="panel padded evidence-note">
                        <span className="eyebrow">REVIEW EVIDENCE</span>
                        <h3>
                          {review?.reasons.map(title).join(" · ") ||
                            "No review decision recorded"}
                        </h3>
                        <p>
                          {review?.unresolved_uncertainty.join(" · ") ||
                            "Labels and boundaries remain provisional."}
                        </p>
                        <div className="note-footer">
                          <span>
                            {review
                              ? `${title(review.actor_type)} observation · ${review.reviewer}`
                              : "Unreviewed source annotation"}
                          </span>
                          <span>Human verification required</span>
                        </div>
                      </section>
                    </>
                  )}
                </div>
                <aside className="detail-panel panel">
                  <div className="panel-header">
                    <h2>Annotation details</h2>
                  </div>
                  {data ? (
                    <div className="detail-content">
                      <span className="eyebrow">SOURCE ANNOTATION</span>
                      <h3>{data.annotation.source_class}</h3>
                      <dl>
                        <dt>Canonical label</dt>
                        <dd>{title(data.annotation.punch_type_provisional)}</dd>
                        <dt>Raw start / end</dt>
                        <dd className="mono">
                          {data.annotation.raw_start ?? "Unknown"} →{" "}
                          {data.annotation.raw_end ?? "Unknown"}
                        </dd>
                        <dt>Mapped frames</dt>
                        <dd className="mono">
                          {data.annotation.candidate_start_index ?? "Unknown"} →{" "}
                          {data.annotation.candidate_end_index ?? "Unknown"}
                        </dd>
                        <dt>End convention</dt>
                        <dd>{data.annotation.annotation_end_assumption}</dd>
                        <dt>Review state</dt>
                        <dd className="amber-text">
                          {title(review?.disposition ?? data.review.status)}
                        </dd>
                        <dt>Label / boundary</dt>
                        <dd>
                          {title(review?.label_status ?? "unreviewed")} /{" "}
                          {title(review?.boundary_status ?? "unreviewed")}
                        </dd>
                      </dl>
                      <hr />
                      <span className="eyebrow">FROZEN MODEL PREDICTION</span>
                      <h3>
                        {prediction
                          ? title(prediction.prediction)
                          : "Unavailable"}
                      </h3>
                      <div className="confidence-row">
                        <span>Class probability</span>
                        <b>{percent(confidence)}</b>
                      </div>
                      <div className="progress-track">
                        <i style={{ width: `${(confidence ?? 0) * 100}%` }} />
                      </div>
                      <p className="caption">
                        {data.evidence.data?.case.selection ??
                          "No diagnostic prediction registered for this case."}{" "}
                        · {data.evidence.data?.case.selection_fold ?? "No fold"}
                        <br />
                        Model confidence is not annotation evidence.
                      </p>
                      <hr />
                      <span className="eyebrow">PROVENANCE & UNCERTAINTY</span>
                      <div className="flag-list">
                        {[
                          ...(data.annotation.flags ?? []),
                          ...(data.annotation.exclusion_reasons ?? []),
                        ].map((flag, i) => (
                          <span key={`${flag}-${i}`}>{title(flag)}</span>
                        ))}
                      </div>
                      <p className="caption">
                        Boxer: {data.recording.data?.boxer_id ?? "unknown"}
                        <br />
                        Session: {data.recording.data?.session_id ?? "unknown"}
                      </p>
                      <details>
                        <summary>Prediction provenance</summary>
                        <p className="break-all caption">
                          {data.evidence.data?.case.prediction_source ??
                            "Not available"}
                        </p>
                      </details>
                    </div>
                  ) : (
                    <Notice>Select a source annotation.</Notice>
                  )}
                </aside>
              </div>
            </>
          ) : section === "Dataset" ? (
            <Dataset
              inventory={inventory.data?.data ?? undefined}
              reviews={queue.data?.data ?? undefined}
            />
          ) : section === "Experiments" ? (
            <ExperimentsView
              experiments={experiments.data?.data ?? undefined}
            />
          ) : (
            <Models experiments={experiments.data?.data ?? undefined} />
          )}
          <footer>
            BOXING MOTION AI{" "}
            <span>
              Provisional data · exact-frame inspection · versioned human
              reviews
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}
