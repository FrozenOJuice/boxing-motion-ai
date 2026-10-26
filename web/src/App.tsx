import { useCallback, useEffect, useMemo, useState } from "react";
import {
  type Case,
  type Experiments,
  type Inventory,
  type Result,
  type Reviews,
  reviewCategory,
  title,
  useRequest,
} from "./api";
import { Notice } from "./Viewer";
import ReviewWorkbench from "./ReviewWorkbench";
import IntervalReview from "./IntervalReview";
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
  const [reviewMode, setReviewMode] = useState("punches");
  const [section, setSection] = useState<Section>("Review Queue");
  const [refresh, setRefresh] = useState(0);
  const [dirty, setDirty] = useState(false);
  const [saved, setSaved] = useState("");
  const [saving, setSaving] = useState(false);
  const [retained, setRetained] = useState<{
    id: string;
    after: string[];
  } | null>(null);
  const handleDirty = useCallback((value: boolean) => {
    setDirty(value);
    if (value) setSaved("");
  }, []);
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
            (status === "dispute" || status === "anchor"
              ? e.priority === status
              : reviewCategory(e) === status)) &&
          `${e.id} ${source?.source_class ?? ""} ${source?.punch_type_provisional ?? ""}`
            .toLowerCase()
            .includes(search.toLowerCase())
        );
      }),
    [queue.data, recording, status, search],
  );
  useEffect(() => {
    if (retained?.id === selected && entries.some((e) => e.id === selected))
      return;
    if (filtered.length && !filtered.some((e) => e.id === selected))
      setSelected(filtered[0].id);
    if (queue.data && !filtered.length) setSelected("");
  }, [filtered, selected, retained, queue.data]);
  const current = useRequest<Result<Case>>(
    selected ? `/api/cases/${selected.replace(":", "/")}` : null,
    refresh,
  );
  const navigate = (id: string) => {
    if (saving || id === selected) return;
    if (!dirty || window.confirm("Discard unsaved review changes?")) {
      setDirty(false);
      setSaved("");
      setRetained(null);
      setSelected(id);
    }
  };
  const data = current.data?.data;
  const position = filtered.findIndex((e) => e.id === selected);
  return (
    <div
      className={`app-shell ${section === "Review Queue" ? "review-screen" : ""}`}
    >
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            B<span>↗</span>
          </div>
          <div>Boxing Motion AI</div>
        </div>

        <nav aria-label="Main navigation">
          {sections.map((s, i) => (
            <button
              key={s === "Review Queue" ? "Review" : s}
              className={section === s ? "active" : ""}
              aria-current={section === s ? "page" : undefined}
              disabled={saving}
              onClick={() => {
                if (
                  !dirty ||
                  window.confirm("Discard unsaved review changes?")
                ) {
                  setDirty(false);
                  setSaved("");
                  setSection(s);
                }
              }}
            >
              <NavIcon index={i} />
              {s === "Review Queue" ? "Review" : s}
              {s === "Review Queue" && queue.data?.data && (
                <small>{queue.data.data.summary.case_count}</small>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <small>Local workspace</small>
        </div>
      </aside>
      <div className="main-shell">
        <main>
          {(queue.error || inventory.error) && (
            <Notice>
              Could not load project metadata: {queue.error || inventory.error}.
              Start the local backend and check registered artifacts.
            </Notice>
          )}
          {section === "Review Queue" && (
            <div className="review-modes" aria-label="Review type">
              {["punches", "intervals"].map((mode) => (
                <button
                  key={mode}
                  className={reviewMode === mode ? "active" : ""}
                  disabled={saving}
                  onClick={() => {
                    if (
                      !dirty ||
                      window.confirm("Discard unsaved review changes?")
                    ) {
                      setReviewMode(mode);
                      setDirty(false);
                    }
                  }}
                >
                  {mode === "punches" ? "Punches" : "Video intervals"}
                </button>
              ))}
            </div>
          )}
          {section === "Review Queue" && reviewMode === "intervals" ? (
            <IntervalReview onDirty={handleDirty} onSaving={setSaving} />
          ) : section === "Review Queue" ? (
            <>
              <div className="review-layout">
                <section className="queue-panel panel">
                  <div className="panel-header">
                    <h2>
                      Punches <span className="count">{filtered.length}</span>
                    </h2>
                  </div>
                  <div className="queue-filters">
                    <input
                      aria-label="Search cases"
                      disabled={saving}
                      placeholder="Find a punch…"
                      value={search}
                      onChange={(e) => {
                        if (
                          !dirty ||
                          window.confirm("Discard unsaved review changes?")
                        ) {
                          setDirty(false);
                          setRetained(null);
                          setSaved("");
                          setSearch(e.target.value);
                        }
                      }}
                    />
                    <div>
                      <select
                        aria-label="Filter recording"
                        disabled={saving}
                        value={recording}
                        onChange={(e) => {
                          if (
                            !dirty ||
                            window.confirm("Discard unsaved review changes?")
                          ) {
                            setDirty(false);
                            setRetained(null);
                            setSaved("");
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
                        disabled={saving}
                        value={status}
                        onChange={(e) => {
                          if (
                            !dirty ||
                            window.confirm("Discard unsaved review changes?")
                          ) {
                            setDirty(false);
                            setRetained(null);
                            setSaved("");
                            setStatus(e.target.value);
                          }
                        }}
                      >
                        <option value="all">All punches</option>
                        <option value="To review">To review</option>
                        <option value="Reviewed">Reviewed</option>
                        <option value="Partially verified">
                          Partially verified
                        </option>
                        <option value="Needs attention">Needs attention</option>
                        <option value="dispute">Disputed punches</option>
                        <option value="anchor">Proposed anchors</option>
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
                    {filtered.map((e, index) => (
                      <button
                        className={`case-item ${selected === e.id ? "selected" : ""}`}
                        key={e.id}
                        disabled={saving}
                        onClick={() => navigate(e.id)}
                      >
                        <strong>
                          {title(
                            e.review.data?.source.punch_type_provisional ??
                              "Unavailable",
                          )}
                        </strong>
                        <span className="queue-meta">
                          <span>
                            {e.id.split(":")[0]} · Punch {index + 1}
                          </span>
                          <span>
                            {e.review.data?.verification_status ===
                            "fully_verified"
                              ? "Fully verified"
                              : reviewCategory(e)}
                          </span>
                        </span>
                      </button>
                    ))}
                  </div>
                </section>
                <div className="review-main">
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
                      <ReviewWorkbench
                        key={`${selected}:${refresh}`}
                        onDirty={handleDirty}
                        onSaving={setSaving}
                        feedback={saved}
                        onSaved={(next) => {
                          setDirty(false);
                          const after =
                            retained?.id === selected
                              ? retained.after.filter((id) =>
                                  filtered.some((e) => e.id === id),
                                )
                              : filtered.slice(position + 1).map((e) => e.id);
                          const nextId = after[0];
                          setSaved(
                            next && !nextId
                              ? "Review saved · End of queue."
                              : "Review saved.",
                          );
                          if (next && nextId) {
                            setSelected(nextId);
                            setRetained(null);
                          } else setRetained({ id: selected, after });
                          setRefresh((n) => n + 1);
                        }}
                        data={data}
                        recording={selected.split(":")[0]}
                      />
                    </>
                  )}
                </div>
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
        </main>
      </div>
    </div>
  );
}
