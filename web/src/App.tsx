import { useEffect, useMemo, useState } from "react";
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
          {section === "Review Queue" ? (
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
                      placeholder="Find a punch…"
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
                      <ReviewWorkbench
                        key={`${selected}:${refresh}`}
                        onDirty={setDirty}
                        onSaved={(next) => {
                          setDirty(false);
                          setSaved(
                            `Review saved${next && !filtered[position + 1] ? " · End of queue" : ""}.`,
                          );
                          if (next && filtered[position + 1])
                            setSelected(filtered[position + 1].id);
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
