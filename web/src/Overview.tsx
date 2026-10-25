import { useState } from "react";
import {
  type Experiments,
  type Inventory,
  type Metrics,
  type Result,
  type Reviews,
  type Stats,
  percent,
  title,
  useRequest,
} from "./api";
import { Notice } from "./Viewer";
export function Dataset({
  inventory,
  reviews,
}: {
  inventory?: Inventory;
  reviews?: Reviews;
}) {
  const stats = useRequest<Record<string, Result<Stats>>>("/api/statistics");
  const proposal = useRequest<
    Result<{
      proposal: {
        training_class_support: Record<string, number>;
        status: string;
        pending_human_queue: string[];
      };
    }>
  >("/api/curation");
  const records = Object.entries(stats.data ?? {});
  const complete =
    records.length > 0 &&
    records.every(([, r]) => r.status === "available" && r.data);
  const classes = [
    ...new Set(
      records.flatMap(([, r]) => Object.keys(r.data?.approved_by_class ?? {})),
    ),
  ];
  const training = records
    .filter(([, r]) => r.data?.role === "training")
    .reduce((n, [, r]) => n + (r.data?.approved_provisional_examples ?? 0), 0);
  const development = records
    .filter(([, r]) => r.data?.role === "development")
    .reduce((n, [, r]) => n + (r.data?.approved_provisional_examples ?? 0), 0);
  return (
    <>
      <div className="metric-grid">
        <div className="metric-card">
          <span>Registered recordings</span>
          <strong>{inventory?.available_recordings ?? "—"}</strong>
          <small>Approved development inventory</small>
        </div>
        <div className="metric-card">
          <span>Training candidates</span>
          <strong>{complete ? training.toLocaleString() : "—"}</strong>
          <small>Existing provisional selection</small>
        </div>
        <div className="metric-card">
          <span>V5 development</span>
          <strong>{complete ? development : "—"}</strong>
          <small>Never used for training</small>
        </div>
        <div className="metric-card">
          <span>Human accepted</span>
          <strong>
            {reviews?.summary.human_accepted ?? "—"}
            <em> / {reviews?.summary.case_count ?? "—"}</em>
          </strong>
          <small>Assistant observations are not approvals</small>
        </div>
      </div>
      {stats.error && <Notice>{stats.error}</Notice>}
      {stats.data && !complete && (
        <Notice>
          Some recording statistics are unavailable. Totals are withheld;
          missing metadata is not zero examples.
        </Notice>
      )}
      <section className="panel padded">
        <div className="section-heading">
          <div>
            <h2>Punch distribution</h2>
            <p>
              Approved mask-aware candidates · source labels remain provisional
            </p>
          </div>
          <span className="badge muted">6-class taxonomy</span>
        </div>
        {!stats.data ? (
          <Notice>Loading dataset statistics…</Notice>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Recording</th>
                  <th>Role</th>
                  {classes.map((c) => (
                    <th key={c}>{title(c)}</th>
                  ))}
                  <th>Total</th>
                </tr>
              </thead>
              <tbody>
                {records.map(([v, result]) => (
                  <tr key={v}>
                    <td>
                      <b>{v}</b>
                    </td>
                    <td>{result.data?.role ?? result.status}</td>
                    {classes.map((c) => (
                      <td key={c}>
                        {result.data
                          ? (result.data.approved_by_class[c] ?? 0)
                          : "—"}
                      </td>
                    ))}
                    <td>
                      <b>{result.data?.approved_provisional_examples ?? "—"}</b>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="distribution-list">
          {records.map(
            ([v, r]) =>
              r.data && (
                <div key={v}>
                  <b>{v}</b>
                  <div className="distribution-bar">
                    {classes.map((c, i) => (
                      <span
                        key={c}
                        title={`${title(c)}: ${r.data!.approved_by_class[c] ?? 0}`}
                        style={{
                          flex: r.data!.approved_by_class[c] ?? 0,
                          background: `var(--class-${i})`,
                        }}
                      />
                    ))}
                  </div>
                  <small>{r.data.approved_provisional_examples}</small>
                </div>
              ),
          )}
        </div>
        <div className="legend">
          {classes.map((c, i) => (
            <span key={c}>
              <i style={{ background: `var(--class-${i})` }} />
              {title(c)}
            </span>
          ))}
        </div>
      </section>
      <div className="two-columns">
        <section className="panel padded">
          <h2>Review readiness</h2>
          <div className="progress-track">
            <i
              style={{
                width: `${(100 * (reviews?.summary.human_accepted ?? 0)) / Math.max(1, reviews?.summary.case_count ?? 0)}%`,
              }}
            />
          </div>
          <p>
            {reviews?.summary.human_accepted ?? 0} human-accepted cases. The
            curated training proposal remains a separate selection.
          </p>
          <dl>
            <dt>Proposal</dt>
            <dd>
              {title(proposal.data?.data?.proposal.status ?? "unavailable")}
            </dd>
            <dt>Pending targeted reviews</dt>
            <dd>
              {proposal.data?.data?.proposal.pending_human_queue.length ?? "—"}
            </dd>
            <dt>Proposed training examples</dt>
            <dd>
              {proposal.data?.data
                ? Object.values(
                    proposal.data.data.proposal.training_class_support,
                  )
                    .reduce((a, b) => a + b, 0)
                    .toLocaleString()
                : "—"}
            </dd>
          </dl>
          {proposal.error && <Notice>{proposal.error}</Notice>}
        </section>
        <section className="panel padded">
          <h2>Source separation</h2>
          {Object.entries(inventory?.records ?? {}).map(([v, r]) => (
            <div className="source-line" key={v}>
              <b>{v}</b>
              <span>{r.data?.source_group ?? r.status}</span>
              <small>{r.data?.boxer_id ?? "Boxer unknown"}</small>
            </div>
          ))}
          <p>
            V7/V8 stay grouped. V1/V6 share a source guard. V9 is reserved and
            is not accessible.
          </p>
        </section>
      </div>
    </>
  );
}
function scoreRows(
  value: unknown,
  path = "",
): { path: string; accuracy: number; f1: number }[] {
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  const obj = value as Record<string, unknown>;
  const number = (v: unknown): number | undefined =>
    typeof v === "number"
      ? v
      : v && typeof v === "object" && "mean" in v && typeof v.mean === "number"
        ? v.mean
        : undefined;
  const accuracy = number(obj.accuracy),
    f1 = number(obj.macro_f1);
  if (accuracy !== undefined && f1 !== undefined)
    return [{ path, accuracy, f1 }];
  return Object.entries(obj).flatMap(([k, v]) =>
    scoreRows(v, path ? `${path} / ${k}` : k),
  );
}
export function ExperimentsView({
  experiments,
}: {
  experiments?: Experiments;
}) {
  const [selected, setSelected] = useState("velocity-reference");
  const result = useRequest<
    Result<{ results: unknown; interpretation: string }>
  >(`/api/experiments/${selected}`);
  const rows = scoreRows(result.data?.data?.results);
  return (
    <>
      <div className="experiment-cards">
        {Object.entries(experiments?.items ?? {}).map(([id, e]) => (
          <button
            className={`panel experiment-card ${selected === id ? "active" : ""}`}
            key={id}
            onClick={() => setSelected(id)}
          >
            <span className="badge muted">{e.status}</span>
            <h2>{title(id.replaceAll("-", "_"))}</h2>
            <p>{e.description}</p>
            <span className="teal">Inspect results ↗</span>
          </button>
        ))}
      </div>
      <section className="panel padded">
        <h2>Preserved experiment results</h2>
        <p>
          Stored accuracy and six-class macro-F1. Aggregate rows show means;
          individual fold rows preserve their evaluation scope.
        </p>
        {result.error ? (
          <Notice>{result.error}</Notice>
        ) : !result.data ? (
          <Notice>Loading results…</Notice>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Evaluation / model / recording</th>
                  <th>Accuracy</th>
                  <th>Macro-F1</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.path}>
                    <td className="mono compact">{r.path}</td>
                    <td>{percent(r.accuracy)}</td>
                    <td className="teal">{percent(r.f1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!rows.length && (
              <Notice>No compatible metric rows in this artifact.</Notice>
            )}
          </div>
        )}
        <details>
          <summary>Original result metadata</summary>
          <pre>{JSON.stringify(result.data?.data?.results, null, 2)}</pre>
        </details>
      </section>
    </>
  );
}
export function Models({ experiments }: { experiments?: Experiments }) {
  const ids = experiments?.items["velocity-reference"]?.metric_ids ?? [];
  const [selected, setSelected] = useState("20261021-xy_velocity-tcn");
  const result = useRequest<Result<Metrics>>(
    `/api/experiments/velocity-reference/metrics/${selected}`,
  );
  const metrics = result.data?.data?.metrics,
    dev = metrics?.development;
  return (
    <>
      <section className="panel padded">
        <div className="section-heading">
          <div>
            <span className="eyebrow">PROVISIONAL REFERENCE</span>
            <h2>Causal TCN · pose only</h2>
            <p>
              XY + validity-gated velocities + 50% mirror augmentation. Frozen
              checkpoints; no inference runs here.
            </p>
          </div>
          <span className="badge">V4 train → V5 development</span>
        </div>
        <p className="caption">
          Selected features:{" "}
          {selected.includes("xy_velocity")
            ? "XY + validity-gated velocity"
            : "XY only"}{" "}
          ·{" "}
          {selected.includes("mask_only")
            ? "mask-only ablation"
            : "pose movement model"}
        </p>
        <label>
          Preserved model run{" "}
          <select
            aria-label="Model run"
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
          >
            {ids.map((id) => (
              <option key={id}>{id}</option>
            ))}
          </select>
        </label>
      </section>
      {result.error && <Notice>{result.error}</Notice>}
      {dev && metrics && (
        <>
          <div className="metric-grid">
            <div className="metric-card">
              <span>Development accuracy</span>
              <strong>{percent(dev.accuracy)}</strong>
            </div>
            <div className="metric-card">
              <span>Development macro-F1</span>
              <strong>{percent(dev.macro_f1)}</strong>
            </div>
            <div className="metric-card">
              <span>Training macro-F1</span>
              <strong>{percent(metrics.train.macro_f1)}</strong>
            </div>
            <div className="metric-card">
              <span>Selected epoch</span>
              <strong>{metrics.selected_epoch}</strong>
            </div>
          </div>
          <div className="two-columns">
            <section className="panel padded">
              <h2>Confusion matrix</h2>
              <p>Rows: source labels · columns: predictions · V5 development</p>
              <div className="table-scroll">
                <table className="confusion">
                  <thead>
                    <tr>
                      <th>Label</th>
                      {dev.labels.map((c) => (
                        <th key={c}>{title(c)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {dev.confusion_matrix.map((row, i) => (
                      <tr key={dev.labels[i]}>
                        <th>{title(dev.labels[i])}</th>
                        {row.map((n, j) => (
                          <td
                            key={j}
                            style={{
                              background: `rgba(55,210,176,${0.06 + (0.65 * n) / Math.max(1, ...row)})`,
                            }}
                          >
                            {n}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
            <section className="panel padded">
              <h2>Class performance</h2>
              <table>
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Support</th>
                  </tr>
                </thead>
                <tbody>
                  {dev.labels.map((c) => (
                    <tr key={c}>
                      <td>{title(c)}</td>
                      <td>
                        {dev.per_class[c].support
                          ? percent(dev.per_class[c].precision)
                          : "Absent"}
                      </td>
                      <td>
                        {dev.per_class[c].support
                          ? percent(dev.per_class[c].recall)
                          : "Absent"}
                      </td>
                      <td>{dev.per_class[c].support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p>
                Development-selected results are not a claim of
                boxer-independent generalization.
              </p>
            </section>
          </div>
        </>
      )}
    </>
  );
}
