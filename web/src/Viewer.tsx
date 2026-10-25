import { useEffect, useState } from "react";
import {
  type Case,
  type Frame,
  type Result,
  frameBounds,
  percent,
  title,
  useRequest,
} from "./api";
export function Notice({ children }: { children: React.ReactNode }) {
  return (
    <div className="notice" role="status">
      {children}
    </div>
  );
}
export function Skeleton({ frame }: { frame: Frame }) {
  const visible = (i: number) =>
    frame.landmarks[i]?.valid &&
    frame.landmarks[i].xyz
      .slice(0, 2)
      .every((n) => n !== null && Number.isFinite(n));
  return (
    <svg
      className="skeleton"
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      aria-label="Valid pose landmarks"
    >
      {frame.edges
        .filter(([a, b]) => visible(a) && visible(b))
        .map(([a, b]) => (
          <line
            key={`${a}-${b}`}
            x1={frame.landmarks[a].xyz[0]!}
            y1={frame.landmarks[a].xyz[1]!}
            x2={frame.landmarks[b].xyz[0]!}
            y2={frame.landmarks[b].xyz[1]!}
          />
        ))}
      {frame.landmarks.map((p, i) =>
        visible(i) && i >= 11 ? (
          <circle key={i} cx={p.xyz[0]!} cy={p.xyz[1]!} r=".004">
            <title>
              {title(p.name)} · visibility{" "}
              {percent(p.confidence[0] ?? undefined)}
            </title>
          </circle>
        ) : null,
      )}
    </svg>
  );
}
function ExactFrame({
  recording,
  index,
  overlay,
  onVisible,
}: {
  recording: string;
  index: number;
  overlay: boolean;
  onVisible?: (frame: Frame) => void;
}) {
  const result = useRequest<Result<Frame>>(
    `/api/recordings/${recording}/frames/${index}`,
  );
  const [ready, setReady] = useState(false);
  const [imageError, setImageError] = useState(false);
  const frame = result.data?.data;
  if (result.error)
    return (
      <Notice>
        Frame unavailable: {result.error}. No approximate frame substituted.
      </Notice>
    );
  if (!frame)
    return (
      <div className="frame-loading" role="status">
        <div className="spinner" />
        Verifying source & decoding exact frame…
        <small>The first request verifies recording and pose hashes.</small>
      </div>
    );
  return (
    <>
      <div
        className="image-stage"
        style={{ aspectRatio: `${frame.width}/${frame.height}` }}
      >
        {!ready && !imageError && (
          <div className="frame-loading" role="status">
            Loading verified preview…
          </div>
        )}
        {imageError && (
          <Notice>Preview unavailable. Reload this case to retry.</Notice>
        )}
        <img
          src={frame.image_url}
          alt={`${recording}, native frame ${index}`}
          style={{ visibility: ready ? "visible" : "hidden" }}
          onLoad={() => {
            setReady(true);
            onVisible?.(frame);
          }}
          onError={() => setImageError(true)}
        />
        {ready && overlay && <Skeleton frame={frame} />}
        {ready && (
          <>
            <span className="image-label">
              {recording} <span>·</span> NATIVE FRAME {index.toLocaleString()}
            </span>
            <span className="image-tag">EXACT PTS</span>
          </>
        )}
      </div>
      <div className="frame-meta">
        <span>
          PTS{" "}
          <b>
            {frame.pts} × {frame.time_base}
          </b>
        </span>
        <span>
          Exact time <b>{frame.exact_timestamp} s</b>
        </span>
        <span>
          Valid joints <b>{frame.landmarks.filter((p) => p.valid).length}/33</b>
        </span>
      </div>
      <div className="frame-meta">
        <span>
          Scene {frame.scene_id} · segment {frame.segment_id}
        </span>
        <span>
          {frame.scene_cut_before
            ? "⚠ Known scene cut before this frame"
            : "Invalid landmarks are hidden, never filled"}
        </span>
      </div>
    </>
  );
}
export default function Viewer({
  data,
  recording,
  onVisible,
  onBoundary,
  jump,
}: {
  data: Case;
  recording: string;
  onVisible?: (frame: Frame) => void;
  jump?: { index: number; token: number };
  onBoundary?: (kind: "start" | "end", index: number) => void;
}) {
  const source = data.annotation;
  const count = data.recording.data?.preparation.source_frame_count ?? 0;
  const bounds = frameBounds(source, count);
  const [index, setIndex] = useState(bounds?.[0] ?? 0);
  const [loaded, setLoaded] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [overlay, setOverlay] = useState(true);
  const [context, setContext] = useState(30);
  const minimum = Math.max(0, Math.min(index, (bounds?.[0] ?? 0) - context));
  const maximum = Math.min(
    count - 1,
    Math.max(index, (bounds?.[1] ?? 0) + context),
  );
  useEffect(() => {
    if (
      jump &&
      Number.isInteger(jump.index) &&
      jump.index >= 0 &&
      jump.index < count
    ) {
      setPlaying(false);
      setIndex(jump.index);
    }
  }, [jump, count]);
  useEffect(() => {
    if (!playing || loaded !== index || index >= maximum) return;
    const timer = window.setTimeout(() => setIndex(index + 1), 120);
    return () => window.clearTimeout(timer);
  }, [playing, loaded, index, maximum]);
  const step = (delta: number) =>
    setIndex((i) => Math.max(minimum, Math.min(maximum, i + delta)));
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (
        (event.target as HTMLElement)?.closest("input,select,textarea,button")
      )
        return;
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        setIndex((i) =>
          Math.max(
            minimum,
            Math.min(maximum, i + (event.key === "ArrowLeft" ? -1 : 1)),
          ),
        );
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [minimum, maximum]);
  if (!bounds)
    return (
      <Notice>
        This interval has unresolved or invalid frame boundaries. Exact-frame
        preview is unavailable; no correction has been applied.
      </Notice>
    );
  const points = [
    ...new Set([
      minimum,
      bounds[0],
      Math.floor((bounds[0] + bounds[1]) / 2),
      bounds[1],
      maximum,
    ]),
  ];
  return (
    <section className="viewer panel">
      <div className="panel-header">
        <span className="eyebrow">RECORDING VIEWER</span>
        <label className="toggle">
          <input
            type="checkbox"
            checked={overlay}
            onChange={(e) => setOverlay(e.target.checked)}
          />{" "}
          Pose overlay
        </label>
      </div>
      <ExactFrame
        key={`${recording}-${index}`}
        recording={recording}
        index={index}
        overlay={overlay}
        onVisible={(frame) => {
          setLoaded(frame.frame_index);
          onVisible?.(frame);
        }}
      />
      <div className="transport">
        <button onClick={() => setPlaying(!playing)}>
          {playing ? "Pause" : "Play frames"}
        </button>
        <small>Decode-paced · not realtime</small>
        <label>
          Go to native frame
          <input
            aria-label="Go to native frame"
            type="number"
            min={0}
            max={count - 1}
            value={index}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (Number.isInteger(n) && n >= 0 && n < count) {
                setPlaying(false);
                setIndex(n);
              }
            }}
          />
        </label>
        {onBoundary && (
          <div className="button-group">
            <button
              disabled={loaded !== index}
              onClick={() => onBoundary("start", index)}
            >
              Use frame as start
            </button>
            <button
              disabled={loaded !== index}
              onClick={() => onBoundary("end", index)}
            >
              Use frame as end
            </button>
          </div>
        )}

        <div className="button-group">
          <button
            aria-label="Previous frame"
            disabled={index <= minimum}
            onClick={() => step(-1)}
          >
            ←
          </button>
          <button onClick={() => setIndex(bounds[0])}>Punch start</button>
          <button
            aria-label="Next frame"
            disabled={index >= maximum}
            onClick={() => step(1)}
          >
            →
          </button>
        </div>
        <span
          className={
            index < bounds[0] || index > bounds[1] ? "badge muted" : "badge"
          }
        >
          {index < bounds[0]
            ? "Before punch"
            : index > bounds[1]
              ? "After punch"
              : "Annotated interval"}
        </span>
        <label>
          Context{" "}
          <select
            aria-label="Context frames"
            value={context}
            onChange={(e) => {
              const n = Number(e.target.value);
              setContext(n);
              setIndex(bounds[0]);
            }}
          >
            <option value={10}>±10 frames</option>
            <option value={30}>±30 frames</option>
            <option value={60}>±60 frames</option>
          </select>
        </label>
      </div>
      <div className="timeline">
        <div
          className="timeline-fill"
          style={{
            left: `${(100 * (bounds[0] - minimum)) / Math.max(1, maximum - minimum)}%`,
            width: `${(100 * (bounds[1] - bounds[0])) / Math.max(1, maximum - minimum)}%`,
          }}
        />
        <input
          aria-label="Native frame"
          type="range"
          min={minimum}
          max={maximum}
          value={index}
          onChange={(e) => setIndex(Number(e.target.value))}
        />
      </div>
      <div className="context-points">
        {points.map((n) => (
          <button
            key={n}
            className={n === index ? "selected" : ""}
            onClick={() => setIndex(n)}
          >
            <span>
              {n === bounds[0]
                ? "START"
                : n === bounds[1]
                  ? "END"
                  : n < bounds[0]
                    ? "PRE-CONTEXT"
                    : n > bounds[1]
                      ? "POST-CONTEXT"
                      : "MIDPOINT"}
            </span>
            <b>{n.toLocaleString()}</b>
          </button>
        ))}
      </div>
      <div className="viewer-footer">
        ← → Step one native frame{" "}
        <span>
          Source mapping is provisional · contextual frames are unlabeled
        </span>
      </div>
    </section>
  );
}
