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
  onFailure,
}: {
  recording: string;
  index: number;
  overlay: boolean;
  onVisible: (frame: Frame) => void;
  onFailure: () => void;
}) {
  const result = useRequest<Result<Frame>>(
    `/api/recordings/${recording}/frames/${index}`,
  );
  const [ready, setReady] = useState(false);
  const [imageError, setImageError] = useState(false);
  const frame = result.data?.data;
  useEffect(() => {
    if (result.error) onFailure();
  }, [result.error]);
  if (result.error)
    return (
      <Notice>
        Frame unavailable: {result.error}. No approximate frame substituted.
      </Notice>
    );
  if (!frame)
    return (
      <div className="frame-loading" role="status">
        Loading frame…
      </div>
    );
  return (
    <div
      className="image-stage"
      style={{ aspectRatio: `${frame.width}/${frame.height}` }}
    >
      {!ready && !imageError && (
        <div className="frame-loading" role="status">
          Loading frame…
        </div>
      )}
      {imageError && (
        <Notice>
          Frame unavailable. Select another frame or reload to retry.
        </Notice>
      )}
      <img
        src={frame.image_url}
        alt={`${recording}, native frame ${index}`}
        style={{ visibility: ready ? "visible" : "hidden" }}
        onLoad={() => {
          setReady(true);
          onVisible(frame);
        }}
        onError={() => {
          setImageError(true);
          onFailure();
        }}
      />
      {ready && overlay && <Skeleton frame={frame} />}
    </div>
  );
}

export default function Viewer({
  data,
  recording,
  onVisible,
  onBoundary,
  jump,
  proposedBounds,
}: {
  data: Case;
  recording: string;
  onVisible?: (frame: Frame) => void;
  onBoundary?: (kind: "start" | "end", index: number) => void;
  jump?: { index: number; token: number };
  proposedBounds?: [number, number];
}) {
  const count = data.recording.data?.preparation.source_frame_count ?? 0;
  const original = frameBounds(data.annotation, count);
  const bounds =
    proposedBounds &&
    proposedBounds.every(Number.isInteger) &&
    proposedBounds[0] >= 0 &&
    proposedBounds[1] >= proposedBounds[0] &&
    proposedBounds[1] < count
      ? proposedBounds
      : original;
  const [index, setIndex] = useState(original?.[0] ?? 0);
  const [loaded, setLoaded] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [overlay, setOverlay] = useState(false);
  const [context, setContext] = useState(30);
  const minimum = Math.max(0, Math.min(index, (bounds?.[0] ?? 0) - context));
  const maximum = Math.min(
    count - 1,
    Math.max(index, (bounds?.[1] ?? 0) + context),
  );
  function seek(n: number) {
    if (Number.isInteger(n) && n >= 0 && n < count) {
      setPlaying(false);
      setIndex(n);
    }
  }
  function toggle() {
    if (!playing && index >= maximum) setIndex(minimum);
    setPlaying((p) => !p);
  }
  useEffect(() => {
    if (jump) seek(jump.index);
  }, [jump, count]);
  useEffect(() => {
    if (!playing || loaded !== index) return;
    if (index >= maximum) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => setIndex(index + 1), 120);
    return () => window.clearTimeout(timer);
  }, [playing, loaded, index, maximum]);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        (e.target as HTMLElement)?.closest(
          "input,select,textarea,button,summary",
        ) ||
        e.altKey ||
        e.ctrlKey ||
        e.metaKey
      )
        return;
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        seek(
          Math.max(
            minimum,
            Math.min(maximum, index + (e.key === "ArrowLeft" ? -1 : 1)),
          ),
        );
      }
      if (e.code === "Space") {
        e.preventDefault();
        toggle();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [index, minimum, maximum, playing]);
  if (!count) return <Notice>Recording frames are unavailable.</Notice>;
  const position = (n: number) =>
    `${(100 * (n - minimum)) / Math.max(1, maximum - minimum)}%`;
  return (
    <section className="viewer panel" aria-label="Punch viewer">
      <ExactFrame
        key={`${recording}-${index}`}
        recording={recording}
        index={index}
        overlay={overlay}
        onVisible={(f) => {
          setLoaded(f.frame_index);
          onVisible?.(f);
        }}
        onFailure={() => setPlaying(false)}
      />
      <div className="playback-controls">
        <button
          aria-label={playing ? "Pause" : "Play"}
          className="play-button"
          onClick={toggle}
        >
          {playing ? "Ⅱ" : "▶"}
        </button>
        <button
          aria-label="Previous frame"
          disabled={index <= minimum}
          onClick={() => seek(index - 1)}
        >
          ‹
        </button>
        <button
          aria-label="Next frame"
          disabled={index >= maximum}
          onClick={() => seek(index + 1)}
        >
          ›
        </button>
        <span className="playback-position">
          Frame {index.toLocaleString()}
        </span>
        <small>Frame-by-frame playback</small>
        <label className="toggle">
          <input
            type="checkbox"
            checked={overlay}
            onChange={(e) => setOverlay(e.target.checked)}
          />
          Skeleton
        </label>
      </div>
      <div className="review-timeline">
        {bounds && (
          <>
            <div
              className="punch-range"
              style={{
                left: position(bounds[0]),
                width: `${(100 * (bounds[1] - bounds[0])) / Math.max(1, maximum - minimum)}%`,
              }}
            />
            <span
              className="boundary-marker"
              style={{ left: position(bounds[0]) }}
            />
            <span
              className="boundary-marker"
              style={{ left: position(bounds[1]) }}
            />
          </>
        )}
        <input
          aria-label="Native frame"
          type="range"
          min={minimum}
          max={maximum}
          value={index}
          onChange={(e) => seek(Number(e.target.value))}
        />
      </div>
      <div className="boundary-navigation">
        {bounds ? (
          <>
            <button onClick={() => seek(bounds[0])}>
              {proposedBounds ? "Proposed start" : "Punch start"} · {bounds[0]}
            </button>
            <button onClick={() => seek(bounds[1])}>
              {proposedBounds ? "Proposed end" : "Punch end"} · {bounds[1]}
            </button>
          </>
        ) : (
          <small>Source boundaries need review.</small>
        )}
      </div>
      {onBoundary && (
        <div className="boundary-edit-tools">
          <button
            disabled={loaded !== index || playing}
            onClick={() => onBoundary("start", index)}
          >
            Use frame as start
          </button>
          <button
            disabled={loaded !== index || playing}
            onClick={() => onBoundary("end", index)}
          >
            Use frame as end
          </button>
        </div>
      )}
      <details className="viewer-options">
        <summary>Playback options</summary>
        <div className="review-fields">
          <label>
            Go to native frame
            <input
              type="number"
              min={0}
              max={count - 1}
              value={index}
              onChange={(e) => seek(Number(e.target.value))}
            />
          </label>
          <label>
            Surrounding frames
            <select
              aria-label="Context frames"
              value={context}
              onChange={(e) => {
                setPlaying(false);
                setContext(Number(e.target.value));
              }}
            >
              {[10, 30, 60].map((n) => (
                <option key={n} value={n}>
                  {n} before and after
                </option>
              ))}
            </select>
          </label>
          <p>
            Space: play/pause. Arrow keys: one frame. Playback waits for each
            exact frame; it is not real time.
          </p>
        </div>
      </details>
    </section>
  );
}
