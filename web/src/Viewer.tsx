import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
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
export function ExactFrame({
  recording,
  index,
  overlay,
  onVisible,
  onFailure,
  retryToken = 0,
}: {
  recording: string;
  index: number;
  overlay: boolean;
  onVisible: (frame: Frame) => void;
  onFailure: () => void;
  retryToken?: number;
}) {
  const [retry, setRetry] = useState(0);
  const result = useRequest<Result<Frame>>(
    `/api/recordings/${recording}/frames/${index}`,
    retry + retryToken,
  );
  const [displayed, setDisplayed] = useState<Frame | null>(null);
  const [failure, setFailure] = useState<{
    key: string;
    message: string;
  } | null>(null);
  const key = `${recording}:${index}:${retry}:${retryToken}`;
  const latest = useRef({ key, generation: 0 });
  if (latest.current.key !== key)
    latest.current = { key, generation: latest.current.generation + 1 };
  const generation = latest.current.generation;
  const callbacks = useRef({ onVisible, onFailure });
  callbacks.current = { onVisible, onFailure };
  const frame = result.data?.data;
  const mismatch = frame && frame.frame_index !== index;
  const error =
    result.error ||
    (mismatch ? "Frame identity mismatch" : undefined) ||
    (failure?.key === key ? failure.message : undefined);
  useEffect(() => {
    if (error) callbacks.current.onFailure();
  }, [error]);
  // Publish metadata before paint, in the same commit as the decoded image and overlay.
  useLayoutEffect(() => {
    if (displayed) callbacks.current.onVisible(displayed);
  }, [displayed]);
  const candidate =
    !error && frame && frame.image_url !== displayed?.image_url ? frame : null;
  const images = [displayed, candidate].filter((f): f is Frame => !!f);
  const pending = displayed?.frame_index !== index || !!candidate;
  return (
    <div className="image-stage" aria-busy={!error && pending}>
      {images.map((f) => (
        <div
          className="frame-plane"
          key={f.image_url}
          style={
            {
              "--frame-ratio": f.width / f.height,
              visibility: f === displayed ? "visible" : "hidden",
            } as React.CSSProperties
          }
        >
          <img
            src={f.image_url}
            alt={`${recording}, native frame ${f.frame_index}`}
            onLoad={async (e) => {
              if (f === displayed) return;
              const element = e.currentTarget;
              try {
                if (element.decode) await element.decode();
                if (
                  latest.current.generation !== generation ||
                  latest.current.key !== key
                )
                  return;
                setDisplayed(f);
              } catch {
                if (
                  latest.current.generation === generation &&
                  latest.current.key === key
                )
                  setFailure({ key, message: "Image could not be decoded" });
              }
            }}
            onError={() => {
              if (
                f !== displayed &&
                latest.current.generation === generation &&
                latest.current.key === key
              )
                setFailure({ key, message: "Image could not be loaded" });
            }}
          />
          {f === displayed && overlay && <Skeleton frame={f} />}
        </div>
      ))}
      {error ? (
        <div className="frame-status frame-error" role="alert">
          Frame {index} unavailable: {error}.{" "}
          {displayed
            ? `Showing frame ${displayed.frame_index}.`
            : "No frame displayed."}
          <button onClick={() => setRetry((n) => n + 1)}>Retry frame</button>
        </div>
      ) : (
        pending && (
          <div className="frame-status" role="status">
            Loading frame {index}…
            {displayed && ` Showing frame ${displayed.frame_index}.`}
          </div>
        )
      )}
    </div>
  );
}

export default function Viewer({
  data,
  recording,
  onVisible,
  onBoundary,
  proposedBounds,
  interval,
}: {
  data?: Case;
  interval?: {
    start: number;
    end: number;
    count: number;
    contextStart: number;
    contextEnd: number;
    delay: number;
  };
  recording: string;
  onVisible?: (frame: Frame) => void;
  onBoundary?: (kind: "start" | "end", index: number) => void;
  proposedBounds?: [number, number];
}) {
  const markerId = useId();
  const count =
    interval?.count ??
    data?.recording.data?.preparation.source_frame_count ??
    0;
  const original: [number, number] | null = interval
    ? [interval.start, interval.end]
    : data
      ? frameBounds(data.annotation, count)
      : null;
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
  const minimum =
    interval?.contextStart ??
    Math.max(0, Math.min(index, loaded ?? index, (bounds?.[0] ?? 0) - context));
  const maximum =
    interval?.contextEnd ??
    Math.min(
      count - 1,
      Math.max(index, loaded ?? index, (bounds?.[1] ?? 0) + context),
    );
  function seek(n: number) {
    if (
      Number.isInteger(n) &&
      n >= (interval?.contextStart ?? 0) &&
      n <= (interval?.contextEnd ?? count - 1)
    ) {
      setPlaying(false);
      setIndex(n);
    }
  }
  function toggle() {
    if (!playing && index >= maximum) setIndex(minimum);
    setPlaying((p) => !p);
  }
  useEffect(() => {
    if (!playing || loaded !== index) return;
    if (index >= maximum) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(
      () => setIndex(index + 1),
      interval?.delay ?? 120,
    );
    return () => window.clearTimeout(timer);
  }, [playing, loaded, index, maximum, interval?.delay]);
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
    <section
      className="viewer panel"
      aria-label={interval ? "Interval viewer" : "Punch viewer"}
    >
      <ExactFrame
        key={recording}
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
        <div className="playback-context">
          <span className="original-punch">
            <small>{interval ? "Review interval" : "Original punch"}</small>
            <strong>
              {interval
                ? `${interval.start}–${interval.end}`
                : title(data?.annotation.punch_type_provisional ?? "unknown")}
            </strong>
          </span>
        </div>
        <div className="playback-buttons">
          <button
            aria-label="Previous frame"
            disabled={index <= minimum}
            onClick={() => seek(index - 1)}
          >
            ‹
          </button>
          <button
            aria-label={playing ? "Pause" : "Play"}
            className="play-button"
            onClick={toggle}
          >
            {playing ? "Ⅱ" : "▶"}
          </button>
          <button
            aria-label="Next frame"
            disabled={index >= maximum}
            onClick={() => seek(index + 1)}
          >
            ›
          </button>
        </div>
        <div className="playback-options">
          <span className="playback-position">
            {loaded === null
              ? "Loading frame…"
              : `Frame ${loaded.toLocaleString()}`}
          </span>
          <label className="toggle">
            <input
              type="checkbox"
              checked={overlay}
              onChange={(e) => setOverlay(e.target.checked)}
            />
            Skeleton
          </label>
        </div>
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
            {(bounds[0] === bounds[1] ? [0] : [0, 1]).map((end) => {
              const label = `${proposedBounds ? "Proposed" : interval ? "Interval" : "Punch"} ${bounds[0] === bounds[1] ? "start/end" : end === 0 ? "start" : "end"} · ${bounds[end]}`;
              return (
                <button
                  key={end}
                  className={`boundary-marker boundary-${end === 0 ? "start" : "end"}`}
                  style={{ left: position(bounds[end]) }}
                  aria-label={label}
                  aria-describedby={`${markerId}-${end}`}
                  onClick={() => seek(bounds[end])}
                >
                  <span
                    role="tooltip"
                    id={`${markerId}-${end}`}
                    className="boundary-tooltip"
                  >
                    {label}
                  </span>
                </button>
              );
            })}
          </>
        )}
        <input
          aria-label="Native frame"
          type="range"
          min={minimum}
          max={maximum}
          value={loaded ?? index}
          aria-valuetext={
            loaded === null ? "No frame displayed" : `Showing frame ${loaded}`
          }
          onChange={(e) => seek(Number(e.target.value))}
        />
      </div>
      {!bounds && (
        <small className="boundary-status">
          Source boundaries need review.
        </small>
      )}
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
              min={interval?.contextStart ?? 0}
              max={interval?.contextEnd ?? count - 1}
              value={index}
              onChange={(e) => seek(Number(e.target.value))}
            />
          </label>
          {!interval && (
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
          )}
          <p>
            Space: play/pause. Arrow keys: one frame. Playback waits for each
            exact frame; it is not real time.
          </p>
        </div>
      </details>
    </section>
  );
}
