import { useEffect, useRef, useState } from "react";
import { type Frame, type Result, useRequest } from "./api";
import { ExactFrame } from "./Viewer";
import {
  type Playback,
  type Presented,
  frameAt,
  creditPresentation,
} from "./playback";

type Interval = {
  start: number;
  end: number;
  count: number;
  contextStart: number;
  contextEnd: number;
  delay: number;
};
export default function BufferedIntervalViewer({
  intervalId,
  interval,
  recording,
  onVisible,
  onPresented,
  seenFrames,
}: {
  intervalId: string;
  interval: Interval;
  recording: string;
  onVisible: (frame: Frame) => void;
  onPresented: (index: number) => void;
  seenFrames: Set<number>;
}) {
  const [frameRetry, setFrameRetry] = useState(0);
  const result = useRequest<Result<Playback>>(
    `/api/intervals/${intervalId}/playback`,
  );
  const clip = result.data?.data;
  const video = useRef<HTMLVideoElement>(null),
    stage = useRef<HTMLDivElement>(null);
  const [index, setIndex] = useState(interval.start),
    [position, setPosition] = useState<number | null>(null);
  const [exact, setExact] = useState(true),
    [exactReady, setExactReady] = useState(false);
  const [playing, setPlaying] = useState(false),
    [overlay, setOverlay] = useState(false);
  const [speed, setSpeed] = useState(1),
    [error, setError] = useState("");
  const [hasPlayed, setHasPlayed] = useState(false);
  const lastExact = useRef<Frame | null>(null);
  const [buffering, setBuffering] = useState(false),
    [supported, setSupported] = useState(true);
  const visible = useRef(false),
    active = useRef(false),
    previous = useRef<Presented | null>(null);
  const shown = useRef<number | null>(null),
    target = useRef(index),
    exactMode = useRef(exact);
  target.current = index;
  exactMode.current = exact;
  const callbacks = useRef({ onVisible, onPresented });
  callbacks.current = { onVisible, onPresented };
  const minimum = clip?.frames[0] ?? interval.contextStart;
  const maximum = clip?.frames.at(-1) ?? interval.contextEnd;
  const invalidate = () => {
    previous.current = null;
    active.current = false;
  };
  useEffect(() => {
    const element = stage.current;
    const observer =
      typeof IntersectionObserver !== "undefined"
        ? new IntersectionObserver(
            (entries) => {
              visible.current = entries[0].intersectionRatio >= 0.95;
              if (!visible.current) {
                invalidate();
                video.current?.pause();
              }
            },
            { threshold: [0, 0.95, 1] },
          )
        : null;
    if (element && observer) observer.observe(element);
    const hide = () => {
      if (document.visibilityState !== "visible") {
        invalidate();
        video.current?.pause();
      }
    };
    document.addEventListener("visibilitychange", hide);
    return () => {
      observer?.disconnect();
      document.removeEventListener("visibilitychange", hide);
    };
  }, []);
  useEffect(() => {
    const v = video.current;
    if (!v || !clip) return;
    if (!v.requestVideoFrameCallback) {
      setSupported(false);
      return;
    }
    let cancelled = false,
      handle = 0;
    const tick = (now: number, metadata: VideoFrameCallbackMetadata) => {
      if (cancelled) return;
      const slot = frameAt(clip, metadata.mediaTime, true);
      if (slot !== null) {
        const current = {
          index: clip.frames[slot],
          mediaTime: metadata.mediaTime,
          frames: metadata.presentedFrames,
          now,
        };
        shown.current = current.index;
        if (!v.paused && !v.seeking) {
          setPosition(current.index);
          setExact(false);
          setExactReady(false);
        }
        const eligible =
          active.current &&
          !v.paused &&
          !v.seeking &&
          v.readyState >= 3 &&
          visible.current &&
          document.visibilityState === "visible";
        if (creditPresentation(previous.current, current, eligible))
          callbacks.current.onPresented(current.index);
        previous.current = eligible ? current : null;
      } else previous.current = null;
      handle = v.requestVideoFrameCallback(tick);
    };
    handle = v.requestVideoFrameCallback(tick);
    return () => {
      cancelled = true;
      v.cancelVideoFrameCallback(handle);
      previous.current = null;
    };
  }, [clip]);
  function seek(n: number) {
    if (!Number.isInteger(n) || n < minimum || n > maximum) return;
    if (
      exactMode.current &&
      exactReady &&
      n === target.current &&
      lastExact.current?.frame_index === n &&
      visible.current &&
      document.visibilityState === "visible"
    ) {
      callbacks.current.onVisible(lastExact.current);
    }
    const wasExact = exactMode.current;
    invalidate();
    exactMode.current = true;
    video.current?.pause();
    setPlaying(false);
    setIndex(n);
    setExact(true);
    // Keep the existing decoded still until its replacement is ready.
    if (!wasExact) setExactReady(false);
  }
  function playPause() {
    const v = video.current;
    if (!v || !clip) return;
    if (!v.paused) {
      v.pause();
      return;
    }
    const n = exact ? index : (shown.current ?? interval.start);
    const slot = clip.frames.indexOf(n);
    if (slot < 0) return;
    invalidate();
    setError("");
    setOverlay(false);
    if (exact || v.ended) v.currentTime = clip.media_times[slot];
    v.playbackRate = speed;
    void v.play().catch(() => {
      setError(
        "Playback could not start. Retry Play or inspect individual frames.",
      );
      setPlaying(false);
    });
  }
  const seekRef = useRef(seek),
    toggleRef = useRef(playPause);
  seekRef.current = seek;
  toggleRef.current = playPause;
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
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
        seekRef.current(
          Math.max(
            minimum,
            Math.min(
              maximum,
              (position ?? index) + (e.key === "ArrowLeft" ? -1 : 1),
            ),
          ),
        );
      }
      if (e.code === "Space") {
        e.preventDefault();
        toggleRef.current();
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [position, index, minimum, maximum]);
  const displayed = position ?? index;
  const slot = clip?.frames.indexOf(displayed) ?? -1;
  const remaining = Array.from(
    { length: interval.end - interval.start + 1 },
    (_, i) => interval.start + i,
  ).filter((n) => !seenFrames.has(n));
  const percent = (n: number) =>
    `${(100 * (n - minimum)) / Math.max(1, maximum - minimum)}%`;
  return (
    <section className="viewer panel" aria-label="Interval viewer">
      <div className="image-stage buffered-stage" ref={stage}>
        {clip && (
          <video
            ref={video}
            src={clip.video_url}
            preload="auto"
            playsInline
            muted
            aria-label="Buffered interval video"
            onLoadedMetadata={() => {
              if (video.current)
                video.current.currentTime =
                  clip.media_times[clip.frames.indexOf(interval.start)];
            }}
            onPlay={() => {
              setHasPlayed(true);
              setPlaying(true);
              setBuffering(false);
            }}
            onPlaying={() => {
              previous.current = null;
              active.current = true;
              setBuffering(false);
            }}
            onSeeking={() => {
              invalidate();
            }}
            onSeeked={() => {
              previous.current = null;
              active.current = !!video.current && !video.current.paused;
            }}
            onWaiting={() => {
              invalidate();
              setBuffering(true);
            }}
            onStalled={() => {
              invalidate();
              setBuffering(true);
            }}
            onTimeUpdate={() => {
              const v = video.current;
              if (v && !v.requestVideoFrameCallback && !v.paused) {
                const i = frameAt(clip, v.currentTime);
                if (i !== null) {
                  shown.current = clip.frames[i];
                  setPosition(clip.frames[i]);
                  setExact(false);
                }
              }
            }}
            onPause={() => {
              invalidate();
              setPlaying(false);
              setBuffering(false);
              if (!exactMode.current && shown.current !== null) {
                setIndex(shown.current);
                setExact(true);
                setExactReady(false);
              }
            }}
            onEnded={() => {
              invalidate();
              setPlaying(false);
            }}
            onError={() => {
              invalidate();
              setPlaying(false);
              setError(
                "Video preview unavailable. Exact-frame inspection remains available.",
              );
            }}
          />
        )}
        {exact && (
          <div
            className="buffered-inspection"
            style={{ opacity: exactReady ? 1 : 0 }}
          >
            <ExactFrame
              recording={recording}
              retryToken={frameRetry}
              index={index}
              overlay={overlay}
              onFailure={() =>
                setError(
                  "Exact frame unavailable. Retry the frame before certifying coverage.",
                )
              }
              onVisible={(f) => {
                if (f.frame_index !== target.current || !exactMode.current)
                  return;
                lastExact.current = f;
                setExactReady(true);
                setPosition(f.frame_index);
                shown.current = f.frame_index;
                requestAnimationFrame(() => {
                  if (
                    exactMode.current &&
                    f.frame_index === target.current &&
                    visible.current &&
                    document.visibilityState === "visible"
                  )
                    callbacks.current.onVisible(f);
                });
              }}
            />
          </div>
        )}
        {(!clip || buffering || error || result.error) && (
          <div
            className="frame-status"
            role={error || result.error ? "alert" : "status"}
          >
            {error ||
              result.error ||
              (clip
                ? "Buffering…"
                : "Preparing cached video… Exact-frame inspection is available.")}
            {error && (
              <button
                onClick={() => {
                  setError("");
                  setFrameRetry((n) => n + 1);
                  video.current?.load();
                }}
              >
                Retry
              </button>
            )}
          </div>
        )}
      </div>
      <div className="playback-controls">
        <div className="playback-context">
          <span className="original-punch">
            <small>Review interval</small>
            <strong>
              {interval.start}–{interval.end}
            </strong>
          </span>
        </div>
        <div className="playback-buttons">
          <button
            aria-label="Previous frame"
            disabled={displayed <= minimum}
            onClick={() => seek(displayed - 1)}
          >
            ‹
          </button>
          <button
            aria-label={playing ? "Pause" : "Play"}
            className="play-button"
            disabled={!clip || !!result.error}
            onClick={playPause}
          >
            {playing ? "Ⅱ" : "▶"}
          </button>
          <button
            aria-label="Next frame"
            disabled={displayed >= maximum}
            onClick={() => seek(displayed + 1)}
          >
            ›
          </button>
        </div>
        <div className="playback-options">
          <span className="playback-position">
            {!supported && !exact ? "≈ " : ""}Frame {displayed.toLocaleString()}
          </span>
          <label className="toggle">
            <input
              type="checkbox"
              checked={overlay}
              onChange={(e) => {
                if (e.target.checked) seek(displayed);
                setOverlay(e.target.checked);
              }}
            />
            Skeleton (paused)
          </label>
          <label className="speed-label">
            <select
              aria-label="Playback speed"
              value={speed}
              onChange={(e) => {
                const rate = Number(e.target.value);
                setSpeed(rate);
                previous.current = null;
                if (video.current) video.current.playbackRate = rate;
              }}
            >
              {[1, 0.5, 0.25].map((r) => (
                <option key={r} value={r}>
                  {r}×
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>
      <div className="review-timeline">
        <div
          className="punch-range"
          style={{
            left: percent(interval.start),
            width: `${(100 * (interval.end - interval.start)) / Math.max(1, maximum - minimum)}%`,
          }}
        />
        {[interval.start, interval.end].map((n, i) => (
          <button
            key={i}
            className={`boundary-marker boundary-${i ? "end" : "start"}`}
            style={{ left: percent(n) }}
            aria-label={`Interval ${i ? "end" : "start"} · ${n}`}
            onClick={() => seek(n)}
          >
            <span className="boundary-tooltip">
              {i ? "End" : "Start"} · {n}
            </span>
          </button>
        ))}
        <input
          type="range"
          aria-label="Native frame"
          min={minimum}
          max={maximum}
          value={displayed}
          aria-valuetext={`Showing frame ${displayed}`}
          onChange={(e) => seek(Number(e.target.value))}
        />
      </div>
      {hasPlayed && !playing && remaining.length > 0 && (
        <div className="coverage-gap">
          <small>
            {remaining.length}{" "}
            {remaining.length === 1 ? "frame still needs" : "frames still need"}{" "}
            inspection.
          </small>
          <button onClick={() => seek(remaining[0])}>Inspect next</button>
        </div>
      )}
      <details className="viewer-options">
        <summary>Playback & inspection</summary>
        <p>
          {clip && slot >= 0
            ? `Source PTS: ${clip.source_times[slot]} s. `
            : ""}
          Frame stepping uses the original footage. Slow playback can help cover
          frames skipped by the browser.
        </p>
        {!supported && (
          <p>
            This browser cannot report individual presented frames. Use
            exact-frame inspection to complete review coverage.
          </p>
        )}
        <label>
          Go to native frame{" "}
          <input
            type="number"
            min={minimum}
            max={maximum}
            value={index}
            onChange={(e) => seek(Number(e.target.value))}
          />
        </label>
        <button disabled={!remaining.length} onClick={() => seek(remaining[0])}>
          Inspect next unreviewed frame
          {remaining.length ? ` (${remaining.length})` : ""}
        </button>
        <p>
          Only individually observed frames count. Hidden playback, buffering
          and skipped frames never fill review coverage; confirmation still
          requires your explicit attestation.
        </p>
      </details>
    </section>
  );
}
