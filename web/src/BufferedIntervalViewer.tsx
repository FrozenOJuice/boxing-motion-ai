import { type ReactNode, useEffect, useRef, useState } from "react";
import { type Frame, type Result, useRequest } from "./api";
import { ExactFrame } from "./Viewer";
import {
  type IntervalAnnotation,
  annotationLabel,
  annotationSegments,
} from "./intervalAnnotations";
import {
  type Playback,
  frameAt,
  ContinuousViewing,
  viewingProgress,
  type ViewingSpan,
  type ViewingUpdate,
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
  pilot = "original",
  interval,
  recording,
  onVisible,
  onViewing,
  seenFrames,
  annotations = [],
  showSourceAnnotations = true,
  humanEvents = [],
  onHumanSelect,
  inspectionControls,
  draftBounds,
}: {
  intervalId: string;
  pilot?: "original" | "motion-v2" | "localization-v1";
  interval: Interval;
  recording: string;
  onVisible: (frame: Frame) => void;
  onViewing: (view: ViewingUpdate) => void;
  seenFrames: Set<number>;
  annotations?: IntervalAnnotation[];
  showSourceAnnotations?: boolean;
  draftBounds?: { start: number | null; end: number | null };
  humanEvents?: {
    event_id: string;
    start: number;
    stop: number;
    punch_type?: string;
    type_uncertain?: boolean;
    kind?: "activity" | "uncertain" | "punch";
    unverified?: boolean;
  }[];
  onHumanSelect?: (id: string) => void;
  inspectionControls?: (
    frame: number,
    ready: boolean,
    inspect: () => void,
  ) => ReactNode;
}) {
  const [frameRetry, setFrameRetry] = useState(0);
  const [mediaReady, setMediaReady] = useState(false);
  const lastToggle = useRef(-Infinity);
  const result = useRequest<Result<Playback>>(
    `/api/intervals/${intervalId}/playback${pilot === "original" ? "" : `?pilot=${pilot}`}`,
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
  const resumeAt = useRef<number | null>(null);
  const [buffering, setBuffering] = useState(false),
    [supported, setSupported] = useState(true);
  const visible = useRef(false),
    active = useRef(false);
  const viewing = useRef(new ContinuousViewing());
  const [viewedSpans, setViewedSpans] = useState<ViewingSpan[]>([]);
  const shown = useRef<number | null>(null),
    target = useRef(index),
    exactMode = useRef(exact);
  target.current = index;
  exactMode.current = exact;
  const callbacks = useRef({ onVisible, onViewing });
  callbacks.current = { onVisible, onViewing };
  const minimum = clip?.frames[0] ?? interval.contextStart;
  const maximum = clip?.frames.at(-1) ?? interval.contextEnd;
  const invalidate = () => {
    viewing.current.break();
    active.current = false;
  };
  useEffect(() => {
    const element = stage.current;
    const observer =
      typeof IntersectionObserver !== "undefined"
        ? new IntersectionObserver(
            (entries) => {
              visible.current =
                entries[entries.length - 1].intersectionRatio >= 0.95;
              if (!visible.current && !canSee()) {
                closeVisibleEpoch();
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
        closeVisibleEpoch();
        invalidate();
        video.current?.pause();
      }
    };
    const blur = () => {
      closeVisibleEpoch();
      invalidate();
      video.current?.pause();
    };
    document.addEventListener("visibilitychange", hide);
    window.addEventListener("blur", blur);
    return () => {
      observer?.disconnect();
      document.removeEventListener("visibilitychange", hide);
      window.removeEventListener("blur", blur);
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
    const tick = (_now: number, metadata: VideoFrameCallbackMetadata) => {
      if (cancelled) return;
      const slot = frameAt(clip, metadata.mediaTime, true);
      if (slot !== null) {
        shown.current = clip.frames[slot];
        if (!v.paused && !v.seeking) {
          setPosition(clip.frames[slot]);
          setExact(false);
          setExactReady(false);
        }
      }
      handle = v.requestVideoFrameCallback(tick);
    };
    handle = v.requestVideoFrameCallback(tick);
    return () => {
      cancelled = true;
      v.cancelVideoFrameCallback(handle);
      viewing.current.break();
    };
  }, [clip]);
  function canSee() {
    const rect = stage.current?.getBoundingClientRect();
    return (
      document.visibilityState === "visible" &&
      !!rect &&
      rect.width > 0 &&
      rect.height > 0 &&
      Math.max(0, Math.min(rect.bottom, innerHeight) - Math.max(rect.top, 0)) *
        Math.max(
          0,
          Math.min(rect.right, innerWidth) - Math.max(rect.left, 0),
        ) >=
        rect.width * rect.height * 0.95
    );
  }
  function closeVisibleEpoch() {
    const v = video.current;
    if (!v || !active.current || v.seeking) return;
    // End the previously visible epoch at the visibility transition, before
    // invalidating it. No subsequent hidden/offscreen clock time is credited.
    if (
      viewing.current.sample({
        media: v.currentTime,
        wall: performance.now() / 1000,
        rate: v.playbackRate as 1 | 0.5 | 0.25,
        eligible: [1, 0.5, 0.25].includes(v.playbackRate),
      })
    )
      setViewedSpans(viewing.current.snapshot());
  }
  function sampleViewing() {
    const v = video.current;
    if (!v) return;
    const rate = v.playbackRate;
    const eligible =
      active.current && canSee() && !v.seeking && [1, 0.5, 0.25].includes(rate);
    if (
      viewing.current.sample({
        media: v.currentTime,
        wall: performance.now() / 1000,
        rate: rate as 1 | 0.5 | 0.25,
        eligible,
      })
    )
      setViewedSpans(viewing.current.snapshot());
    if (v.paused || v.ended || v.readyState < 3) {
      viewing.current.break();
      active.current = false;
    }
    if (active.current && !canSee()) {
      invalidate();
      v.pause();
      setError("Playback paused. Keep the video visible to record viewing.");
    }
  }
  const sampleRef = useRef(sampleViewing);
  sampleRef.current = sampleViewing;
  useEffect(() => {
    const timer = window.setInterval(() => sampleRef.current(), 100);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    if (!clip) return;
    const progress = viewingProgress(
      clip,
      interval.start,
      interval.end + 1,
      viewedSpans,
      seenFrames,
    );
    callbacks.current.onViewing({
      ...progress,
      evidence: {
        schema_version: "continuous-viewing-1.0",
        policy: "visible-realtime-1.0",
        cache_key: clip.cache_key,
        interval_id: intervalId,
        segments: viewedSpans,
      },
    });
  }, [clip, interval.start, interval.end, intervalId, viewedSpans, seenFrames]);
  const viewProgress = clip
    ? viewingProgress(
        clip,
        interval.start,
        interval.end + 1,
        viewedSpans,
        seenFrames,
      )
    : { complete: false, playbackComplete: false, percent: 0 };
  function seek(n: number) {
    if (!Number.isInteger(n) || n < minimum || n > maximum) return;
    if (
      exactMode.current &&
      exactReady &&
      n === target.current &&
      lastExact.current?.frame_index === n &&
      canSee()
    ) {
      callbacks.current.onVisible(lastExact.current);
    }
    resumeAt.current = null;
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
    if (!v || !clip || !mediaReady) return;
    const now = performance.now();
    if (now - lastToggle.current < 200) return;
    lastToggle.current = now;
    if (playing || !v.paused) {
      v.pause();
      setPlaying(false);
      return;
    }
    const n = exact ? index : (shown.current ?? interval.start);
    const replay =
      v.ended || n >= maximum || (!hasPlayed && n === interval.start);
    const slot = replay ? 0 : clip.frames.indexOf(n);
    if (slot < 0) return;
    invalidate();
    setError("");
    setOverlay(false);
    stage.current?.scrollIntoView({ block: "start", behavior: "instant" });
    if (replay) {
      resumeAt.current = null;
      v.currentTime = clip.media_times[0];
    } else if (resumeAt.current !== null) {
      if (Math.abs(v.currentTime - resumeAt.current) > 0.0001)
        v.currentTime = resumeAt.current;
    } else if (exact) v.currentTime = clip.media_times[slot];
    v.playbackRate = speed;
    // Anchor before play() advances the clock; the asynchronous playing event
    // may arrive after the first displayed fraction of a second.
    active.current = true;
    viewing.current.sample({
      media: v.currentTime,
      wall: performance.now() / 1000,
      rate: speed as 1 | 0.5 | 0.25,
      eligible: canSee() && !v.seeking,
    });
    void v.play().catch(() => {
      invalidate();
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
  function inspectLoaded() {
    const frame = lastExact.current;
    if (!frame || !exactMode.current || !exactReady || playing) return;
    stage.current?.scrollIntoView({ block: "start", behavior: "instant" });
    requestAnimationFrame(() => {
      if (exactMode.current && frame.frame_index === target.current && canSee())
        callbacks.current.onVisible(frame);
    });
  }
  const displayed = position ?? index;
  const slot = clip?.frames.indexOf(displayed) ?? -1;
  const remaining = Array.from(
    { length: interval.end - interval.start + 1 },
    (_, i) => interval.start + i,
  ).filter((n) => !seenFrames.has(n));
  const segments = annotationSegments(
    annotations,
    interval.start,
    interval.end + 1,
  );
  const lanes = Math.max(1, ...segments.map((s) => s.lane + 1));
  const typeCounts = new Map<string, number>();
  for (const a of annotations) {
    const label = annotationLabel(a);
    typeCounts.set(label, (typeCounts.get(label) ?? 0) + 1);
  }
  const segmentName = (s: (typeof segments)[number]) =>
    `${s.label} · Row ${s.source.source_row} · Frames ${s.originalStart}–${s.originalStop - 1}${s.partial ? " · Partial overlap" : ""}`;
  const percent = (n: number) =>
    `${(100 * (n - minimum)) / Math.max(1, maximum - minimum)}%`;
  return (
    <section
      className="viewer panel"
      aria-label="Interval viewer"
      data-viewing-evidence={JSON.stringify({
        progress: viewProgress,
        segments: viewedSpans,
      })}
    >
      <div className="image-stage buffered-stage" ref={stage}>
        {clip && (
          <video
            ref={video}
            src={clip.video_url}
            preload="auto"
            playsInline
            muted
            aria-label="Buffered interval video"
            onLoadStart={() => setMediaReady(false)}
            onLoadedData={() => setMediaReady(true)}
            onLoadedMetadata={() => {
              if (video.current && !active.current && !hasPlayed)
                video.current.currentTime =
                  clip.media_times[clip.frames.indexOf(interval.start)];
            }}
            onPlay={() => {
              setHasPlayed(true);
              setPlaying(true);
              setBuffering(false);
            }}
            onPlaying={() => {
              if (!active.current) viewing.current.break();
              active.current = true;
              sampleViewing();
              setBuffering(false);
            }}
            onSeeking={() => {
              invalidate();
            }}
            onSeeked={() => {
              viewing.current.break();
              active.current = !!video.current && !video.current.paused;
              sampleViewing();
            }}
            onWaiting={() => {
              sampleViewing();
              invalidate();
              setBuffering(true);
            }}
            onStalled={() => {
              sampleViewing();
              invalidate();
              setBuffering(true);
            }}
            onTimeUpdate={() => {
              sampleViewing();
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
              resumeAt.current = video.current?.currentTime ?? null;
              sampleViewing();
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
              sampleViewing();
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
                    canSee()
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
            key={playing ? "pause" : "play"}
            aria-label={playing ? "Pause" : "Play"}
            className="play-button"
            disabled={!clip || !mediaReady || !!result.error}
            onClick={
              playing
                ? () => {
                    video.current?.pause();
                    setPlaying(false);
                  }
                : playPause
            }
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
                sampleViewing();
                viewing.current.break();
                setSpeed(rate);
                if (video.current) video.current.playbackRate = rate;
                sampleViewing();
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
      {inspectionControls?.(
        displayed,
        exact && exactReady && !playing,
        inspectLoaded,
      )}
      {humanEvents.length > 0 && (
        <div
          className="annotation-summary"
          aria-label="Human annotation summary"
        >
          <strong>
            {humanEvents.filter((e) => !e.unverified).length}{" "}
            {humanEvents.some((e) => e.kind && e.kind !== "punch")
              ? "sections you added"
              : "punches you added"}
          </strong>
          <span>
            Human labels
            {humanEvents.some((e) => e.unverified)
              ? ` · ${humanEvents.filter((e) => e.unverified).length} source suggestions (unverified)`
              : ""}
          </span>
        </div>
      )}
      {humanEvents.length > 0 && (
        <div
          className="human-event-track"
          aria-label={
            humanEvents.some((e) => e.kind && e.kind !== "punch")
              ? "Human activity timeline"
              : "Human punch timeline"
          }
          style={{
            height: `${(humanEvents.filter((e) => !e.unverified).length + Number(humanEvents.some((e) => e.unverified))) * 20 + 4}px`,
          }}
        >
          {humanEvents.map((event, lane) => (
            <button
              key={event.event_id}
              className={`source-annotation-segment human-event-segment${event.unverified ? " activity-suggestion-segment" : ""}${event.kind === "uncertain" || event.start < interval.start || event.stop > interval.end + 1 ? " partial" : ""}`}
              style={{
                left: percent(event.start),
                width: `${(100 * (event.stop - event.start)) / Math.max(1, maximum - minimum + 1)}%`,
                top: `${(event.unverified ? 0 : lane + Number(humanEvents.some((e) => e.unverified))) * 20 + 2}px`,
              }}
              title={`${event.unverified ? "Unverified source suggestion" : "Human"} · ${event.type_uncertain ? "Type uncertain" : event.kind && event.kind !== "punch" ? (event.kind === "activity" ? "Punching activity" : "Uncertain section") : event.punch_type?.replaceAll("_", " ")} · ${event.start}–${event.stop - 1}${event.start < interval.start || event.stop > interval.end + 1 ? " · Partial overlap" : ""}`}
              aria-label={`${event.unverified ? "Unverified suggestion" : "Human"} ${event.kind && event.kind !== "punch" ? "section" : "punch"} ${lane + 1}: ${event.type_uncertain ? "Type uncertain" : event.kind && event.kind !== "punch" ? (event.kind === "activity" ? "Punching activity" : "Uncertain section") : event.punch_type?.replaceAll("_", " ")}`}
              onClick={() => {
                seek(event.start);
                onHumanSelect?.(event.event_id);
              }}
            />
          ))}
          <span
            className="annotation-playhead"
            aria-hidden="true"
            style={{ left: percent(displayed) }}
          />
        </div>
      )}
      {showSourceAnnotations && (
        <div
          className="annotation-summary"
          aria-label="Original annotation summary"
        >
          <strong>
            {annotations.length} annotated{" "}
            {annotations.length === 1 ? "punch" : "punches"}
          </strong>
          <span>Original dataset · Provisional</span>
          {annotations.length > 0 && (
            <small>
              {[...typeCounts]
                .map(([label, count]) => `${label} ${count}`)
                .join(" · ")}
            </small>
          )}
        </div>
      )}
      {showSourceAnnotations && segments.length > 0 && (
        <div
          className="source-annotation-track"
          aria-label="Original punch annotation timeline"
          style={{ height: `${lanes * 18 + 4}px` }}
        >
          {segments.map((s) => (
            <button
              key={s.source.source_row}
              className={`source-annotation-segment${s.partial ? " partial" : ""}${displayed >= s.originalStart && displayed < s.originalStop ? " current" : ""}`}
              style={{
                left: percent(s.start),
                width: `${(100 * (s.stop - 1 - s.start)) / Math.max(1, maximum - minimum)}%`,
                top: `${s.lane * 18 + 2}px`,
              }}
              aria-label={segmentName(s)}
              title={segmentName(s)}
              aria-current={
                displayed >= s.originalStart && displayed < s.originalStop
                  ? "true"
                  : undefined
              }
              onClick={() => seek(Math.max(minimum, s.originalStart))}
            >
              <span className="annotation-tooltip">{segmentName(s)}</span>
            </button>
          ))}
          <span
            className="annotation-playhead"
            aria-hidden="true"
            style={{ left: percent(displayed) }}
          />
        </div>
      )}
      <div className="review-timeline">
        {draftBounds &&
          ([draftBounds.start, draftBounds.end] as const).map(
            (n, i) =>
              n !== null && (
                <button
                  key={`draft-${i}`}
                  className={`boundary-marker boundary-${i ? "end" : "start"}`}
                  style={{ left: percent(n) }}
                  aria-label={`Proposed ${i ? "end" : "start"} · ${n}`}
                  onClick={() => seek(n)}
                >
                  <span className="boundary-tooltip">
                    Proposed {i ? "end" : "start"} · {n}
                  </span>
                </button>
              ),
          )}
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
      {showSourceAnnotations && annotations.length > 0 && (
        <details className="source-annotation-list">
          <summary>Source annotations ({annotations.length})</summary>
          <div>
            {annotations.map((a) => {
              const s = segments.find(
                (segment) => segment.source.source_row === a.source_row,
              );
              const current =
                !!s &&
                displayed >= s.originalStart &&
                displayed < s.originalStop;
              return (
                <button
                  key={a.source_row}
                  disabled={!s}
                  aria-current={current ? "true" : undefined}
                  className={current ? "current" : ""}
                  onClick={() => s && seek(Math.max(minimum, s.originalStart))}
                >
                  <strong>{annotationLabel(a)}</strong>
                  <span>
                    Row {a.source_row} · Source frames {a.raw_start}–{a.raw_end}
                  </span>
                  {s?.partial && (
                    <span className="partial-label">
                      Partial overlap ·{" "}
                      {s.originalStart < interval.start
                        ? "Starts before this interval"
                        : ""}
                      {s.originalStart < interval.start &&
                      s.originalStop > interval.end + 1
                        ? "; "
                        : ""}
                      {s.originalStop > interval.end + 1
                        ? "Ends after this interval"
                        : ""}
                    </span>
                  )}
                  {!s && <span>Frame mapping unavailable</span>}
                  {!!a.exclusion_reasons?.length && (
                    <span>Previously excluded</span>
                  )}
                </button>
              );
            })}
          </div>
          <p>
            Provisional references; boundaries use the existing dataset mapping.
            Marks do not establish exhaustive coverage or verified labels.
            Dashed segments overlap only part of the interval.
          </p>
        </details>
      )}
      {hasPlayed && !playing && !viewProgress.complete && (
        <div className="coverage-gap">
          <small>Some of this interval still needs viewing.</small>
          <button
            onClick={() => {
              seek(minimum);
            }}
          >
            Replay interval
          </button>
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
            Playback frame positions are approximate in this browser. Pause for
            exact-frame inspection; continuous viewing is recorded separately.
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
          Next frame not individually inspected
          {remaining.length ? ` (${remaining.length})` : ""}
        </button>
        <p>
          Viewing credit follows visible, real-time playback. Seeking, hidden
          video and buffering do not fill gaps. Exact-frame inspection is
          tracked separately; explicit confirmation is always required.
        </p>
      </details>
    </section>
  );
}
