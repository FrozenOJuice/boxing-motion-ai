import { useRef, type KeyboardEvent, type PointerEvent } from "react";
import {
  clamp,
  clock,
  frameTime,
  nextBoundary,
  snapBoundary,
  type Bounds,
  type VideoItem,
} from "./model";

type Props = {
  video: VideoItem;
  current: number;
  bounds: Bounds | null;
  onSeek: (time: number) => void;
  onBounds: (bounds: Bounds) => void;
};
export default function Timeline({
  video,
  current,
  bounds,
  onSeek,
  onBounds,
}: Props) {
  const track = useRef<HTMLDivElement>(null);
  const drag = useRef<{
    mode: "start" | "end" | "range";
    anchor: number;
    x: number;
    moved: boolean;
  } | null>(null);
  const position = (time: number) => `${(time / video.duration) * 100}%`;
  const timeAt = (x: number) => {
    const rect = track.current!.getBoundingClientRect();
    return clamp((x - rect.left) / rect.width, 0, 1) * video.duration;
  };
  const seekBoundary = (mode: "start" | "end", value: number) => {
    const time = snapBoundary(video, value);
    onSeek(
      mode === "end"
        ? Math.max(
            0,
            video.frameTimes ? frameTime(video, time - 1e-6) : time - 0.001,
          )
        : time,
    );
  };
  function change(mode: "start" | "end", time: number) {
    if (!bounds) return;
    const lastStart = video.frameTimes
      ? frameTime(video, bounds.end - 1e-6)
      : Math.max(0, bounds.end - 0.01);
    const firstEnd = nextBoundary(video, bounds.start);
    const n = snapBoundary(video, time);
    onBounds(
      mode === "start"
        ? { start: Math.min(n, lastStart), end: bounds.end }
        : { start: bounds.start, end: Math.max(n, firstEnd) },
    );
  }
  function pointerDown(e: PointerEvent<HTMLDivElement>) {
    if (e.button !== 0) return;
    const mode = (e.target as HTMLElement).closest<HTMLElement>("[data-handle]")
      ?.dataset.handle as "start" | "end" | undefined;
    const anchor = timeAt(e.clientX);
    drag.current = {
      mode: mode ?? "range",
      anchor,
      x: e.clientX,
      moved: false,
    };
    e.currentTarget.setPointerCapture(e.pointerId);
    if (mode) {
      change(mode, anchor);
      seekBoundary(mode, anchor);
    }
  }
  function pointerMove(e: PointerEvent<HTMLDivElement>) {
    const value = drag.current;
    if (!value) return;
    if (Math.abs(e.clientX - value.x) > 4) value.moved = true;
    const now = timeAt(e.clientX);
    if (value.mode !== "range") {
      change(value.mode, now);
      seekBoundary(value.mode, now);
    } else if (value.moved) {
      const start = snapBoundary(video, Math.min(value.anchor, now));
      const end = snapBoundary(video, Math.max(value.anchor, now));
      if (end > start) onBounds({ start, end });
      seekBoundary(now >= value.anchor ? "end" : "start", now);
    }
  }
  function pointerUp(e: PointerEvent<HTMLDivElement>) {
    if (drag.current?.mode === "range" && !drag.current.moved)
      onSeek(timeAt(e.clientX));
    drag.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId))
      e.currentTarget.releasePointerCapture(e.pointerId);
  }
  function handleKey(
    e: KeyboardEvent<HTMLButtonElement>,
    mode: "start" | "end",
  ) {
    if (!bounds) return;
    const value = bounds[mode],
      times = [...(video.frameTimes ?? []), video.duration];
    let next = value;
    const step = e.shiftKey ? 0.5 : 0.01;
    if (e.key === "ArrowLeft")
      next = video.frameTimes
        ? (times.filter((t) => t < value - 1e-6).at(-1) ?? 0)
        : value - step;
    else if (e.key === "ArrowRight")
      next = video.frameTimes
        ? (times.find((t) => t > value + 1e-6) ?? video.duration)
        : value + step;
    else if (e.key === "Home")
      next = mode === "start" ? 0 : nextBoundary(video, bounds.start);
    else if (e.key === "End")
      next =
        mode === "end"
          ? video.duration
          : video.frameTimes
            ? frameTime(video, bounds.end - 1e-6)
            : Math.max(0, bounds.end - 0.01);
    else return;
    e.preventDefault();
    e.stopPropagation();
    change(mode, clamp(next, 0, video.duration));
    seekBoundary(mode, next);
  }
  return (
    <div className="timeline" aria-label="Video timeline">
      <div
        ref={track}
        className="timeline-track"
        onPointerDown={pointerDown}
        onPointerMove={pointerMove}
        onPointerUp={pointerUp}
        onPointerCancel={() => {
          drag.current = null;
        }}
      >
        <div className="timeline-film" aria-hidden="true">
          {Array.from({ length: 8 }, (_, i) => (
            <div key={i}>
              {video.kind === "sample" ? (
                <img
                  src={`/demo/frames/${Math.round(
                    (i * (video.frameTimes!.length - 1)) / 7,
                  )
                    .toString()
                    .padStart(4, "0")}.png`}
                  alt=""
                  draggable="false"
                />
              ) : (
                <span />
              )}
            </div>
          ))}
        </div>
        {bounds && (
          <div
            className="timeline-selection"
            style={{
              left: position(bounds.start),
              width: position(bounds.end - bounds.start),
            }}
            aria-hidden="true"
          />
        )}
        <input
          type="range"
          className="timeline-seek"
          aria-label="Video position"
          min="0"
          max={video.duration}
          step="0.001"
          value={Math.min(current, video.duration)}
          onChange={(e) => onSeek(Number(e.target.value))}
          onKeyDown={(e) => {
            if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
              e.preventDefault();
              const times = video.frameTimes;
              const next =
                e.key === "ArrowLeft"
                  ? (times?.filter((t) => t < current - 1e-6).at(-1) ??
                    current - 0.01)
                  : (times?.find((t) => t > current + 1e-6) ?? current + 0.01);
              onSeek(clamp(next, 0, video.duration));
            }
          }}
          aria-valuetext={clock(current, true)}
        />
        {bounds &&
          (["start", "end"] as const).map((mode) => (
            <button
              key={mode}
              className={`timeline-handle timeline-handle--${mode}`}
              style={{ left: position(bounds[mode]) }}
              data-handle={mode}
              role="slider"
              aria-label={`Punch ${mode}`}
              aria-valuemin={0}
              aria-valuemax={video.duration}
              aria-valuenow={bounds[mode]}
              aria-valuetext={clock(bounds[mode], true)}
              onKeyDown={(e) => handleKey(e, mode)}
            >
              <span />
            </button>
          ))}
        <div
          className="timeline-playhead"
          style={{ left: position(current) }}
          aria-hidden="true"
        >
          <span />
        </div>
      </div>
      <div className="timeline-ruler" aria-hidden="true">
        {Array.from({ length: 7 }, (_, i) => (
          <span key={i}>{clock((video.duration * i) / 6)}</span>
        ))}
      </div>
    </div>
  );
}
