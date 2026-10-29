import { useCallback, useEffect, useRef, useState } from "react";
import {
  clock,
  frameIndex,
  frameTime,
  type Bounds,
  type VideoItem,
} from "./model";
import { Icon, IconButton } from "./ui";

export function usePlayer(video: VideoItem) {
  const media = useRef<HTMLVideoElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const [current, setCurrent] = useState(0),
    [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1),
    [ready, setReady] = useState(false),
    [error, setError] = useState("");
  const stopAt = useRef<Bounds | null>(null);
  const presented = useRef(0);
  const seek = useCallback(
    (time: number) => {
      const value = frameTime(video, time);
      media.current?.pause();
      stopAt.current = null;
      setPlaying(false);
      setCurrent(value);
      presented.current = value;
      if (media.current) media.current.currentTime = value;
    },
    [video],
  );
  const step = useCallback(
    (direction: number) => {
      if (!video.frameTimes) return;
      const index = frameIndex(video.frameTimes, presented.current);
      seek(
        video.frameTimes[
          Math.max(0, Math.min(video.frameTimes.length - 1, index + direction))
        ],
      );
    },
    [video, seek],
  );
  async function play(bounds?: Bounds) {
    const element = media.current;
    if (!element) return;
    setError("");
    if (bounds) {
      element.currentTime = bounds.start;
      setCurrent(bounds.start);
    } else if (element.ended) element.currentTime = 0;
    stopAt.current = bounds ?? null;
    try {
      await element.play();
      setPlaying(true);
    } catch {
      setError("Playback could not start. Press Play to try again.");
    }
  }
  function pause() {
    seek(media.current?.currentTime ?? current);
  }
  const update = useCallback(
    (time: number) => {
      const element = media.current;
      if (!element) return;
      const range = stopAt.current;
      if (range && time >= range.end - 1e-5) {
        const last = video.frameTimes
          ? (video.frameTimes.filter((t) => t < range.end - 1e-6).at(-1) ??
            range.start)
          : Math.max(range.start, range.end - 0.001);
        seek(last);
        // Late compositor callbacks must remain bounded until the next explicit seek/play.
        stopAt.current = range;
        return;
      }
      presented.current = time;
      setCurrent(time);
    },
    [seek, video],
  );
  useEffect(() => {
    const element = media.current;
    if (!element || !element.requestVideoFrameCallback) return;
    let id = 0,
      disposed = false;
    const tick: VideoFrameRequestCallback = (_now, data) => {
      if (disposed) return;
      if (!element.paused) update(data.mediaTime);
      id = element.requestVideoFrameCallback(tick);
    };
    id = element.requestVideoFrameCallback(tick);
    return () => {
      disposed = true;
      element.cancelVideoFrameCallback(id);
    };
  }, [update]);
  async function fullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await stage.current?.requestFullscreen();
    } catch {
      setError("Fullscreen is unavailable in this browser.");
    }
  }
  return {
    media,
    stage,
    current,
    playing,
    speed,
    ready,
    error,
    seek,
    step,
    play,
    pause,
    fullscreen,
    setPlaying,
    setReady,
    setError,
    update,
    cycleSpeed: () => {
      const value = speed === 1 ? 0.5 : speed === 0.5 ? 0.25 : 1;
      setSpeed(value);
      if (media.current) media.current.playbackRate = value;
    },
  };
}
export type Player = ReturnType<typeof usePlayer>;
export function VideoStage({
  video,
  player,
  initialBounds,
}: {
  video: VideoItem;
  player: Player;
  initialBounds?: Bounds;
}) {
  const [loadedFrame, setLoadedFrame] = useState<number | null>(null);
  const index = video.frameTimes
    ? frameIndex(video.frameTimes, player.current)
    : 0;
  useEffect(() => {
    if (initialBounds) player.seek(initialBounds.start);
  }, [initialBounds?.start]);
  return (
    <div
      className="video-stage"
      ref={player.stage}
      tabIndex={0}
      aria-label="Video preview. Space to play or pause."
      style={{ aspectRatio: `${video.width} / ${video.height}` }}
    >
      <video
        ref={player.media}
        src={video.src}
        poster={video.poster}
        preload="auto"
        playsInline
        onLoadedData={() => player.setReady(true)}
        onPlay={() => player.setPlaying(true)}
        onPause={() => player.setPlaying(false)}
        onEnded={() => {
          player.setPlaying(false);
          player.seek(
            video.frameTimes?.at(-1) ?? Math.max(0, video.duration - 0.001),
          );
        }}
        onError={() =>
          player.setError(
            "This video could not be played. Return to Videos and choose another file.",
          )
        }
        onTimeUpdate={(e) => {
          if (player.playing && !e.currentTarget.requestVideoFrameCallback)
            player.update(e.currentTarget.currentTime);
        }}
      />
      {video.kind === "sample" && !player.playing && (
        <img
          key={index}
          className="exact-frame"
          src={`/demo/frames/${index.toString().padStart(4, "0")}.png`}
          alt={`Sample source frame ${index + 1}`}
          onLoad={() => setLoadedFrame(index)}
        />
      )}
      {video.kind === "sample" && !player.playing && loadedFrame !== index && (
        <span className="stage-status" role="status">
          Loading frame…
        </span>
      )}
      {!player.playing && (
        <button
          className="stage-play"
          aria-label={initialBounds ? "Replay selected punch" : "Play video"}
          disabled={!player.ready}
          onClick={() => player.play(initialBounds)}
        >
          <Icon name="play" size={28} />
        </button>
      )}
      {player.error && (
        <div className="stage-error" role="alert">
          {player.error}
        </div>
      )}
    </div>
  );
}
export function Transport({
  video,
  player,
}: {
  video: VideoItem;
  player: Player;
}) {
  return (
    <div className="transport">
      <div className="cluster transport-buttons">
        <IconButton
          icon={player.playing ? "pause" : "play"}
          label={player.playing ? "Pause" : "Play"}
          disabled={!player.ready}
          onClick={() => (player.playing ? player.pause() : player.play())}
        />
        <span className="control-divider" />
        <IconButton
          icon="previous"
          label="Previous frame"
          disabled={!video.frameTimes || player.current < 0.001}
          onClick={() => player.step(-1)}
        />
        <IconButton
          icon="next"
          label="Next frame"
          disabled={
            !video.frameTimes || player.current >= video.frameTimes.at(-1)!
          }
          onClick={() => player.step(1)}
        />
      </div>
      <div className="timecode" aria-label="Playback time">
        <span>{clock(player.current, true)}</span>
        <span className="timecode-separator">/</span>
        <span className="muted">{clock(video.duration, true)}</span>
      </div>
      <div className="cluster">
        <button
          className="speed-button"
          aria-label={`Playback speed ${player.speed} times. Change speed`}
          onClick={player.cycleSpeed}
        >
          {player.speed}×
        </button>
        <IconButton
          icon="expand"
          label="Fullscreen"
          onClick={player.fullscreen}
        />
      </div>
    </div>
  );
}
