import { useCallback, useEffect, useRef, useState } from "react";
import ActivityScan from "./ActivityScan";

export type Prediction = {
  status: "available" | "unavailable";
  prediction?: string;
  scores?: Record<string, number>;
  reason?: string;
  start_frame: number;
  stop_frame_exclusive: number;
  inference_seconds?: number;
  classification_wall_seconds?: number;
};
type Video = {
  frame_count: number;
  duration_seconds: number;
  width: number;
  height: number;
  rotation: number;
  decoding_seconds: number;
  frames: {
    frame_index: number;
    media_time: number;
    timestamp_exact: string;
  }[];
};
export type Job = {
  id: string;
  filename: string;
  status: "uploading" | "queued" | "processing" | "ready" | "failed";
  message: string;
  progress?: { stage: string; frames: number; total?: number };
  video?: Video;
  poses?: {
    pose_seconds: number;
    pose_coverage: {
      anchor_frames: number;
      one_arm_frames: number;
      missing_frames: number;
    };
  };
  processing_wall_seconds?: number;
  preview_url?: string;
  predictions?: Prediction[];
};
const label = (s: string) => s.replaceAll("_", " ");
const headers = {
  "Content-Type": "application/json",
  "X-Review-Client": "dashboard-v0.2",
};
async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok)
    throw new Error(
      typeof value.detail === "string"
        ? value.detail
        : "The request failed. Please try again.",
    );
  return value as T;
}

export default function Analyze() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selected, setSelected] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const load = useCallback(async () => {
    const value = await api<{ videos: Job[] }>("/api/analyze/videos");
    setJobs(value.videos);
    setSelected((current) => current || value.videos.at(-1)?.id || "");
  }, []);
  useEffect(() => {
    load().catch((e) => setError(e.message));
  }, [load]);
  useEffect(() => {
    setJob(null);
    if (!selected) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    const poll = async () => {
      try {
        const value = await api<Job>(`/api/analyze/videos/${selected}`, {
          signal: controller.signal,
        });
        if (disposed) return;
        setJob(value);
        if (["uploading", "queued", "processing"].includes(value.status))
          timer = setTimeout(poll, 1000);
      } catch (e) {
        if (!disposed) setError((e as Error).message);
      }
    };
    poll();
    return () => {
      disposed = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, [selected]);
  async function upload(file?: File) {
    if (!file || uploading) return;
    setError("");
    if (
      !/\.(mp4|mov|mkv|webm|m4v)$/i.test(file.name) ||
      !file.size ||
      file.size > 150 * 1024 * 1024
    ) {
      setError(
        "Choose a nonempty MP4, MOV, MKV, WebM or M4V video, up to 150 MiB.",
      );
      return;
    }
    setUploading(true);
    try {
      const value = await api<{ id: string }>("/api/analyze/uploads", {
        method: "POST",
        headers: {
          ...headers,
          "Content-Type": "application/octet-stream",
          "X-Upload-Name": encodeURIComponent(file.name),
        },
        body: file,
      });
      await load();
      setSelected(value.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
      if (input.current) input.current.value = "";
    }
  }
  async function remove() {
    if (!selected || removing) return;
    setRemoving(true);
    try {
      await api(`/api/analyze/videos/${selected}/remove`, {
        method: "POST",
        headers,
        body: "{}",
      });
      setSelected("");
      setJob(null);
      setError("");
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRemoving(false);
    }
  }
  return (
    <div
      className={`analyze-page ${job?.status === "ready" ? "has-video" : ""}`}
    >
      <header className="analyze-heading">
        <span className="eyebrow">ASSISTED PUNCH RECOGNITION</span>
        <h1>See what your punch looks like to AI.</h1>
        <p>
          Upload a short shadowboxing video. Scan for possible activity, or
          select one punch for an experimental classification.
        </p>
      </header>
      <section className="panel upload-panel">
        <div>
          <h2>Your video. Processed locally.</h2>
          <p>
            One boxer, a continuous take, arms and hips in view. Up to 30
            seconds, 60 FPS and 150 MiB.
          </p>
          <small>MP4 · MOV · MKV · WebM · M4V · No cloud upload</small>
        </div>
        <input
          ref={input}
          aria-label="Upload shadowboxing video"
          type="file"
          accept=".mp4,.mov,.mkv,.webm,.m4v"
          disabled={uploading}
          onChange={(e) => upload(e.target.files?.[0])}
        />
        <button
          className="primary"
          disabled={uploading || analyzing}
          onClick={() => input.current?.click()}
        >
          {uploading ? "Uploading locally…" : "Upload video"}
        </button>
      </section>
      {error && (
        <div className="analysis-error" role="alert">
          {error}
        </div>
      )}
      {jobs.length > 0 && (
        <div className="local-video-bar">
          <label>
            Local videos{" "}
            <select
              aria-label="Local video"
              disabled={analyzing || removing}
              value={selected}
              onChange={(e) => {
                setSelected(e.target.value);
                setError("");
              }}
            >
              {jobs.map((j) => (
                <option key={j.id} value={j.id}>
                  {j.filename}
                </option>
              ))}
            </select>
          </label>
          <button
            disabled={
              !job ||
              ["uploading", "queued", "processing"].includes(job.status) ||
              removing ||
              analyzing
            }
            onClick={remove}
          >
            {removing ? "Removing…" : "Remove local video"}
          </button>
        </div>
      )}
      {selected && !job && <p role="status">Loading local video…</p>}
      {job && job.status !== "ready" && (
        <section className="panel processing-panel" role="status">
          <h2>
            {job.status === "failed"
              ? "Couldn’t process this video"
              : job.message}
          </h2>
          <p>
            {job.status === "failed"
              ? job.message
              : job.progress
                ? `${label(job.progress.stage)} · ${job.progress.frames}${job.progress.total ? ` / ${job.progress.total}` : ""} frames`
                : "Your video stays on this computer. Processing may take a few minutes."}
          </p>
          {job.progress?.total && (
            <progress value={job.progress.frames} max={job.progress.total} />
          )}
        </section>
      )}
      {job?.status === "ready" && (
        <PunchSelection key={job.id} job={job} onBusy={setAnalyzing} />
      )}
      <p className="analysis-footnote">
        Experimental predictions, not verified punch labels. Activity scans show
        possible bouts; six-class recognition still requires selecting one
        punch. Local videos and results stay here until you remove them.
      </p>
    </div>
  );
}

export function frameAtTime(frames: Video["frames"], time: number) {
  let lo = 0,
    hi = frames.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (frames[mid].media_time <= time + 0.000001) lo = mid + 1;
    else hi = mid;
  }
  return Math.max(0, lo - 1);
}

export function PunchSelection({
  job,
  onBusy,
}: {
  job: Job;
  onBusy: (busy: boolean) => void;
}) {
  const video = job.video!;
  const element = useRef<HTMLVideoElement>(null);
  const [index, setIndex] = useState(0);
  const [exact, setExact] = useState(true);
  const [loaded, setLoaded] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const [start, setStart] = useState<number | null>(null);
  const [end, setEnd] = useState<number | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [history, setHistory] = useState(job.predictions ?? []);
  const [busy, setBusy] = useState(false);
  const [scanning, setScanning] = useState(false);
  useEffect(() => {
    onBusy(busy || scanning);
  }, [busy, scanning, onBusy]);
  const lock = useRef(false);
  const replayEnd = useRef<number | null>(null);
  const replayLast = useRef<number | null>(null);
  const request = useRef(0);
  const [retry, setRetry] = useState(0);
  const canClassify = start !== null && end !== null && end > start;
  const frames = video.frames;
  const selectFrame = (n: number) => {
    if (busy) return;
    element.current?.pause();
    replayEnd.current = null;
    replayLast.current = null;
    setPlaying(false);
    setIndex(Math.max(0, Math.min(video.frame_count - 1, n)));
    setExact(true);
    setError("");
  };
  useEffect(() => {
    setLoaded(null);
  }, [index, retry]);
  useEffect(() => {
    const media = element.current;
    if (!media) return;
    type FrameVideo = HTMLVideoElement & {
      requestVideoFrameCallback?: (
        callback: (now: number, metadata: { mediaTime: number }) => void,
      ) => number;
      cancelVideoFrameCallback?: (id: number) => void;
    };
    const source = media as FrameVideo;
    let callback = 0,
      disposed = false;
    const update = (_now: number, metadata: { mediaTime: number }) => {
      if (disposed) return;
      if (!media.paused) {
        setIndex(frameAtTime(frames, metadata.mediaTime));
        if (
          replayEnd.current !== null &&
          metadata.mediaTime >= replayEnd.current
        ) {
          media.pause();
          setPlaying(false);
          if (replayLast.current !== null) {
            setIndex(replayLast.current);
            setExact(true);
          }
        }
      }
      callback = source.requestVideoFrameCallback?.(update) ?? 0;
    };
    callback = source.requestVideoFrameCallback?.(update) ?? 0;
    return () => {
      disposed = true;
      if (callback) source.cancelVideoFrameCallback?.(callback);
    };
  }, [frames]);
  useEffect(
    () => () => {
      request.current++;
    },
    [],
  );
  async function play(replay = false) {
    const media = element.current;
    if (!media || !ready) return;
    setError("");
    if (replay && start !== null && end !== null) {
      media.currentTime = frames[start].media_time;
      replayEnd.current = frames[end + 1]?.media_time ?? video.duration_seconds;
      replayLast.current = end;
    } else {
      replayEnd.current = null;
      replayLast.current = null;
      if (exact) media.currentTime = frames[index].media_time;
      if (media.ended) media.currentTime = 0;
    }
    setExact(false);
    try {
      await media.play();
      setPlaying(true);
    } catch {
      setError("Playback could not start. Try Play again.");
    }
  }
  function pause() {
    const media = element.current;
    if (!media) return;
    media.pause();
    setPlaying(false);
    selectFrame(frameAtTime(frames, media.currentTime));
  }
  async function classify() {
    if (!canClassify || lock.current || scanning) return;
    lock.current = true;
    setBusy(true);
    setError("");
    const generation = ++request.current;
    try {
      const result = await api<Prediction>(
        `/api/analyze/videos/${job.id}/classify`,
        {
          method: "POST",
          headers,
          body: JSON.stringify({
            start_frame: start,
            stop_frame_exclusive: end! + 1,
          }),
        },
      );
      if (generation !== request.current) return;
      setPrediction(result);
      setHistory((items) => [
        ...items.filter(
          (p) =>
            p.start_frame !== result.start_frame ||
            p.stop_frame_exclusive !== result.stop_frame_exclusive,
        ),
        result,
      ]);
    } catch (e) {
      if (generation === request.current) setError((e as Error).message);
    } finally {
      lock.current = false;
      setBusy(false);
    }
  }
  return (
    <>
      <div className="analysis-grid">
        <section className="panel selection-panel">
          <div className="selection-title">
            <h2>Review your video</h2>
            <small>
              {video.duration_seconds.toFixed(2)} s · {video.frame_count} frames
            </small>
          </div>
          <div
            className="analysis-stage"
            style={{ aspectRatio: `${video.width} / ${video.height}` }}
          >
            <video
              ref={element}
              src={job.preview_url}
              preload="auto"
              playsInline
              onLoadedData={() => setReady(true)}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => {
                setPlaying(false);
                setIndex(replayLast.current ?? video.frame_count - 1);
                setExact(true);
              }}
              onTimeUpdate={(e) => {
                if (!playing) return;
                const media = e.currentTarget;
                setIndex(frameAtTime(frames, media.currentTime));
                if (
                  replayEnd.current !== null &&
                  media.currentTime >= replayEnd.current
                ) {
                  media.pause();
                  setPlaying(false);
                  if (replayLast.current !== null) {
                    setIndex(replayLast.current);
                    setExact(true);
                  }
                }
              }}
              onError={() =>
                setError(
                  "Video playback failed. Remove this video and upload it again.",
                )
              }
            />
            {exact && (
              <img
                key={`${index}:${retry}`}
                src={`/api/analyze/videos/${job.id}/frames/${index}`}
                alt={`Selected source frame ${index + 1}`}
                onLoad={() => setLoaded(index)}
                onError={() =>
                  setError(
                    "This exact frame could not load. Retry frame or choose another frame.",
                  )
                }
              />
            )}
            {exact && loaded !== index && (
              <span className="frame-loading">Loading exact frame…</span>
            )}
          </div>
          <div className="analysis-playback">
            <button
              disabled={!ready || busy}
              onClick={() => (playing ? pause() : play())}
            >
              {playing ? "Pause" : "Play"}
            </button>
            <button
              aria-label="Previous source frame"
              disabled={busy || index === 0}
              onClick={() => selectFrame(index - 1)}
            >
              ← Frame
            </button>
            <button
              aria-label="Next source frame"
              disabled={busy || index === video.frame_count - 1}
              onClick={() => selectFrame(index + 1)}
            >
              Frame →
            </button>
            <span>
              Frame {index + 1} / {video.frame_count} ·{" "}
              {frames[index].media_time.toFixed(3)} s
            </span>
          </div>
          <label className="analysis-timeline">
            Video timeline
            <input
              aria-label="Video timeline"
              type="range"
              min="0"
              max={video.frame_count - 1}
              value={index}
              disabled={busy}
              onChange={(e) => selectFrame(Number(e.target.value))}
            />
          </label>
          <ActivityScan
            videoId={job.id}
            disabled={busy}
            onBusy={setScanning}
            onSeek={selectFrame}
          />
          <div className="boundary-controls">
            <button
              disabled={busy || !exact || loaded !== index}
              onClick={() => {
                setStart(index);
                setPrediction(null);
              }}
            >
              Set punch start
            </button>
            <button
              disabled={busy || !exact || loaded !== index}
              onClick={() => {
                setEnd(index);
                setPrediction(null);
              }}
            >
              Set punch end
            </button>
            <button
              disabled={!canClassify || !ready || busy}
              onClick={() => play(true)}
            >
              Replay selection
            </button>
          </div>
          <div className="selected-bounds">
            <button
              disabled={start === null || busy}
              onClick={() => start !== null && selectFrame(start)}
            >
              Start: {start === null ? "not set" : `frame ${start + 1}`}
            </button>
            <span>→</span>
            <button
              disabled={end === null || busy}
              onClick={() => end !== null && selectFrame(end)}
            >
              End: {end === null ? "not set" : `frame ${end + 1}`}
            </button>
            <small>
              {canClassify
                ? `${(frames[end!].media_time - frames[start!].media_time).toFixed(3)} s between boundary frames`
                : "Include one complete punch. Start must precede end."}
            </small>
          </div>
          <p>
            Pause or step to inspect a frame, then set each boundary. End frame
            is included. Select up to 3 seconds.
          </p>
        </section>
        <section className="panel prediction-panel">
          <h2>2. Classify your selection</h2>
          <p>
            Choose a movement you believe is one punch. The model assigns one of
            six punch types; it cannot verify that a punch occurred.
          </p>
          <button
            className="primary classify-button"
            disabled={!canClassify || busy || scanning}
            onClick={classify}
          >
            {busy ? "Running trained model…" : "Classify selected punch"}
          </button>
          {prediction && (
            <div className="prediction-result" role="status">
              {prediction.status === "unavailable" ? (
                <>
                  <h3>Prediction unavailable</h3>
                  <p>{prediction.reason}</p>
                </>
              ) : (
                <>
                  <span className="eyebrow">MODEL PREDICTION</span>
                  <h3>{label(prediction.prediction!)}</h3>
                  <p>Uncalibrated scores · Not verified confidence</p>
                  <div className="model-scores">
                    {Object.entries(prediction.scores!)
                      .sort((a, b) => b[1] - a[1])
                      .map(([name, score]) => (
                        <div key={name}>
                          <span>{label(name)}</span>
                          <meter
                            min="0"
                            max="1"
                            value={score}
                            aria-label={`${label(name)} model score`}
                          />
                          <strong>{score.toFixed(3)}</strong>
                        </div>
                      ))}
                  </div>
                </>
              )}
            </div>
          )}
          {!prediction && (
            <div className="prediction-empty">
              Your prediction will appear here after you select a punch.
            </div>
          )}
          {history.length > 0 && (
            <div className="analysis-history">
              <h3>Selections</h3>
              {history.map((p) => (
                <button
                  key={`${p.start_frame}:${p.stop_frame_exclusive}`}
                  disabled={busy}
                  onClick={() => {
                    selectFrame(p.start_frame);
                    setStart(p.start_frame);
                    setEnd(p.stop_frame_exclusive - 1);
                    setPrediction(p);
                  }}
                >
                  Frames {p.start_frame + 1}–{p.stop_frame_exclusive}
                  <span>
                    {p.prediction ? label(p.prediction) : "Unavailable"}
                  </span>
                </button>
              ))}
            </div>
          )}
        </section>
      </div>
      {error && (
        <div className="analysis-error" role="alert">
          {error}{" "}
          <button
            onClick={() => {
              setError("");
              setRetry((n) => n + 1);
            }}
          >
            Retry frame
          </button>
        </div>
      )}
      <details className="analysis-details">
        <summary>Processing & model details</summary>
        <p>
          Pose extraction: {job.poses!.pose_seconds.toFixed(2)} s · Total
          processing: {job.processing_wall_seconds?.toFixed(2)} s · Reliable
          torso frames:{" "}
          {(job.poses!.pose_coverage.anchor_frames * 100).toFixed(1)}% · At
          least one reliable arm:{" "}
          {(job.poses!.pose_coverage.one_arm_frames * 100).toFixed(1)}%.
        </p>
        <p>
          These measure processing and pose coverage, not recognition accuracy.
          Poses are cached for additional selections. Pinned Level 1 XY +
          velocity TCN, seed 20261021, epoch 32. Experimental transfer to new
          boxers is unvalidated. Continuous footage and single-boxer framing are
          user requirements, not automatically verified.
        </p>
        {prediction?.inference_seconds !== undefined && (
          <p>
            Model inference: {prediction.inference_seconds.toFixed(4)} s ·
            Classification including worker/model startup:{" "}
            {prediction.classification_wall_seconds?.toFixed(2)} s.
          </p>
        )}
      </details>
    </>
  );
}
