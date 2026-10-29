import { useEffect, useRef, useState } from "react";

type Range = {
  bout_id: string;
  start_frame: number;
  stop_frame_exclusive: number;
  start_media_time: number;
  end_media_time: number;
  left_censored: boolean;
  right_censored: boolean;
};
type Scan = {
  video_id: string;
  status: "not_started" | "processing" | "ready" | "failed";
  message?: string;
  scan_wall_seconds?: number;
  result?: {
    bouts: Range[];
    observable_frames: number;
    total_frames: number;
    unobservable_intervals: {
      reason: string;
      start_media_time: number;
      end_media_time: number;
    }[];
  };
};

export default function ActivityScan({
  videoId,
  disabled,
  onBusy,
  onSeek,
}: {
  videoId: string;
  disabled: boolean;
  onBusy: (value: boolean) => void;
  onSeek: (frame: number) => void;
}) {
  const [value, setScan] = useState<Scan | null>(null);
  const scan = value?.video_id === videoId ? value : null;
  const [error, setError] = useState("");
  const [requesting, setRequesting] = useState(false);
  const generation = useRef(0);
  const locked = useRef(false);
  const processing = requesting || scan?.status === "processing";
  useEffect(() => {
    onBusy(processing);
    return () => onBusy(false);
  }, [processing, onBusy]);

  async function read(method: "GET" | "POST", signal?: AbortSignal) {
    const response = await fetch(
      `/api/analyze/videos/${videoId}/${method === "POST" ? "scan" : "activity"}`,
      method === "POST"
        ? {
            method,
            headers: {
              "Content-Type": "application/json",
              "X-Review-Client": "dashboard-v0.2",
            },
            body: "{}",
          }
        : { signal },
    );
    const value = await response.json();
    if (!response.ok)
      throw new Error(value.detail || "Activity scan failed. Retry.");
    if (value.video_id !== videoId)
      throw new Error("Activity result does not match this video.");
    return value as Scan;
  }

  useEffect(() => {
    const controller = new AbortController();
    const current = ++generation.current;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const value = await read("GET", controller.signal);
        if (current !== generation.current) return;
        setScan(value);
        setError("");
        if (value.status === "processing") timer = setTimeout(poll, 1000);
      } catch (e) {
        if (!controller.signal.aborted && current === generation.current) {
          setError((e as Error).message);
          setScan(null);
        }
      }
    }
    void poll();
    return () => {
      generation.current++;
      controller.abort();
      clearTimeout(timer);
    };
    // A new processing scan starts polling; a video change invalidates old requests.
  }, [videoId, scan?.status === "processing"]);

  async function start() {
    if (locked.current || disabled || processing) return;
    locked.current = true;
    const current = ++generation.current;
    setRequesting(true);
    setError("");
    try {
      const value = await read("POST");
      if (current === generation.current) setScan(value);
    } catch (e) {
      if (current === generation.current) {
        setError((e as Error).message);
        setScan(null);
      }
    } finally {
      locked.current = false;
      setRequesting(false);
    }
  }

  return (
    <section className="activity-scan" aria-label="Automatic activity scan">
      <div className="activity-scan-heading">
        <strong>Possible punching activity</strong>
        <button disabled={disabled || processing} onClick={start}>
          {processing ? "Scanning activity…" : "Scan for activity"}
        </button>
      </div>
      <small>
        Experimental · Training-only smoke model · Activity bouts, not
        individual punches
      </small>
      {processing && (
        <p role="status">Scanning the full video using cached poses…</p>
      )}
      {(error || scan?.status === "failed") && (
        <p role="alert">
          {error || scan?.message} Use Scan for activity to retry.
        </p>
      )}
      {scan?.status === "ready" && scan.result && (
        <div aria-live="polite">
          {scan.result.bouts.length === 0 ? (
            <p>No activity ranges emitted. This does not confirm background.</p>
          ) : (
            <ul
              className="activity-ranges"
              aria-label="Detected activity ranges"
            >
              {scan.result.bouts.map((bout) => (
                <li key={bout.bout_id}>
                  <button
                    disabled={disabled}
                    onClick={() => onSeek(bout.start_frame)}
                  >
                    {bout.start_media_time.toFixed(3)}–
                    {bout.end_media_time.toFixed(3)} s
                    <span>
                      Frames {bout.start_frame + 1}–{bout.stop_frame_exclusive}
                    </span>
                    {(bout.left_censored || bout.right_censored) && (
                      <span>Boundary interrupted / censored</span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
          {scan.result.unobservable_intervals.length > 0 && (
            <details>
              <summary>Unobservable intervals — not background</summary>
              <ul>
                {scan.result.unobservable_intervals.map((range, i) => (
                  <li key={i}>
                    {range.start_media_time.toFixed(3)}–
                    {range.end_media_time.toFixed(3)} s ·{" "}
                    {range.reason.replaceAll("_", " ")}
                  </li>
                ))}
              </ul>
            </details>
          )}
          <small>
            Observable pose frames: {scan.result.observable_frames}/
            {scan.result.total_frames}
            {scan.scan_wall_seconds !== undefined &&
              ` · Scan: ${scan.scan_wall_seconds.toFixed(2)} s`}
            {" · Processing and coverage, not recognition accuracy"}
          </small>
        </div>
      )}
    </section>
  );
}
