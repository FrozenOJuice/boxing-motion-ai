import { useAnalysisTask } from "./useAnalysisTask";

type Range = {
  bout_id: string;
  start_frame: number;
  stop_frame_exclusive: number;
  start_media_time: number;
  end_media_time: number;
  left_censored: boolean;
  right_censored: boolean;
};
type ActivityResult = {
  bouts: Range[];
  observable_frames: number;
  total_frames: number;
  unobservable_intervals: {
    reason: string;
    start_media_time: number;
    end_media_time: number;
  }[];
};

export default function ActivityScan({
  videoId,
  disabled,
  onBusy,
  onSeek,
  refreshKey = 0,
}: {
  videoId: string;
  disabled: boolean;
  onBusy: (value: boolean) => void;
  onSeek: (frame: number) => void;
  refreshKey?: number;
}) {
  const {
    task: scan,
    error,
    processing,
    start,
  } = useAnalysisTask<ActivityResult>(
    videoId,
    "activity",
    "scan",
    disabled,
    onBusy,
    refreshKey,
  );

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
