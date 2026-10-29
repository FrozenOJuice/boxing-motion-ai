import { useEffect, useRef } from "react";
import { useAnalysisTask } from "./useAnalysisTask";

type Candidate = {
  candidate_id: string;
  start_frame: number;
  stop_frame_exclusive: number;
  start_media_time: number;
  end_media_time: number;
  classification: {
    status: "available" | "unavailable";
    prediction?: string;
    scores?: Record<string, number>;
    reason?: string;
  };
};
type Result = {
  candidates: Candidate[];
  unresolved_activity: {
    start_frame: number;
    stop_frame_exclusive: number;
    start_media_time: number;
    end_media_time: number;
    reason: string;
  }[];
  classifier_checkpoint: { sha256: string };
};
export default function AutomaticPunches({
  videoId,
  disabled,
  ready,
  onBusy,
  onReplay,
  onComplete,
}: {
  videoId: string;
  disabled: boolean;
  ready: boolean;
  onBusy: (value: boolean) => void;
  onReplay: (start: number, stop: number) => void;
  onComplete: () => void;
}) {
  const { task, error, processing, start } = useAnalysisTask<Result>(
    videoId,
    "automatic",
    "automatic",
    disabled,
    onBusy,
  );
  const completed = useRef("");
  useEffect(() => {
    if (
      task?.status === "ready" &&
      completed.current !== `${videoId}:${task.analysis_wall_seconds}`
    ) {
      completed.current = `${videoId}:${task.analysis_wall_seconds}`;
      onComplete();
    }
  }, [task, videoId, onComplete]);
  return (
    <section className="activity-scan" aria-label="Automatic punch candidates">
      <div className="activity-scan-heading">
        <strong>Automatic punch candidates</strong>
        <button disabled={disabled || processing} onClick={start}>
          {processing ? "Analyzing automatically…" : "Analyze automatically"}
        </button>
      </div>
      <small>
        Experimental · Offline motion segmentation · Candidates are not verified
        punches
      </small>
      {processing && (
        <p role="status">
          Reusing poses, checking activity, and classifying eligible arm cycles…
        </p>
      )}
      {(error || task?.status === "failed") && (
        <p role="alert">
          {error || task?.message} Run automatic analysis to retry.
        </p>
      )}
      {task?.status === "ready" && task.result && (
        <div aria-live="polite">
          {task.result.candidates.length === 0 && (
            <p>No isolated punch candidates. Activity may remain unresolved.</p>
          )}
          <ul className="automatic-candidates">
            {task.result.candidates.map((p) => (
              <li key={p.candidate_id}>
                <div className="activity-scan-heading">
                  <strong>
                    {p.classification.status === "available"
                      ? `${p.classification.prediction!.replaceAll("_", " ")} · predicted type`
                      : "Classification unavailable"}
                  </strong>
                  <button
                    disabled={disabled || !ready}
                    onClick={() =>
                      onReplay(p.start_frame, p.stop_frame_exclusive)
                    }
                  >
                    Replay candidate
                  </button>
                </div>
                <small>
                  {p.start_media_time.toFixed(3)}–{p.end_media_time.toFixed(3)}{" "}
                  s · Frames {p.start_frame + 1}–{p.stop_frame_exclusive}
                </small>
                {p.classification.status === "unavailable" ? (
                  <p>{p.classification.reason}</p>
                ) : (
                  <details>
                    <summary>
                      Six model scores · Uncalibrated, not punch evidence
                    </summary>
                    <div className="model-scores">
                      {Object.entries(p.classification.scores!)
                        .sort((a, b) => b[1] - a[1])
                        .map(([name, score]) => (
                          <div key={name}>
                            <span>{name.replaceAll("_", " ")}</span>
                            <meter
                              min="0"
                              max="1"
                              value={score}
                              aria-label={`${name.replaceAll("_", " ")} candidate score`}
                            />
                            <strong>{score.toFixed(3)}</strong>
                          </div>
                        ))}
                    </div>
                  </details>
                )}
              </li>
            ))}
          </ul>
          {task.result.unresolved_activity.length > 0 && (
            <details open>
              <summary>
                Unresolved activity · individual boundaries not established
              </summary>
              <ul className="activity-ranges">
                {task.result.unresolved_activity.map((p, i) => (
                  <li key={i}>
                    <button
                      disabled={disabled || !ready}
                      onClick={() =>
                        onReplay(p.start_frame, p.stop_frame_exclusive)
                      }
                    >
                      Replay unresolved activity {p.start_media_time.toFixed(3)}
                      –{p.end_media_time.toFixed(3)} s
                    </button>
                  </li>
                ))}
              </ul>
            </details>
          )}
          <small>
            Classifier {task.result.classifier_checkpoint.sha256.slice(0, 12)} ·{" "}
            {task.analysis_wall_seconds?.toFixed(2)} s processing · No accuracy
            claim
          </small>
        </div>
      )}
    </section>
  );
}
