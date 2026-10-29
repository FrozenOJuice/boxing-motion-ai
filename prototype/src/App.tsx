import { useEffect, useRef, useState } from "react";
import {
  checkpoint,
  clock,
  nextBoundary,
  snapBoundary,
  previewSelection,
  punchName,
  recordedSelections,
  sample,
  type Bounds,
  type Selection,
  type VideoItem,
} from "./model";
import {
  importVideo,
  loadPreviews,
  recentVideos,
  savePreviews,
} from "./storage";
import { Brand, Button, Dialog, Icon, IconButton } from "./ui";
import { Transport, usePlayer, VideoStage } from "./Player";
import Timeline from "./Timeline";

type Modal = "about" | "shortcuts" | "development" | null;
export default function App() {
  const [screen, setScreen] = useState<"analyze" | "videos">("analyze");
  const [video, setVideo] = useState(sample),
    [recent, setRecent] = useState<VideoItem[]>([]);
  const [ready, setReady] = useState(false),
    [opening, setOpening] = useState(false);
  const [error, setError] = useState(""),
    [notice, setNotice] = useState("");
  const [modal, setModal] = useState<Modal>(null);
  const [selections, setSelections] = useState<Selection[]>(() => [
    ...recordedSelections,
    ...loadPreviews(),
  ]);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    recentVideos()
      .then(setRecent)
      .catch(() => setNotice("Recent videos could not be loaded."));
    fetch("/demo/manifest.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((v) => setReady(!!v?.source_sha256))
      .catch(() => {});
  }, []);
  useEffect(() => {
    savePreviews(selections);
  }, [selections]);
  function open(item: VideoItem) {
    setVideo(item);
    setScreen("analyze");
    setError("");
  }
  async function upload(file?: File) {
    if (!file || opening) return;
    setOpening(true);
    setError("");
    setNotice("");
    try {
      const result = await importVideo(file);
      setRecent((items) => [result.item, ...items]);
      if (!result.stored)
        setNotice("Opened for this session. Browser storage is unavailable.");
      open(result.item);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setOpening(false);
      if (input.current) input.current.value = "";
    }
  }
  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="app-header">
        <Brand onClick={() => setScreen("videos")} />
        <nav aria-label="Main navigation">
          <button
            aria-current={screen === "analyze" ? "page" : undefined}
            onClick={() => setScreen("analyze")}
          >
            Analyze
          </button>
          <button
            aria-current={screen === "videos" ? "page" : undefined}
            onClick={() => setScreen("videos")}
          >
            Videos
          </button>
        </nav>
        <div className="header-secondary">
          <button
            className="text-button development-link"
            onClick={() => setModal("development")}
          >
            Development
          </button>
          <button className="prototype-link" onClick={() => setModal("about")}>
            Design prototype
          </button>
        </div>
      </header>
      <input
        ref={input}
        type="file"
        accept="video/mp4,video/quicktime,video/webm,.m4v"
        hidden
        aria-label="Upload video file"
        onChange={(e) => void upload(e.target.files?.[0])}
      />
      {(error || notice || opening) && (
        <div className="notice" role={error ? "alert" : "status"}>
          {error || (opening ? "Opening video…" : notice)}
          <button
            aria-label="Dismiss message"
            onClick={() => {
              setError("");
              setNotice("");
            }}
            hidden={opening}
          >
            ×
          </button>
        </div>
      )}
      <main id="main">
        {screen === "videos" ? (
          <section
            className="library"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              void upload(e.dataTransfer.files[0]);
            }}
          >
            <div className="page-heading">
              <div>
                <h1>Videos</h1>
                <p>Open a video to analyze a punch.</p>
              </div>
              <Button
                variant="primary"
                icon="upload"
                disabled={opening}
                onClick={() => input.current?.click()}
              >
                Upload video
              </Button>
            </div>
            <div className="library-heading">
              <h2>Recent videos</h2>
              <span>Up to 30 seconds · 150 MB · Stored in this browser</span>
            </div>
            <div className="video-grid">
              {recent.map((item) => (
                <VideoTile
                  key={item.id}
                  video={item}
                  onOpen={() => open(item)}
                />
              ))}
              <VideoTile
                video={sample}
                disabled={!ready}
                onOpen={() => open(sample)}
              />
              <button
                className="upload-tile"
                disabled={opening}
                onClick={() => input.current?.click()}
              >
                <Icon name="upload" size={24} />
                <strong>Upload footage</strong>
                <span>MP4, MOV or WebM</span>
              </button>
            </div>
          </section>
        ) : ready || video.kind === "local" ? (
          <Workspace
            key={video.id}
            video={video}
            selections={selections.filter((s) => s.videoId === video.id)}
            onSave={(s) => setSelections((items) => [...items, s])}
            onVideos={() => setScreen("videos")}
            onShortcuts={() => setModal("shortcuts")}
          />
        ) : (
          <div className="loading-state">
            <h1>Analyze</h1>
            <p>Sample media unavailable. Upload a video to begin.</p>
            <Button icon="upload" onClick={() => input.current?.click()}>
              Upload video
            </Button>
          </div>
        )}
      </main>
      {modal && (
        <Dialog
          title={
            modal === "about"
              ? "Design prototype"
              : modal === "shortcuts"
                ? "Keyboard shortcuts"
                : "Development"
          }
          onClose={() => setModal(null)}
        >
          {modal === "about" ? (
            <>
              <p>
                This isolated prototype tests video playback, selection and the
                classification interface. New classifications show example
                output; no model runs.
              </p>
              <p>
                Sample selections contain recorded model outputs from existing
                reference footage. They do not establish accuracy on new boxers.
              </p>
              <p>
                Uploads remain in this browser. Clearing this site's browser
                data removes local videos and selections.
              </p>
            </>
          ) : modal === "development" ? (
            <>
              <p>
                Review, Dataset, Experiments and Models belong in a separate
                development area.
              </p>
              <p>
                This prototype does not open or modify the production tools.
              </p>
            </>
          ) : (
            <dl className="shortcuts">
              <dt>Space</dt>
              <dd>Play / pause</dd>
              <dt>← / →</dt>
              <dd>Previous / next source frame (sample)</dd>
              <dt>I / O</dt>
              <dd>Set punch start / end</dd>
              <dt>F</dt>
              <dd>Fullscreen</dd>
              <dt>Tab + arrows</dt>
              <dd>Adjust timeline handles</dd>
            </dl>
          )}
        </Dialog>
      )}
    </div>
  );
}
function VideoTile({
  video,
  onOpen,
  disabled = false,
}: {
  video: VideoItem;
  onOpen: () => void;
  disabled?: boolean;
}) {
  const [failed, setFailed] = useState(false);
  return (
    <button
      className="video-tile"
      onClick={onOpen}
      disabled={disabled}
      aria-label={`Open ${video.title}`}
    >
      <div className="thumbnail">
        {!failed ? (
          <img
            src={video.poster}
            alt=""
            onError={() => setFailed(true)}
            onLoad={(e) => {
              try {
                const canvas = document.createElement("canvas");
                canvas.width = 8;
                canvas.height = 8;
                const context = canvas.getContext("2d")!;
                context.drawImage(e.currentTarget, 0, 0, 8, 8);
                const pixels = context.getImageData(0, 0, 8, 8).data;
                if (Array.from(pixels).every((v, i) => i % 4 === 3 || v < 16))
                  setFailed(true);
              } catch {
                /* A readable poster is optional. */
              }
            }}
          />
        ) : (
          <Icon name="file" size={32} />
        )}
        <span className="video-duration">{clock(video.duration)}</span>
        <span className="thumbnail-play">
          <Icon name="play" />
        </span>
      </div>
      <strong>{video.title}</strong>
      <span>
        {video.kind === "sample"
          ? "Reference sample · 2 saved punches"
          : `${video.width} × ${video.height} · Local video`}
      </span>
    </button>
  );
}
function Workspace({
  video,
  selections,
  onSave,
  onVideos,
  onShortcuts,
}: {
  video: VideoItem;
  selections: Selection[];
  onSave: (s: Selection) => void;
  onVideos: () => void;
  onShortcuts: () => void;
}) {
  const player = usePlayer(video);
  const [bounds, setBounds] = useState<Bounds | null>(null),
    [result, setResult] = useState<Selection | null>(null),
    [busy, setBusy] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  function change(next: Bounds) {
    if (busy) return;
    setBounds(next);
    setResult(null);
  }
  function mark(mode: "start" | "end") {
    const t = snapBoundary(video, player.current);
    change(
      mode === "start"
        ? {
            start: Math.min(
              t,
              video.frameTimes?.at(-1) ?? video.duration - 0.001,
            ),
            end: bounds && bounds.end > t ? bounds.end : nextBoundary(video, t),
          }
        : {
            start: bounds && bounds.start < t ? bounds.start : 0,
            end: nextBoundary(video, t),
          },
    );
  }
  useEffect(() => {
    function key(e: KeyboardEvent) {
      if (
        (e.target instanceof HTMLElement &&
          e.target.closest("button,input,dialog,a")) ||
        e.altKey ||
        e.metaKey ||
        e.ctrlKey
      )
        return;
      if (e.code === "Space") {
        e.preventDefault();
        if (player.playing) player.pause();
        else void player.play();
      }
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        player.step(e.key === "ArrowLeft" ? -1 : 1);
      }
      if (e.key.toLowerCase() === "i") mark("start");
      if (e.key.toLowerCase() === "o") mark("end");
      if (e.key.toLowerCase() === "f") void player.fullscreen();
    }
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });
  function classify() {
    if (!bounds || busy) return;
    player.pause();
    setBusy(true);
    timer.current = setTimeout(() => {
      const s = previewSelection(video.id, bounds);
      onSave(s);
      setResult(s);
      setBusy(false);
    }, 450);
  }
  return (
    <section className="workspace" aria-label="Analyze workspace">
      <div className="workspace-heading">
        <div className="cluster">
          <IconButton icon="back" label="Back to videos" onClick={onVideos} />
          <div>
            <h1>{video.title}</h1>
            <p>
              {video.kind === "sample" ? "Reference sample" : "Local video"} ·{" "}
              {video.width} × {video.height} · {clock(video.duration)}
            </p>
          </div>
        </div>
        <IconButton
          icon="keyboard"
          label="Keyboard shortcuts"
          onClick={onShortcuts}
        />
      </div>
      <div className="workspace-grid">
        <div className="video-workspace">
          <VideoStage video={video} player={player} />
          <Transport video={video} player={player} />
          <div className="selection-tools">
            <div className="cluster">
              <Button
                icon="bracket-in"
                disabled={busy}
                onClick={() => mark("start")}
                aria-label="Set punch start (I)"
              >
                Set start <kbd>I</kbd>
              </Button>
              <Button
                icon="bracket-out"
                disabled={busy}
                onClick={() => mark("end")}
                aria-label="Set punch end (O)"
              >
                Set end <kbd>O</kbd>
              </Button>
            </div>
            <span>Drag on the timeline to select a punch</span>
          </div>
          <Timeline
            video={video}
            current={player.current}
            bounds={bounds}
            onSeek={player.seek}
            onBounds={change}
          />
          <div className="saved-selections">
            <div className="saved-heading">
              <h2>Saved punches</h2>
              <span>{selections.length}</span>
            </div>
            <div className="saved-list">
              {selections.length ? (
                selections.map((s, i) => (
                  <button
                    key={s.id}
                    disabled={busy}
                    className={result?.id === s.id ? "is-active" : ""}
                    onClick={() => {
                      setBounds(s.bounds);
                      setResult(s);
                      player.seek(s.bounds.start);
                    }}
                  >
                    <span>
                      {s.kind === "recorded"
                        ? punchName(s.prediction)
                        : `Selection ${i + 1}`}
                    </span>
                    <small>
                      {clock(s.bounds.start, true)} –{" "}
                      {clock(s.bounds.end, true)}
                      {s.kind === "preview" ? " · Example" : ""}
                    </small>
                  </button>
                ))
              ) : (
                <p>No saved punches yet</p>
              )}
            </div>
          </div>
        </div>
        <aside className="analysis-panel" aria-label="Punch analysis">
          <div className="panel-heading">
            <h2>Selected punch</h2>
            {bounds && (
              <IconButton
                icon="close"
                label="Clear selection"
                disabled={busy}
                onClick={() => {
                  setBounds(null);
                  setResult(null);
                }}
              />
            )}
          </div>
          <div className="panel-content">
            {bounds ? (
              <>
                <div className="selection-times">
                  <div>
                    <span>Start</span>
                    <strong>{clock(bounds.start, true)}</strong>
                  </div>
                  <div>
                    <span>End</span>
                    <strong>{clock(bounds.end, true)}</strong>
                  </div>
                  <div>
                    <span>Duration</span>
                    <strong>{(bounds.end - bounds.start).toFixed(3)}s</strong>
                  </div>
                </div>
                <Button icon="play" onClick={() => void player.play(bounds)}>
                  Replay selection
                </Button>
              </>
            ) : (
              <div className="selection-empty">
                <Icon name="bracket-in" size={24} />
                <h3>Select one punch</h3>
                <p>
                  Drag a range on the timeline, or set the start and end during
                  playback.
                </p>
              </div>
            )}
            <section className="prediction" aria-live="polite" aria-busy={busy}>
              <h3>AI prediction</h3>
              {busy ? (
                <p role="status">Loading example output…</p>
              ) : result ? (
                <>
                  <div className="result-source">
                    {result.kind === "recorded"
                      ? "Saved model output"
                      : "Prototype example"}
                  </div>
                  <h4>{punchName(result.prediction)}</h4>
                  <p>
                    {result.kind === "recorded"
                      ? "Recorded output for this sample selection."
                      : "No model ran for this selection."}
                  </p>
                  <div className="scores">
                    {Object.entries(result.scores)
                      .sort((a, b) => b[1] - a[1])
                      .map(([name, score]) => (
                        <div
                          className={`score ${name === result.prediction ? "score--top" : ""}`}
                          key={name}
                        >
                          <div>
                            <span>{punchName(name)}</span>
                            <span>{score.toFixed(3)}</span>
                          </div>
                          <div
                            className="score-track"
                            role="meter"
                            aria-label={punchName(name)}
                            aria-valuemin={0}
                            aria-valuemax={1}
                            aria-valuenow={score}
                          >
                            <span style={{ width: `${score * 100}%` }} />
                          </div>
                        </div>
                      ))}
                  </div>
                  <p className="score-note">
                    Uncalibrated model scores, not probabilities of correctness.
                  </p>
                </>
              ) : (
                <p className="prediction-placeholder">
                  {bounds
                    ? "Ready to classify the selected punch."
                    : "Prediction appears here after classification."}
                </p>
              )}
            </section>
            <section className="quality">
              <h3>Selection quality</h3>
              <dl>
                <dt>Time range</dt>
                <dd>{bounds ? "Valid" : "Not selected"}</dd>
                <dt>Pose coverage</dt>
                <dd>Not evaluated live</dd>
                <dt>Frame precision</dt>
                <dd>{video.frameTimes ? "Source frames" : "Time based"}</dd>
              </dl>
            </section>
            <details className="model-details">
              <summary>Model information</summary>
              <p>Level 1 · Six punch categories</p>
              <p>
                New selections use example output. Sample outputs are recorded
                from the existing trained checkpoint.
              </p>
              <p>
                Checkpoint <code>{checkpoint.slice(0, 12)}</code>
              </p>
              <p>
                Experimental recognition. New-boxer performance is unvalidated.
              </p>
            </details>
          </div>
          <div className="panel-action">
            <Button
              variant="primary"
              disabled={!bounds || busy}
              onClick={classify}
            >
              {busy ? "Loading…" : "Classify punch"}
            </Button>
            <span>Prototype · New predictions are example output</span>
          </div>
        </aside>
      </div>
    </section>
  );
}
