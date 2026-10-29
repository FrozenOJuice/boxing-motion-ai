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

type Screen = "library" | "analyze" | "results";
type Modal = "about" | "shortcuts" | "development" | null;
export default function App() {
  const [screen, setScreen] = useState<Screen>("library"),
    [video, setVideo] = useState<VideoItem>(sample);
  const [recent, setRecent] = useState<VideoItem[]>([]),
    [sampleReady, setSampleReady] = useState(false);
  const [selections, setSelections] = useState<Selection[]>(() => [
    ...recordedSelections,
    ...loadPreviews(),
  ]);
  const [activeResult, setActiveResult] = useState(recordedSelections[0].id),
    [bounds, setBounds] = useState<Bounds | null>(null);
  const [modal, setModal] = useState<Modal>(null),
    [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [opening, setOpening] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const main = useRef<HTMLElement>(null);
  useEffect(() => {
    recentVideos()
      .then(setRecent)
      .catch(() =>
        setNotice(
          "Recent videos are unavailable. You can still open a video for this session.",
        ),
      );
    fetch("/demo/manifest.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((value) => setSampleReady(!!value?.source_sha256))
      .catch(() => {});
  }, []);
  useEffect(() => {
    savePreviews(selections);
  }, [selections]);
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "instant" });
    if (screen !== "library") main.current?.focus({ preventScroll: true });
  }, [screen]);
  function open(item: VideoItem) {
    setVideo(item);
    setBounds(null);
    setError("");
    setScreen("analyze");
  }
  async function upload(file?: File) {
    if (!file || opening) return;
    setOpening(true);
    setError("");
    setNotice("");
    try {
      const value = await importVideo(file);
      setRecent((items) => [value.item, ...items]);
      if (!value.stored)
        setNotice(
          "Opened for this session. Browser storage could not save it to Recents.",
        );
      open(value.item);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setOpening(false);
      if (input.current) input.current.value = "";
    }
  }
  function add(selection: Selection) {
    setSelections((items) => [...items, selection]);
    setActiveResult(selection.id);
    setScreen("results");
  }
  const ownSelections = selections.filter((item) => item.videoId === video.id);
  const selectedResult =
    ownSelections.find((item) => item.id === activeResult) ?? ownSelections[0];
  return (
    <div className={`app app--${screen}`}>
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <header className="app-header">
        <Brand onClick={() => setScreen("library")} />
        {screen !== "library" && (
          <div className="header-workspace">
            <div className="cluster context">
              <IconButton
                icon="back"
                label="Back to videos"
                onClick={() => setScreen("library")}
              />
              <div className="video-heading">
                <h1>{video.title}</h1>
                <span>
                  {clock(video.duration)} ·{" "}
                  {video.kind === "sample" ? "Sample footage" : "Local video"}
                </span>
              </div>
            </div>
            <nav className="workspace-tabs" aria-label="Video workspace">
              <button
                aria-current={screen === "analyze" ? "page" : undefined}
                onClick={() => setScreen("analyze")}
              >
                Analyze
              </button>
              <button
                aria-label={`Results, ${ownSelections.length} saved selections`}
                aria-current={screen === "results" ? "page" : undefined}
                onClick={() => setScreen("results")}
              >
                Results
                {ownSelections.length > 0 && (
                  <span>{ownSelections.length}</span>
                )}
              </button>
            </nav>
          </div>
        )}
        <div className="header-right">
          <button className="prototype-label" onClick={() => setModal("about")}>
            Design prototype <Icon name="info" size={15} />
          </button>
          {screen !== "library" && (
            <IconButton
              icon="keyboard"
              label="Keyboard shortcuts"
              onClick={() => setModal("shortcuts")}
            />
          )}
        </div>
      </header>
      <input
        ref={input}
        className="visually-hidden"
        type="file"
        aria-label="Choose a local video"
        accept=".mp4,.mov,.webm,.m4v"
        onChange={(e) => upload(e.target.files?.[0])}
      />
      <main id="main" ref={main} tabIndex={-1}>
        {screen === "library" ? (
          <Library
            recent={recent}
            sampleReady={sampleReady}
            opening={opening}
            onChoose={() => input.current?.click()}
            onDrop={upload}
            onOpen={open}
          />
        ) : (
          <>
            {screen === "analyze" ? (
              <Analyze
                key={video.id}
                video={video}
                bounds={bounds}
                onBounds={setBounds}
                onClassify={add}
              />
            ) : (
              <Results
                key={video.id}
                video={video}
                selections={ownSelections}
                selected={selectedResult}
                onSelect={setActiveResult}
                onAnalyze={(selection) => {
                  setBounds(selection?.bounds ?? null);
                  setScreen("analyze");
                }}
              />
            )}
          </>
        )}
        {error && (
          <div className="message message--error" role="alert">
            <Icon name="info" />
            <span>{error}</span>
            <IconButton
              icon="close"
              label="Dismiss error"
              onClick={() => setError("")}
            />
          </div>
        )}
        {notice && (
          <div className="message" role="status">
            <Icon name="info" />
            <span>{notice}</span>
            <IconButton
              icon="close"
              label="Dismiss notice"
              onClick={() => setNotice("")}
            />
          </div>
        )}
      </main>
      <footer className="app-footer">
        <span>Local media. No live inference.</span>
        <button className="text-link" onClick={() => setModal("development")}>
          Development tools <Icon name="arrow" size={13} />
        </button>
      </footer>
      {modal && (
        <Dialog
          title={
            modal === "about"
              ? "About this prototype"
              : modal === "shortcuts"
                ? "Keyboard shortcuts"
                : "Development tools"
          }
          onClose={() => setModal(null)}
        >
          {modal === "about" ? (
            <div className="dialog-copy">
              <p>
                This is an isolated design prototype. It does not connect to the
                production dashboard or run a model.
              </p>
              <p>
                Uploads play locally and are saved in this browser. New
                classifications are <strong>layout previews</strong>, clearly
                marked in Results.
              </p>
              <p>
                The sample includes two saved outputs from the existing trained
                model. These are labeled separately from prototype previews.
              </p>
              <p className="muted">
                Exact frame stepping is available for the indexed sample. Other
                uploads use the browser’s video clock. Sample footage stays
                outside Git.
              </p>
            </div>
          ) : modal === "shortcuts" ? (
            <>
              <p className="dialog-intro">
                When the video workspace has focus:
              </p>
              <dl className="shortcut-list">
                <div>
                  <dt>Play / pause</dt>
                  <dd>
                    <kbd>Space</kbd>
                  </dd>
                </div>
                <div>
                  <dt>Previous / next sample frame</dt>
                  <dd>
                    <kbd>←</kbd> <kbd>→</kbd>
                  </dd>
                </div>
                <div>
                  <dt>Set punch start / end</dt>
                  <dd>
                    <kbd>I</kbd> <kbd>O</kbd>
                  </dd>
                </div>
                <div>
                  <dt>Fullscreen</dt>
                  <dd>
                    <kbd>F</kbd>
                  </dd>
                </div>
              </dl>
              <p className="muted">
                Tab reaches every control. Focus a timeline handle and use arrow
                keys to adjust it. Uploaded videos use time-based selection;
                sample footage supports exact frames.
              </p>
            </>
          ) : (
            <div className="dialog-copy">
              <p>
                Review, datasets, experiments and training remain in the
                existing application.
              </p>
              <p>
                This prototype does not open or modify those tools or their
                records.
              </p>
            </div>
          )}
        </Dialog>
      )}
    </div>
  );
}

function Library({
  recent,
  sampleReady,
  opening,
  onChoose,
  onDrop,
  onOpen,
}: {
  recent: VideoItem[];
  sampleReady: boolean;
  opening: boolean;
  onChoose: () => void;
  onDrop: (file?: File) => void;
  onOpen: (video: VideoItem) => void;
}) {
  const [dragging, setDragging] = useState(false),
    depth = useRef(0);
  return (
    <div className="library page-width">
      <div className="library-heading">
        <h1>Your videos</h1>
        <p>Upload a short shadowboxing clip to get started.</p>
      </div>
      <section
        className={`upload-zone ${dragging ? "is-dragging" : ""}`}
        aria-label="Upload a video"
        aria-busy={opening}
        onDragEnter={(e) => {
          e.preventDefault();
          depth.current++;
          setDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          depth.current--;
          if (!depth.current) setDragging(false);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          depth.current = 0;
          setDragging(false);
          onDrop(e.dataTransfer.files[0]);
        }}
      >
        <div className="upload-symbol">
          <Icon name="upload" size={26} />
        </div>
        <h2>
          {opening
            ? "Opening video…"
            : dragging
              ? "Drop to open"
              : "Drop your video here"}
        </h2>
        <Button variant="primary" onClick={onChoose} disabled={opening}>
          Choose video
        </Button>
        <p>Up to 30 seconds · 150 MB · Stored in this browser</p>
      </section>
      <section className="recent-section" aria-labelledby="recent-title">
        <div className="section-heading">
          <h2 id="recent-title">Recent videos</h2>
          {recent.length > 0 && (
            <span className="muted small">On this browser</span>
          )}
        </div>
        {recent.length ? (
          <div className="video-list">
            <div className="video-list-labels" aria-hidden="true">
              <span>Video</span>
              <span>Length</span>
              <span>Opened</span>
              <span />
            </div>
            {recent.map((item) => (
              <button
                className="video-row"
                key={item.id}
                onClick={() => onOpen(item)}
              >
                <span className="video-row-name">
                  <img src={item.poster} alt="" />
                  <strong>{item.title}</strong>
                </span>
                <span className="mono muted">{clock(item.duration)}</span>
                <span className="muted small">
                  {new Intl.DateTimeFormat("en-CA", {
                    month: "short",
                    day: "numeric",
                    timeZone: "America/Edmonton",
                  }).format(item.openedAt)}
                </span>
                <Icon name="chevron" size={17} />
              </button>
            ))}
          </div>
        ) : (
          <div className="recent-empty">
            <Icon name="file" size={19} />
            <p>Your videos will appear here after you open them.</p>
          </div>
        )}
      </section>
      <section className="sample-row">
        <div className="sample-image">
          {sampleReady ? (
            <img src={sample.poster} alt="A boxer shadowboxing in a gym" />
          ) : (
            <Icon name="play" size={25} />
          )}
        </div>
        <div>
          <h2>Try the sample</h2>
          <p>Explore the workspace and two saved model predictions.</p>
          {!sampleReady && (
            <p className="small muted">
              Sample media is not installed. See the prototype README.
            </p>
          )}
        </div>
        <Button
          variant="ghost"
          icon="arrow"
          disabled={!sampleReady}
          onClick={() => onOpen(sample)}
        >
          Open sample
        </Button>
      </section>
    </div>
  );
}

function Analyze({
  video,
  bounds,
  onBounds,
  onClassify,
}: {
  video: VideoItem;
  bounds: Bounds | null;
  onBounds: (bounds: Bounds | null) => void;
  onClassify: (selection: Selection) => void;
}) {
  const player = usePlayer(video);
  const initialBounds = useRef(bounds);
  const [busy, setBusy] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const duration = bounds ? bounds.end - bounds.start : 0;
  const valid = !!bounds && duration >= 0.04 && duration <= 3.000001;
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  const mark = (end: boolean) => {
    const time = player.current;
    if (end) {
      const stop = nextBoundary(video, time);
      onBounds({
        start:
          bounds && bounds.start < stop
            ? bounds.start
            : snapBoundary(video, Math.max(0, stop - 0.5)),
        end: stop,
      });
    } else
      onBounds({
        start: time,
        end:
          bounds && bounds.end > time
            ? bounds.end
            : snapBoundary(video, Math.min(video.duration, time + 0.5)),
      });
  };
  useEffect(() => {
    const key = (e: globalThis.KeyboardEvent) => {
      const element = e.target as HTMLElement;
      if (
        e.metaKey ||
        e.ctrlKey ||
        e.altKey ||
        element.closest("button, input, dialog, [role='slider']")
      )
        return;
      if (e.code === "Space") {
        e.preventDefault();
        player.playing ? player.pause() : player.play();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        player.step(-1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        player.step(1);
      } else if (e.key.toLowerCase() === "i") mark(false);
      else if (e.key.toLowerCase() === "o") mark(true);
      else if (e.key.toLowerCase() === "f") player.fullscreen();
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });
  function classify() {
    if (!valid || busy || !bounds) return;
    setBusy(true);
    player.pause();
    // Deliberately simulated transition. No production API or model request.
    const chosen = { ...bounds };
    timer.current = setTimeout(
      () => onClassify(previewSelection(video.id, chosen)),
      450,
    );
  }
  return (
    <div className="analyze-workspace workspace-width">
      <h2 className="visually-hidden">Select a punch</h2>
      <VideoStage
        video={video}
        player={player}
        initialBounds={initialBounds.current ?? undefined}
      />
      <Transport video={video} player={player} />
      <Timeline
        video={video}
        current={player.current}
        bounds={bounds}
        onSeek={player.seek}
        onBounds={onBounds}
      />
      <div className="selection-actions">
        <div className="selection-summary">
          {bounds ? (
            <>
              <span className="selection-label">Selected punch</span>
              <span className="mono selection-range">
                {clock(bounds.start, true)}
                <span className="muted">—</span>
                {clock(bounds.end, true)}
              </span>
              <span className="muted small">{duration.toFixed(3)} s</span>
              <button
                className="text-link small"
                disabled={busy}
                onClick={() => onBounds(null)}
              >
                Clear
              </button>
            </>
          ) : (
            <p>Drag on the timeline to select one punch.</p>
          )}
          {bounds && !valid && (
            <p className="selection-error">
              Select between 0.04 and 3 seconds.
            </p>
          )}
        </div>
        <div className="cluster action-buttons">
          <IconButton
            icon="bracket-in"
            label="Set punch start (I)"
            disabled={busy}
            onClick={() => mark(false)}
          />
          <IconButton
            icon="bracket-out"
            label="Set punch end (O)"
            disabled={busy}
            onClick={() => mark(true)}
          />
          <Button
            icon="play"
            variant="ghost"
            disabled={!valid || busy}
            onClick={() => bounds && player.play(bounds)}
          >
            Replay
          </Button>
          <Button
            variant="primary"
            disabled={!valid || busy}
            onClick={classify}
          >
            {busy ? "Previewing…" : "Classify punch"}
            <Icon name="arrow" size={17} />
          </Button>
        </div>
      </div>
    </div>
  );
}

function Results({
  video,
  selections,
  selected,
  onSelect,
  onAnalyze,
}: {
  video: VideoItem;
  selections: Selection[];
  selected?: Selection;
  onSelect: (id: string) => void;
  onAnalyze: (selection?: Selection) => void;
}) {
  const player = usePlayer(video);
  const [details, setDetails] = useState(false);
  return (
    <div className="results-workspace page-width">
      <div className="results-heading">
        <div>
          <h2>Punch results</h2>
          <p>
            {selections.length} saved{" "}
            {selections.length === 1 ? "selection" : "selections"}
          </p>
        </div>
        <Button variant="primary" icon="back" onClick={() => onAnalyze()}>
          Select another punch
        </Button>
      </div>
      {!selected ? (
        <div className="results-empty">
          <Icon name="bracket-in" size={30} />
          <h3>No selections yet</h3>
          <p>Select a punch in Analyze to preview a result.</p>
          <Button onClick={() => onAnalyze()}>Go to Analyze</Button>
        </div>
      ) : (
        <div className="results-layout">
          <aside className="selection-list" aria-label="Saved punch selections">
            {selections.map((item, i) => (
              <button
                key={item.id}
                className="selection-row"
                aria-current={selected.id === item.id ? "true" : undefined}
                onClick={() => {
                  player.pause();
                  onSelect(item.id);
                }}
              >
                <span className="selection-index mono">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span>
                  <strong>{punchName(item.prediction)}</strong>
                  <span className="mono small muted">
                    {clock(item.bounds.start, true)} —{" "}
                    {clock(item.bounds.end, true)}
                  </span>
                  <span className="selection-kind">
                    {item.kind === "recorded"
                      ? "Saved sample output"
                      : "Prototype preview"}
                  </span>
                </span>
                <Icon name="chevron" size={15} />
              </button>
            ))}
          </aside>
          <article className="result-detail">
            <div className="result-visual">
              <VideoStage
                video={video}
                player={player}
                initialBounds={selected.bounds}
              />
              <div className="result-replay">
                <Button
                  variant="ghost"
                  icon={player.playing ? "pause" : "play"}
                  onClick={() =>
                    player.playing
                      ? player.pause()
                      : player.play(selected.bounds)
                  }
                >
                  {player.playing ? "Pause" : "Replay selection"}
                </Button>
                <span className="mono small muted">
                  {clock(selected.bounds.start, true)} —{" "}
                  {clock(selected.bounds.end, true)}
                </span>
                <button
                  className="text-link small"
                  onClick={() => onAnalyze(selected)}
                >
                  Adjust selection
                </button>
              </div>
            </div>
            <div className="result-reading">
              <div className="prediction-heading">
                <span className="eyebrow">
                  {selected.kind === "recorded"
                    ? "SAVED MODEL PREDICTION"
                    : "EXAMPLE RESULT"}
                </span>
                <h3>{punchName(selected.prediction)}</h3>
                <p>
                  {selected.kind === "recorded"
                    ? "Recorded output from the sample video."
                    : "Layout preview using sample output. This selection was not classified."}
                </p>
              </div>
              <div className="score-heading">
                <h4>Model scores</h4>
                <span>Uncalibrated</span>
              </div>
              <ol className="score-list">
                {Object.entries(selected.scores)
                  .sort((a, b) => b[1] - a[1])
                  .map(([name, score], i) => (
                    <li key={name} className={i === 0 ? "is-leading" : ""}>
                      <span>{punchName(name)}</span>
                      <meter
                        value={score}
                        min="0"
                        max="1"
                        aria-label={`${punchName(name)} uncalibrated score`}
                      />
                      <span className="mono">{score.toFixed(3)}</span>
                    </li>
                  ))}
              </ol>
              <p className="score-note">
                Scores are not probabilities of correctness.
              </p>
              <button
                className="model-details-toggle text-link"
                onClick={() => setDetails(!details)}
                aria-expanded={details}
              >
                Model information <Icon name="chevron" size={14} />
              </button>
              {details && (
                <dl className="model-details">
                  <div>
                    <dt>Model</dt>
                    <dd>Level 1 · Pose + velocity TCN</dd>
                  </div>
                  <div>
                    <dt>Classes</dt>
                    <dd>Six punch types · No “no punch” class</dd>
                  </div>
                  <div>
                    <dt>Status</dt>
                    <dd>Experimental · New-boxer performance unvalidated</dd>
                  </div>
                  <div>
                    <dt>Source</dt>
                    <dd>
                      {selected.kind === "recorded"
                        ? "Saved local inference, seed 20261021"
                        : "Sample output, displayed for design review"}
                    </dd>
                  </div>
                  <div>
                    <dt>Checkpoint</dt>
                    <dd className="mono">{checkpoint.slice(0, 16)}…</dd>
                  </div>
                </dl>
              )}
            </div>
          </article>
        </div>
      )}
    </div>
  );
}
