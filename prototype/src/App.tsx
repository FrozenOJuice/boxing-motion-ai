import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import {
  clock,
  recordedSelections,
  sample,
  type Selection,
  type VideoItem,
} from "./model";
import Workspace from "./Workspace";
import {
  readSession,
  writeSession,
  useNavigation,
  type WorkspaceSession,
} from "./navigation";
import {
  importVideo,
  loadPreviews,
  recentVideos,
  savePreviews,
} from "./storage";
import { Brand, Button, Dialog, Icon, IconButton } from "./ui";

type Modal = "about" | "shortcuts" | "development" | null;
export default function App() {
  const { route, navigate } = useNavigation();
  const [recent, setRecent] = useState<VideoItem[]>([]);
  const [libraryLoaded, setLibraryLoaded] = useState(false),
    [sampleLoaded, setSampleLoaded] = useState(false);
  const sessions = useRef(new Map<string, WorkspaceSession>());
  const library = useRef<HTMLElement>(null),
    libraryHeading = useRef<HTMLHeadingElement>(null);
  const tiles = useRef(new Map<string, HTMLButtonElement>());
  const returnedVideo = useRef<string | null>(null),
    libraryScroll = useRef(0);
  const onSession = useCallback((id: string, state: WorkspaceSession) => {
    sessions.current.set(id, state);
    writeSession(id, state);
  }, []);
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
      .catch(() => setNotice("Recent videos could not be loaded."))
      .finally(() => setLibraryLoaded(true));
    fetch("/demo/manifest.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((v) => setReady(!!v?.source_sha256))
      .catch(() => {})
      .finally(() => setSampleLoaded(true));
  }, []);
  useEffect(() => {
    savePreviews(selections);
  }, [selections]);
  const video =
    route.page === "analyze"
      ? [sample, ...recent].find(
          (v) => v.id === route.videoId && (v.kind !== "sample" || ready),
        )
      : undefined;
  const pageKey = route.page === "analyze" ? route.videoId : route.page;
  const previousPage = useRef(pageKey);
  useLayoutEffect(() => {
    if (previousPage.current !== pageKey && previousPage.current !== "videos")
      returnedVideo.current = previousPage.current;
    previousPage.current = pageKey;
    if (route.page === "videos") {
      if (library.current) library.current.scrollTop = libraryScroll.current;
      (
        tiles.current.get(returnedVideo.current ?? "") ?? libraryHeading.current
      )?.focus({ preventScroll: true });
    }
  }, [pageKey, route.page, libraryLoaded, ready]);
  function home() {
    navigate("/videos");
  }
  function open(item: VideoItem) {
    libraryScroll.current = library.current?.scrollTop ?? libraryScroll.current;
    returnedVideo.current = item.id;
    setError("");
    navigate(`/videos/${encodeURIComponent(item.id)}`);
  }
  function saveResult(s: Selection) {
    setSelections((items) =>
      items.some((item) => item.id === s.id) ? items : [...items, s],
    );
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
        <div className="header-inner">
          <Brand onClick={home} />
          <nav aria-label="Main navigation">
            <button
              aria-current={route.page === "videos" ? "page" : undefined}
              data-active="true"
              onClick={home}
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
            <button
              className="prototype-link"
              onClick={() => setModal("about")}
            >
              Design prototype
            </button>
          </div>
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
        {route.page === "videos" ? (
          <section
            ref={library}
            className="library"
            onScroll={(e) => {
              libraryScroll.current = e.currentTarget.scrollTop;
            }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              void upload(e.dataTransfer.files[0]);
            }}
          >
            <div className="page-heading">
              <div>
                <h1 ref={libraryHeading} tabIndex={-1}>
                  Videos
                </h1>
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
              <h2>{recent.length ? "Recent videos" : "Library"}</h2>
              <span>Up to 30 seconds · 150 MB · MP4, MOV, WebM</span>
            </div>
            <div className="video-grid">
              {recent.map((item) => (
                <VideoTile
                  key={item.id}
                  video={item}
                  buttonRef={(el) => {
                    if (el) tiles.current.set(item.id, el);
                    else tiles.current.delete(item.id);
                  }}
                  onOpen={() => open(item)}
                />
              ))}
              {ready && (
                <VideoTile
                  video={sample}
                  buttonRef={(el) => {
                    if (el) tiles.current.set(sample.id, el);
                    else tiles.current.delete(sample.id);
                  }}
                  onOpen={() => open(sample)}
                />
              )}
            </div>
            {!recent.length && !ready && (
              <div className="library-empty">
                <Icon name="file" size={32} />
                <h2>No videos yet</h2>
                <p>Upload a short shadowboxing video to get started.</p>
              </div>
            )}
          </section>
        ) : video ? (
          <Workspace
            key={video.id}
            video={video}
            selections={selections.filter((s) => s.videoId === video.id)}
            initialSession={
              sessions.current.get(video.id) ?? readSession(video.id)
            }
            onSession={onSession}
            onSave={saveResult}
            onVideos={home}
            onShortcuts={() => setModal("shortcuts")}
          />
        ) : (
          <section className="loading-state" aria-live="polite">
            <h1>
              {!libraryLoaded || !sampleLoaded
                ? "Opening saved video…"
                : "Video unavailable"}
            </h1>
            <p>
              {!libraryLoaded || !sampleLoaded
                ? "Restoring this browser's local workspace."
                : "This video is unavailable in this browser, or its sample media is missing. Your other videos and saved results are unchanged."}
            </p>
            <Button icon="back" onClick={home}>
              Back to Videos
            </Button>
          </section>
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
  buttonRef,
}: {
  video: VideoItem;
  onOpen: () => void;
  disabled?: boolean;
  buttonRef?: (element: HTMLButtonElement | null) => void;
}) {
  const [failed, setFailed] = useState(false);
  return (
    <button
      ref={buttonRef}
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
