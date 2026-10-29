import { useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  clamp,
  frameTime,
  nextBoundary,
  previewSelection,
  type Bounds,
  type Selection,
  type VideoItem,
} from "./model";
import {
  checkSelection,
  emptyDraft,
  sameBounds,
  type Draft,
} from "./selection";
import { Button, IconButton } from "./ui";
import { Transport, usePlayer, VideoStage } from "./Player";
import Timeline from "./Timeline";
import Inspector from "./Inspector";
import type { InspectorTab, WorkspaceSession } from "./navigation";

export default function Workspace({
  video,
  selections,
  initialSession,
  onSession,
  onSave,
  onVideos,
  onShortcuts,
}: {
  video: VideoItem;
  selections: Selection[];
  initialSession?: WorkspaceSession;
  onSession: (id: string, s: WorkspaceSession) => void;
  onSave: (s: Selection) => void;
  onVideos: () => void;
  onShortcuts: () => void;
}) {
  const initial = useRef(initialSession).current;
  const player = usePlayer(
    video,
    clamp(initial?.current ?? 0, 0, video.duration),
  );
  const [draft, setDraft] = useState<Draft>(
    () => initial?.draft ?? { ...emptyDraft },
  );
  const [result, setResult] = useState<Selection | null>(() => {
    const s = selections.find((s) => s.id === initial?.resultId);
    const b = initial && checkSelection(video, initial.draft).bounds;
    return s && b && sameBounds(s.bounds, b) ? s : null;
  });
  const [tab, setTab] = useState<InspectorTab>(initial?.tab ?? "selection");
  const [busy, setBusy] = useState(false);
  const generation = useRef(0),
    timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const heading = useRef<HTMLHeadingElement>(null),
    viewer = useRef<HTMLDivElement>(null),
    footer = useRef<HTMLDivElement>(null);
  const [viewerWidth, setViewerWidth] = useState<number>();
  const latest = useRef<WorkspaceSession>({
    current: player.current,
    draft,
    resultId: result?.id ?? null,
    tab,
  });
  latest.current = {
    current: player.current,
    draft,
    resultId: result?.id ?? null,
    tab,
  };
  useEffect(() => {
    heading.current?.focus({ preventScroll: true });
  }, []);
  useEffect(() => {
    const save = () => onSession(video.id, latest.current);
    window.addEventListener("pagehide", save);
    return () => {
      generation.current++;
      if (timer.current) clearTimeout(timer.current);
      save();
      window.removeEventListener("pagehide", save);
    };
  }, [video.id, onSession]);
  useEffect(() => {
    const save = setTimeout(() => onSession(video.id, latest.current), 150);
    return () => clearTimeout(save);
  }, [draft, result, tab, player.current, video.id, onSession]);
  useLayoutEffect(() => {
    const area = viewer.current!,
      tools = footer.current!;
    const resize = () =>
      setViewerWidth(
        Math.min(
          area.clientWidth,
          (Math.max(1, area.clientHeight - tools.offsetHeight) * video.width) /
            video.height,
        ),
      );
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(area);
    observer.observe(tools);
    return () => observer.disconnect();
  }, [video.width, video.height]);
  function change(next: Draft) {
    if (busy) return;
    setDraft(next);
    setResult(null);
    setTab("selection");
  }
  function mark(mode: "start" | "end") {
    if (!player.inspectionReady) return;
    const time = frameTime(video, player.current);
    change({
      ...draft,
      [mode]: mode === "end" ? nextBoundary(video, time) : time,
    });
  }
  function select(next: Bounds) {
    change(next);
  }
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (
        (e.target instanceof HTMLElement &&
          e.target.closest("button,input,dialog,a,summary,[role=tablist]")) ||
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
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });
  function classify() {
    const bounds = checkSelection(video, draft).bounds;
    if (!bounds || busy || !player.inspectionReady) return;
    player.pause();
    setBusy(true);
    setTab("selection");
    const ticket = ++generation.current;
    timer.current = setTimeout(() => {
      if (ticket !== generation.current) return;
      const saved =
        selections.find((s) => sameBounds(s.bounds, bounds)) ??
        previewSelection(video.id, bounds);
      onSave(saved);
      setResult(saved);
      setBusy(false);
    }, 450);
  }
  const checked = checkSelection(video, draft);
  return (
    <section className="workspace" aria-label="Analyze workspace">
      <div className="workspace-heading">
        <div className="cluster">
          <Button variant="ghost" icon="back" onClick={onVideos}>
            Back to Videos
          </Button>
          <div>
            <h1 ref={heading} tabIndex={-1}>
              {video.title}
            </h1>
            <p>
              {video.kind === "sample" ? "Reference sample" : "Local video"} ·{" "}
              {video.width} × {video.height}
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
        <div className="video-workspace" ref={viewer}>
          <div className="viewer-stack" style={{ width: viewerWidth }}>
            <VideoStage video={video} player={player} />
            <div ref={footer}>
              <Timeline
                video={video}
                current={player.current}
                bounds={
                  checked.bounds ??
                  (draft.start !== null &&
                  draft.end !== null &&
                  draft.end > draft.start &&
                  draft.start >= 0 &&
                  draft.end <= video.duration
                    ? { start: draft.start, end: draft.end }
                    : null)
                }
                draft={draft}
                disabled={busy}
                onSeek={player.seek}
                onBounds={select}
              />
              <div className="viewer-controls">
                <Transport video={video} player={player} />
                <div className="selection-tools">
                  <Button
                    icon="bracket-in"
                    disabled={busy || !player.inspectionReady}
                    onClick={() => mark("start")}
                    aria-label="Set punch start (I)"
                  >
                    Set start<kbd>I</kbd>
                  </Button>
                  <Button
                    icon="bracket-out"
                    disabled={busy || !player.inspectionReady}
                    onClick={() => mark("end")}
                    aria-label="Set punch end (O)"
                  >
                    Set end<kbd>O</kbd>
                  </Button>
                </div>
              </div>
              <p className="precision-note">
                {video.frameTimes
                  ? "Source-frame selection"
                  : "Time-based selection · Exact frames unavailable in this prototype"}
              </p>
            </div>
          </div>
        </div>
        <Inspector
          video={video}
          draft={draft}
          result={result}
          busy={busy}
          inspectionReady={player.inspectionReady}
          tab={tab}
          selections={selections}
          onTab={setTab}
          onClear={() => change({ ...emptyDraft })}
          onReplay={() => {
            if (checked.bounds) void player.play(checked.bounds);
          }}
          onClassify={classify}
          onOpen={(s) => {
            if (busy) return;
            setDraft(s.bounds);
            setResult(
              checkSelection(video, s.bounds).state === "valid" ? s : null,
            );
            setTab("selection");
            player.seek(s.bounds.start);
          }}
        />
      </div>
    </section>
  );
}
