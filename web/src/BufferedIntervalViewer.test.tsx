import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import BufferedIntervalViewer from "./BufferedIntervalViewer";
import { type IntervalAnnotation } from "./intervalAnnotations";

let callback: VideoFrameRequestCallback | null = null;
let intersection: IntersectionObserverCallback;
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  callback = null;
  vi.useRealTimers();
});
function mount(
  annotations: IntervalAnnotation[] = [],
  pilot: "original" | "motion-v2" = "original",
) {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
    x: 0,
    y: 0,
    top: 0,
    left: 0,
    width: 640,
    height: 360,
    right: 640,
    bottom: 360,
    toJSON: () => ({}),
  });
  HTMLElement.prototype.scrollIntoView = vi.fn();
  vi.stubGlobal(
    "IntersectionObserver",
    class {
      constructor(cb: IntersectionObserverCallback) {
        intersection = cb;
      }
      observe() {
        intersection(
          [{ intersectionRatio: 1 } as IntersectionObserverEntry],
          {} as IntersectionObserver,
        );
      }
      disconnect() {}
    },
  );
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    cb(0);
    return 1;
  });
  vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(function (
    this: HTMLVideoElement,
  ) {
    Object.defineProperty(this, "paused", { configurable: true, value: false });
    fireEvent.play(this);
    fireEvent.playing(this);
    return Promise.resolve();
  });
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(function (
    this: HTMLVideoElement,
  ) {
    Object.defineProperty(this, "paused", { configurable: true, value: true });
    fireEvent.pause(this);
  });
  HTMLVideoElement.prototype.requestVideoFrameCallback = vi.fn((cb) => {
    callback = cb;
    return 1;
  });
  HTMLVideoElement.prototype.cancelVideoFrameCallback = vi.fn();
  const fetchMock = vi.fn(async (url: RequestInfo | URL) => {
    if (String(url).includes("/playback"))
      return {
        ok: true,
        json: async () => ({
          data: {
            interval_id: "V4-bg-1",
            cache_key: "a".repeat(64),
            video_url: "/clip.mp4",
            frames: [10, 11, 12, 13],
            media_times: [0, 0.04, 0.08, 0.12],
            source_times: ["1", "26/25", "27/25", "28/25"],
            source_start: "1",
            duration: 0.16,
            width: 32,
            height: 32,
          },
        }),
      };
    const n = Number(String(url).split("/").at(-1));
    return {
      ok: true,
      json: async () => ({
        data: {
          frame_index: n,
          image_url: `/frame-${n}.jpg`,
          width: 32,
          height: 32,
          landmarks: [],
          edges: [],
        },
      }),
    };
  });
  vi.stubGlobal("fetch", fetchMock);
  const viewing = vi.fn(),
    visible = vi.fn();
  render(
    <BufferedIntervalViewer
      intervalId={pilot === "original" ? "V4-bg-1" : "V4-motion-1"}
      pilot={pilot}
      annotations={annotations}
      recording="V4"
      interval={{
        start: 10,
        end: 12,
        count: 50,
        contextStart: 10,
        contextEnd: 13,
        delay: 40,
      }}
      seenFrames={new Set()}
      onVisible={visible}
      onViewing={viewing}
    />,
  );
  return { viewing, visible, fetchMock };
}
async function readyPlayer() {
  const v = await screen.findByLabelText("Buffered interval video");
  Object.defineProperty(v, "readyState", { configurable: true, value: 4 });
  fireEvent.loadedData(v);
  return v;
}
function tick(frame: number, count: number) {
  act(() =>
    callback?.(100 + frame * 40, {
      mediaTime: (frame - 10) * 0.04,
      presentedFrames: count,
    } as VideoFrameCallbackMetadata),
  );
}

it("records real-time visible playback independently of frame callbacks and HTTP inspection", async () => {
  const { viewing, visible, fetchMock } = mount();
  await readyPlayer();
  const v = screen.getByLabelText(
    "Buffered interval video",
  ) as HTMLVideoElement;
  Object.defineProperty(v, "readyState", { configurable: true, value: 4 });
  let now = 1000;
  vi.spyOn(performance, "now").mockImplementation(() => now);
  fireEvent.click(screen.getByRole("button", { name: "Play" }));
  const before = fetchMock.mock.calls.length;
  v.currentTime = 0.1;
  now = 1100;
  fireEvent.timeUpdate(v);
  v.currentTime = 0.16;
  now = 1200;
  fireEvent.timeUpdate(v);
  expect(viewing.mock.lastCall?.[0]).toMatchObject({
    complete: true,
    playbackComplete: true,
  });
  expect(viewing.mock.lastCall?.[0].playbackComplete).toBe(true);
  expect(visible).not.toHaveBeenCalled();
  expect(fetchMock.mock.calls.length).toBe(before);
  expect(HTMLElement.prototype.scrollIntoView).toHaveBeenCalled();
});

it("pauses and stops viewing credit outside the viewport", async () => {
  const { viewing } = mount();
  await readyPlayer();
  const v = screen.getByLabelText(
    "Buffered interval video",
  ) as HTMLVideoElement;
  Object.defineProperty(v, "readyState", { configurable: true, value: 4 });
  fireEvent.click(screen.getByRole("button", { name: "Play" }));
  vi.mocked(HTMLElement.prototype.getBoundingClientRect).mockReturnValue({
    x: 0,
    y: -200,
    top: -200,
    left: 0,
    width: 640,
    height: 360,
    right: 640,
    bottom: 160,
    toJSON: () => ({}),
  });
  act(() =>
    intersection(
      [{ intersectionRatio: 0 } as IntersectionObserverEntry],
      {} as IntersectionObserver,
    ),
  );
  expect(v.paused).toBe(true);
  expect(viewing.mock.lastCall?.[0].complete).toBe(false);
});

it("keeps exact frame stepping and skeletons separate from browser seeking", async () => {
  const { visible } = mount();
  const first = await screen.findByAltText("V4, native frame 10");
  fireEvent.load(first);
  await waitFor(() =>
    expect(visible).toHaveBeenCalledWith(
      expect.objectContaining({ frame_index: 10 }),
    ),
  );
  fireEvent.change(screen.getByRole("combobox", { name: "Playback speed" }), {
    target: { value: "0.25" },
  });
  expect(
    (screen.getByLabelText("Buffered interval video") as HTMLVideoElement)
      .playbackRate,
  ).toBe(0.25);
  fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
  const next = await screen.findByAltText("V4, native frame 11");
  fireEvent.load(next);
  await waitFor(() =>
    expect(visible).toHaveBeenLastCalledWith(
      expect.objectContaining({ frame_index: 11 }),
    ),
  );
  expect(screen.getByText("Frame 11")).toBeInTheDocument();
});

it("does not pretend timeupdate events are individual-frame inspection when callbacks are missing", async () => {
  const { viewing } = mount();
  HTMLVideoElement.prototype.requestVideoFrameCallback =
    undefined as unknown as typeof HTMLVideoElement.prototype.requestVideoFrameCallback;
  await readyPlayer();
  await screen.findByText(/Playback frame positions are approximate/);
  const v = screen.getByLabelText(
    "Buffered interval video",
  ) as HTMLVideoElement;
  fireEvent.click(screen.getByRole("button", { name: "Play" }));
  v.currentTime = 0.12;
  fireEvent.timeUpdate(v);
  expect(viewing.mock.lastCall?.[0].complete).toBe(false);
});

it("keeps the review player height stable and frame inspection usable after a video error", async () => {
  mount();
  await readyPlayer();
  fireEvent.error(screen.getByLabelText("Buffered interval video"));
  expect(screen.getByRole("alert")).toHaveTextContent(
    "Exact-frame inspection remains available",
  );
  expect(document.querySelector(".buffered-stage")).toHaveClass("image-stage");
  fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
  expect(await screen.findByAltText("V4, native frame 11")).toBeInTheDocument();
});

it("shows source counts/types, partial markers and exact jumps without filling skipped coverage", async () => {
  const annotations = [
    {
      source_row: 2,
      source_class: "Jab",
      punch_type_provisional: "jab",
      raw_start: 9,
      raw_end: 10,
      source_interval_half_open: [9, 11],
    },
    {
      source_row: 3,
      source_class: "Rear Hook",
      punch_type_provisional: "rear_hook",
      raw_start: 12,
      raw_end: 16,
      source_interval_half_open: [12, 17],
    },
  ] as IntervalAnnotation[];
  const { viewing, visible } = mount(annotations);
  expect(
    screen.getByLabelText("Original annotation summary"),
  ).toHaveTextContent("2 annotated punches");
  expect(
    screen.getByLabelText("Original annotation summary"),
  ).toHaveTextContent("Jab 1 · Rear Hook 1");
  expect(
    screen.getByText("Original dataset · Provisional"),
  ).toBeInTheDocument();
  const partial = screen.getByRole("button", {
    name: "Jab · Row 2 · Frames 9–10 · Partial overlap",
  });
  expect(partial).toHaveClass("partial");
  const first = await screen.findByAltText("V4, native frame 10");
  fireEvent.load(first);
  const marker = screen.getByRole("button", {
    name: "Rear Hook · Row 3 · Frames 12–16 · Partial overlap",
  });
  fireEvent.click(marker);
  const target = await screen.findByAltText("V4, native frame 12");
  expect(viewing.mock.lastCall?.[0].complete).toBe(false);
  fireEvent.load(target);
  await waitFor(() =>
    expect(visible).toHaveBeenLastCalledWith(
      expect.objectContaining({ frame_index: 12 }),
    ),
  );
  expect(marker).toHaveAttribute("aria-current", "true");
  expect(partial).not.toHaveAttribute("aria-current");
  expect(visible.mock.calls.some(([f]) => f.frame_index === 11)).toBe(false);
  expect(screen.getByText(/Ends after this interval/)).toBeInTheDocument();
});

it("updates the current annotation from buffered presentation callbacks", async () => {
  mount([
    {
      source_row: 7,
      source_class: "Cross",
      source_interval_half_open: [11, 12],
    } as IntervalAnnotation,
  ]);
  await readyPlayer();
  const v = screen.getByLabelText(
    "Buffered interval video",
  ) as HTMLVideoElement;
  Object.defineProperty(v, "readyState", { configurable: true, value: 4 });
  fireEvent.click(screen.getByRole("button", { name: "Play" }));
  tick(10, 1);
  tick(11, 2);
  expect(
    screen.getByRole("button", { name: "Cross · Row 7 · Frames 11–11" }),
  ).toHaveAttribute("aria-current", "true");
  tick(12, 3);
  expect(
    screen.getByRole("button", { name: "Cross · Row 7 · Frames 11–11" }),
  ).not.toHaveAttribute("aria-current");
});

it("requests playback metadata from the selected pilot", async () => {
  const { fetchMock } = mount([], "motion-v2");
  await readyPlayer();
  expect(
    fetchMock.mock.calls.some(
      ([url]) =>
        String(url) === "/api/intervals/V4-motion-1/playback?pilot=motion-v2",
    ),
  ).toBe(true);
});

it("flushes native pause progress and resumes at the media clock, not a rounded frame index", async () => {
  const { viewing } = mount();
  await readyPlayer();
  const v = screen.getByLabelText(
    "Buffered interval video",
  ) as HTMLVideoElement;
  Object.defineProperty(v, "readyState", { configurable: true, value: 4 });
  let now = 1000;
  vi.spyOn(performance, "now").mockImplementation(() => now);
  fireEvent.click(screen.getByRole("button", { name: "Play" }));
  now = 1050;
  v.currentTime = 0.05;
  Object.defineProperty(v, "paused", { configurable: true, value: true });
  fireEvent.timeUpdate(v);
  fireEvent.pause(v);
  expect(viewing.mock.lastCall?.[0].evidence.segments[0].end).toBe(0.05);
  now = 2000;
  fireEvent.click(screen.getByRole("button", { name: "Play" }));
  expect(v.currentTime).toBe(0.05);
  now = 2100;
  v.currentTime = 0.15;
  fireEvent.timeUpdate(v);
  expect(viewing.mock.lastCall?.[0].complete).toBe(true);
});

for (const event of ["visibilitychange", "blur"] as const) {
  it(`ends the visible epoch on ${event} and never credits hidden time`, async () => {
    const { viewing } = mount();
    const v = (await readyPlayer()) as HTMLVideoElement;
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    now = 1040;
    v.currentTime = 0.04;
    if (event === "visibilitychange") {
      vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden");
      fireEvent(document, new Event(event));
    } else fireEvent(window, new Event(event));
    expect(v.paused).toBe(true);
    const before = viewing.mock.lastCall?.[0];
    expect(before.evidence.segments[0].end).toBe(0.04);
    now = 1200;
    v.currentTime = 0.16;
    fireEvent.timeUpdate(v);
    expect(viewing.mock.lastCall?.[0]).toEqual(before);
    expect(before.complete).toBe(false);
  });
}
