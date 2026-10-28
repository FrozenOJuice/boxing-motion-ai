import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import BufferedIntervalViewer from "./BufferedIntervalViewer";

let callback: VideoFrameRequestCallback | null = null;
let intersection: IntersectionObserverCallback;
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  callback = null;
});
function mount() {
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
    if (String(url).endsWith("/playback"))
      return {
        ok: true,
        json: async () => ({
          data: {
            interval_id: "V4-bg-1",
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
  const presented = vi.fn(),
    visible = vi.fn();
  render(
    <BufferedIntervalViewer
      intervalId="V4-bg-1"
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
      onPresented={presented}
    />,
  );
  return { presented, visible, fetchMock };
}
function tick(frame: number, count: number) {
  act(() =>
    callback?.(100 + frame * 40, {
      mediaTime: (frame - 10) * 0.04,
      presentedFrames: count,
    } as VideoFrameCallbackMetadata),
  );
}

it("plays buffered video without per-frame HTTP requests; skips never fill coverage", async () => {
  const { presented, fetchMock } = mount();
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Play" })).not.toBeDisabled(),
  );
  const v = screen.getByLabelText(
    "Buffered interval video",
  ) as HTMLVideoElement;
  Object.defineProperty(v, "readyState", { configurable: true, value: 4 });
  fireEvent.click(screen.getByRole("button", { name: "Play" }));
  const before = fetchMock.mock.calls.length;
  tick(10, 1);
  tick(11, 2);
  tick(13, 4);
  expect(presented.mock.calls.map(([n]) => n)).toEqual([11]);
  expect(fetchMock.mock.calls.length).toBe(before);
  fireEvent.seeking(v);
  tick(12, 5);
  fireEvent.seeked(v);
  tick(12, 6);
  expect(presented.mock.calls.map(([n]) => n)).toEqual([11]);
  fireEvent.waiting(v);
  tick(13, 7);
  expect(presented.mock.calls.map(([n]) => n)).toEqual([11]);
});

it("pauses outside the viewport and accepts neither hidden nor paused playback as review evidence", async () => {
  const { presented } = mount();
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Play" })).not.toBeDisabled(),
  );
  const v = screen.getByLabelText(
    "Buffered interval video",
  ) as HTMLVideoElement;
  Object.defineProperty(v, "readyState", { configurable: true, value: 4 });
  fireEvent.click(screen.getByRole("button", { name: "Play" }));
  tick(10, 1);
  act(() =>
    intersection(
      [{ intersectionRatio: 0 } as IntersectionObserverEntry],
      {} as IntersectionObserver,
    ),
  );
  tick(11, 2);
  expect(presented).not.toHaveBeenCalled();
  expect(v.paused).toBe(true);
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

it("does not count timeupdate ranges when the frame callback API is missing", async () => {
  const { presented } = mount();
  HTMLVideoElement.prototype.requestVideoFrameCallback =
    undefined as unknown as typeof HTMLVideoElement.prototype.requestVideoFrameCallback;
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Play" })).not.toBeDisabled(),
  );
  await screen.findByText(
    /This browser cannot report individual presented frames/,
  );
  const v = screen.getByLabelText(
    "Buffered interval video",
  ) as HTMLVideoElement;
  fireEvent.click(screen.getByRole("button", { name: "Play" }));
  v.currentTime = 0.12;
  fireEvent.timeUpdate(v);
  expect(presented).not.toHaveBeenCalled();
});

it("keeps the review player height stable and frame inspection usable after a video error", async () => {
  mount();
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Play" })).not.toBeDisabled(),
  );
  fireEvent.error(screen.getByLabelText("Buffered interval video"));
  expect(screen.getByRole("alert")).toHaveTextContent(
    "Exact-frame inspection remains available",
  );
  expect(document.querySelector(".buffered-stage")).toHaveClass("image-stage");
  fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
  expect(await screen.findByAltText("V4, native frame 11")).toBeInTheDocument();
});
