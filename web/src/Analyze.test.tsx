import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import Analyze, { frameAtTime, PunchSelection, type Job } from "./Analyze";
import App from "./App";

const job: Job = {
  id: "a".repeat(32),
  filename: "synthetic.mp4",
  status: "ready",
  message: "Ready",
  video: {
    frame_count: 5,
    duration_seconds: 0.24,
    width: 640,
    height: 480,
    rotation: 0,
    decoding_seconds: 0.1,
    frames: [0, 0.04, 0.1, 0.14, 0.2].map((media_time, frame_index) => ({
      media_time,
      frame_index,
      timestamp_exact: String(media_time + 2),
    })),
  },
  poses: {
    pose_seconds: 0.5,
    pose_coverage: {
      anchor_frames: 0.95,
      one_arm_frames: 0.9,
      missing_frames: 0,
    },
  },
  processing_wall_seconds: 1,
  preview_url: "/preview.mp4",
  predictions: [],
};
const prediction = {
  status: "available",
  prediction: "lead_hook",
  scores: {
    jab: 0.1,
    cross: 0.1,
    lead_hook: 0.5,
    rear_hook: 0.1,
    lead_uppercut: 0.1,
    rear_uppercut: 0.1,
  },
  start_frame: 0,
  stop_frame_exclusive: 3,
  inference_seconds: 0.01,
};
function response(value: unknown, ok = true) {
  return { ok, json: async () => value } as Response;
}
function selectPunch() {
  fireEvent.load(screen.getByAltText("Selected source frame 1"));
  fireEvent.click(screen.getByRole("button", { name: "Set punch start" }));
  fireEvent.change(screen.getByRole("slider", { name: "Video timeline" }), {
    target: { value: "2" },
  });
  fireEvent.load(screen.getByAltText("Selected source frame 3"));
  fireEvent.click(screen.getByRole("button", { name: "Set punch end" }));
}
beforeEach(() => {
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("defaults to Analyze without querying development artifacts", async () => {
  const fetcher = vi.fn().mockResolvedValue(response({ videos: [] }));
  vi.stubGlobal("fetch", (url: string, options?: RequestInit) =>
    url.endsWith("/activity")
      ? Promise.resolve(response({ video_id: job.id, status: "not_started" }))
      : fetcher(url, options),
  );
  render(<App />);
  await screen.findByRole("heading", {
    name: "See what your punch looks like to AI.",
  });
  expect(screen.getByRole("button", { name: "Analyze" })).toHaveAttribute(
    "aria-current",
    "page",
  );
  expect(screen.getByText("Development")).toBeInTheDocument();
  expect(fetcher.mock.calls.map(([url]) => url)).toEqual([
    "/api/analyze/videos",
  ]);
});

it("maps variable timestamps to source frames without a nominal FPS", () => {
  expect(frameAtTime(job.video!.frames, 0.099)).toBe(1);
  expect(frameAtTime(job.video!.frames, 0.1)).toBe(2);
  expect(frameAtTime(job.video!.frames, 0.239)).toBe(4);
});

it("requires displayed boundaries, submits exclusive source bounds and shows uncalibrated scores", async () => {
  const fetcher = vi.fn().mockResolvedValue(response(prediction));
  vi.stubGlobal("fetch", (url: string, options?: RequestInit) =>
    url.endsWith("/activity")
      ? Promise.resolve(response({ video_id: job.id, status: "not_started" }))
      : fetcher(url, options),
  );
  render(<PunchSelection job={job} onBusy={() => {}} />);
  expect(
    screen.getByRole("button", { name: "Set punch start" }),
  ).toBeDisabled();
  expect(
    screen.getByRole("button", { name: "Classify selected punch" }),
  ).toBeDisabled();
  selectPunch();
  fireEvent.click(
    screen.getByRole("button", { name: "Classify selected punch" }),
  );
  await screen.findByRole("heading", { name: "lead hook" });
  expect(
    screen.getByText("Uncalibrated scores · Not verified confidence"),
  ).toBeInTheDocument();
  expect(screen.getAllByRole("meter")).toHaveLength(6);
  expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({
    start_frame: 0,
    stop_frame_exclusive: 3,
  });
  fireEvent.click(screen.getByRole("button", { name: "Next source frame" }));
  expect(screen.getByAltText("Selected source frame 4")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Set punch end" })).toBeDisabled();
});

it("shows unavailable pose results without inventing scores", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockImplementation((url: string) =>
      Promise.resolve(
        url.endsWith("/activity")
          ? response({ video_id: job.id, status: "not_started" })
          : response({
              status: "unavailable",
              reason: "Not enough reliable poses",
              start_frame: 0,
              stop_frame_exclusive: 3,
            }),
      ),
    ),
  );
  render(<PunchSelection job={job} onBusy={() => {}} />);
  selectPunch();
  fireEvent.click(
    screen.getByRole("button", { name: "Classify selected punch" }),
  );
  await screen.findByRole("heading", { name: "Prediction unavailable" });
  expect(screen.getByText("Not enough reliable poses")).toBeInTheDocument();
  expect(screen.queryAllByRole("meter")).toHaveLength(0);
});

it("locks concurrent classifications and preserves errors for retry", async () => {
  let finish!: (value: Response) => void;
  const fetcher = vi.fn().mockImplementation(
    () =>
      new Promise<Response>((r) => {
        finish = r;
      }),
  );
  vi.stubGlobal("fetch", (url: string, options?: RequestInit) =>
    url.endsWith("/activity")
      ? Promise.resolve(response({ video_id: job.id, status: "not_started" }))
      : fetcher(url, options),
  );
  render(<PunchSelection job={job} onBusy={() => {}} />);
  selectPunch();
  const classify = screen.getByRole("button", {
    name: "Classify selected punch",
  });
  fireEvent.click(classify);
  fireEvent.click(classify);
  expect(fetcher).toHaveBeenCalledTimes(1);
  await act(async () => {
    finish(response({ detail: "Analysis is busy. Try again." }, false));
  });
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Analysis is busy",
  );
  expect(
    screen.getByRole("button", { name: "Classify selected punch" }),
  ).toBeEnabled();
});

it("uses buffered media for play and exact source images for stepping", async () => {
  const play = vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  const { container } = render(<PunchSelection job={job} onBusy={() => {}} />);
  const media = container.querySelector("video")!;
  fireEvent.loadedData(media);
  fireEvent.load(screen.getByAltText("Selected source frame 1"));
  fireEvent.change(screen.getByRole("slider"), { target: { value: "2" } });
  fireEvent.load(screen.getByAltText("Selected source frame 3"));
  await act(async () =>
    fireEvent.click(screen.getByRole("button", { name: "Play" })),
  );
  expect(play).toHaveBeenCalledOnce();
  expect(media.currentTime).toBe(0.1);
  expect(
    screen.queryByAltText("Selected source frame 3"),
  ).not.toBeInTheDocument();
  media.currentTime = 0.15;
  fireEvent.timeUpdate(media);
  fireEvent.click(screen.getByRole("button", { name: "Pause" }));
  expect(screen.getByAltText("Selected source frame 4")).toBeInTheDocument();
});

it("uploads the real file body and follows processing status without development calls", async () => {
  let count = 0;
  const fetcher = vi.fn(async (url: string, _options?: RequestInit) => {
    if (url === "/api/analyze/videos")
      return response({ videos: count++ ? [job] : [] });
    if (url === "/api/analyze/uploads") return response({ id: job.id });
    return response({
      ...job,
      status: "failed",
      message: "Video has invalid timestamps",
    });
  });
  vi.stubGlobal("fetch", (url: string, options?: RequestInit) =>
    url.endsWith("/activity")
      ? Promise.resolve(response({ video_id: job.id, status: "not_started" }))
      : fetcher(url, options),
  );
  render(<Analyze />);
  await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1));
  const file = new File([new Uint8Array([1, 2, 3])], "shadowboxing.mp4", {
    type: "video/mp4",
  });
  fireEvent.change(screen.getByLabelText("Upload shadowboxing video"), {
    target: { files: [file] },
  });
  await screen.findByText("Video has invalid timestamps");
  const call = fetcher.mock.calls.find(
    ([url]) => url === "/api/analyze/uploads",
  )! as unknown as [string, RequestInit];
  expect(call[1].body).toBe(file);
  expect(call[1].headers).toMatchObject({
    "Content-Type": "application/octet-stream",
  });
});

it("rejects unsupported local files before sending them", async () => {
  const fetcher = vi.fn().mockResolvedValue(response({ videos: [] }));
  vi.stubGlobal("fetch", (url: string, options?: RequestInit) =>
    url.endsWith("/activity")
      ? Promise.resolve(response({ video_id: job.id, status: "not_started" }))
      : fetcher(url, options),
  );
  render(<Analyze />);
  fireEvent.change(screen.getByLabelText("Upload shadowboxing video"), {
    target: { files: [new File(["x"], "unsafe.html")] },
  });
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Choose a nonempty MP4",
  );
  expect(fetcher.mock.calls.some(([url]) => url.includes("/uploads"))).toBe(
    false,
  );
});

it("replays only the selected span and returns to its exact last frame", async () => {
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
  const { container } = render(<PunchSelection job={job} onBusy={() => {}} />);
  const media = container.querySelector("video")!;
  fireEvent.loadedData(media);
  selectPunch();
  await act(async () =>
    fireEvent.click(screen.getByRole("button", { name: "Replay selection" })),
  );
  expect(media.currentTime).toBe(0);
  media.currentTime = 0.14;
  fireEvent.timeUpdate(media);
  expect(screen.getByAltText("Selected source frame 3")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
});
