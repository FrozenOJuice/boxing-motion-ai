import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import ActivityScan from "./ActivityScan";
const reply = (value: unknown, ok = true) => ({ ok, json: async () => value });
const result = (id = "a") => ({
  video_id: id,
  status: "ready",
  scan_wall_seconds: 1.2,
  result: {
    bouts: [
      {
        bout_id: "one",
        start_frame: 12,
        stop_frame_exclusive: 30,
        start_media_time: 0.5,
        end_media_time: 1.25,
        left_censored: false,
        right_censored: true,
      },
    ],
    observable_frames: 40,
    total_frames: 50,
    unobservable_intervals: [
      {
        reason: "missing_or_unreliable_pose",
        start_media_time: 2,
        end_media_time: 2.4,
      },
    ],
  },
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});
it("scans without a manual selection, polls, shows experimental ranges and seeks without classifying", async () => {
  let count = 0;
  const fetcher = vi.fn(async (url: string) =>
    reply(
      url.endsWith("/scan")
        ? { video_id: "a", status: "processing" }
        : count++
          ? result()
          : { video_id: "a", status: "not_started" },
    ),
  );
  vi.stubGlobal("fetch", fetcher);
  const seek = vi.fn(),
    busy = vi.fn();
  render(
    <ActivityScan videoId="a" disabled={false} onBusy={busy} onSeek={seek} />,
  );
  await waitFor(() => expect(fetcher).toHaveBeenCalledOnce());
  fireEvent.click(screen.getByRole("button", { name: "Scan for activity" }));
  const range = await screen.findByRole("button", { name: /0.500–1.250 s/ });
  fireEvent.click(range);
  expect(seek).toHaveBeenCalledWith(12);
  expect(screen.getByText(/Training-only smoke model/)).toBeInTheDocument();
  expect(
    screen.getByText(/Unobservable intervals — not background/),
  ).toBeInTheDocument();
  expect(fetcher.mock.calls.some(([url]) => url.includes("classify"))).toBe(
    false,
  );
  expect(busy).toHaveBeenCalledWith(true);
});
it("does not display a previous video's late result", async () => {
  let finish!: (value: unknown) => void;
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) =>
      url.includes("/a/")
        ? new Promise((r) => {
            finish = r;
          })
        : Promise.resolve(reply({ video_id: "b", status: "not_started" })),
    ),
  );
  const busy = vi.fn(),
    seek = vi.fn();
  const { rerender } = render(
    <ActivityScan videoId="a" disabled={false} onBusy={busy} onSeek={seek} />,
  );
  rerender(
    <ActivityScan videoId="b" disabled={false} onBusy={busy} onSeek={seek} />,
  );
  await act(async () => finish(reply(result("a"))));
  expect(
    screen.queryByRole("button", { name: /0.500/ }),
  ).not.toBeInTheDocument();
});
it("handles scan errors and rejects mismatched cached video results", async () => {
  const fetcher = vi
    .fn()
    .mockResolvedValueOnce(reply(result("wrong")))
    .mockResolvedValueOnce(reply({ detail: "Checkpoint unavailable" }, false))
    .mockResolvedValueOnce(
      reply({ ...result(), result: { ...result().result, bouts: [] } }),
    );
  vi.stubGlobal("fetch", fetcher);
  render(
    <ActivityScan
      videoId="a"
      disabled={false}
      onBusy={() => {}}
      onSeek={() => {}}
    />,
  );
  expect(await screen.findByRole("alert")).toHaveTextContent("does not match");
  fireEvent.click(screen.getByRole("button", { name: "Scan for activity" }));
  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Checkpoint unavailable",
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Scan for activity" }));
  expect(
    await screen.findByText(
      "No activity ranges emitted. This does not confirm background.",
    ),
  ).toBeInTheDocument();
});
