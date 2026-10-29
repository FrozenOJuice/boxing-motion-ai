import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import AutomaticPunches from "./AutomaticPunches";
const response = (value: unknown, ok = true) => ({
  ok,
  json: async () => value,
});
const fixture = (id = "a") => ({
  video_id: id,
  status: "ready",
  analysis_wall_seconds: 1,
  result: {
    classifier_checkpoint: { sha256: "synthetic-checkpoint" },
    candidates: [
      {
        candidate_id: "candidate-1",
        start_frame: 10,
        stop_frame_exclusive: 31,
        start_media_time: 0.2,
        end_media_time: 0.62,
        classification: {
          status: "available",
          prediction: "jab",
          scores: {
            jab: 0.4,
            cross: 0.2,
            lead_hook: 0.1,
            rear_hook: 0.1,
            lead_uppercut: 0.1,
            rear_uppercut: 0.1,
          },
        },
      },
      {
        candidate_id: "candidate-2",
        start_frame: 40,
        stop_frame_exclusive: 50,
        start_media_time: 0.8,
        end_media_time: 1,
        classification: {
          status: "unavailable",
          reason: "Insufficient pose coverage",
        },
      },
    ],
    unresolved_activity: [
      {
        start_frame: 60,
        stop_frame_exclusive: 70,
        start_media_time: 1.2,
        end_media_time: 1.4,
        reason: "unresolved",
      },
    ],
  },
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
it("runs one automatic task, distinguishes predictions from candidates and replays exact exclusive bounds", async () => {
  let reads = 0;
  const fetcher = vi.fn(async (_url: string, opts?: RequestInit) =>
    response(
      opts?.method === "POST"
        ? { video_id: "a", status: "processing" }
        : reads++
          ? fixture()
          : { video_id: "a", status: "not_started" },
    ),
  );
  vi.stubGlobal("fetch", fetcher);
  const replay = vi.fn(),
    complete = vi.fn(),
    busy = vi.fn();
  render(
    <AutomaticPunches
      videoId="a"
      disabled={false}
      ready
      onBusy={busy}
      onReplay={replay}
      onComplete={complete}
    />,
  );
  await waitFor(() => expect(fetcher).toHaveBeenCalledOnce());
  fireEvent.click(
    screen.getByRole("button", { name: "Analyze automatically" }),
  );
  await screen.findByText("jab · predicted type");
  expect(
    screen.getByText(/Candidates are not verified punches/),
  ).toBeInTheDocument();
  fireEvent.click(
    screen.getAllByRole("button", { name: "Replay candidate" })[0],
  );
  expect(replay).toHaveBeenCalledWith(10, 31);
  fireEvent.click(screen.getByText(/Six model scores/));
  expect(screen.getAllByRole("meter", { hidden: true })).toHaveLength(6);
  expect(screen.getByText("Classification unavailable")).toBeInTheDocument();
  expect(screen.getByText("Insufficient pose coverage")).toBeInTheDocument();
  expect(
    screen.getByText(/Unresolved activity · individual boundaries/),
  ).toBeInTheDocument();
  expect(complete).toHaveBeenCalledOnce();
  expect(busy).toHaveBeenCalledWith(true);
});
it("ignores a previous video's late result", async () => {
  let finish!: (v: unknown) => void;
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) =>
      url.includes("/a/")
        ? new Promise((r) => {
            finish = r;
          })
        : Promise.resolve(response({ video_id: "b", status: "not_started" })),
    ),
  );
  const busy = vi.fn(),
    replay = vi.fn(),
    complete = vi.fn();
  const { rerender } = render(
    <AutomaticPunches
      videoId="a"
      disabled={false}
      ready
      onBusy={busy}
      onReplay={replay}
      onComplete={complete}
    />,
  );
  rerender(
    <AutomaticPunches
      videoId="b"
      disabled={false}
      ready
      onBusy={busy}
      onReplay={replay}
      onComplete={complete}
    />,
  );
  await act(async () => finish(response(fixture())));
  expect(screen.queryByText("jab · predicted type")).not.toBeInTheDocument();
});
