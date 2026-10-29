import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { recentVideos } from "./storage";
import { sample } from "./model";
vi.mock("./storage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./storage")>()),
  recentVideos: vi.fn().mockResolvedValue([]),
  importVideo: vi.fn(),
}));
beforeEach(() => {
  history.replaceState(null, "", "/videos");
  sessionStorage.clear();
  vi.mocked(recentVideos).mockResolvedValue([]);
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ source_sha256: "sample" }),
    }),
  );
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
  vi.spyOn(window, "scrollTo").mockImplementation(() => {});
  HTMLDialogElement.prototype.showModal = function () {
    this.setAttribute("open", "");
  };
  HTMLDialogElement.prototype.close = function () {
    this.removeAttribute("open");
  };
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

async function openSample() {
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Open Shadowboxing sample" }),
    ).toBeEnabled(),
  );
  fireEvent.click(
    screen.getByRole("button", { name: "Open Shadowboxing sample" }),
  );
  loadedFrame();
}
function loadedFrame() {
  const image = screen.queryByRole("img", { name: /Sample source frame/ });
  if (image) fireEvent.load(image);
}

it("starts in Videos with one upload action and no independent Analyze destination", async () => {
  render(<App />);
  expect(screen.getByRole("heading", { name: "Videos" })).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Analyze" }),
  ).not.toBeInTheDocument();
  expect(screen.getAllByRole("button", { name: "Upload video" })).toHaveLength(
    1,
  );
  await openSample();
  expect(
    screen.getByRole("heading", { name: "Select one punch" }),
  ).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Classify punch" }),
  ).not.toBeInTheDocument();
  expect(screen.queryByText("Selection quality")).not.toBeInTheDocument();
  expect(screen.queryByText("AI prediction")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Back to Videos" }));
  expect(screen.getByRole("heading", { name: "Videos" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Development" }));
  expect(screen.getByRole("dialog")).toHaveTextContent(
    "does not open or modify",
  );
});
it("handles an empty library without duplicating the upload action", async () => {
  vi.mocked(fetch).mockResolvedValue({ ok: false } as Response);
  render(<App />);
  expect(await screen.findByText("No videos yet")).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: /^Open / }),
  ).not.toBeInTheDocument();
  expect(screen.getAllByRole("button", { name: "Upload video" })).toHaveLength(
    1,
  );
});
it("renders a single available sample as a normal library item", async () => {
  render(<App />);
  expect(
    await screen.findByRole("button", { name: "Open Shadowboxing sample" }),
  ).toBeInTheDocument();
  expect(screen.getAllByRole("button", { name: /^Open / })).toHaveLength(1);
});
it("keeps all videos accessible in a larger library", async () => {
  vi.mocked(recentVideos).mockResolvedValue(
    Array.from({ length: 20 }, (_, i) => ({
      ...sample,
      id: `local-${i}`,
      title: `Video ${i + 1}`,
      kind: "local",
    })),
  );
  render(<App />);
  expect(
    await screen.findByRole("button", { name: "Open Video 20" }),
  ).toBeInTheDocument();
  expect(screen.getAllByRole("button", { name: /^Open / })).toHaveLength(21);
  expect(screen.getAllByRole("button", { name: "Upload video" })).toHaveLength(
    1,
  );
});
it("opens saved predictions beside the video and clears stale output when bounds change", async () => {
  render(<App />);
  await openSample();
  fireEvent.click(screen.getByRole("tab", { name: /Saved punches/ }));
  fireEvent.click(screen.getByRole("button", { name: /lead hook/i }));
  loadedFrame();
  expect(screen.getByText("Saved model output")).toBeInTheDocument();
  expect(screen.getAllByRole("meter")).toHaveLength(6);
  expect(screen.getByText(/Uncalibrated model scores/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Set punch end (O)" }));
  expect(screen.queryByRole("meter")).not.toBeInTheDocument();
  expect(screen.queryByText("Saved model output")).not.toBeInTheDocument();
});
it("shows example classification in the same workspace without a model API", async () => {
  render(<App />);
  await openSample();
  fireEvent.click(screen.getByRole("button", { name: "Set punch start (I)" }));
  fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
  loadedFrame();
  fireEvent.click(screen.getByRole("button", { name: "Set punch end (O)" }));
  fireEvent.click(screen.getByRole("button", { name: "Classify punch" }));
  expect(
    screen.getByRole("button", { name: "Loading example…" }),
  ).toHaveAttribute("aria-disabled", "true");
  await screen.findByText("Prototype example");
  expect(
    screen.getByText("No model ran for this selection."),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("region", { name: "Analyze workspace" }),
  ).toBeInTheDocument();
  expect(vi.mocked(fetch).mock.calls.map(([url]) => url)).toEqual([
    "/demo/manifest.json",
  ]);
  fireEvent.click(screen.getByRole("button", { name: "Clear selection" }));
  expect(screen.queryByRole("meter")).not.toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Classify punch" }),
  ).not.toBeInTheDocument();
});
it("restores keyboard focus after a dialog closes", () => {
  render(<App />);
  const trigger = screen.getByRole("button", { name: "Design prototype" });
  trigger.focus();
  fireEvent.click(trigger);
  fireEvent.click(screen.getByRole("button", { name: "Close dialog" }));
  expect(trigger).toHaveFocus();
});

it("requires both explicit boundaries and rejects a one-frame range", async () => {
  render(<App />);
  await openSample();
  const classify = () => screen.getByRole("button", { name: "Classify punch" });
  fireEvent.click(screen.getByRole("button", { name: "Set punch start (I)" }));
  expect(classify()).toBeDisabled();
  expect(screen.getByText("Set the end of the punch.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Set punch end (O)" }));
  expect(classify()).toBeDisabled();
  expect(
    screen.getByText(/Select at least two source frames/),
  ).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
  loadedFrame();
  fireEvent.click(screen.getByRole("button", { name: "Set punch end (O)" }));
  expect(classify()).toBeEnabled();
});
it("preserves a draft, playhead, tab and focus through Back, reopen and browser history", async () => {
  render(<App />);
  await openSample();
  expect(location.pathname).toBe("/videos/reference-sample");
  expect(screen.getByRole("heading", { name: sample.title })).toHaveFocus();
  fireEvent.click(screen.getByRole("button", { name: "Set punch start (I)" }));
  fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
  loadedFrame();
  fireEvent.click(screen.getByRole("button", { name: "Set punch end (O)" }));
  fireEvent.click(screen.getByRole("tab", { name: /Saved punches/ }));
  fireEvent.click(screen.getByRole("button", { name: "Back to Videos" }));
  expect(location.pathname).toBe("/videos");
  expect(
    screen.getByRole("button", { name: "Open Shadowboxing sample" }),
  ).toHaveFocus();
  await openSample();
  expect(screen.getByRole("tab", { name: /Saved punches/ })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  expect(
    Number(
      screen
        .getByRole("slider", { name: "Video position" })
        .getAttribute("value"),
    ),
  ).toBeCloseTo(sample.frameTimes![1], 3);
  expect(screen.getByRole("button", { name: "Classify punch" })).toBeEnabled();
  history.back();
  await waitFor(() =>
    expect(screen.getByRole("heading", { name: "Videos" })).toBeInTheDocument(),
  );
  history.forward();
  await screen.findByRole("region", { name: "Analyze workspace" });
  loadedFrame();
  expect(screen.getByRole("button", { name: "Classify punch" })).toBeEnabled();
});
it("restores a source-labeled saved prediction after refresh without promoting example output", async () => {
  const first = render(<App />);
  await openSample();
  fireEvent.click(screen.getByRole("tab", { name: /Saved punches/ }));
  fireEvent.click(screen.getByRole("button", { name: /lead hook/i }));
  first.unmount();
  render(<App />);
  await screen.findByText("Saved model output");
  expect(screen.getAllByRole("meter")).toHaveLength(6);
  expect(location.pathname).toBe("/videos/reference-sample");
});
it("deduplicates repeat classification and preserves its recorded provenance", async () => {
  render(<App />);
  await openSample();
  fireEvent.click(screen.getByRole("tab", { name: /Saved punches/ }));
  fireEvent.click(screen.getByRole("button", { name: /lead hook/i }));
  loadedFrame();
  const classify = screen.getByRole("button", { name: "Classify punch" });
  classify.focus();
  fireEvent.click(classify);
  await screen.findByText("Saved model output");
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Classify punch" }),
    ).toHaveFocus(),
  );
  expect(
    screen.getByRole("tab", { name: "Saved punches (2)" }),
  ).toBeInTheDocument();
});
it("cancels a pending example when navigating so it cannot attach to another video", async () => {
  render(<App />);
  await openSample();
  fireEvent.click(screen.getByRole("button", { name: "Set punch start (I)" }));
  fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
  loadedFrame();
  fireEvent.click(screen.getByRole("button", { name: "Set punch end (O)" }));
  fireEvent.click(screen.getByRole("button", { name: "Classify punch" }));
  fireEvent.click(screen.getByRole("button", { name: "Back to Videos" }));
  await openSample();
  await new Promise((resolve) => setTimeout(resolve, 500));
  expect(screen.queryByRole("meter")).not.toBeInTheDocument();
  expect(
    screen.getByRole("tab", { name: "Saved punches (2)" }),
  ).toBeInTheDocument();
});
it("recovers a missing direct video URL without falling back to another video's results", async () => {
  history.replaceState(null, "", "/videos/missing-video");
  render(<App />);
  await screen.findByRole("heading", { name: "Video unavailable" });
  expect(
    screen.queryByRole("region", { name: "Analyze workspace" }),
  ).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Back to Videos" }));
  expect(location.pathname).toBe("/videos");
});

it("waits for the requested exact frame before enabling boundary controls", async () => {
  render(<App />);
  await openSample();
  fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
  expect(
    screen.getByRole("button", { name: "Set punch start (I)" }),
  ).toBeDisabled();
  loadedFrame();
  expect(
    screen.getByRole("button", { name: "Set punch start (I)" }),
  ).toBeEnabled();
});
it("persists a new example across refresh and reuses its saved row on repeat classification", async () => {
  const first = render(<App />);
  await openSample();
  fireEvent.click(screen.getByRole("button", { name: "Set punch start (I)" }));
  fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
  loadedFrame();
  fireEvent.click(screen.getByRole("button", { name: "Set punch end (O)" }));
  fireEvent.click(screen.getByRole("button", { name: "Classify punch" }));
  await screen.findByText("Prototype example");
  first.unmount();
  render(<App />);
  await screen.findByText("Prototype example");
  loadedFrame();
  expect(
    screen.getByText("No model ran for this selection."),
  ).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Classify punch" }));
  await screen.findByText("Prototype example");
  expect(
    screen.getByRole("tab", { name: "Saved punches (3)" }),
  ).toBeInTheDocument();
});
it("isolates each video's draft, playhead and output", async () => {
  vi.mocked(recentVideos).mockResolvedValue([
    {
      ...sample,
      id: "local-one",
      title: "Local one",
      kind: "local",
      frameTimes: undefined,
    },
  ]);
  render(<App />);
  await openSample();
  fireEvent.click(screen.getByRole("tab", { name: /Saved punches/ }));
  fireEvent.click(screen.getByRole("button", { name: /lead hook/i }));
  loadedFrame();
  fireEvent.click(screen.getByRole("button", { name: "Back to Videos" }));
  fireEvent.click(screen.getByRole("button", { name: "Open Local one" }));
  expect(screen.queryByRole("meter")).not.toBeInTheDocument();
  expect(
    screen.getByRole("tab", { name: "Saved punches (0)" }),
  ).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Set punch start (I)" }));
  fireEvent.change(screen.getByRole("slider", { name: "Video position" }), {
    target: { value: "0.5" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Set punch end (O)" }));
  fireEvent.click(screen.getByRole("button", { name: "Back to Videos" }));
  await openSample();
  expect(screen.getByText("Saved model output")).toBeInTheDocument();
  expect(
    Number(
      screen
        .getByRole("slider", { name: "Video position" })
        .getAttribute("value"),
    ),
  ).toBeCloseTo(sample.frameTimes![26], 3);
});

it("supports arrow-key inspector tabs without moving the playhead", async () => {
  render(<App />);
  await openSample();
  const tab = screen.getByRole("tab", { name: "Selection" });
  tab.focus();
  fireEvent.keyDown(tab, { key: "ArrowRight" });
  const saved = screen.getByRole("tab", { name: /Saved punches/ });
  expect(saved).toHaveAttribute("aria-selected", "true");
  expect(saved).toHaveFocus();
  expect(screen.getByRole("slider", { name: "Video position" })).toHaveValue(
    "0",
  );
  fireEvent.keyDown(saved, { key: "Home" });
  expect(tab).toHaveAttribute("aria-selected", "true");
  expect(tab).toHaveFocus();
});
it("restores library scroll and the originating tile when returning from a video", async () => {
  vi.mocked(recentVideos).mockResolvedValue(
    Array.from({ length: 20 }, (_, i) => ({
      ...sample,
      id: `local-${i}`,
      title: `Video ${i + 1}`,
      kind: "local",
      frameTimes: undefined,
    })),
  );
  render(<App />);
  const tile = await screen.findByRole("button", { name: "Open Video 20" });
  const library = tile.closest("section")!;
  library.scrollTop = 460;
  fireEvent.scroll(library);
  fireEvent.click(tile);
  fireEvent.click(screen.getByRole("button", { name: "Back to Videos" }));
  const returned = screen.getByRole("button", { name: "Open Video 20" });
  expect(returned).toHaveFocus();
  expect(returned.closest("section")!.scrollTop).toBe(460);
});
