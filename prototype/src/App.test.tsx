import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { recentVideos } from "./storage";
import { sample } from "./model";
vi.mock("./storage", () => ({
  recentVideos: vi.fn().mockResolvedValue([]),
  importVideo: vi.fn(),
  loadPreviews: () => [],
  savePreviews: vi.fn(),
}));
beforeEach(() => {
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
  fireEvent.click(screen.getByRole("button", { name: /lead hook/i }));
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
  fireEvent.click(screen.getByRole("button", { name: "Classify punch" }));
  expect(screen.getByRole("button", { name: "Loading…" })).toBeDisabled();
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
