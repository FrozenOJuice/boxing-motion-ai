import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
vi.mock("./storage", () => ({
  recentVideos: vi.fn().mockResolvedValue([]),
  importVideo: vi.fn(),
  loadPreviews: () => [],
  savePreviews: vi.fn(),
}));
beforeEach(() => {
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

it("opens Analyze by default and keeps Videos and Development separate", async () => {
  render(<App />);
  await screen.findByRole("region", { name: "Analyze workspace" });
  expect(
    screen.queryByRole("button", { name: /^Results/ }),
  ).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Classify punch" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Videos" }));
  expect(
    screen.getByRole("heading", { name: "Recent videos" }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Upload video" }),
  ).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Development" }));
  expect(screen.getByRole("dialog")).toHaveTextContent(
    "does not open or modify",
  );
});
it("opens saved predictions beside the video and clears stale output when bounds change", async () => {
  render(<App />);
  await screen.findByRole("region", { name: "Analyze workspace" });
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
  await screen.findByRole("region", { name: "Analyze workspace" });
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
  expect(screen.getByRole("button", { name: "Classify punch" })).toBeDisabled();
});
it("restores keyboard focus after a dialog closes", () => {
  render(<App />);
  const trigger = screen.getByRole("button", { name: "Design prototype" });
  trigger.focus();
  fireEvent.click(trigger);
  fireEvent.click(screen.getByRole("button", { name: "Close dialog" }));
  expect(trigger).toHaveFocus();
});
