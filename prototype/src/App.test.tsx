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

it("starts with Upload and keeps Development outside the boxer workspace", async () => {
  render(<App />);
  expect(
    screen.getByRole("heading", { name: "Your videos" }),
  ).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Review" }),
  ).not.toBeInTheDocument();
  expect(
    screen.getByText("Local media. No live inference."),
  ).toBeInTheDocument();
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Open sample" })).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: "Development tools" }));
  expect(screen.getByRole("dialog")).toHaveTextContent(
    "does not open or modify",
  );
});

it("shows clearly sourced, saved model predictions and opt-in model information", async () => {
  render(<App />);
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Open sample" })).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: "Open sample" }));
  fireEvent.click(
    screen.getByRole("button", { name: "Results, 2 saved selections" }),
  );
  expect(screen.getByText("SAVED MODEL PREDICTION")).toBeInTheDocument();
  expect(
    screen.getByText("Recorded output from the sample video."),
  ).toBeInTheDocument();
  expect(screen.getAllByRole("meter")).toHaveLength(6);
  expect(
    screen.getByText("Scores are not probabilities of correctness."),
  ).toBeInTheDocument();
  const details = screen.getByRole("button", { name: "Model information" });
  expect(details).toHaveAttribute("aria-expanded", "false");
  fireEvent.click(details);
  expect(
    screen.getByText(/New-boxer performance unvalidated/),
  ).toBeInTheDocument();
});

it("labels every new classification as an example, without calling a model API", async () => {
  render(<App />);
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Open sample" })).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: "Open sample" }));
  expect(screen.getByRole("button", { name: "Classify punch" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Set punch start (I)" }));
  fireEvent.click(screen.getByRole("button", { name: "Classify punch" }));
  await screen.findByText("EXAMPLE RESULT");
  expect(
    screen.getByText(/This selection was not classified/),
  ).toBeInTheDocument();
  expect(vi.mocked(fetch).mock.calls.map(([url]) => url)).toEqual([
    "/demo/manifest.json",
  ]);
});

it("restores keyboard focus to the dialog trigger after dismissal", async () => {
  render(<App />);
  const trigger = screen.getByRole("button", { name: "Design prototype" });
  trigger.focus();
  fireEvent.click(trigger);
  expect(screen.getByRole("dialog")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Close dialog" }));
  expect(trigger).toHaveFocus();
});
