import { describe, expect, it } from "vitest";
import {
  clock,
  frameIndex,
  frameTime,
  nextBoundary,
  previewSelection,
  recordedSelections,
  sample,
  snapBoundary,
} from "./model";
import { loadPreviews } from "./storage";

describe("source-time selection", () => {
  const video = {
    ...sample,
    frameTimes: [0, 0.04, 0.1, 0.14, 0.2],
    duration: 0.24,
  };
  it("uses the supplied variable timestamps rather than a nominal FPS", () => {
    expect(frameIndex(video.frameTimes, 0.099)).toBe(1);
    expect(frameTime(video, 0.099)).toBe(0.04);
    expect(nextBoundary(video, 0.1)).toBe(0.14);
    expect(nextBoundary(video, 0.2)).toBe(0.24);
    expect(snapBoundary(video, 0.115)).toBe(0.1);
    expect(snapBoundary(video, 0.135)).toBe(0.14);
  });
  it("formats times without inventing duration or rounding into the next frame", () => {
    expect(clock(1.9999, true)).toBe("00:01.999");
    expect(clock(60.1, true)).toBe("01:00.100");
    expect(clock(Number.NaN)).toBe("00:00");
  });
});

it("separates recorded model outputs from every newly simulated result", () => {
  expect(recordedSelections).toHaveLength(2);
  expect(recordedSelections.every((s) => s.kind === "recorded")).toBe(true);
  const preview = previewSelection("a-new-upload", { start: 0, end: 0.5 });
  expect(preview.kind).toBe("preview");
  expect(preview.videoId).toBe("a-new-upload");
  expect(preview.scores).toEqual(recordedSelections[0].scores);
  expect(recordedSelections[0].scores.lead_hook).toBe(0.7781179547309875);
});

it("restores saved prototype selections without upgrading them to real model results", () => {
  localStorage.setItem(
    "boxing-motion-design-prototype:selections:v1",
    JSON.stringify([
      {
        id: "changed",
        videoId: "upload",
        kind: "recorded",
        prediction: "made_up",
        scores: { made_up: 1 },
        bounds: { start: 0, end: 0.5 },
      },
    ]),
  );
  const [value] = loadPreviews();
  expect(value.kind).toBe("preview");
  expect(value.prediction).toBe(recordedSelections[0].prediction);
  expect(value.scores).toEqual(recordedSelections[0].scores);
  localStorage.clear();
});
