import { expect, it } from "vitest";
import {
  type Playback,
  frameAt,
  ContinuousViewing,
  viewingProgress,
} from "./playback";
const clip = {
  frames: [20, 21, 22, 23],
  media_times: [0, 0.033, 0.067, 0.1],
  duration: 0.134,
} as Playback;
it("maps VFR PTS without assuming FPS", () => {
  expect(frameAt(clip, 0.067, true)).toBe(2);
  expect(frameAt(clip, 0.05, true)).toBeNull();
  expect(frameAt(clip, 0.05)).toBe(1);
  expect(frameAt(clip, 0.134)).toBeNull();
});
const point = (
  media: number,
  wall: number,
  rate: 1 | 0.5 | 0.25 = 1,
  eligible = true,
) => ({ media, wall, rate, eligible });
it("recognizes complete visible real-time viewing without inventing inspected frames", () => {
  const t = new ContinuousViewing();
  t.sample(point(0, 0));
  t.sample(point(0.1, 0.1));
  t.sample(point(0.134, 0.134));
  const status = viewingProgress(clip, 20, 24, t.snapshot(), new Set());
  expect(status.complete).toBe(true);
  expect(status.playbackComplete).toBe(true);
  expect(t.snapshot()[0].samples).toBe(3);
});
it("retains gaps from seeks, hidden time, clock jumps and long scheduling gaps", () => {
  for (const interruption of ["seek", "hidden", "jump", "long"]) {
    const t = new ContinuousViewing();
    t.sample(point(0, 0));
    t.sample(point(0.033, 0.033));
    if (interruption === "seek") t.break();
    if (interruption === "hidden") t.sample(point(0.067, 0.067, 1, false));
    if (interruption === "jump") t.sample(point(0.1, 0.034));
    if (interruption === "long") t.sample(point(0.1, 2));
    t.sample(point(0.1, 3));
    t.sample(point(0.134, 3.034));
    expect(
      viewingProgress(clip, 20, 24, t.snapshot(), new Set()).complete,
    ).toBe(false);
  }
});
it("combines actual pause/resume and rate changes, while replay can genuinely fill a gap", () => {
  const t = new ContinuousViewing();
  t.sample(point(0, 0));
  t.sample(point(0.033, 0.033));
  t.break();
  t.sample(point(0.033, 5, 0.5));
  t.sample(point(0.067, 5.068, 0.5));
  t.break();
  t.sample(point(0.067, 6, 0.25));
  t.sample(point(0.134, 6.268, 0.25));
  expect(
    viewingProgress(clip, 20, 24, t.snapshot(), new Set()).playbackComplete,
  ).toBe(true);
  const replay = new ContinuousViewing();
  replay.sample(point(0.1, 0));
  replay.sample(point(0.134, 0.034));
  expect(
    viewingProgress(clip, 20, 24, replay.snapshot(), new Set()).complete,
  ).toBe(false);
  replay.break();
  replay.sample(point(0, 1));
  replay.sample(point(0.134, 1.134));
  expect(
    viewingProgress(clip, 20, 24, replay.snapshot(), new Set()).complete,
  ).toBe(true);
});
it("keeps partial viewing distinct from exact inspection and does not fill a missing frame", () => {
  const t = new ContinuousViewing();
  t.sample(point(0.033, 0));
  t.sample(point(0.134, 0.101));
  expect(viewingProgress(clip, 20, 24, t.snapshot(), new Set()).complete).toBe(
    false,
  );
  const mixed = viewingProgress(clip, 20, 24, t.snapshot(), new Set([20]));
  expect(mixed.complete).toBe(true);
  expect(mixed.playbackComplete).toBe(false);
  expect(
    viewingProgress(clip, 20, 24, [], new Set([20, 22, 23])).complete,
  ).toBe(false);
});
