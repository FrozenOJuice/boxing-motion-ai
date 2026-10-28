import { expect, it } from "vitest";
import { type Playback, frameAt, creditPresentation } from "./playback";
const clip = {
  frames: [20, 21, 22, 23],
  media_times: [0, 0.033, 0.067, 0.1],
  duration: 0.134,
} as Playback;
it("maps actual VFR media PTS rather than an assumed FPS", () => {
  expect(frameAt(clip, 0.067, true)).toBe(2);
  expect(frameAt(clip, 0.05, true)).toBeNull();
  expect(frameAt(clip, 0.05)).toBe(1);
  expect(frameAt(clip, 0.134, true)).toBeNull();
  expect(frameAt(clip, -1)).toBeNull();
});
it("credits only consecutive individually presented frames", () => {
  const a = { index: 20, mediaTime: 0, frames: 1, now: 100 };
  const b = { index: 21, mediaTime: 0.033, frames: 2, now: 133 };
  expect(creditPresentation(a, b, true)).toBe(true);
  expect(creditPresentation(null, b, true)).toBe(false);
  expect(creditPresentation(a, b, false)).toBe(false);
  expect(creditPresentation(a, { ...b, index: 22 }, true)).toBe(false);
  expect(creditPresentation(a, { ...b, frames: 3 }, true)).toBe(false);
  expect(creditPresentation(a, { ...b, mediaTime: 0 }, true)).toBe(false);
  expect(creditPresentation(a, { ...b, now: 2000 }, true)).toBe(false);
});
