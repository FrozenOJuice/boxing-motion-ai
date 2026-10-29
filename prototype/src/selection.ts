import { frameIndex, type Bounds, type VideoItem } from "./model";

// End is the exclusive boundary after the last included source frame.
export type Draft = { start: number | null; end: number | null };
export const emptyDraft: Draft = { start: null, end: null };
export type CheckedSelection =
  | { state: "empty" | "incomplete" | "invalid"; reason: string; bounds: null }
  | { state: "valid"; reason: ""; bounds: Bounds };
export function checkSelection(
  video: VideoItem,
  draft: Draft,
): CheckedSelection {
  const { start, end } = draft;
  if (start === null && end === null)
    return { state: "empty", reason: "Select one punch", bounds: null };
  if (start === null || end === null)
    return {
      state: "incomplete",
      reason:
        start === null
          ? "Set the start of the punch."
          : "Set the end of the punch.",
      bounds: null,
    };
  const invalid = (reason: string): CheckedSelection => ({
    state: "invalid",
    reason,
    bounds: null,
  });
  if (
    !Number.isFinite(start) ||
    !Number.isFinite(end) ||
    start < 0 ||
    end > video.duration ||
    end <= start
  )
    return invalid("The start must precede the end, within this video.");
  if (video.frameTimes) {
    const times = [...video.frameTimes, video.duration];
    const first = times.findIndex((t) => Math.abs(t - start) < 1e-6);
    const stop = times.findIndex((t) => Math.abs(t - end) < 1e-6);
    if (first < 0 || stop < 0)
      return invalid("Choose boundaries on source frames.");
    if (stop - first < 2)
      return invalid("Select at least two source frames of one punch.");
    // Match the existing Python gate: elapsed time between first and last included frames.
    if (video.frameTimes[stop - 1] - video.frameTimes[first] > 3)
      return invalid("Select one punch spanning at most 3 seconds.");
  } else if (end - start > 3) {
    return invalid("Select one punch spanning at most 3 seconds.");
  }
  return { state: "valid", reason: "", bounds: { start, end } };
}
export const sameBounds = (a: Bounds, b: Bounds) =>
  Math.abs(a.start - b.start) < 1e-6 && Math.abs(a.end - b.end) < 1e-6;
export function lastIncluded(video: VideoItem, end: number) {
  return video.frameTimes
    ? video.frameTimes[frameIndex(video.frameTimes, end - 1e-6)]
    : Math.max(0, end - 0.001);
}
