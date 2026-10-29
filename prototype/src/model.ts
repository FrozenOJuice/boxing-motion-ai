import reference from "./demo-data.json";

export type Bounds = { start: number; end: number };
export type VideoItem = {
  id: string;
  title: string;
  src: string;
  poster: string;
  duration: number;
  width: number;
  height: number;
  kind: "sample" | "local";
  frameTimes?: number[];
  openedAt: number;
};
export type Selection = {
  id: string;
  videoId: string;
  bounds: Bounds;
  kind: "recorded" | "preview";
  prediction: string;
  scores: Record<string, number>;
};
export const sample: VideoItem = {
  id: "reference-sample",
  title: "Shadowboxing sample",
  src: "/demo/shadowboxing.mp4",
  poster: "/demo/poster.png",
  duration: reference.duration,
  width: reference.width,
  height: reference.height,
  kind: "sample",
  frameTimes: reference.frameTimes,
  openedAt: 0,
};
export const recordedSelections: Selection[] = reference.results.map(
  (r, i) => ({
    id: `sample-${i + 1}`,
    videoId: sample.id,
    bounds: {
      start: reference.frameTimes[r.start_frame],
      end: reference.frameTimes[r.stop_frame_exclusive] ?? reference.duration,
    },
    kind: "recorded",
    prediction: r.prediction,
    scores: r.scores,
  }),
);
export const checkpoint = reference.results[0].checkpoint_sha256;
export const punchName = (name: string) => name.replaceAll("_", " ");
export const clamp = (n: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, n));
export function clock(time: number, precise = false) {
  const n = Math.max(0, Number.isFinite(time) ? time : 0);
  const seconds = Math.floor(n % 60)
    .toString()
    .padStart(2, "0");
  return `${Math.floor(n / 60)
    .toString()
    .padStart(2, "0")}:${seconds}${
    precise
      ? `.${Math.floor((n % 1) * 1000 + 0.0001)
          .toString()
          .padStart(3, "0")}`
      : ""
  }`;
}
export function frameIndex(times: number[], time: number) {
  let a = 0,
    b = times.length;
  while (a < b) {
    const m = (a + b) >>> 1;
    if (times[m] <= time + 1e-7) a = m + 1;
    else b = m;
  }
  return Math.max(0, a - 1);
}
export function frameTime(video: VideoItem, time: number) {
  return video.frameTimes
    ? video.frameTimes[frameIndex(video.frameTimes, time)]
    : clamp(time, 0, video.duration);
}
export function nextBoundary(video: VideoItem, time: number) {
  return video.frameTimes
    ? (video.frameTimes[frameIndex(video.frameTimes, time) + 1] ??
        video.duration)
    : Math.min(video.duration, time + 0.001);
}
export function snapBoundary(video: VideoItem, time: number) {
  const value = clamp(time, 0, video.duration);
  if (!video.frameTimes) return value;
  const before = frameTime(video, value),
    after = nextBoundary(video, value);
  return value - before < after - value ? before : after;
}
export function previewSelection(videoId: string, bounds: Bounds): Selection {
  return {
    id: crypto.randomUUID(),
    videoId,
    bounds,
    kind: "preview",
    prediction: recordedSelections[0].prediction,
    scores: recordedSelections[0].scores,
  };
}
