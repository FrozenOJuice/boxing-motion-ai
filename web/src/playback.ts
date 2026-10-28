/** Preview PTS mapping and conservative presentation evidence. Never interpolate coverage. */
export type Playback = {
  interval_id: string;
  cache_key: string;
  video_url: string;
  frames: number[];
  media_times: number[];
  source_times: string[];
  source_start: string;
  duration: number;
  width: number;
  height: number;
};
export type Presented = {
  index: number;
  mediaTime: number;
  frames: number;
  now: number;
};
export function frameAt(
  clip: Playback,
  time: number,
  exact = false,
): number | null {
  if (!Number.isFinite(time) || time < 0 || time >= clip.duration) return null;
  let lo = 0,
    hi = clip.media_times.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (clip.media_times[mid] <= time + 0.00001) lo = mid + 1;
    else hi = mid;
  }
  const slot = lo - 1;
  if (slot < 0 || (exact && Math.abs(clip.media_times[slot] - time) > 0.00001))
    return null;
  return slot;
}
export function creditPresentation(
  previous: Presented | null,
  current: Presented,
  eligible: boolean,
): boolean {
  // Consecutive callbacks AND native frame IDs; a timeupdate range or seek never fills a gap.
  return (
    eligible &&
    previous !== null &&
    current.frames === previous.frames + 1 &&
    current.index === previous.index + 1 &&
    current.mediaTime > previous.mediaTime &&
    current.now > previous.now &&
    current.now - previous.now <= 1000
  );
}
