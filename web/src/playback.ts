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

export type ViewingSpan = {
  start: number;
  end: number;
  rate: 1 | 0.5 | 0.25;
  wall_seconds: number;
  samples: number;
};
export type PlaybackCoverage = {
  schema_version: "continuous-viewing-1.0";
  policy: "visible-realtime-1.0";
  cache_key: string;
  interval_id: string;
  segments: ViewingSpan[];
};
type Sample = {
  media: number;
  wall: number;
  rate: 1 | 0.5 | 0.25;
  eligible: boolean;
};
export class ContinuousViewing {
  private last: Sample | null = null;
  private join = false;
  private spans: ViewingSpan[] = [];
  break() {
    this.last = null;
    this.join = false;
  }
  sample(point: Sample): boolean {
    if (!point.eligible) {
      this.break();
      return false;
    }
    const previous = this.last;
    this.last = point;
    if (!previous) {
      this.join = false;
      return false;
    }
    const media = point.media - previous.media,
      wall = point.wall - previous.wall;
    if (
      point.rate !== previous.rate ||
      wall <= 0 ||
      wall > 1 ||
      media < 0 ||
      media > wall * point.rate + 0.015
    ) {
      this.join = false;
      return false;
    }
    if (media === 0) {
      this.last = previous;
      return false;
    }
    const tail = this.spans.at(-1);
    if (
      this.join &&
      tail &&
      tail.rate === point.rate &&
      Math.abs(tail.end - previous.media) < 0.0001
    ) {
      tail.end = point.media;
      tail.wall_seconds += wall;
      tail.samples++;
    } else {
      this.spans.push({
        start: previous.media,
        end: point.media,
        rate: point.rate,
        wall_seconds: wall,
        samples: 2,
      });
    }
    this.join = true;
    return true;
  }
  snapshot() {
    return this.spans.map((s) => ({ ...s }));
  }
}
export function viewingProgress(
  clip: Playback,
  startFrame: number,
  stopFrame: number,
  spans: ViewingSpan[],
  inspected: Set<number>,
) {
  const first = clip.frames.indexOf(startFrame),
    stopIndex = stopFrame - clip.frames[0];
  const start = clip.media_times[first],
    stop = clip.media_times[stopIndex] ?? clip.duration;
  const exact = [...inspected]
    .filter((n) => n >= startFrame && n < stopFrame)
    .map((n) => {
      const i = clip.frames.indexOf(n);
      return [
        clip.media_times[i],
        clip.media_times[i + 1] ?? clip.duration,
      ] as [number, number];
    });
  const ranges: [number, number][] = spans.map((s) => [s.start, s.end]);
  function duration(input: [number, number][]) {
    let end = start,
      total = 0;
    for (const [a, b] of [...input].sort((x, y) => x[0] - y[0])) {
      const left = Math.max(start, a, end),
        right = Math.min(stop, b);
      if (right > left) {
        total += right - left;
        end = right;
      }
    }
    return total;
  }
  const viewed = duration(ranges.concat(exact)),
    required = stop - start;
  return {
    complete: required > 0 && viewed >= required - 0.0001,
    playbackComplete: duration(ranges) >= required - 0.0001,
    percent: Math.min(100, Math.floor((100 * viewed) / required)),
  };
}
export type ViewingUpdate = ReturnType<typeof viewingProgress> & {
  evidence: PlaybackCoverage;
};
