/** Use the supplied provisional native mapping; never infer bounds from raw spreadsheet cells. */
import { type Source, title } from "./api";
export type IntervalAnnotation = Source & {
  source_interval_half_open?: [number, number] | null;
  schema_version?: string;
  array_slice_half_open?: [number, number] | null;
};
export type AnnotationSegment = {
  source: IntervalAnnotation;
  label: string;
  start: number;
  stop: number;
  originalStart: number;
  originalStop: number;
  partial: boolean;
  lane: number;
};
export function annotationLabel(source: IntervalAnnotation): string {
  return (
    source.source_class?.trim() ||
    title(source.punch_type_provisional || "Unknown")
  );
}
export function annotationSegments(
  annotations: IntervalAnnotation[],
  start: number,
  stop: number,
): AnnotationSegment[] {
  const mapped: AnnotationSegment[] = [];
  for (const source of annotations) {
    // The original V5 schema predates source_interval_half_open. Its prepared
    // native array slice is the existing provisional mapping, not a new offset.
    const bounds =
      source.source_interval_half_open === undefined &&
      source.schema_version === "v5-candidate-1.0" &&
      source.video_id === "V5"
        ? source.array_slice_half_open
        : source.source_interval_half_open;
    if (
      !bounds ||
      bounds.length !== 2 ||
      !bounds.every(Number.isInteger) ||
      bounds[0] < 0 ||
      bounds[1] <= bounds[0]
    )
      continue;
    if (bounds[0] >= stop || bounds[1] <= start) continue;
    mapped.push({
      source,
      label: annotationLabel(source),
      start: Math.max(start, bounds[0]),
      stop: Math.min(stop, bounds[1]),
      originalStart: bounds[0],
      originalStop: bounds[1],
      partial: bounds[0] < start || bounds[1] > stop,
      lane: 0,
    });
  }
  mapped.sort(
    (a, b) => a.start - b.start || a.source.source_row - b.source.source_row,
  );
  const ends: number[] = [];
  for (const segment of mapped) {
    let lane = ends.findIndex((end) => end <= segment.start);
    if (lane === -1) lane = ends.length;
    segment.lane = lane;
    ends[lane] = segment.stop;
  }
  return mapped;
}
