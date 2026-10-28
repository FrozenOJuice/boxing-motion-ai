import { expect, it } from "vitest";
import {
  type IntervalAnnotation,
  annotationLabel,
  annotationSegments,
} from "./intervalAnnotations";
const source = (row: number, start: number, stop: number, label = "Jab") =>
  ({
    source_row: row,
    source_class: label,
    punch_type_provisional: "jab",
    raw_start: start,
    raw_end: stop - 1,
    source_interval_half_open: [start, stop],
  }) as IntervalAnnotation;
it("clips markers to half-open review bounds and flags partial annotations without changing sources", () => {
  const rows = [
    source(1, 8, 12),
    source(2, 13, 15, "Cross"),
    source(3, 18, 23, "Rear Hook"),
  ];
  const before = JSON.stringify(rows);
  const mapped = annotationSegments(rows, 10, 20);
  expect(mapped.map((s) => [s.start, s.stop, s.partial])).toEqual([
    [10, 12, true],
    [13, 15, false],
    [18, 20, true],
  ]);
  expect(mapped[0].originalStart).toBe(8);
  expect(mapped[2].originalStop).toBe(23);
  expect(JSON.stringify(rows)).toBe(before);
});
it("does not guess native bounds from raw columns or include touching intervals", () => {
  const missing = {
    ...source(1, 11, 13),
    source_interval_half_open: undefined,
  };
  const reversed = source(2, 15, 12);
  expect(
    annotationSegments(
      [missing, reversed, source(3, 8, 10), source(4, 20, 25)],
      10,
      20,
    ),
  ).toEqual([]);
});
it("keeps overlapping and excluded source rows distinct and reachable in separate lanes", () => {
  const rows = [
    source(4, 10, 14),
    source(5, 11, 15),
    { ...source(6, 10, 14), exclusion_reasons: ["suspect_overlap"] },
    source(7, 15, 16),
  ];
  const mapped = annotationSegments(rows, 10, 20);
  expect(mapped).toHaveLength(4);
  expect(new Set(mapped.slice(0, 3).map((s) => s.lane)).size).toBe(3);
  expect(mapped.at(-1)?.lane).toBe(0);
});
it("preserves literal source label wording with an honest unknown fallback", () => {
  expect(annotationLabel(source(1, 10, 11, "Rear Uppercut"))).toBe(
    "Rear Uppercut",
  );
  expect(annotationLabel({ source_row: 1 } as IntervalAnnotation)).toBe(
    "Unknown",
  );
});
