import { expect, it } from "vitest";
import { checkSelection } from "./selection";
import { sample } from "./model";
it("does not infer a missing start or end", () => {
  expect(checkSelection(sample, { start: 0, end: null }).state).toBe(
    "incomplete",
  );
  expect(checkSelection(sample, { start: null, end: 0.5 }).state).toBe(
    "incomplete",
  );
});
it("enforces source boundaries, minimum frame count and the existing three-second elapsed-frame gate", () => {
  const video = { ...sample, duration: 3.5, frameTimes: [0, 0.1, 3, 3.1, 3.4] };
  expect(checkSelection(video, { start: 0, end: 0.1 }).state).toBe("invalid");
  expect(checkSelection(video, { start: 0.1, end: 0 }).state).toBe("invalid");
  expect(checkSelection(video, { start: 0, end: 0.15 }).state).toBe("invalid");
  expect(checkSelection(video, { start: 0, end: 3.1 }).state).toBe("valid");
  expect(checkSelection(video, { start: 0, end: 3.4 }).reason).toMatch(
    /at most 3 seconds/,
  );
  expect(checkSelection(video, { start: 3.1, end: 3.5 }).state).toBe("valid");
});
it("does not pretend to validate unknown source-frame counts for local uploads", () => {
  const video = { ...sample, kind: "local" as const, frameTimes: undefined };
  expect(checkSelection(video, { start: 0, end: null }).state).toBe(
    "incomplete",
  );
  expect(checkSelection(video, { start: 0, end: 0.5 }).state).toBe("valid");
  expect(checkSelection(video, { start: 0, end: 3.1 }).state).toBe("invalid");
});
