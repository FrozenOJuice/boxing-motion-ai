import { act, renderHook } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { usePlayer } from "./Player";
import { sample } from "./model";

it("keeps late compositor callbacks inside the exclusive replay boundary", async () => {
  const video = { ...sample, frameTimes: [0, 0.04, 0.1, 0.14], duration: 0.18 };
  const { result } = renderHook(() => usePlayer(video));
  const element = document.createElement("video");
  const pause = vi.spyOn(element, "pause").mockImplementation(() => {});
  const play = vi.spyOn(element, "play").mockResolvedValue();
  result.current.media.current = element;
  await act(async () => {
    await result.current.play({ start: 0.04, end: 0.14 });
  });
  act(() => result.current.update(0.14));
  expect(result.current.current).toBe(0.1);
  act(() => result.current.update(0.14));
  expect(result.current.current).toBe(0.1);
  act(() => result.current.seek(0.14));
  expect(result.current.current).toBe(0.14);
  pause.mockRestore();
  play.mockRestore();
});

it("restarts after natural completion even though the final still seek clears native ended", async () => {
  const { result } = renderHook(() => usePlayer(sample));
  const element = document.createElement("video");
  vi.spyOn(element, "pause").mockImplementation(() => {});
  vi.spyOn(element, "play").mockResolvedValue();
  result.current.media.current = element;
  act(() => result.current.finish());
  expect(result.current.current).toBe(sample.frameTimes!.at(-1));
  expect(element.ended).toBe(false);
  await act(async () => result.current.play());
  expect(element.currentTime).toBe(0);
  expect(result.current.current).toBe(0);
  expect(result.current.playing).toBe(true);
});
it("replays the selected range after completion and steps/seeks from the displayed time", async () => {
  const { result } = renderHook(() => usePlayer(sample));
  const element = document.createElement("video");
  vi.spyOn(element, "pause").mockImplementation(() => {});
  vi.spyOn(element, "play").mockResolvedValue();
  result.current.media.current = element;
  const range = { start: sample.frameTimes![26], end: sample.frameTimes![35] };
  await act(async () => result.current.play(range));
  act(() => result.current.finish());
  expect(result.current.current).toBe(sample.frameTimes![34]);
  await act(async () => result.current.play(range));
  expect(result.current.current).toBe(range.start);
  act(() => result.current.pause());
  act(() => result.current.step(1));
  expect(result.current.current).toBe(sample.frameTimes![27]);
  act(() => result.current.seek(sample.frameTimes![30]));
  act(() => result.current.step(-1));
  expect(result.current.current).toBe(sample.frameTimes![29]);
});
it("ignores a late play promise after an explicit pause", async () => {
  const { result } = renderHook(() => usePlayer(sample));
  const element = document.createElement("video");
  vi.spyOn(element, "pause").mockImplementation(() => {});
  let resolve!: () => void;
  vi.spyOn(element, "play").mockImplementation(
    () =>
      new Promise((r) => {
        resolve = r;
      }),
  );
  result.current.media.current = element;
  let pending!: Promise<void>;
  act(() => {
    pending = result.current.play();
  });
  act(() => result.current.pause());
  await act(async () => {
    resolve();
    await pending;
  });
  expect(result.current.playing).toBe(false);
});
