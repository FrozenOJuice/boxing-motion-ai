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
