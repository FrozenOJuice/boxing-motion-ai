import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import Timeline from "./Timeline";
import { sample } from "./model";

const video = {
  ...sample,
  frameTimes: [0, 0.04, 0.1, 0.14, 0.2],
  duration: 0.24,
};
it("keeps both keyboard handles on real boundaries and prevents them crossing", () => {
  const seek = vi.fn();
  function Harness() {
    const [bounds, setBounds] = useState({ start: 0.04, end: 0.14 });
    return (
      <Timeline
        video={video}
        current={0.04}
        bounds={bounds}
        onBounds={setBounds}
        onSeek={seek}
      />
    );
  }
  render(<Harness />);
  const start = screen.getByRole("slider", { name: "Punch start" });
  const end = screen.getByRole("slider", { name: "Punch end" });
  fireEvent.keyDown(start, { key: "ArrowRight" });
  expect(start).toHaveAttribute("aria-valuenow", "0.1");
  fireEvent.keyDown(start, { key: "ArrowRight" });
  expect(start).toHaveAttribute("aria-valuenow", "0.1");
  expect(seek).toHaveBeenLastCalledWith(0.1);
  fireEvent.keyDown(end, { key: "ArrowLeft" });
  expect(end).toHaveAttribute("aria-valuenow", "0.14");
  fireEvent.keyDown(end, { key: "End" });
  expect(end).toHaveAttribute("aria-valuenow", "0.24");
  expect(
    screen.getByRole("slider", { name: "Video position" }),
  ).toHaveAttribute("max", "0.24");
});

it("seeks adjacent variable-time frames from the single playhead", () => {
  const seek = vi.fn();
  render(
    <Timeline
      video={video}
      current={0.04}
      bounds={null}
      onBounds={() => {}}
      onSeek={seek}
    />,
  );
  const playhead = screen.getByRole("slider", { name: "Video position" });
  fireEvent.keyDown(playhead, { key: "ArrowRight" });
  expect(seek).toHaveBeenLastCalledWith(0.1);
  fireEvent.keyDown(playhead, { key: "ArrowLeft" });
  expect(seek).toHaveBeenLastCalledWith(0);
});
