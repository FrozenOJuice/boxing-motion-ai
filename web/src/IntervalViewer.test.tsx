import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import Viewer from "./Viewer";

afterEach(() => vi.unstubAllGlobals());
it("uses the exact-frame viewer with bounded interval context and displayed-frame evidence", async () => {
  const visible = vi.fn();
  const fetchMock = vi.fn(async (url: RequestInfo | URL) => {
    const index = Number(String(url).split("/").at(-1));
    return {
      ok: true,
      json: async () => ({
        status: "available",
        issues: [],
        data: {
          frame_index: index,
          pts: String(index * 40),
          time_base: "1/1000",
          exact_timestamp: `${index}/25`,
          width: 32,
          height: 32,
          image_url: `/frame-${index}.jpg`,
          landmarks: [],
          edges: [],
          scene_id: 0,
          segment_id: 0,
          scene_cut_before: false,
        },
      }),
    };
  });
  vi.stubGlobal("fetch", fetchMock);
  render(
    <Viewer
      recording="V4"
      interval={{
        start: 10,
        end: 12,
        count: 100,
        contextStart: 9,
        contextEnd: 13,
        delay: 40,
      }}
      onVisible={visible}
    />,
  );
  const first = await screen.findByAltText("V4, native frame 10");
  expect(visible).not.toHaveBeenCalled();
  fireEvent.load(first);
  await waitFor(() =>
    expect(visible).toHaveBeenCalledWith(
      expect.objectContaining({ frame_index: 10 }),
    ),
  );
  const slider = screen.getByRole("slider", { name: "Native frame" });
  expect(slider).toHaveAttribute("min", "9");
  expect(slider).toHaveAttribute("max", "13");
  fireEvent.click(screen.getByRole("button", { name: "Interval end · 12" }));
  const end = await screen.findByAltText("V4, native frame 12");
  expect(visible).toHaveBeenLastCalledWith(
    expect.objectContaining({ frame_index: 10 }),
  );
  fireEvent.load(end);
  await waitFor(() =>
    expect(visible).toHaveBeenLastCalledWith(
      expect.objectContaining({ frame_index: 12 }),
    ),
  );
  fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "99" } });
  expect(
    fetchMock.mock.calls.every(([url]) => !String(url).endsWith("/99")),
  ).toBe(true);
  expect(screen.queryByText("Original punch")).not.toBeInTheDocument();
});
