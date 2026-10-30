import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ExactFrame } from "./Viewer";
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
it("times out a stalled metadata request visibly and retries without crediting unseen frames", async () => {
  vi.useFakeTimers();
  const visible = vi.fn(),
    failure = vi.fn();
  const fetch = vi
    .fn()
    .mockImplementationOnce(() => new Promise(() => {}))
    .mockResolvedValue({
      ok: true,
      json: async () => ({
        data: {
          frame_index: 8138,
          image_url: "/v5-8138.jpg",
          width: 32,
          height: 32,
          landmarks: [],
          edges: [],
        },
      }),
    });
  vi.stubGlobal("fetch", fetch);
  render(
    <ExactFrame
      recording="V5"
      index={8138}
      overlay={false}
      onVisible={visible}
      onFailure={failure}
    />,
  );
  await act(async () => {
    await vi.advanceTimersByTimeAsync(15000);
  });
  expect(screen.getByRole("alert")).toHaveTextContent("Frame load timed out");
  expect(failure).toHaveBeenCalledOnce();
  expect(visible).not.toHaveBeenCalled();
  await act(async () => {
    fireEvent.click(screen.getByText("Retry frame"));
  });
  fireEvent.load(screen.getByAltText("V5, native frame 8138"));
  expect(visible).toHaveBeenCalledOnce();
  expect(screen.queryByRole("alert")).toBeNull();
});
it("recovers from image failure and does not publish a superseded decoded frame", async () => {
  const visible = vi.fn(),
    failure = vi.fn();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => ({
      ok: true,
      json: async () => ({
        data: {
          frame_index: Number(url.split("/").at(-1)),
          image_url: "/image-" + url.split("/").at(-1) + ".jpg",
          width: 32,
          height: 32,
          landmarks: [],
          edges: [],
        },
      }),
    })),
  );
  const props = {
    recording: "V5",
    overlay: false,
    onVisible: visible,
    onFailure: failure,
  };
  const view = render(<ExactFrame {...props} index={8138} />);
  fireEvent.error(await screen.findByAltText("V5, native frame 8138"));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Image could not be loaded",
  );
  view.rerender(<ExactFrame {...props} index={8137} />);
  fireEvent.load(await screen.findByAltText("V5, native frame 8137"));
  expect(visible).toHaveBeenLastCalledWith(
    expect.objectContaining({ frame_index: 8137 }),
  );
  expect(screen.queryByRole("alert")).toBeNull();
});

it("rejects a decode that resolves after its timeout", async () => {
  vi.useFakeTimers();
  const visible = vi.fn();
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      json: async () => ({
        data: {
          frame_index: 8138,
          image_url: "/hung-decode.jpg",
          width: 32,
          height: 32,
          landmarks: [],
          edges: [],
        },
      }),
    })),
  );
  render(
    <ExactFrame
      recording="V5"
      index={8138}
      overlay={false}
      onVisible={visible}
      onFailure={vi.fn()}
    />,
  );
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
  const image = screen.getByAltText("V5, native frame 8138");
  let release!: () => void;
  Object.defineProperty(image, "decode", {
    value: () =>
      new Promise<void>((resolve) => {
        release = resolve;
      }),
  });
  fireEvent.load(image);
  await act(async () => {
    await vi.advanceTimersByTimeAsync(15000);
  });
  expect(screen.getByRole("alert")).toHaveTextContent("timed out");
  await act(async () => release());
  expect(visible).not.toHaveBeenCalled();
  expect(screen.queryByAltText("V5, native frame 8138")).toBeNull();
});

it("still times out metadata when returning to an already decoded frame", async () => {
  vi.useFakeTimers();
  const visible = vi.fn();
  const fetch = vi
    .fn()
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        data: {
          frame_index: 8138,
          image_url: "/retained.jpg",
          width: 32,
          height: 32,
          landmarks: [],
          edges: [],
        },
      }),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        data: {
          frame_index: 8139,
          image_url: "/not-yet-decoded.jpg",
          width: 32,
          height: 32,
          landmarks: [],
          edges: [],
        },
      }),
    })
    .mockImplementation(() => new Promise(() => {}));
  vi.stubGlobal("fetch", fetch);
  const props = {
    recording: "V5",
    overlay: false,
    onVisible: visible,
    onFailure: vi.fn(),
  };
  const view = render(<ExactFrame {...props} index={8138} />);
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
  fireEvent.load(screen.getByAltText("V5, native frame 8138"));
  view.rerender(<ExactFrame {...props} index={8139} />);
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
  view.rerender(<ExactFrame {...props} index={8138} />);
  await act(async () => {
    await vi.advanceTimersByTimeAsync(15000);
  });
  expect(screen.getByRole("alert")).toHaveTextContent("timed out");
  expect(screen.getByAltText("V5, native frame 8138")).toBeVisible();
  expect(visible).toHaveBeenCalledOnce();
});
