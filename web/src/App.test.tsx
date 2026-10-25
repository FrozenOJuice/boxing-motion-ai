import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { frameBounds, type Source } from "./api";
import { Skeleton } from "./Viewer";
const source: Source = {
  source_row: 1,
  video_id: "V4",
  source_class: "Jab",
  punch_type_provisional: "jab",
  raw_start: 2,
  raw_end: 4,
  candidate_start_index: 2,
  candidate_end_index: 4,
  annotation_end_assumption: "inclusive",
  flags: ["provisional_mapping"],
};
const revision = {
  actor_type: "assistant",
  reviewer: "Synthetic reviewer",
  disposition: "hold",
  label_status: "ambiguous",
  boundary_status: "mixed_action",
  reasons: ["Needs human review"],
  unresolved_uncertainty: ["Unverified stance"],
};
const ok = (data: unknown) => ({ status: "available", issues: [], data });
const review = { source, latest: revision, human_accepted: false };
const frame = {
  frame_index: 2,
  pts: "80",
  time_base: "1/1000",
  exact_timestamp: "2/25",
  width: 32,
  height: 32,
  image_url: "/synthetic.jpg",
  landmarks: Array.from({ length: 33 }, (_, i) => ({
    name: `joint${i}`,
    xyz: [0.4, 0.4, 0],
    confidence: [0.9, 0.9],
    valid: i !== 13,
  })),
  edges: [
    [11, 12],
    [11, 13],
  ] as [number, number][],
  scene_id: 0,
  segment_id: 0,
  scene_cut_before: false,
};
function mockApi() {
  const fetchMock = vi.fn(async (url: RequestInfo | URL) => {
    const path = String(url);
    let value: unknown;
    if (path === "/api/reviews")
      value = ok({
        entries: [
          { id: "V4:1", priority: "dispute", review: ok(review) },
          {
            id: "V5:2",
            priority: "diagnostic",
            review: ok({
              ...review,
              source: { ...source, video_id: "V5", source_row: 2 },
            }),
          },
        ],
        summary: {
          case_count: 2,
          human_accepted: 0,
          dispositions: { hold: 2 },
        },
      });
    else if (path === "/api/inventory")
      value = ok({
        available_recordings: 2,
        records: {
          V4: ok({
            recording_id: "V4",
            role: "training",
            source_group: "unknown",
            boxer_id: null,
            session_id: null,
            preparation: { source_frame_count: 10 },
          }),
          V5: ok({
            recording_id: "V5",
            role: "development",
            source_group: "unknown",
            boxer_id: null,
            session_id: null,
            preparation: { source_frame_count: 10 },
          }),
        },
      });
    else if (path === "/api/experiments") value = ok({ items: {} });
    else if (path.startsWith("/api/cases/"))
      value = ok({
        annotation: source,
        labels: ["jab", "cross"],
        review: ok(review),
        evidence: ok({
          case: {
            prediction: { prediction: "cross", probabilities: [0.2, 0.8] },
            selection: "synthetic held-out",
            selection_fold: "hold-V4",
          },
        }),
        recording: ok({
          preparation: { source_frame_count: 10 },
          boxer_id: null,
          session_id: null,
        }),
      });
    else if (path.includes("/frames/")) {
      const index = Number(path.split("/").at(-1));
      value = ok({
        ...frame,
        frame_index: index,
        image_url: `/synthetic-${index}.jpg`,
      });
    } else value = ok({});
    return { ok: true, json: async () => value } as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
afterEach(() => vi.unstubAllGlobals());
describe("read-only dashboard", () => {
  it("shows real response labels, uncertainty and model confidence without edit controls", async () => {
    mockApi();
    render(<App />);
    expect(await screen.findByText("80.0%")).toBeInTheDocument();
    expect(screen.getByText("ambiguous / mixed action")).toBeInTheDocument();
    expect(
      screen.getByText(/Model confidence is not annotation evidence/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /save|approve|confirm/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("navigation")).toBeInTheDocument();
  });
  it("steps integer frames and filters to V5 without touching V9", async () => {
    const fetchMock = mockApi();
    render(<App />);
    await screen.findByAltText("V4, native frame 2");
    fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    await screen.findByAltText("V4, native frame 3");
    fireEvent.change(screen.getByLabelText("Filter recording"), {
      target: { value: "V5" },
    });
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/cases/V5/2",
        expect.anything(),
      ),
    );
    expect(
      fetchMock.mock.calls.some(([url]) => String(url).includes("/V9")),
    ).toBe(false);
  });
  it("does not show an old frame or fabricate pixels after a retrieval error", async () => {
    const fetchMock = mockApi();
    const original = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (url, ...args) =>
      String(url).endsWith("/frames/3")
        ? ({
            ok: false,
            json: async () => ({ issues: ["exact_frame_not_found"] }),
          } as Response)
        : original(url, ...args),
    );
    render(<App />);
    await screen.findByAltText("V4, native frame 2");
    fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    expect(
      await screen.findByText(/No approximate frame substituted/),
    ).toBeInTheDocument();
    expect(screen.queryByAltText("V4, native frame 2")).not.toBeInTheDocument();
  });
  it("shows an actionable backend failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("Backend unavailable")),
    );
    render(<App />);
    expect(
      await screen.findByText(/Start the local backend/),
    ).toBeInTheDocument();
  });
  it("draws only valid joint connections", () => {
    const { container } = render(<Skeleton frame={frame} />);
    expect(container.querySelectorAll("line")).toHaveLength(1);
    expect(container.querySelectorAll("circle")).toHaveLength(21);
  });
  it("preserves boundary conventions and refuses reversed or out-of-range intervals", () => {
    expect(frameBounds(source, 10)).toEqual([2, 4]);
    expect(
      frameBounds({ ...source, annotation_end_assumption: "exclusive" }, 10),
    ).toEqual([2, 3]);
    expect(frameBounds({ ...source, candidate_end_index: 1 }, 10)).toBeNull();
    expect(frameBounds({ ...source, candidate_end_index: 11 }, 10)).toBeNull();
    expect(
      frameBounds({ ...source, candidate_start_index: null }, 10),
    ).toBeNull();
  });
});

it("recognizes documented end conventions and rejects an unspecified convention", () => {
  expect(
    frameBounds(
      {
        ...source,
        annotation_end_assumption: "exclusive; author convention unverified",
      },
      10,
    ),
  ).toEqual([2, 3]);
  expect(
    frameBounds(
      {
        ...source,
        annotation_end_assumption:
          "inclusive assumption; author convention unverified",
      },
      10,
    ),
  ).toEqual([2, 4]);
  expect(
    frameBounds({ ...source, annotation_end_assumption: "unverified" }, 10),
  ).toBeNull();
});

it("clears the viewer when no annotation matches the filters", async () => {
  mockApi();
  render(<App />);
  await screen.findByAltText("V4, native frame 2");
  fireEvent.change(screen.getByLabelText("Search cases"), {
    target: { value: "no matching case" },
  });
  await screen.findByText("No cases match these filters.");
  await waitFor(() =>
    expect(screen.queryByAltText("V4, native frame 2")).not.toBeInTheDocument(),
  );
});
