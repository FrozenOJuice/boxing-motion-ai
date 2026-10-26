import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { frameBounds, reviewCategory, type Source } from "./api";
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
const ok = <T,>(data: T) => ({ status: "available", issues: [], data });
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
  const fetchMock = vi.fn(
    async (url: RequestInfo | URL, _init?: RequestInit) => {
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
          edit: ok({
            ledger_sha256: "a".repeat(64),
            revision_sha256: "b".repeat(64),
            source_candidate_sha256: "c".repeat(64),
            ledger_version: 1,
            history: [revision],
            decisions: [],
          }),
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
    },
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
afterEach(() => vi.unstubAllGlobals());
describe("interactive dashboard", () => {
  it("shows real response labels, uncertainty and model confidence with explicit human review controls", async () => {
    mockApi();
    render(<App />);
    await screen.findByRole("button", { name: "Yes" });
    expect(
      screen.queryByLabelText("Correct punch label"),
    ).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Your name")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save & Next" })).toBeDisabled();
    expect(
      screen.getByText("Advanced details").closest("details"),
    ).not.toHaveAttribute("open");
    fireEvent.click(screen.getByText("Advanced details"));
    expect(screen.getByText(/cross · 80.0%/)).toBeVisible();
    expect(
      screen.getByText(/Model confidence is not annotation evidence/),
    ).toBeVisible();
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

it("saves an explicitly attested hold and refreshes review history", async () => {
  const fetchMock = mockApi();
  render(<App />);
  await screen.findByRole("button", { name: "Not sure" });
  fireEvent.click(screen.getByRole("button", { name: "Not sure" }));
  fireEvent.change(screen.getByLabelText("What should happen to this punch?"), {
    target: { value: "hold" },
  });
  fireEvent.change(screen.getByLabelText("Your name"), {
    target: { value: "Human tester" },
  });
  fireEvent.change(screen.getByLabelText("What did you observe?"), {
    target: { value: "Unclear movement" },
  });
  fireEvent.change(screen.getByLabelText(/What is still uncertain/), {
    target: { value: "Stance not established" },
  });
  expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  fireEvent.click(screen.getByLabelText(/I personally reviewed/));
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText(/Review saved/);
  const call = fetchMock.mock.calls.find(
    ([url, init]) =>
      String(url) === "/api/reviews/V4/1" &&
      (init as RequestInit)?.method === "POST",
  );
  expect(call).toBeDefined();
  const payload = JSON.parse((call![1] as RequestInit).body as string);
  expect(payload).toMatchObject({
    action: "hold",
    reviewer: "Human tester",
    attest_human: true,
    inspected_frames: [],
    expected_ledger_sha256: "a".repeat(64),
  });
  await waitFor(() =>
    expect(
      screen.queryByLabelText(/I personally reviewed/),
    ).not.toBeInTheDocument(),
  );
});

it("requires displayed boundary frames and stance evidence for confirmation", async () => {
  mockApi();
  render(<App />);
  fireEvent.load(await screen.findByAltText("V4, native frame 2"));
  fireEvent.click(screen.getByRole("button", { name: "Yes" }));
  fireEvent.change(screen.getByLabelText("Your name"), {
    target: { value: "Human tester" },
  });
  fireEvent.change(screen.getByLabelText("What did you observe?"), {
    target: { value: "Punch reviewed" },
  });
  fireEvent.change(screen.getByLabelText(/How did you identify/), {
    target: { value: "Observed stance" },
  });
  fireEvent.click(screen.getByLabelText(/I personally reviewed/));
  expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: /^View end/ }));
  fireEvent.load(await screen.findByAltText("V4, native frame 4"));
  expect(screen.getByRole("button", { name: "Save" })).toBeEnabled();
  fireEvent.click(screen.getByRole("button", { name: "Change it" }));
  fireEvent.change(screen.getByLabelText("Correct punch label"), {
    target: { value: "cross" },
  });
  expect(screen.getByLabelText(/I personally reviewed/)).not.toBeChecked();
});

it("preserves a draft on conflict and guards navigation", async () => {
  const mock = mockApi();
  const original = mock.getMockImplementation()!;
  mock.mockImplementation(async (url, init) =>
    (init as RequestInit)?.method === "POST"
      ? ({
          ok: false,
          status: 409,
          json: async () => ({ issues: ["stale_review_reload"] }),
        } as Response)
      : original(url, init),
  );
  vi.spyOn(window, "confirm").mockReturnValue(false);
  render(<App />);
  await screen.findByRole("button", { name: "Not sure" });
  fireEvent.click(screen.getByRole("button", { name: "Not sure" }));
  fireEvent.change(screen.getByLabelText("What should happen to this punch?"), {
    target: { value: "exclude" },
  });
  fireEvent.change(screen.getByLabelText("Your name"), {
    target: { value: "Human tester" },
  });
  fireEvent.change(screen.getByLabelText("What did you observe?"), {
    target: { value: "Mixed actions" },
  });
  fireEvent.click(screen.getByLabelText(/I personally reviewed/));
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText(/Review changed or source verification failed/);
  expect(screen.getByLabelText("What did you observe?")).toHaveValue(
    "Mixed actions",
  );
  fireEvent.click(screen.getByRole("button", { name: "Next annotation" }));
  expect(window.confirm).toHaveBeenCalled();
  expect(screen.getByLabelText("What did you observe?")).toHaveValue(
    "Mixed actions",
  );
  vi.restoreAllMocks();
});

it.each(["ambiguous", "hold", "exclude"])(
  "preserves the %s backend decision under Not sure",
  async (action) => {
    const fetchMock = mockApi();
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Not sure" }));
    fireEvent.change(
      screen.getByLabelText("What should happen to this punch?"),
      { target: { value: action } },
    );
    fireEvent.change(screen.getByLabelText("Your name"), {
      target: { value: "Tester" },
    });
    fireEvent.change(screen.getByLabelText("What did you observe?"), {
      target: { value: "Synthetic review" },
    });
    if (action !== "exclude")
      fireEvent.change(screen.getByLabelText("What is still uncertain?"), {
        target: { value: "Unclear arm" },
      });
    fireEvent.click(screen.getByLabelText(/I personally reviewed/));
    fireEvent.click(screen.getByRole("button", { name: "Save & Next" }));
    await screen.findByText(/Review saved/);
    const call = fetchMock.mock.calls.find(
      ([, init]) => init?.method === "POST",
    );
    expect(JSON.parse(call![1]!.body as string).action).toBe(action);
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/cases/V5/2",
        expect.anything(),
      ),
    );
  },
);

it("Yes resets a proposed correction and preserves the original label and bounds", async () => {
  const mock = mockApi();
  render(<App />);
  fireEvent.load(await screen.findByAltText("V4, native frame 2"));
  fireEvent.click(screen.getByRole("button", { name: "Change it" }));
  fireEvent.change(screen.getByLabelText("Correct punch label"), {
    target: { value: "cross" },
  });
  fireEvent.change(screen.getByLabelText("Start frame"), {
    target: { value: "1" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Yes" }));
  expect(screen.queryByLabelText("Start frame")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /^View end/ }));
  fireEvent.load(await screen.findByAltText("V4, native frame 4"));
  fireEvent.change(screen.getByLabelText("Your name"), {
    target: { value: "Tester" },
  });
  fireEvent.change(screen.getByLabelText("What did you observe?"), {
    target: { value: "Synthetic review" },
  });
  fireEvent.change(screen.getByLabelText(/How did you identify/), {
    target: { value: "Observed stance" },
  });
  fireEvent.click(screen.getByLabelText(/I personally reviewed/));
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText(/Review saved/);
  const call = mock.mock.calls.find(([, init]) => init?.method === "POST");
  expect(JSON.parse(call![1]!.body as string)).toMatchObject({
    action: "confirm",
    proposed_label: "jab",
    proposed_bounds: { start: 2, end: 4 },
  });
});

it("groups for display without certifying automated observations", () => {
  const entry = { id: "V4:1", priority: "anchor", review: ok(review) };
  expect(reviewCategory(entry)).toBe("Needs attention");
  const accepted = {
    ...review,
    latest: {
      ...revision,
      disposition: "accept",
      label_status: "confirmed",
      boundary_status: "confirmed",
    },
  };
  expect(reviewCategory({ ...entry, review: ok(accepted) })).toBe("To review");
  expect(
    reviewCategory({
      ...entry,
      review: ok({
        ...accepted,
        latest: { ...accepted.latest, actor_type: "human" },
      }),
    }),
  ).toBe("Reviewed");
  expect(
    reviewCategory({
      ...entry,
      review: ok({
        ...review,
        latest: { ...revision, actor_type: "human", disposition: "exclude" },
      }),
    }),
  ).toBe("Reviewed");
});

it("waits for displayed frames, pauses on stepping, and stops at the window end", async () => {
  mockApi();
  render(<App />);
  const first = await screen.findByAltText("V4, native frame 2");
  fireEvent.click(screen.getByRole("button", { name: "Play" }));
  await new Promise((resolve) => setTimeout(resolve, 160));
  expect(screen.queryByAltText("V4, native frame 3")).not.toBeInTheDocument();
  fireEvent.load(first);
  await screen.findByAltText("V4, native frame 3");
  fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
  expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
  fireEvent.change(screen.getByRole("slider", { name: "Native frame" }), {
    target: { value: "8" },
  });
  fireEvent.load(await screen.findByAltText("V4, native frame 8"));
  fireEvent.click(screen.getByRole("button", { name: "Play" }));
  fireEvent.load(await screen.findByAltText("V4, native frame 9"));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument(),
  );
});
