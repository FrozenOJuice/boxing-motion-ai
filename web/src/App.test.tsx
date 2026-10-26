import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { frameBounds, reviewCategory, reviewSource, type Source } from "./api";
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
function confirmSelectedReview() {
  if (screen.queryByLabelText("Punch type is correct.")) {
    for (const label of [
      "Punch type is correct.",
      "Lead/rear hand is correct.",
      "Start/end timing is correct.",
    ]) {
      const input = screen.getByLabelText(label) as HTMLInputElement;
      if (!input.checked) fireEvent.click(input);
    }
  } else fireEvent.click(screen.getByLabelText(/I personally reviewed/));
}

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
  vi.unstubAllGlobals();
});
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
  it("retains the displayed frame and identifies a failed requested frame", async () => {
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
    const old = await screen.findByAltText("V4, native frame 2");
    fireEvent.load(old);
    const stage = old.closest(".image-stage");
    fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Frame 3 unavailable",
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Showing frame 2");
    expect(old).toBeVisible();
    expect(old.closest(".image-stage")).toBe(stage);
    expect(screen.getByText("Frame 2")).toBeVisible();
    expect(screen.queryByAltText("V4, native frame 3")).not.toBeInTheDocument();
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
  fireEvent.change(screen.getByLabelText("Why are you not sure?"), {
    target: { value: "later" },
  });
  fireEvent.click(screen.getByText("Add a note (optional)"));
  fireEvent.change(screen.getByLabelText("Additional explanation (optional)"), {
    target: { value: "Unclear movement" },
  });

  expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  confirmSelectedReview();
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

    attest_human: true,
    inspected_frames: [],
    expected_ledger_sha256: "a".repeat(64),
  });
  await waitFor(() =>
    expect(
      screen.queryByLabelText(/I personally (?:reviewed|verified)/),
    ).not.toBeInTheDocument(),
  );
});

it("requires displayed boundary frames and stance evidence for confirmation", async () => {
  mockApi();
  render(<App />);
  fireEvent.load(await screen.findByAltText("V4, native frame 2"));
  fireEvent.click(screen.getByRole("button", { name: "Yes" }));
  fireEvent.click(screen.getByText("Add a note (optional)"));
  fireEvent.change(screen.getByLabelText("Additional explanation (optional)"), {
    target: { value: "Punch reviewed" },
  });

  confirmSelectedReview();
  expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: /^View end/ }));
  fireEvent.load(await screen.findByAltText("V4, native frame 4"));
  expect(screen.getByRole("button", { name: "Save" })).toBeEnabled();
  fireEvent.click(screen.getByRole("button", { name: "Change it" }));
  fireEvent.change(screen.getByLabelText("Correct punch label"), {
    target: { value: "cross" },
  });
  expect(screen.getByLabelText("Punch type is correct.")).not.toBeChecked();
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
  fireEvent.change(screen.getByLabelText("Why are you not sure?"), {
    target: { value: "mixed" },
  });
  fireEvent.click(screen.getByText("Add a note (optional)"));
  fireEvent.change(screen.getByLabelText("Additional explanation (optional)"), {
    target: { value: "Mixed actions" },
  });
  confirmSelectedReview();
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText(/Review changed or source verification failed/);
  expect(
    screen.getByLabelText("Additional explanation (optional)"),
  ).toHaveValue("Mixed actions");
  fireEvent.click(screen.getByRole("button", { name: /V5 · Punch 2/ }));
  expect(window.confirm).toHaveBeenCalled();
  expect(
    screen.getByLabelText("Additional explanation (optional)"),
  ).toHaveValue("Mixed actions");
  vi.restoreAllMocks();
});

it.each(["ambiguous", "hold", "exclude"])(
  "preserves the %s backend decision under Not sure",
  async (action) => {
    const fetchMock = mockApi();
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Not sure" }));
    fireEvent.change(screen.getByLabelText("Why are you not sure?"), {
      target: {
        value:
          action === "ambiguous"
            ? "arm"
            : action === "hold"
              ? "later"
              : "mixed",
      },
    });
    fireEvent.click(screen.getByText("Add a note (optional)"));
    fireEvent.change(
      screen.getByLabelText("Additional explanation (optional)"),
      {
        target: { value: "Synthetic review" },
      },
    );
    if (action === "exclude") {
      fireEvent.click(screen.getByText("More options"));
      fireEvent.click(
        screen.getByLabelText("Exclude this example from future candidates"),
      );
    }
    confirmSelectedReview();
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
  fireEvent.click(screen.getByText("Add a note (optional)"));
  fireEvent.change(screen.getByLabelText("Additional explanation (optional)"), {
    target: { value: "Synthetic review" },
  });

  confirmSelectedReview();
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText(/Review saved/);
  const call = mock.mock.calls.find(([, init]) => init?.method === "POST");
  expect(JSON.parse(call![1]!.body as string)).toMatchObject({
    action: "verify",
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

it("atomically swaps decoded pixels, frame label, timestamp and overlay; late decodes cannot win", async () => {
  const mock = mockApi();
  const original = mock.getMockImplementation()!;
  mock.mockImplementation(async (url, init) => {
    if (!String(url).includes("/frames/")) return original(url, init);
    const index = Number(String(url).split("/").at(-1));
    return {
      ok: true,
      json: async () =>
        ok({
          ...frame,
          frame_index: index,
          pts: String(index * 40),
          exact_timestamp: `${index}/25`,
          image_url: `/synthetic-${index}.jpg`,
          landmarks: frame.landmarks.map((p) => ({
            ...p,
            xyz: [index / 10, 0.4, 0],
          })),
        }),
    } as Response;
  });
  const { container } = render(<App />);
  fireEvent.load(await screen.findByAltText("V4, native frame 2"));
  fireEvent.click(screen.getByLabelText("Skeleton"));
  fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
  const third = await screen.findByAltText("V4, native frame 3");
  let resolveThird!: () => void;
  Object.defineProperty(third, "decode", {
    value: () =>
      new Promise<void>((r) => {
        resolveThird = r;
      }),
  });
  fireEvent.load(third);
  expect(third).not.toBeVisible();
  expect(screen.getByAltText("V4, native frame 2")).toBeVisible();
  expect(screen.getByText("Frame 2")).toBeVisible();
  expect(container.querySelector(".skeleton line")).toHaveAttribute(
    "x1",
    "0.2",
  );
  fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
  const fourth = await screen.findByAltText("V4, native frame 4");
  let resolveFourth!: () => void;
  Object.defineProperty(fourth, "decode", {
    value: () =>
      new Promise<void>((r) => {
        resolveFourth = r;
      }),
  });
  fireEvent.load(fourth);
  await act(async () => resolveThird());
  expect(screen.getByText("Frame 2")).toBeVisible();
  expect(screen.queryByAltText("V4, native frame 3")).not.toBeInTheDocument();
  await act(async () => resolveFourth());
  expect(fourth).toBeVisible();
  expect(screen.queryByAltText("V4, native frame 2")).not.toBeInTheDocument();
  expect(screen.getByText("Frame 4")).toBeVisible();
  expect(container.querySelector(".skeleton line")).toHaveAttribute(
    "x1",
    "0.4",
  );
  expect(screen.getByText(/"pts": "160"/)).toHaveTextContent(
    '"exact_timestamp": "4/25"',
  );
  expect(screen.getByRole("slider", { name: "Native frame" })).toHaveValue("4");
});

it.each(["load", "decode"])(
  "keeps the stage and old image after a %s failure, then retries",
  async (kind) => {
    mockApi();
    render(<App />);
    const old = await screen.findByAltText("V4, native frame 2");
    fireEvent.load(old);
    const stage = old.closest(".image-stage");
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    const next = await screen.findByAltText("V4, native frame 3");
    if (kind === "load") fireEvent.error(next);
    else {
      Object.defineProperty(next, "decode", {
        value: () => Promise.reject(new Error("decode failed")),
      });
      fireEvent.load(next);
    }
    await screen.findByRole("alert");
    expect(old).toBeVisible();
    expect(old.closest(".image-stage")).toBe(stage);
    expect(
      await screen.findByRole("button", { name: "Play" }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry frame" }));
    const retry = await screen.findByAltText("V4, native frame 3");
    fireEvent.load(retry);
    expect(retry).toBeVisible();
    expect(retry.closest(".image-stage")).toBe(stage);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  },
);

it("ignores late metadata during rapid stepping and retains pixels across viewport changes", async () => {
  const mock = mockApi();
  const original = mock.getMockImplementation()!;
  let resolveThird!: (r: Response) => void;
  mock.mockImplementation((url, init) =>
    String(url).endsWith("/frames/3")
      ? new Promise<Response>((r) => {
          resolveThird = r;
        })
      : original(url, init),
  );
  render(<App />);
  const old = await screen.findByAltText("V4, native frame 2");
  fireEvent.load(old);
  const stage = old.closest(".image-stage");
  fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
  fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
  const fourth = await screen.findByAltText("V4, native frame 4");
  fireEvent.load(fourth);
  await act(async () =>
    resolveThird({
      ok: true,
      json: async () =>
        ok({ ...frame, frame_index: 3, image_url: "/late.jpg" }),
    } as Response),
  );
  expect(fourth).toBeVisible();
  expect(screen.queryByAltText("V4, native frame 3")).not.toBeInTheDocument();
  for (const [width, height] of [
    [1440, 900],
    [1366, 768],
    [1280, 720],
    [390, 844],
  ]) {
    vi.stubGlobal("innerWidth", width);
    vi.stubGlobal("innerHeight", height);
    fireEvent(window, new Event("resize"));
    expect(fourth.closest(".image-stage")).toBe(stage);
    expect(fourth).toBeVisible();
    expect(screen.getByText("Frame 4")).toBeVisible();
  }
});

it("keeps the initial empty stage stable on failure and removes redundant page titles", async () => {
  mockApi();
  render(<App />);
  const image = await screen.findByAltText("V4, native frame 2");
  const stage = image.closest(".image-stage");
  fireEvent.error(image);
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "No frame displayed",
  );
  expect(document.querySelector(".image-stage")).toBe(stage);
  expect(screen.queryByRole("heading", { level: 1 })).not.toBeInTheDocument();
  for (const name of ["Dataset", "Experiments", "Models"]) {
    fireEvent.click(screen.getByRole("button", { name }));
    expect(screen.queryByRole("heading", { level: 1 })).not.toBeInTheDocument();
  }
});

it("confirms anonymously without written notes across next and remount", async () => {
  const mock = mockApi();
  const first = render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Yes" }));
  expect(screen.queryByLabelText("Your name")).not.toBeInTheDocument();
  expect(
    screen.getByText(/Review/, { selector: "p.confirmation-scope" }),
  ).toHaveTextContent("jab");
  fireEvent.load(await screen.findByAltText("V4, native frame 2"));
  fireEvent.click(screen.getByRole("button", { name: /^View end/ }));
  fireEvent.load(await screen.findByAltText("V4, native frame 4"));
  expect(screen.getByRole("button", { name: "Save & Next" })).toBeDisabled();
  const attestation = screen.getByLabelText("Start/end timing is correct.");
  expect(attestation).not.toBeChecked();
  expect(screen.getByLabelText("Lead/rear hand is correct.")).not.toBeChecked();
  confirmSelectedReview();
  fireEvent.click(screen.getByRole("button", { name: "Save & Next" }));
  await screen.findByText(/Review saved/);
  const call = mock.mock.calls.find(([, init]) => init?.method === "POST");
  const payload = JSON.parse(call![1]!.body as string);
  expect(payload).toMatchObject({
    action: "verify",
    attest_human: true,
    proposed_label: "jab",
    proposed_bounds: { start: 2, end: 4 },
  });
  expect(payload).not.toHaveProperty("reviewer");
  expect(payload).not.toHaveProperty("source"); // The server chooses the source.
  expect(payload.notes).toContain("Selected manual verification state");
  expect(payload.notes).not.toContain("Reviewer note:");
  expect(payload.lead_rear_evidence).toBe("");
  expect(payload.lead_rear_evidence).not.toMatch(/orthodox|southpaw/);
  await waitFor(() =>
    expect(mock).toHaveBeenCalledWith("/api/cases/V5/2", expect.anything()),
  );
  fireEvent.click(await screen.findByRole("button", { name: "Not sure" }));
  expect(screen.queryByLabelText("Your name")).not.toBeInTheDocument();
  expect(screen.getByLabelText(/I personally reviewed/)).not.toBeChecked();
  first.unmount();
  // A new application instance still requires no identity and restores no draft decisions.
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Yes" }));
  expect(screen.queryByLabelText("Your name")).not.toBeInTheDocument();
  expect(
    screen.getByLabelText("Start/end timing is correct."),
  ).not.toBeChecked();
  expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
});

it("requires an explanation for corrections and sends the corrected interval with original source bindings", async () => {
  const mock = mockApi();
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Change it" }));
  fireEvent.change(screen.getByLabelText("Correct punch label"), {
    target: { value: "cross" },
  });
  fireEvent.change(screen.getByLabelText("Start frame"), {
    target: { value: "3" },
  });
  expect(screen.getByText("Briefly explain your correction.")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: /^View start/ }));
  fireEvent.load(await screen.findByAltText("V4, native frame 3"));
  fireEvent.click(screen.getByRole("button", { name: /^View end/ }));
  fireEvent.load(await screen.findByAltText("V4, native frame 4"));
  confirmSelectedReview();
  expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  fireEvent.change(screen.getByLabelText("Why change it?"), {
    target: {
      value:
        "The opening frame belongs to the previous movement; the punch is a cross.",
    },
  });
  expect(screen.getByLabelText("Start/end timing is correct.")).toBeChecked();
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText(/Review saved/);
  const call = mock.mock.calls.find(([, init]) => init?.method === "POST");
  expect(JSON.parse(call![1]!.body as string)).toMatchObject({
    action: "verify",

    proposed_label: "cross",
    proposed_bounds: { start: 3, end: 4 },
    source_candidate_sha256: "c".repeat(64),
    expected_revision_sha256: "b".repeat(64),
  });
  expect(JSON.parse(call![1]!.body as string).notes).toContain(
    "Reviewer note: The opening frame belongs",
  );
});

it("flags uncertainty using only a selected reason, without certifying label, stance or boundaries", async () => {
  const mock = mockApi();
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Not sure" }));
  expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  fireEvent.change(screen.getByLabelText("Why are you not sure?"), {
    target: { value: "arm" },
  });
  fireEvent.click(screen.getByLabelText(/I personally reviewed/));
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText(/Review saved/);
  const call = mock.mock.calls.find(([, init]) => init?.method === "POST");
  expect(JSON.parse(call![1]!.body as string)).toMatchObject({
    action: "ambiguous",
    uncertainty: "Lead or rear arm is unclear",
    lead_rear_evidence: "",
    proposed_label: null,
    proposed_bounds: null,
    inspected_frames: [],
  });
});

it("validates Other reason without a reviewer profile", async () => {
  const mock = mockApi();
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Not sure" }));
  fireEvent.change(screen.getByLabelText("Why are you not sure?"), {
    target: { value: "other" },
  });
  expect(
    screen.getByText("Add a short explanation for Other reason."),
  ).toBeVisible();
  fireEvent.change(screen.getByLabelText("Additional explanation"), {
    target: { value: "Two athletes overlap in the crop." },
  });
  fireEvent.click(screen.getByLabelText(/I personally reviewed/));
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText(/Review saved/);
  const call = mock.mock.calls.find(([, init]) => init?.method === "POST");
  expect(JSON.parse(call![1]!.body as string)).not.toHaveProperty("reviewer");
  expect(
    screen.queryByRole("button", { name: "Change name" }),
  ).not.toBeInTheDocument();
});

it("ignores old stored names and needs no browser profile or storage access", async () => {
  window.localStorage.setItem("boxing-motion-ai.reviewer.v1", "Alice");
  const read = vi.spyOn(window.localStorage, "getItem");
  const write = vi.spyOn(window.localStorage, "setItem");
  mockApi();
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Yes" }));
  expect(screen.queryByLabelText("Your name")).not.toBeInTheDocument();
  expect(screen.queryByText("Alice")).not.toBeInTheDocument();
  expect(read).not.toHaveBeenCalled();
  expect(write).not.toHaveBeenCalled();
});

it("explains server boundary validation without losing the review draft", async () => {
  const mock = mockApi();
  const original = mock.getMockImplementation()!;
  mock.mockImplementation(async (url, init) =>
    init?.method === "POST"
      ? ({
          ok: false,
          status: 422,
          json: async () => ({ issues: ["confirmation_crosses_known_cut"] }),
        } as Response)
      : original(url, init),
  );
  render(<App />);
  fireEvent.load(await screen.findByAltText("V4, native frame 2"));
  fireEvent.click(screen.getByRole("button", { name: "Yes" }));
  fireEvent.click(screen.getByRole("button", { name: /^View end/ }));
  fireEvent.load(await screen.findByAltText("V4, native frame 4"));
  confirmSelectedReview();
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "This interval crosses a scene cut",
  );
  expect(screen.getByRole("button", { name: "Yes" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
});

it("blocks invalid correction intervals with a clear message", async () => {
  mockApi();
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Change it" }));
  fireEvent.change(screen.getByLabelText("Start frame"), {
    target: { value: "9" },
  });
  expect(
    screen.getByText(/Enter whole-number start\/end frames/),
  ).toBeVisible();
  expect(screen.getByRole("button", { name: "Save & Next" })).toBeDisabled();
  fireEvent.change(screen.getByLabelText("Start frame"), {
    target: { value: "-1" },
  });
  expect(
    screen.getByText(/Enter whole-number start\/end frames/),
  ).toBeVisible();
  fireEvent.change(screen.getByLabelText("Start frame"), {
    target: { value: "2" },
  });
  fireEvent.change(screen.getByLabelText("End frame (included)"), {
    target: { value: "10" },
  });
  expect(
    screen.getByText(/Enter whole-number start\/end frames/),
  ).toBeVisible();
  expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
});

it.each([
  ["Punch type is correct.", "punch_type"],
  ["Lead/rear hand is correct.", "hand"],
  ["Start/end timing is correct.", "timing"],
])(
  "saves %s independently without additional explanations",
  async (label, axis) => {
    const mock = mockApi();
    render(<App />);
    fireEvent.load(await screen.findByAltText("V4, native frame 2"));
    fireEvent.click(screen.getByRole("button", { name: "Yes" }));
    for (const text of [
      "Punch type is correct.",
      "Lead/rear hand is correct.",
      "Start/end timing is correct.",
    ])
      expect(screen.getByLabelText(text)).not.toBeChecked();
    fireEvent.click(screen.getByLabelText(label));
    expect(screen.getByText("Partially verified · 1/3 selected")).toBeVisible();
    if (axis === "timing") {
      expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
      fireEvent.click(screen.getByRole("button", { name: /^View end/ }));
      fireEvent.load(await screen.findByAltText("V4, native frame 4"));
    }
    fireEvent.click(screen.getByRole("button", { name: "Save & Next" }));
    await screen.findByText(/Review saved/);
    const call = mock.mock.calls.find(([, init]) => init?.method === "POST");
    expect(JSON.parse(call![1]!.body as string).verification).toEqual({
      schema_version: "annotation-verification-1.0",
      punch_type: axis === "punch_type",
      hand: axis === "hand",
      timing: axis === "timing",
    });
  },
);

function mockPriorVerification(
  actor: "human" | "assistant" = "human",
  legacy = false,
) {
  const mock = mockApi();
  const original = mock.getMockImplementation()!;
  mock.mockImplementation(async (url, init) => {
    const response = await original(url, init);
    if (!String(url).startsWith("/api/cases/")) return response;
    const value = await response.json();
    const human = {
      ...revision,
      revision_id: "human-1",
      actor_type: actor,
      reviewer: "Alice",
      reviewed_at: "2026-10-25T00:00:00+00:00",
      label_status: legacy ? "confirmed" : "unreviewed",
      boundary_status: legacy ? "confirmed" : "unreviewed",
      confirmed_label: legacy ? "jab" : null,
      confirmed_bounds: legacy
        ? { start: 2, end: 4, convention: "zero_based_decoded_inclusive" }
        : null,
      ...(legacy
        ? {}
        : {
            verification: {
              schema_version: "annotation-verification-1.0",
              punch_type: true,
              hand: false,
              timing: false,
              label: "jab",
              bounds: {
                start: 2,
                end: 4,
                convention: "zero_based_decoded_inclusive",
              },
              origins: { punch_type: "human-1" },
            },
          }),
    };
    value.data.edit.data.history = [revision, human];
    value.data.review.data.latest = human;
    value.data.review.data.verification_status = legacy
      ? "fully_verified"
      : "partially_verified";
    return { ok: true, json: async () => value } as Response;
  });
  return mock;
}

it("restores only real human checks and allows withdrawing a prior check", async () => {
  const mock = mockPriorVerification();
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Yes" }));
  expect(screen.getByLabelText("Punch type is correct.")).toBeChecked();
  expect(screen.getByLabelText("Lead/rear hand is correct.")).not.toBeChecked();
  expect(
    screen.getByLabelText("Start/end timing is correct."),
  ).not.toBeChecked();
  expect(screen.getByText("Previously verified manually")).toBeVisible();
  expect(screen.getByText("Saved: Partially verified")).toBeVisible();
  fireEvent.click(screen.getByLabelText("Punch type is correct."));
  expect(screen.getByText("No confirmations selected")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText(/Review saved/);
  const call = mock.mock.calls.find(([, init]) => init?.method === "POST");
  expect(JSON.parse(call![1]!.body as string).verification).toMatchObject({
    punch_type: false,
    hand: false,
    timing: false,
  });
});

it("never restores assistant checks and clears checks after a target edit", async () => {
  mockPriorVerification("assistant");
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Change it" }));
  expect(screen.getByLabelText("Punch type is correct.")).not.toBeChecked();
  fireEvent.click(screen.getByLabelText("Punch type is correct."));
  fireEvent.click(screen.getByLabelText("Lead/rear hand is correct."));
  expect(screen.getByLabelText("Punch type is correct.")).toBeChecked();
  fireEvent.change(screen.getByLabelText("Correct punch label"), {
    target: { value: "cross" },
  });
  expect(screen.getByLabelText("Punch type is correct.")).not.toBeChecked();
  expect(screen.getByLabelText("Lead/rear hand is correct.")).not.toBeChecked();
});

it("reads legacy human verification as three separate prior checks", async () => {
  mockPriorVerification("human", true);
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Yes" }));
  for (const text of [
    "Punch type is correct.",
    "Lead/rear hand is correct.",
    "Start/end timing is correct.",
  ])
    expect(screen.getByLabelText(text)).toBeChecked();
  expect(screen.getByText("Saved: Fully verified")).toBeVisible();
});

it("keeps partially verified cases distinct from Reviewed", () => {
  const entry = {
    id: "V4:1",
    priority: "dispute",
    review: ok({
      ...review,
      verification_status: "partially_verified" as const,
    }),
  };
  expect(reviewCategory(entry)).toBe("Partially verified");
});

it("labels manual and automated origins while retaining legacy names in history", async () => {
  expect(reviewSource({ ...revision, actor_type: "assistant" })).toBe(
    "automated",
  );
  expect(
    reviewSource({ ...revision, actor_type: "human", reviewer: "Alice" }),
  ).toBe("manual");
  mockPriorVerification();
  render(<App />);
  await screen.findByRole("button", { name: "Yes" });
  fireEvent.click(screen.getByText("Advanced details"));
  fireEvent.click(screen.getByText(/Review history \(/));
  expect(screen.getByText("Manual review · hold")).toBeVisible();
  expect(screen.getByText(/Legacy reviewer: Alice/)).toBeVisible();
  expect(screen.getByText("Automated observation · hold")).toBeVisible();
});
