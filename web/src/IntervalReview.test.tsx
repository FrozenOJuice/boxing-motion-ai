import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import IntervalReview from "./IntervalReview";

// Viewer decoding, stale image handling and exact stepping have separate real-viewer tests.
vi.mock("./BufferedIntervalViewer", () => ({
  default: ({
    onVisible,
    onViewing,
  }: {
    onVisible: (f: { frame_index: number }) => void;
    onViewing: (v: unknown) => void;
  }) => (
    <div>
      <button
        onClick={() =>
          onViewing({
            complete: true,
            playbackComplete: true,
            percent: 100,
            evidence: {
              schema_version: "continuous-viewing-1.0",
              policy: "visible-realtime-1.0",
              cache_key: "a".repeat(64),
              interval_id: "V4-bg-1",
              segments: [
                {
                  start: 0,
                  end: 0.12,
                  rate: 1,
                  wall_seconds: 0.12,
                  samples: 3,
                },
              ],
            },
          })
        }
      >
        Watch complete interval
      </button>
      {[10, 11, 12].map((n) => (
        <button key={n} onClick={() => onVisible({ frame_index: n })}>
          Show frame {n}
        </button>
      ))}
    </div>
  ),
}));
afterEach(() => vi.unstubAllGlobals());
const entry = {
  id: "V4-bg-1",
  recording: "V4",
  role: "train_candidate",
  purpose: "background_candidate",
  start: 10,
  stop: 13,
  frame_count: 100,
  context_start: 9,
  context_end: 14,
  timing: { start: "1", stop_exclusive: "13/10" },
};
function api(
  options: {
    fail?: boolean;
    conflict?: boolean;
    second?: boolean;
    pending?: Promise<Response>;
  } = {},
) {
  const entries = options.second
    ? [entry, { ...entry, id: "V4-bg-2" }]
    : [entry];
  const histories: Record<string, unknown[]> = { original: [], motion: [] };
  const mock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
    const motion = String(url).includes("pilot=motion-v2");
    const activeEntries = motion
      ? [
          {
            ...entry,
            id: "V6-motion-1",
            recording: "V6",
            discovery_hint: "Less overall movement",
          },
        ]
      : entries;
    const history = histories[motion ? "motion" : "original"];
    if (init?.method === "POST") {
      if (options.pending) return options.pending;
      if (options.fail)
        return {
          ok: false,
          json: async () => ({ issues: ["interval_review_changed_reload"] }),
        };
      const payload = JSON.parse(String(init.body));
      history.push({
        interval_id: String(url).split("?")[0].split("/").at(-1),
        payload,
        source: "manual",
        reviewed_at: "2026-10-26",
        status:
          payload.decision === "no_punch"
            ? "verified_background"
            : payload.decision === "punch_present"
              ? "needs_event_labeling"
              : "unresolved",
      });
      return { ok: true, json: async () => ({ data: {} }) };
    }
    const data =
      String(url).split("?")[0] === "/api/intervals"
        ? {
            manifest: { entries: activeEntries },
            head: (motion ? "b" : "a").repeat(64),
            history,
          }
        : {
            entry: activeEntries.find((e) =>
              String(url).split("?")[0].endsWith(e.id),
            ),
            head: (motion ? "b" : "a").repeat(64),
            history,
            annotations: options.conflict
              ? [
                  {
                    source_row: 1,
                    punch_type_provisional: "jab",
                    raw_start: 10,
                    raw_end: 12,
                  },
                ]
              : [],
            status: "unknown",
          };
    return {
      ok: true,
      json: async () => ({ status: "available", issues: [], data }),
    };
  });
  vi.stubGlobal("fetch", mock);
  render(<IntervalReview onDirty={vi.fn()} onSaving={vi.fn()} />);
  return mock;
}
const posts = (mock: ReturnType<typeof api>) =>
  mock.mock.calls.filter(([, init]) => init?.method === "POST");
async function watchAll() {
  await screen.findByText("Show frame 10");
  [10, 11, 12].forEach((n) =>
    fireEvent.click(screen.getByText(`Show frame ${n}`)),
  );
}

it("requires every frame and explicit whole-interval attestation, then saves without advancing", async () => {
  const mock = api({ second: true });
  await screen.findByRole("button", { name: "No punches" });
  fireEvent.click(screen.getByRole("button", { name: "No punches" }));
  expect(
    screen.getByLabelText(
      "I watched the entire interval and observed no punches.",
    ),
  ).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  expect(posts(mock)).toHaveLength(0);
  await watchAll();
  fireEvent.click(
    screen.getByLabelText(
      "I watched the entire interval and observed no punches.",
    ),
  );
  fireEvent.click(
    screen.getByLabelText(
      "The boxer is visible throughout, with no scene changes.",
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText("Review saved.");
  expect(posts(mock)).toHaveLength(1);
  const body = JSON.parse(String(posts(mock)[0][1]?.body));
  expect(body.inspected_frames).toEqual([10, 11, 12]);
  expect(body.decision).toBe("no_punch");
  expect(body).not.toHaveProperty("source");
  expect(body).not.toHaveProperty("reviewer");
  expect(
    screen.getByLabelText(
      "I watched the entire interval and observed no punches.",
    ),
  ).not.toBeChecked();
});

it("flags unannotated punches, advances only on successful save, and handles the final item", async () => {
  const mock = api();
  await watchAll();
  fireEvent.click(screen.getByRole("button", { name: "Punch present" }));
  expect(
    screen.getByLabelText("Did you see any punches that weren't marked?"),
  ).toBeInTheDocument();
  expect(
    within(
      screen.getByLabelText("Did you see any punches that weren't marked?"),
    )
      .getAllByRole("option")
      .map((o) => o.textContent),
  ).toEqual(["Not sure", "Yes", "No"]);
  fireEvent.change(
    screen.getByLabelText("Did you see any punches that weren't marked?"),
    { target: { value: "yes" } },
  );
  fireEvent.click(screen.getByRole("button", { name: "Save & Next" }));
  await screen.findByText("Review saved · End of queue.");
  expect(JSON.parse(String(posts(mock)[0][1]?.body))).toMatchObject({
    decision: "punch_present",
    unannotated_punches: "yes",
    watched_entire: false,
  });
});

it("keeps uncertainty unresolved and preserves edits after save failure", async () => {
  const mock = api({ fail: true });
  await watchAll();
  fireEvent.click(screen.getByRole("button", { name: "Not sure" }));
  fireEvent.change(screen.getByLabelText("Interval notes"), {
    target: { value: "Arm is hidden" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save & Next" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Another review was saved",
  );
  expect(screen.getByLabelText("Interval notes")).toHaveValue("Arm is hidden");
  expect(JSON.parse(String(posts(mock)[0][1]?.body)).decision).toBe("not_sure");
});

it("cannot certify background over a source punch", async () => {
  const mock = api({ conflict: true });
  await watchAll();
  fireEvent.click(screen.getByRole("button", { name: "No punches" }));
  expect(
    screen.getByLabelText(
      "I watched the entire interval and observed no punches.",
    ),
  ).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  expect(posts(mock)).toHaveLength(0);
  expect(screen.getByRole("alert")).toHaveTextContent("source punch overlaps");
});

it("prevents duplicate submissions while a save is pending", async () => {
  const mock = api({ pending: new Promise(() => {}) });
  await watchAll();
  fireEvent.click(screen.getByRole("button", { name: "Not sure" }));
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Saving…" })).toBeDisabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  expect(posts(mock)).toHaveLength(1);
});

it("keeps the motion pilot separate and saves to its own endpoint/head", async () => {
  const mock = api();
  await screen.findByText("Show frame 10");
  fireEvent.change(screen.getByRole("combobox", { name: "Review pilot" }), {
    target: { value: "motion-v2" },
  });
  await screen.findByRole("button", { name: /V6 · Background candidate 1/ });
  expect(
    screen.queryByRole("button", { name: /V4 · Background candidate/ }),
  ).not.toBeInTheDocument();
  expect(
    screen.getByText("Less overall movement · Unverified"),
  ).toBeInTheDocument();
  fireEvent.click(await screen.findByText("Show frame 10"));
  fireEvent.click(screen.getByRole("button", { name: "Not sure" }));
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText("Review saved.");
  expect(posts(mock)[0][0]).toBe("/api/intervals/V6-motion-1?pilot=motion-v2");
  expect(JSON.parse(String(posts(mock)[0][1]?.body)).expected_head).toBe(
    "b".repeat(64),
  );
  fireEvent.change(screen.getByRole("combobox", { name: "Review pilot" }), {
    target: { value: "original" },
  });
  await screen.findByRole("button", { name: /V4 · Background candidate 1/ });
  expect(
    screen.queryByText("Less overall movement · Unverified"),
  ).not.toBeInTheDocument();
});

it("enables explicit no-punch confirmation after complete playback without inventing exact inspection", async () => {
  const mock = api();
  await screen.findByText("Watch complete interval");
  fireEvent.click(screen.getByRole("button", { name: "No punches" }));
  expect(
    screen.getByLabelText(
      "I watched the entire interval and observed no punches.",
    ),
  ).toBeDisabled();
  fireEvent.click(screen.getByText("Watch complete interval"));
  expect(
    screen.getByText(/Interval viewed · 0 frames inspected individually/),
  ).toBeInTheDocument();
  const attest = screen.getByLabelText(
    "I watched the entire interval and observed no punches.",
  );
  expect(attest).toBeEnabled();
  expect(attest).not.toBeChecked();
  fireEvent.click(attest);
  fireEvent.click(
    screen.getByLabelText(
      "The boxer is visible throughout, with no scene changes.",
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText("Review saved.");
  const body = JSON.parse(String(posts(mock)[0][1]?.body));
  expect(body.inspected_frames).toEqual([]);
  expect(body.playback_coverage.schema_version).toBe("continuous-viewing-1.0");
});

it("offers only the six reserved blocks in activity mode and requests no predictions", async () => {
  const blocks = ["V7", "V8", "V10"].flatMap((recording) =>
    [1, 2].map((n) => ({
      ...entry,
      id: `${recording}-eval-${n}`,
      recording,
      role: "evaluation_reserved",
      purpose: "evaluation_block",
    })),
  );
  const mock = vi.fn(async (url: RequestInfo | URL) => {
    const path = String(url);
    if (path === "/api/intervals")
      return {
        ok: true,
        json: async () => ({
          data: {
            manifest: { entries: [entry, ...blocks] },
            history: [],
            head: "a".repeat(64),
          },
        }),
      };
    if (path === "/api/activity-evaluation/status")
      return { ok: true, json: async () => ({ data: {} }) };
    if (path.startsWith("/api/activity-evaluation/"))
      return {
        ok: true,
        json: async () => ({
          data: {
            head: "a".repeat(64),
            events: [],
            complete: false,
            history: [],
          },
        }),
      };
    const selected =
      [entry, ...blocks].find((e) => path.includes(e.id)) ?? entry;
    return {
      ok: true,
      json: async () => ({
        data: {
          entry: selected,
          annotations: [],
          history: [],
          head: "a".repeat(64),
          status: "unknown",
        },
      }),
    };
  });
  vi.stubGlobal("fetch", mock);
  render(<IntervalReview onDirty={vi.fn()} onSaving={vi.fn()} />);
  fireEvent.change(
    await screen.findByRole("combobox", { name: "Review pilot" }),
    { target: { value: "activity" } },
  );
  await screen.findByRole("button", { name: "Add activity" });
  expect(
    screen.getAllByRole("button", { name: /Activity block/ }),
  ).toHaveLength(6);
  expect(
    screen.queryByRole("button", { name: /Background candidate/ }),
  ).toBeNull();
  expect(screen.queryByRole("combobox", { name: "Punch type" })).toBeNull();
  expect(
    mock.mock.calls.some(([url]) => String(url).includes("prediction")),
  ).toBe(false);
});
