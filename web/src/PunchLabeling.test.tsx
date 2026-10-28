import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import PunchLabeling from "./PunchLabeling";

vi.mock("./BufferedIntervalViewer", async () => {
  const { useState } = await import("react");
  return {
    default: ({
      onVisible,
      onViewing,
      inspectionControls,
      humanEvents,
      onHumanSelect,
    }: any) => {
      const [frame, setFrame] = useState(10);
      return (
        <div>
          {[9, 10, 11, 12, 13].map((n) => (
            <button
              key={n}
              onClick={() => {
                setFrame(n);
                onVisible({ frame_index: n });
              }}
            >
              Inspect {n}
            </button>
          ))}
          {inspectionControls(frame, true)}
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
                      end: 0.2,
                      rate: 1,
                      wall_seconds: 0.2,
                      samples: 3,
                    },
                  ],
                },
              })
            }
          >
            Watch interval
          </button>
          {humanEvents.map((p: any, i: number) => (
            <button key={p.event_id} onClick={() => onHumanSelect(p.event_id)}>
              Human timeline {i + 1}
            </button>
          ))}
        </div>
      );
    },
  };
});
afterEach(() => vi.unstubAllGlobals());
const entry = {
  id: "V4-bg-1",
  recording: "V4",
  start: 10,
  stop: 13,
  context_start: 9,
  context_end: 14,
  frame_count: 100,
};
function setup(
  options: {
    fail?: boolean;
    failComplete?: boolean;
    pending?: Promise<void>;
  } = {},
) {
  let state: any = {
    head: "a".repeat(64),
    events: [],
    history: [],
    complete: false,
  };
  let fail = options.fail;
  const bodies: any[] = [];
  const mock = vi.fn(async (_url: any, init?: RequestInit) => {
    if (init?.method !== "POST")
      return { ok: true, json: async () => ({ data: state }) };
    const p = JSON.parse(init.body as string);
    bodies.push(p);
    if (options.pending) await options.pending;
    if (fail || (options.failComplete && p.operation === "complete")) {
      fail = false;
      return { ok: false, json: async () => ({ issues: ["failed"] }) };
    }
    const events =
      p.operation === "upsert"
        ? [
            ...state.events.filter((e: any) => e.event_id !== p.event.event_id),
            p.event,
          ]
        : p.operation === "remove"
          ? state.events.filter((e: any) => e.event_id !== p.event_id)
          : state.events;
    const record = {
      sha256: String(state.history.length + 1).repeat(64),
      events,
      complete: p.operation === "complete",
      reviewed_at: "2026-10-28T16:00:00Z",
      source: "manual",
      payload: p,
    };
    state = {
      head: record.sha256,
      events,
      complete: record.complete,
      history: [...state.history, record],
    };
    return { ok: true, json: async () => ({ data: record }) };
  });
  vi.stubGlobal("fetch", mock);
  const next = vi.fn(),
    dirty = vi.fn();
  const mount = () =>
    render(
      <PunchLabeling
        entry={entry}
        annotations={[]}
        onDirty={dirty}
        onSaving={vi.fn()}
        onNext={next}
        onProgress={vi.fn()}
      />,
    );
  const result = mount();
  return { bodies, next, dirty, result, mount };
}
async function add() {
  fireEvent.click(await screen.findByRole("button", { name: "Add punch" }));
  fireEvent.click(screen.getByRole("button", { name: "Inspect 10" }));
  fireEvent.click(screen.getByRole("button", { name: "Set start here" }));
  fireEvent.click(screen.getByRole("button", { name: "Inspect 11" }));
  fireEvent.click(screen.getByRole("button", { name: "Set end here" }));
  fireEvent.change(screen.getByRole("combobox", { name: "Punch type" }), {
    target: { value: "jab" },
  });
}
it("requires explicit boundaries/type and saves half-open human events without completion", async () => {
  const { bodies } = setup();
  fireEvent.click(await screen.findByRole("button", { name: "Add punch" }));
  fireEvent.click(screen.getByRole("button", { name: "Save punch" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Mark the first and last",
  );
  expect(bodies).toHaveLength(0);
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  await add();
  fireEvent.click(screen.getByRole("button", { name: "Save punch" }));
  await screen.findByText("Punch saved. Continue looking for others.");
  expect(bodies[0].event).toMatchObject({
    start: 10,
    stop: 12,
    punch_type: "jab",
    type_uncertain: false,
  });
  expect(bodies[0]).not.toHaveProperty("watched_entire");
  expect(
    screen.getByRole("button", { name: "Human timeline 1" }),
  ).toBeVisible();
});
it("edits to uncertain, persists across reload, removes without losing history", async () => {
  const { bodies, result, mount } = setup();
  await add();
  fireEvent.click(screen.getByRole("button", { name: "Save punch" }));
  await screen.findByText("Punch saved. Continue looking for others.");
  fireEvent.click(screen.getByRole("button", { name: "Edit punch 1" }));
  fireEvent.click(screen.getByRole("checkbox", { name: "Type uncertain" }));
  fireEvent.click(screen.getByRole("button", { name: "Save punch" }));
  await waitFor(() => expect(bodies).toHaveLength(2));
  expect(bodies[1].event).toMatchObject({
    punch_type: "unknown",
    type_uncertain: true,
  });
  await screen.findByRole("button", { name: "Edit punch 1" });
  result.unmount();
  mount();
  expect(await screen.findByText(/1\. Type uncertain/)).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Remove punch 1" }));
  await screen.findByText("Punch removed. Its history is preserved.");
  expect(bodies[2].operation).toBe("remove");
  expect(screen.queryByRole("button", { name: "Human timeline 1" })).toBeNull();
});
it("requires complete viewing and explicit attestation before Save complete & Next", async () => {
  const { bodies, next } = setup();
  await add();
  fireEvent.click(screen.getByRole("button", { name: "Save punch" }));
  await screen.findByText("Punch saved. Continue looking for others.");
  const checkbox = screen.getByRole("checkbox", {
    name: "I watched the entire interval and labeled every punch.",
  });
  expect(checkbox).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Watch interval" }));
  expect(checkbox).toBeEnabled();
  expect(checkbox).not.toBeChecked();
  expect(
    screen.getByRole("button", { name: "Save complete & Next" }),
  ).toBeDisabled();
  fireEvent.click(checkbox);
  fireEvent.click(screen.getByRole("button", { name: "Save complete & Next" }));
  await waitFor(() => expect(next).toHaveBeenCalledOnce());
  expect(bodies[1]).toMatchObject({
    operation: "complete",
    watched_entire: true,
    all_punches_labeled: true,
  });
  expect(bodies[1].playback_coverage.interval_id).toBe(entry.id);
});
it("retains a failed draft and retries idempotently without duplicate submissions", async () => {
  const { bodies, next } = setup({ fail: true });
  await add();
  fireEvent.click(screen.getByRole("button", { name: "Save punch" }));
  await screen.findByRole("alert");
  expect(screen.getByRole("combobox", { name: "Punch type" })).toHaveValue(
    "jab",
  );
  fireEvent.click(screen.getByRole("button", { name: "Save punch" }));
  fireEvent.click(screen.getByRole("button", { name: "Save punch" }));
  await screen.findByText("Punch saved. Continue looking for others.");
  expect(bodies).toHaveLength(2);
  expect(bodies[0].request_id).toBe(bodies[1].request_id);
  expect(next).not.toHaveBeenCalled();
});

it("never advances when completion persistence fails", async () => {
  const { next } = setup({ failComplete: true });
  await add();
  fireEvent.click(screen.getByRole("button", { name: "Save punch" }));
  await screen.findByText("Punch saved. Continue looking for others.");
  fireEvent.click(screen.getByRole("button", { name: "Watch interval" }));
  fireEvent.click(
    screen.getByRole("checkbox", {
      name: "I watched the entire interval and labeled every punch.",
    }),
  );
  fireEvent.click(screen.getByRole("button", { name: "Save complete & Next" }));
  await screen.findByRole("alert");
  expect(next).not.toHaveBeenCalled();
  expect(
    screen.getByRole("checkbox", {
      name: "I watched the entire interval and labeled every punch.",
    }),
  ).toBeChecked();
  expect(screen.getByText(/In progress/)).toBeVisible();
});
