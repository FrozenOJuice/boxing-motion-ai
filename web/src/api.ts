import { useEffect, useState } from "react";
export type Result<T> = { status: string; issues: string[]; data: T | null };
export type Source = {
  source_row: number;
  video_id: string;
  source_class: string;
  punch_type_provisional: string;
  raw_start: number | null;
  raw_end: number | null;
  candidate_start_index: number | null;
  candidate_end_index: number | null;
  annotation_end_assumption: string;
  flags?: string[];
  exclusion_reasons?: string[];
};
export type VerificationChecks = {
  punch_type: boolean;
  hand: boolean;
  timing: boolean;
};
export type Verification = VerificationChecks & {
  schema_version: "annotation-verification-1.0";
  label: string;
  bounds: Bounds | null;
  origins: Record<string, string>;
};
export type Revision = {
  verification?: Verification;
  revision_id?: string;
  reviewed_at?: string;
  confirmed_label?: string | null;
  confirmed_bounds?: Bounds | null;
  actor_type: string;
  reviewer: string;
  disposition: string;
  label_status: string;
  boundary_status: string;
  reasons: string[];
  unresolved_uncertainty: string[];
};
export type Review = {
  source: Source;
  latest: Revision | null;
  human_accepted: boolean;
  verification_status?: "fully_verified" | "partially_verified" | "unverified";
  latest_action?: string;
};
export type QueueEntry = {
  id: string;
  priority: string;
  review: Result<Review>;
};
export type Reviews = {
  entries: QueueEntry[];
  summary: {
    case_count: number;
    human_accepted: number;
    human_reviewed?: number;
    dispositions: Record<string, number>;
  };
};
export type Recording = {
  recording_id: string;
  role: string;
  source_group: string;
  boxer_id: string | null;
  session_id: string | null;
  preparation: { source_frame_count: number };
};
export type Inventory = {
  records: Record<string, Result<Recording>>;
  available_recordings: number;
};
export type Case = {
  edit?: Result<EditState>;
  annotation: Source;
  labels: string[];
  review: Result<Review>;
  recording: Result<Recording>;
  evidence: Result<{
    case: {
      prediction?: {
        prediction: string;
        target: string;
        probabilities: number[];
      };
      selection?: string;
      prediction_source?: string;
      selection_fold?: string;
      exact_pts?: string[];
    };
  }>;
};
export type Point = {
  name: string;
  xyz: (number | null)[];
  confidence: (number | null)[];
  valid: boolean;
};
export type Frame = {
  frame_index: number;
  pts: string;
  time_base: string;
  exact_timestamp: string;
  width: number;
  height: number;
  image_url: string;
  landmarks: Point[];
  edges: [number, number][];
  scene_id: number;
  segment_id: number;
  scene_cut_before: boolean;
};
export type Stats = {
  approved_provisional_examples: number;
  approved_by_class: Record<string, number>;
  annotation_rows: number;
  role: string;
  absent_approved_classes: string[];
};
export type Experiments = {
  items: Record<
    string,
    { status: string; description: string; metric_ids: string[] }
  >;
};
export type Scores = {
  accuracy: number;
  macro_f1: number;
  labels: string[];
  confusion_matrix: number[][];
  per_class: Record<
    string,
    { precision: number; recall: number; f1: number; support: number }
  >;
};
export type Metrics = {
  metrics: { selected_epoch: number; train: Scores; development: Scores };
  absent_classes: Record<string, string[]>;
};
export async function request<T>(
  url: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(url, { signal });
  const value = await response.json();
  if (!response.ok)
    throw new Error(
      value.issues?.join(", ") || value.detail || `HTTP ${response.status}`,
    );
  return value as T;
}
export function useRequest<T>(url: string | null, refresh = 0) {
  const key = `${url}:${refresh}`;
  const [state, setState] = useState<{
    key: string | null;
    data?: T;
    error?: string;
  }>({ key: null });
  useEffect(() => {
    if (!url) return;
    const controller = new AbortController();
    request<T>(url, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setState({ key, data });
      })
      .catch((error) => {
        if (!controller.signal.aborted)
          setState({ key, error: String(error.message) });
      });
    return () => controller.abort();
  }, [url, refresh]);
  return state.key === key ? state : { key };
}
export const title = (value: string) => value.replaceAll("_", " ");
export const percent = (n: number | undefined) =>
  n === undefined ? "Unavailable" : `${(n * 100).toFixed(1)}%`;
export function frameBounds(
  source: Source,
  count: number,
): [number, number] | null {
  const start = source.candidate_start_index,
    rawEnd = source.candidate_end_index;
  if (
    start === null ||
    rawEnd === null ||
    !Number.isInteger(start) ||
    !Number.isInteger(rawEnd)
  )
    return null;
  const convention = /^(inclusive|exclusive)(?:[ ;]|$)/.exec(
    source.annotation_end_assumption,
  );
  if (!convention) return null;
  const end = rawEnd - (convention[1] === "exclusive" ? 1 : 0);
  return start >= 0 && end >= start && end < count ? [start, end] : null;
}

export type Bounds = {
  start: number;
  end: number;
  convention: "zero_based_decoded_inclusive";
};
export type EditState = {
  verification?: VerificationChecks;
  ledger_sha256: string;
  revision_sha256: string | null;
  source_candidate_sha256: string;
  ledger_version: number;
  history: Revision[];
  decisions: { decision: Record<string, unknown>; saved_at: string }[];
};
export async function saveReview(
  recording: string,
  row: number,
  decision: object,
) {
  const response = await fetch(`/api/reviews/${recording}/${row}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Review-Client": "dashboard-v0.2",
    },
    body: JSON.stringify(decision),
  });
  const value = await response.json();
  if (!response.ok) {
    const messages: Record<string, string> = {
      inspect_punch_before_verifying:
        "Display a frame within the punch before verifying its type or hand.",
      select_at_least_one_confirmation:
        "Select at least one aspect you verified.",
      inspect_both_proposed_boundaries:
        "View both start and end frames before confirming.",
      proposed_bounds_out_of_range:
        "The start/end frames must be within the recording.",
      confirmation_crosses_known_cut:
        "This interval crosses a scene cut. Adjust the boundaries or flag it as uncertain.",
      confirmation_crosses_scene_or_segment:
        "The boundaries belong to different scenes. Adjust them or flag uncertainty.",
      lead_rear_evidence_required:
        "Explicitly verify the lead/rear arm before confirming.",
      unresolved_uncertainty_required: "Choose a reason for uncertainty.",
      review_source_unavailable:
        "The source annotation is unavailable. Reload the case.",
    };
    const code = value.issues?.[0];
    throw new Error(
      response.status === 409
        ? "Review changed or source verification failed. Reload the case before retrying; your draft has not been saved."
        : (messages[code] ??
            (response.status === 422
              ? "Review could not be saved. Check your name, reason and frame boundaries."
              : "Review could not be saved. Please try again; your draft has been kept.")),
    );
  }
  return value;
}

/** Display grouping only. Never used as a persisted curation disposition. */
export function reviewCategory(
  entry: QueueEntry,
): "To review" | "Reviewed" | "Needs attention" | "Partially verified" {
  const review = entry.review.data;
  if (entry.review.status !== "available" || !review) return "Needs attention";
  const latest = review.latest;
  if (review.verification_status === "partially_verified")
    return "Partially verified";
  if (
    latest?.actor_type === "human" &&
    ["accept", "exclude"].includes(latest.disposition)
  )
    return "Reviewed";
  if (
    entry.priority === "dispute" ||
    (latest && ["hold", "exclude"].includes(latest.disposition)) ||
    latest?.label_status === "ambiguous" ||
    ["ambiguous", "mixed_action"].includes(latest?.boundary_status ?? "")
  )
    return "Needs attention";
  return "To review";
}
