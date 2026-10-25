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
export type Revision = {
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
export function useRequest<T>(url: string | null) {
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
        if (!controller.signal.aborted) setState({ key: url, data });
      })
      .catch((error) => {
        if (!controller.signal.aborted)
          setState({ key: url, error: String(error.message) });
      });
    return () => controller.abort();
  }, [url]);
  return state.key === url ? state : { key: url };
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
