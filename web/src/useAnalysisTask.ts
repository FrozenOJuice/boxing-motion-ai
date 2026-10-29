import { useEffect, useRef, useState } from "react";

export type Task<T> = {
  video_id: string;
  status: "not_started" | "processing" | "ready" | "failed";
  message?: string;
  result?: T;
  scan_wall_seconds?: number;
  analysis_wall_seconds?: number;
};

/** Shared local task lifecycle: stale responses never cross videos or replace a newer request. */
export function useAnalysisTask<T>(
  videoId: string,
  readPath: string,
  startPath: string,
  disabled: boolean,
  onBusy: (value: boolean) => void,
  refreshKey = 0,
) {
  const [value, setValue] = useState<Task<T> | null>(null);
  const task = value?.video_id === videoId ? value : null;
  const [error, setError] = useState("");
  const [requesting, setRequesting] = useState(false);
  const generation = useRef(0);
  const locked = useRef(false);
  const processing = requesting || task?.status === "processing";
  useEffect(() => {
    onBusy(processing);
    return () => onBusy(false);
  }, [processing, onBusy]);

  async function read(method: "GET" | "POST", signal?: AbortSignal) {
    const response = await fetch(
      `/api/analyze/videos/${videoId}/${method === "POST" ? startPath : readPath}`,
      method === "POST"
        ? {
            method,
            headers: {
              "Content-Type": "application/json",
              "X-Review-Client": "dashboard-v0.2",
            },
            body: "{}",
          }
        : { signal },
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.detail || "Analysis request failed. Retry.");
    if (result.video_id !== videoId)
      throw new Error("Analysis result does not match this video.");
    return result as Task<T>;
  }

  useEffect(() => {
    const controller = new AbortController(),
      current = ++generation.current;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const result = await read("GET", controller.signal);
        if (current !== generation.current) return;
        setValue(result);
        setError("");
        if (result.status === "processing") timer = setTimeout(poll, 1000);
      } catch (e) {
        if (!controller.signal.aborted && current === generation.current) {
          setError((e as Error).message);
          setValue(null);
        }
      }
    }
    void poll();
    return () => {
      generation.current++;
      controller.abort();
      clearTimeout(timer);
    };
  }, [videoId, readPath, startPath, task?.status === "processing", refreshKey]);

  async function start() {
    if (locked.current || disabled || processing) return;
    locked.current = true;
    const current = ++generation.current;
    setRequesting(true);
    setValue(null);
    setError("");
    try {
      const result = await read("POST");
      if (current === generation.current) setValue(result);
    } catch (e) {
      if (current === generation.current) {
        setError((e as Error).message);
        setValue(null);
      }
    } finally {
      locked.current = false;
      setRequesting(false);
    }
  }
  return { task, error, processing, start };
}
