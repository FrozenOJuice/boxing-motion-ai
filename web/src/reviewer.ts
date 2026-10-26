import { useState } from "react";

export const reviewerKey = "boxing-motion-ai.reviewer.v1";

export function loadReviewer(): string {
  try {
    const name = window.localStorage.getItem(reviewerKey)?.trim() ?? "";
    return name.length <= 160 ? name : "";
  } catch {
    return "";
  }
}

/** Remember identity only. Decisions and attestations are never remembered. */
export function useReviewer() {
  const [reviewer, setReviewer] = useState(loadReviewer);
  const [editing, setEditing] = useState(!reviewer);
  const [storageError, setStorageError] = useState("");
  function update(name: string) {
    setReviewer(name);
    try {
      if (name.trim()) window.localStorage.setItem(reviewerKey, name.trim());
      else window.localStorage.removeItem(reviewerKey);
      setStorageError("");
    } catch {
      setStorageError(
        "Your browser could not remember your name. Allow local storage to keep it for future reviews.",
      );
    }
  }
  return { reviewer, update, editing, setEditing, storageError };
}
