import type { Selection, VideoItem } from "./model";
import { recordedSelections } from "./model";

type StoredVideo = Omit<VideoItem, "src" | "poster"> & {
  file: Blob;
  thumbnail: Blob;
};
const RESULT_KEY = "boxing-motion-design-prototype:selections:v1";
function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("boxing-motion-design-prototype-v1", 1);
    request.onupgradeneeded = () =>
      request.result.createObjectStore("videos", { keyPath: "id" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(new Error("Recent videos are unavailable in this browser."));
  });
}
export async function recentVideos(): Promise<VideoItem[]> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction("videos", "readonly");
    const request = transaction.objectStore("videos").getAll();
    request.onsuccess = () =>
      resolve(
        (request.result as StoredVideo[])
          .map(({ file, thumbnail, ...item }) => ({
            ...item,
            src: URL.createObjectURL(file),
            poster: URL.createObjectURL(thumbnail),
          }))
          .sort((a, b) => b.openedAt - a.openedAt),
      );
    request.onerror = () => reject(new Error("Could not read recent videos."));
    transaction.oncomplete = () => db.close();
  });
}
export async function importVideo(
  file: File,
): Promise<{ item: VideoItem; stored: boolean }> {
  if (!file.size || file.size > 150 * 1024 * 1024)
    throw new Error("Choose a video under 150 MB.");
  if (!/\.(mp4|mov|webm|m4v)$/i.test(file.name))
    throw new Error("Choose an MP4, MOV, M4V or WebM video.");
  const src = URL.createObjectURL(file);
  const media = document.createElement("video");
  media.preload = "auto";
  media.muted = true;
  media.src = src;
  try {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(
        () =>
          reject(
            new Error("This video took too long to open. Try an MP4 export."),
          ),
        10000,
      );
      media.onloadeddata = () => {
        clearTimeout(timer);
        resolve();
      };
      media.onerror = () => {
        clearTimeout(timer);
        reject(
          new Error(
            "Your browser cannot play this video. Try an MP4 or WebM export.",
          ),
        );
      };
    });
    if (
      !Number.isFinite(media.duration) ||
      media.duration <= 0 ||
      media.duration > 30
    )
      throw new Error("Choose a video no longer than 30 seconds.");
    const canvas = document.createElement("canvas");
    canvas.width = 480;
    canvas.height = Math.round((480 * media.videoHeight) / media.videoWidth);
    canvas
      .getContext("2d")!
      .drawImage(media, 0, 0, canvas.width, canvas.height);
    const thumbnail = await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob(
        (blob) =>
          blob
            ? resolve(blob)
            : reject(new Error("Could not read a video thumbnail.")),
        "image/jpeg",
        0.8,
      ),
    );
    const item: VideoItem = {
      id: crypto.randomUUID(),
      title: file.name.replace(/\.[^.]+$/, ""),
      src,
      poster: URL.createObjectURL(thumbnail),
      duration: media.duration,
      width: media.videoWidth,
      height: media.videoHeight,
      kind: "local",
      openedAt: Date.now(),
    };
    let stored = true;
    try {
      const db = await database();
      await new Promise<void>((resolve, reject) => {
        const transaction = db.transaction("videos", "readwrite");
        const { src: _src, poster: _poster, ...value } = item;
        transaction.objectStore("videos").put({ ...value, file, thumbnail });
        transaction.oncomplete = () => {
          db.close();
          resolve();
        };
        transaction.onerror = () => {
          db.close();
          reject(transaction.error);
        };
      });
    } catch {
      stored = false;
    }
    return { item, stored };
  } catch (error) {
    URL.revokeObjectURL(src);
    throw error;
  } finally {
    media.removeAttribute("src");
    media.load();
  }
}
export function loadPreviews(): Selection[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(RESULT_KEY) ?? "[]");
    if (!Array.isArray(value)) return [];
    return value
      .filter(
        (p) =>
          p &&
          typeof p.id === "string" &&
          typeof p.videoId === "string" &&
          Number.isFinite(p.bounds?.start) &&
          Number.isFinite(p.bounds?.end) &&
          p.bounds.start >= 0 &&
          p.bounds.end > p.bounds.start,
      )
      .map((p) => ({
        id: p.id,
        videoId: p.videoId,
        bounds: p.bounds,
        kind: "preview",
        prediction: recordedSelections[0].prediction,
        scores: recordedSelections[0].scores,
      }));
  } catch {
    return [];
  }
}
export function savePreviews(selections: Selection[]) {
  try {
    localStorage.setItem(
      RESULT_KEY,
      JSON.stringify(
        selections
          .filter((p) => p.kind === "preview")
          .map(({ id, videoId, bounds }) => ({ id, videoId, bounds })),
      ),
    );
  } catch {
    /* Preview still works for this session. */
  }
}
