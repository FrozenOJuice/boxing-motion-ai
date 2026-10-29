import { useCallback, useEffect, useState } from "react";
import type { Draft } from "./selection";
export type InspectorTab = "selection" | "saved";
export type WorkspaceSession = {
  current: number;
  draft: Draft;
  resultId: string | null;
  tab: InspectorTab;
};
export type Route =
  | { page: "videos" }
  | { page: "analyze"; videoId: string }
  | { page: "missing" };
const SESSION_KEY = "boxing-motion-design-prototype:workspace:v1:";
export function readSession(id: string): WorkspaceSession | undefined {
  try {
    const s = JSON.parse(sessionStorage.getItem(SESSION_KEY + id) ?? "null");
    const boundary = (n: unknown) =>
      n === null || (typeof n === "number" && Number.isFinite(n) && n >= 0);
    if (
      s &&
      Number.isFinite(s.current) &&
      s.current >= 0 &&
      s.draft &&
      boundary(s.draft.start) &&
      boundary(s.draft.end) &&
      (s.resultId === null || typeof s.resultId === "string") &&
      ["selection", "saved"].includes(s.tab)
    )
      return s;
  } catch {
    /* Session restoration is optional when browser storage is unavailable. */
  }
}
export function writeSession(id: string, state: WorkspaceSession) {
  try {
    sessionStorage.setItem(SESSION_KEY + id, JSON.stringify(state));
  } catch {
    /* App memory still preserves navigation state. */
  }
}
function routeAtLocation(): Route {
  if (
    location.pathname === "/" ||
    location.pathname === "/videos" ||
    location.pathname === "/videos/"
  )
    return { page: "videos" };
  const match = location.pathname.match(/^\/videos\/([^/]+)\/?$/);
  try {
    if (match)
      return { page: "analyze", videoId: decodeURIComponent(match[1]) };
  } catch {
    /* malformed URL */
  }
  return { page: "missing" };
}
export function useNavigation() {
  const [route, setRoute] = useState<Route>(routeAtLocation);
  useEffect(() => {
    if (location.pathname === "/") history.replaceState(null, "", "/videos");
    const pop = () => setRoute(routeAtLocation());
    window.addEventListener("popstate", pop);
    return () => window.removeEventListener("popstate", pop);
  }, []);
  const navigate = useCallback((path: string) => {
    if (location.pathname !== path) history.pushState(null, "", path);
    setRoute(routeAtLocation());
  }, []);
  return { route, navigate };
}
