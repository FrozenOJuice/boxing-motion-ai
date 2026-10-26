import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";
afterEach(cleanup);

// Node 26 exposes an unavailable localStorage global without --localstorage-file.
// Supply the browser Storage contract for jsdom, retaining values across remounts.
const stored = new Map<string, string>();
Object.defineProperty(window, "localStorage", {
  configurable: true,
  value: {
    get length() {
      return stored.size;
    },
    key: (index: number) => [...stored.keys()][index] ?? null,
    getItem: (key: string) => stored.get(key) ?? null,
    setItem: (key: string, value: string) => {
      stored.set(key, String(value));
    },
    removeItem: (key: string) => {
      stored.delete(key);
    },
    clear: () => {
      stored.clear();
    },
  } satisfies Storage,
});
