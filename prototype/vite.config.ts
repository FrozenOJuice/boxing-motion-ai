import { defineConfig } from "vitest/config";
export default defineConfig({
  server: { host: "127.0.0.1", port: 5186, strictPort: true },
  preview: { host: "127.0.0.1", port: 5186, strictPort: true },
  build: {
    rolldownOptions: { input: { app: "index.html", review: "review.html" } },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test-setup.ts",
    clearMocks: true,
  },
});
