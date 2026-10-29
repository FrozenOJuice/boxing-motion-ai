import { createHash } from "node:crypto";
import { readFile, copyFile, cp, mkdir } from "node:fs/promises";
import { resolve, join } from "node:path";

// Copy only local preview artifacts. Never write to the production upload or research folders.
const source = resolve(
  process.argv[2] ?? "../.analysis-local/a0e809913dc44d229bcc6f15bb98ad2d",
);
const target = resolve("public/demo");
const expected = JSON.parse(await readFile("src/demo-data.json", "utf8"));
const video = JSON.parse(await readFile(join(source, "video.json"), "utf8"));
const digest = (value) => createHash("sha256").update(value).digest("hex");
if (
  video.source_sha256 !== expected.source_sha256 ||
  digest(await readFile(join(source, "preview.mp4"))) !==
    expected.preview_sha256
) {
  throw new Error(
    "This is not the reference clip. Choose its original local upload directory.",
  );
}
await mkdir(target, { recursive: true });
await copyFile(join(source, "preview.mp4"), join(target, "shadowboxing.mp4"));
await cp(join(source, "frames"), join(target, "frames"), { recursive: true });
await copyFile(join(source, "frames/0034.png"), join(target, "poster.png"));
await import("node:fs/promises").then(({ writeFile }) =>
  writeFile(
    join(target, "manifest.json"),
    JSON.stringify({ source_sha256: expected.source_sha256 }),
  ),
);
console.log(
  "Local sample prepared. Media stays ignored by Git; no production files were modified.",
);
