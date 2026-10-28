# Buffered interval review playback

2026-10-27. Video Intervals now use locally cached H.264 MP4 playback at source speed, with 1×, 0.5× and 0.25× controls. Exact inspection remains tied to original decoded frames. The dashboard is running at http://127.0.0.1:8000; refresh the page to load the update. Existing review decisions and history remain unchanged.

## What changed

The previous player waited for a metadata request, a JPEG request, image decoding and an explicit per-frame timer before moving forward. Source-video keyframe seeking/decode happened separately for each uncached image. The result was serialized request/decode latency added to every frame's intended display duration.

`api/clips.py` decodes a registered interval plus its bounded context sequentially once and creates a silent, browser-compatible H.264/yuv420p derivative (up to 1280×720). It uses FFmpeg through the existing PyAV dependency, with two encoder threads; no new runtime dependency or download is required. Original MKV files are read-only and retain their hashes. Audio is omitted from this visual-review preview.

Source frames are selected by exact rational PTS, not nominal FPS or approximate seeking. The preview has one frame per selected source frame, including variable timestamp spacing. Source timestamps are rebased only in the derivative. Before publishing a clip, a separate decode verifies every MP4 frame's PTS and the final frame duration against the expected source timeline. A mismatch fails generation rather than inventing a mapping. Context cannot cross known cuts, estimator segments, missing frames or timestamp gaps. A final source frame lacking a following timeline PTS is omitted from context, never from the reviewed interval.

Fragmented MP4 places initialization before media for buffered playback. Byte-range responses support seeking; cache keys bind source video/pose hashes, interval ID, exact bounds, PTS and encoder version. Generation is locked across processes, writes are atomic, cache hashes are checked, and symlinks/unsafe paths are refused. No arbitrary video path or V9 route is exposed. Sources are revalidated even when a clip is cached.

The HTML video element handles normal playback without HTTP requests for individual frames. `requestVideoFrameCallback` maps the presented preview PTS to a verified source index; the timeline and source timestamp use that mapping. Previous/next frame and timeline inspection pause playback and retrieve the exact original frame through the existing decoder. Browser `currentTime` is not used as an exact boundary-verification claim. Frozen footage remains visible while an exact image loads; source skeletons are available while paused, avoiding an unsynchronized overlay during smooth playback. The existing stage dimensions and dark theme are retained.

The regular Punches screen continues using its existing player. Both players share the exact-frame image/overlay implementation. Extending smooth previews to arbitrary punch/context selections would require another bounded registration contract and is deferred.

## Review coverage and its limits

The backend verification contract is unchanged: `No punches` requires **every native frame** in the interval in `inspected_frames`, explicit whole-interval/visibility attestations, no conflicting source punch, and the existing source/timing checks. No new method can mark an elapsed time range as fully reviewed.

During buffered playback, only individually mapped frames from consecutive presentation callbacks are credited. Both the native index and compositor counter must advance by exactly one; seeks, missing callbacks, repeated frames and large timing jumps leave gaps. Paused/seeking/buffering playback receives no credit. Hiding the page or moving the player substantially outside the viewport pauses it and resets the callback sequence. More than one missing callback never produces a fabricated intervening frame range.

The [browser presentation callback specification](https://wicg.github.io/video-rvfc/) describes frames submitted for composition, not proof of physical display or human attention. Callbacks can be missed, and the browser cannot prove that a person watched. Evidence remains a conservative client report, as with the original image-viewer mechanism; explicit human attestation remains necessary. Viewport checks cannot detect every external window obscuring the browser.

If frames remain uncredited after playback, a small **Inspect next** action appears. It loads the next missing source frame through the exact decoder; repeat as needed. Quarter/half-speed replay can also recover missing callbacks. **Playback & inspection → Inspect next unreviewed frame** is always available. A browser without presentation callbacks can still play smoothly, but playback adds no coverage: use exact-frame inspection to satisfy the full set. Coverage resets on reopening, as before. No checkbox is selected automatically.

The actual normal-speed runs below left one frame uncredited; exact inspection recovered it. This is intentional conservative behavior rather than silently accepting a clock-based playback-complete flag. No actual review was saved or attested during these tests.

## Measured performance on actual footage

The bounded benchmark used V4, V5 and V10 background candidate 1, including context. Rates below come from the exact frame/timestamp mapping. Chrome ran headlessly against the real local dashboard, with its real React controls and local MP4s. Measurements cover playback after initial readiness/seek; encoding and initial buffering are separate.

| Recording | Native rate | Speed | Expected wall time | Measured wall time | Reported dropped frames |
|---|---:|---:|---:|---:|---:|
| V4 | 23.976 FPS | 1× | 5.0459 s | 5.0477 s | 0 |
| V5 | 29.970 FPS | 1× | 5.0371 s | 5.0422 s | 0 |
| V10 | 25 FPS | 1× | 5.0393 s | 5.0420 s | 0 |
| V5 | 29.970 FPS | 0.5× | 10.0756 s | 10.0778 s | 0 |
| V5 | 29.970 FPS | 0.25× | 20.1520 s | 20.1582 s | 0 |

All runs were within **0.11%** of expected wall time. Each reported one waiting event before the first playing event, none during measured playback. The single source-frame request recorded near the end of each run is paused exact inspection; there is no per-frame HTTP loop during playback. Callback coverage completed after targeted exact recovery in every run, with zero checked attestations. These short local tests are not a guarantee for every browser/display or under heavy load.

For comparison, twelve original metadata+JPEG pairs per recording measured mean server times (excluding the first request) of 5.4 ms for cached V4, **841 ms for V5**, and **144 ms for V10**. Adding the old native-frame timer alone limits those latter paths to approximately 1.14 and 5.44 FPS, before browser image decoding/rendering. Cache warmth differs, so this comparison diagnoses the serialized bottleneck rather than claiming a universal speedup factor.

All **36 pilot previews are pre-generated**: 371.642 seconds including context, 77.97 MB total generated MP4 data. Median generation/round-trip verification time was **2.85 seconds per clip**. Warm preparation lookup median was **10.4 ms**, maximum **49.5 ms**, in the local batch test. Cache hits do not encode again. A fresh installation or changed source/config must generate/validate its own previews; the UI displays preparation status while exact inspection remains available. Long evaluation blocks incur a one-time generation delay. Missing/corrupt inputs fail honestly; no source file or annotation is repaired.

## Reproduction and preservation

New routes: `GET /api/intervals/{registered-id}/playback` returns mapping/preview metadata; `GET /api/intervals/{registered-id}/clip` streams the cached MP4 with range support. Existing localhost/origin guards apply. Derivatives and receipts live under ignored `.dashboard-cache/clips/`. Generated benchmarks, scripts, source-frame checks and the actual dashboard screenshot are preserved under ignored `reports/boxingvi/buffered-playback-v1/`.

The ordinary launch remains:

```sh
npm --prefix web run build
.venv/bin/python -m boxing_motion_ai.api
```

Open Review → Video intervals. Play uses the MP4; pause/arrow keys/timeline clicks inspect exact frames. Use the speed select for slow motion. Checking Skeleton pauses for exact overlay inspection. If coverage is incomplete, use Inspect next before selecting the whole-interval attestation. Save and Save & Next retain their existing persistence behavior.

No classifier training, model/quality-policy changes, new human decisions or dataset modifications occurred. All **4,647 protected input hashes** matched after generation and browser tests; review histories remain intact. Existing experiment artifacts, the curated 1,123-example pool and RTMPose comparisons are preserved. V9 was never accessed.

Verification: **462 Python tests** passed; the **20 targeted clip/interval tests** passed after final byte-range handling changes. **55 frontend tests** passed, covering actual-frame mapping, seek/drop/buffering/visibility coverage gates, missing callback support, exact stepping, slow speed, failure fallback and save contracts. Ruff, TypeScript, formatting and production build pass. One existing Starlette/httpx deprecation warning remains. Real browser testing is isolated/headless; no Safari or visible browser window was opened.
