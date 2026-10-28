# Interval viewing coverage correction

2026-10-28. This supersedes the callback-counting coverage instructions in the original buffered-playback and interval-pilot notes.

## Root cause and corrected workflow

Reproduced in the actual Chrome dashboard with V4-motion-2: normal playback finished but coverage remained 1/96. The old player conflated consecutive compositor callbacks with individual native-frame inspection. Its 95% viewport gate could already be false when Play was pressed, allowing playback without credit. Missing callbacks also produced gaps despite watching normally.

Play now brings the footage into view. Continuous viewing is recorded as bounded, visible media-time spans at 1×, 0.5× or 0.25×, independently of exact-frame inspection. The dashboard shows **Interval viewed** and a separate number of **frames inspected individually**. Watching normally does not manufacture an `inspected_frames` list.

Watch the interval, select No punches, explicitly confirm the whole-interval and visibility statements, then Save or Save & Next. Attestations remain unchecked until the reviewer selects them. No source punch may conflict with a No punches decision. Seeking leaves skipped ranges uncovered; replay can cover them. Completed coverage is local to this viewing session and resets when switching intervals or reopening.

Pause flushes the final visible clock increment. Resume retains the exact media-clock position instead of seeking to a rounded frame PTS. Playback begins with an anchor before asynchronous `playing` events; visibility transitions close the previous epoch before invalidation. These details avoid small artificial gaps. Play waits for loaded media; Pause has its own pause-only action.

## Evidence and persistence

Optional `playback_coverage` is versioned as `continuous-viewing-1.0`, policy `visible-realtime-1.0`. Spans contain media start/end, playback rate, elapsed wall seconds and sample count. The server binds evidence to the registered interval and source-derived preview cache key, rejects impossible rates/ranges, and recomputes coverage against exact source PTS. Completion distinguishes continuous playback, exact inspection and their union. Numeric endpoint tolerance is 100 microseconds, not a missing-frame allowance.

Seeking, buffering, source/interval changes, hidden tabs, loss of window focus, offscreen footage, long sampling gaps and implausible clock jumps break continuity. Timers and media events sample progress; compositor callbacks only synchronize displayed source-frame information. Exact source decoding remains available for boundaries and skeletons.

This is evidence of continuous visible playback plus explicit human attestation, **not proof that every frame was individually inspected or that the person paid attention**. Browser visibility cannot detect every form of external occlusion. No punches certification still depends on the reviewer's explicit truthful confirmation. The protocol deliberately replaces the previous mandatory-all-frame-inspection gate for new decisions; it does not reinterpret old records or weaken source/attestation validation.

Legacy payloads omit the new optional field entirely. Their bytes, chain hashes, replay validation and idempotency are preserved. New decisions store a server-computed `viewing-validation-1.0` summary alongside evidence. No original review or annotation is rewritten.

## Verification

Actual Chrome, actual V4 motion-pilot footage:

- Reproduced the reported 1/96 failure before the fix.
- Normal complete playback enables the unchecked No punches confirmation and reports Interval viewed with only one individually inspected frame.
- Pause/resume, 0.5× playback and a change to 0.25× complete successfully without filling missing frames.
- Seeking to the end remains incomplete; replay from the beginning completes it.
- Switching intervals resets coverage.
- Native Chrome tab switching pauses playback; returning after background time leaves the fresh interval incomplete at 1%, rather than crediting hidden playback. This check was performed without debugger focus emulation, which otherwise makes background tabs appear visible to page APIs.
- No real review was saved or attested during testing. Browser screenshots and the protected-input hash inventory remain under ignored `reports/boxingvi/viewing-coverage-fix-20261028/`.

Automated regression tests cover source binding, gaps, invalid rates, explicit attestations, mixed speeds, exact/continuous separation, pause-event ordering, visibility/focus loss, persistence/restart, legacy records and idempotency. Original source files, review histories, historical experiment artifacts and V9 protections remain unchanged.

Final checks: **479 Python tests**, **70 frontend tests**, Ruff lint, TypeScript, frontend formatting, production build and diff whitespace checks pass. Changed Python files pass Ruff formatting. Repository-wide Ruff formatting reports a pre-existing quote-style difference in untouched `src/boxing_motion_ai/cli.py`; it was left unchanged. One existing Starlette/httpx deprecation warning remains. All **4,657 protected file hashes** match the pre-change snapshot, including both interval-review ledgers.
