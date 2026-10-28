# Continuous interval review pilot

Implemented 2026-10-26 from [the approved plan](CONTINUOUS_DETECTION_PLAN.md). This is review preparation, not a detection dataset release or a trained detector. All 36 prepared intervals initially remain **unknown**, with zero new human decisions. MediaPipe, original annotations, the 1,123-example curated pool, prior reviews/results and the RTMPose comparison remain unchanged. V9 is inaccessible.

## How to review

Restart the existing local backend and refresh the dashboard after rebuilding:

```sh
npm --prefix web run build
.venv/bin/python -m boxing_motion_ai.api
```

Open the existing local dashboard at **http://127.0.0.1:8000**, then **Review → Video intervals**. The original **Punches** tab is unchanged. Development remains available with `npm --prefix web run dev` on port 5174. No browser is opened automatically.

1. Select an interval. Its queue entry states Training candidate, Development only or Evaluation only.
2. Play or step through the footage. The highlighted timeline range is the interval being reviewed; surrounding frames are context only. Click the start marker to replay it. Skeletons remain optional. Play now uses a cached MP4 at source speed; paused stepping inspects exact original frames. See the [buffered playback update](BUFFERED_REVIEW_PLAYBACK.md) for coverage rules and measured performance.
3. Choose **No punches**, **Punch present** or **Not sure**.
4. **Save & Next** saves first, then advances. **Save** stays on the current interval. The final item reports End of queue. Save failures retain edits; duplicate submissions are blocked.

**No punches:** all interval frames must have been displayed in this review session, then explicitly check that you watched the entire interval and observed no punches, and that the boxer was visible without scene changes. Viewing frames does not check these boxes automatically. Optional tags describe ordinary movement, guard/recovery, defense, pause or feint; optional notes need no identifying information. Do not certify title cards, edits, hidden actions or ambiguous feints. Existing overlapping source punch annotations block negative certification and require conflict review instead.

**Punch present:** indicate whether any punches are missing from the source annotations (Yes / No / Not sure). This saves `needs_event_labeling`; it never becomes a negative and does not certify event labels, timings or exhaustive coverage. Existing source rows are available under **Source punch annotations**, including exclusion indicators. No excluded event is restored.

**Not sure:** saves unresolved uncertainty with optional notes. It remains excluded from negative examples. For an interval with only a small clear negative section, flag it for later targeted selection; this bounded version does not alter the immutable candidate boundaries.

Reopening shows prior decisions and notes. Revising creates another version; old decisions remain in history. Absence attestations start unchecked on reopening and require rewatching, so earlier evidence is never silently extended. The human source is `manual`; candidate selection is `automated`. No names/accounts are required. Local manual provenance is explicit user input, not authenticated identity.

## Prepared footage and split boundaries

| Recording | Background candidates | Continuous blocks | New pilot role |
|---|---:|---:|---|
| V4 | 5 | 0 | Training candidates |
| V6 | 5 | 0 | Training candidates; V1/V6 guard retained |
| V5 | 5 | 0 | Development only |
| V7 | 5 | 2 | Evaluation only; grouped with V8 |
| V8 | 5 | 2 | Evaluation only; grouped with V7 |
| V10 | 5 | 2 | Evaluation only |

There are **30 approximately four-second background candidates**, and **six continuous blocks totaling 179.526 seconds**, within the approved six-minute cap. The 15 background candidates on evaluation recordings also remain evaluation-only. Background windows and continuous blocks do not overlap and have at least a one-second separation. Known cuts, extraction discontinuities and PTS gaps are rejected. Unknown visual edits still need review.

To meet the request to keep evaluation separate from training/development, this pilot reserves **whole V7/V8 and V10 source groups** for future detector evaluation; V4/V6 supply prospective detector training and V5 development. This narrows the earlier six-recording evaluation proposal and stays below its time budget. It does **not** change historical clip-classifier splits, curation selections or experiments. Any future detector loader must consume this reservation and exclude all positives/negatives from those held-out groups. No training loader or existing training configuration was changed now. These historically investigated sources support diagnostic holdouts, not a fresh independent test or boxer-independent claim.

Candidate preparation is deterministic and label-neutral. Background windows are centered within the longest usable internal source-annotation gaps, with at most five per recording. Continuous blocks are selected by higher/lower *source annotation coverage*, not by detector performance or fabricated movement judgments. “Lower coverage” does not mean no punches; instructional footage, missing annotations and movement diversity remain review questions. Feints/defenses have not been claimed present. Use the optional tags to record actual observed content; missing categories will remain a readiness limitation.

## Storage and engineering

- `configs/interval_review_pilot_v1.json` pins the private manifest by SHA256.
- `reports/boxingvi/interval-review-pilot-v1/manifest.json` contains exact native half-open bounds, rational PTS endpoints, source/pose hashes, scene/segment IDs, null boxer/session identities, source-group guards, role and automated selection provenance. Original source files remain read-only.
- `reports/boxingvi/interval-reviews-v1/vNNNNNNNN/review.json` stores append-only manual decisions, full submitted evidence, notes, tags, uncertainty, timestamp, manifest/provenance, parent hash and request ID. Snapshots replay revisions and validate their semantics. Interrupted pending directories are not published decisions.
- The existing review workspace's POSIX no-symlink locking/traversal is reused with a separate fixed interval directory. Publishing uses atomic directory rename and fsync. Source changes, stale heads, invalid frames, conflicting annotations, unsafe paths and remote writes are rejected. Idempotent retries do not append duplicates.
- Existing exact-frame endpoints serve the viewer; no browser video seeking or new media decoder is introduced. Source annotations are reference information, never automatically human-confirmed. No model runs in this workflow.
- The interval ledger is separate because the punch contract's three independent confirmations cannot represent absence across an interval. Existing punch-review records and their revision hashes are untouched.

New API surface: `GET /api/intervals`, `GET /api/intervals/{registered-id}`, `POST /api/intervals/{registered-id}` under the existing localhost/origin/JSON guards. IDs resolve only to the pinned manifest, never caller-selected paths. Saved no-punch evidence requires the entire native frame set and explicit strict boolean attestations. This records the user's assertion; client evidence cannot prove attention or eliminate human error.

To reproduce preparation on the same registered inputs when the private manifest is absent:

```sh
.venv/bin/python -m boxing_motion_ai.data.interval_pilot
```

Preparation refuses to replace an existing manifest and verifies an existing config pin before rebuilding. Source/config changes require a separately versioned pilot, not overwriting this one. All manifests, reviews, images and caches remain Git-ignored; only code, tests, the small pinned config and documentation are committed.

## Readiness and limits

Ready for human interval review; **not ready for detector training or complete event-level evaluation**. No new background has yet been human-certified. Positive continuous blocks still need event enumeration and timing review after Punch present, including unannotated punches and overlaps. The three-way pilot intentionally flags that work rather than certifying an incomplete reference. There is no event-editing platform, automatic negative labeling, classifier training or new architecture.

Verification includes synthetic all-three-path saves, persistence through a fresh app instance, immutable revision history, stale/idempotent requests, partial-view rejection, source conflict, unsafe storage, split restrictions, timing/cut/gap checks and frontend save/failure behavior. Read-only real-footage checks loaded all 36 interval details and verified 72 exact boundary images against native frame IDs and source PTS. No POST was sent to the real review workspace. Frontend interaction tests use synthetic fixtures; no browser-based human review is claimed.

Checks: **457 Python tests passed**, **48 frontend tests passed**, Ruff passed, TypeScript passed, Python/frontend formatting passed, production frontend build passed, and `git diff --check` passed. Python reported one existing Starlette/httpx deprecation warning. All **4,647 protected hashes matched**, and the real interval ledger remained empty after verification.
