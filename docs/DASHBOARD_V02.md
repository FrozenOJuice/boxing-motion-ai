# Dashboard v0.2 — Interactive Annotation Review

The [simplified Review experience](DASHBOARD_REVIEW_REDESIGN.md) supersedes the UI
workflow below. The persistence, safety and launch instructions remain current.

Implemented 2026-10-25. This adds human decisions to the local v0.1 viewer; it does
not run training, change original annotations or materialize corrected datasets.
The six disputes and twelve proposed anchors retain their original priorities.

## Launch

From the repository root:

```sh
uv sync --locked --extra dashboard
npm --prefix web ci
npm --prefix web run build
uv run --locked --extra dashboard python -m boxing_motion_ai.api
```

Restart an existing v0.1 backend to load the new routes, then open
http://127.0.0.1:8000 yourself. No browser is launched automatically. Frontend
HMR remains available with `npm --prefix web run dev` (loopback port 5174).

## Review workflow

1. Select **Review Queue** and filter to **Priority disputes** or **Proposed
   anchors**, or search a recording/source row. The complete ledger has 83 cases:
   the 79 diagnostic cases plus four precautionary neighbors. Review counts are
   live; a human hold is reviewed but is not accepted.
2. Use **Play frames**, frame stepping, the native-frame input and contextual
   frames to inspect the movement. Playback advances only after each exact image
   loads; it is decode-paced, not realtime, and has no audio. Pose overlays hide
   invalid joints. Source mappings and anatomical interpretation remain provisional.
3. Choose **Confirm**, **Ambiguous**, **Hold** or **Exclude**. Original labels and
   model predictions remain separately identified. Neither a prediction nor an
   assistant observation certifies the human decision.
4. If appropriate, propose a label and zero-based decoded **inclusive** start/end
   frames. Use **Use frame as start/end** or numeric inputs, then **Inspect proposed
   start/end**. Confirmation requires both endpoints to have loaded in this review,
   reviewer name, evidence/notes and explicit lead/rear or stance evidence. It also
   rejects a known scene/segment crossing. Seeing endpoints is necessary, not proof
   that the entire interval or label is correct: inspect the movement yourself.
5. Record uncertainty, required for Ambiguous/Hold. Unresolved proposed corrections
   remain proposals; both actions map to curation holds. Exclude requires a reason
   in the notes. Check the explicit personal-review attestation, then **Save review**
   or **Save & next annotation**. Editing fields clears the attestation.
6. Open **Review history** to see the original automated observations, every human
   revision and saved proposals. Revisit any case to revise it: another save appends
   a revision; it never replaces history. Previously proposed/confirmed fields are
   shown as a starting point, but no action or attestation is preselected.

Unsaved navigation prompts before discarding a draft. A concurrent save or changed
source produces an error and keeps the draft. Reload to obtain the current head,
then recheck and resubmit. A retry of the identical request is idempotent. Current
reviews record a manual source without a name; historical names remain preserved.

## Persistence and provenance

`services/review_workspace.py` wraps the existing DashboardService, ReviewLedger,
Revision, Bounds and CurationConfig contracts. A `LiveDashboardService` snapshot
reuses existing metadata loaders and `build_plan`; the CLI/training paths are unchanged.

Every successful transaction atomically publishes a new directory under:

```text
reports/boxingvi/dashboard-reviews-v1/v00000002/
  ledger.json
  config.json
  receipt.json
```

Subsequent saves increment the version. A process-shared file lock serializes writers;
ledger and revision hashes detect stale edits. Staging directories named `pending-*`
are not committed versions and cannot become active after a crash. No mutable head
pointer is necessary. Restart replays and validates the contiguous committed history.

The new config pins the new ledger while retaining original sources, group guards,
exclusions and policy hashes. Receipts include the original candidate fingerprint,
source input hashes, proposed fields, manual source and explicit attestation, server UTC
time, and verified native indices, exact PTS/time bases, scenes and segments for
attested displayed frames. The revision binds this audit content by hash. Displayed
frames are tracked only after image load (up to the last 512 unique frames); users
can revisit boundaries. No browser seeking or nominal-FPS conversion is used.

**Back up the complete `dashboard-reviews-v1/` directory.** It is durable review data,
although Git ignores it under the existing reports rule. It is distinct from
`.dashboard-cache/`, which holds disposable frame previews. Do not clean, reorder or
edit committed versions manually. Existing base ledger/config/catalogue files remain
unchanged. Malformed, missing, tampered or symlinked review storage fails closed.

## Safety and dataset effects

- Only registered cases may be written; V9 and arbitrary filesystem paths are refused.
- Writes require a local allowlisted Origin, JSON and the dashboard request header.
  The launcher binds loopback. This prevents ordinary cross-site browser writes; it
  is not authentication against other programs on the same machine.
- Request schemas reject unknown fields. The server chooses paths, actor type and
  timestamps. An explicit human attestation is mandatory; no automatic certification.
- Accepted changed labels/bounds are separate curation proposals and require a later
  controlled preparation step. Existing exclusions still win. Nothing restores a
  previously excluded row, rewrites poses/media or changes historical evaluations.
- Dataset statistics describe existing frozen provisional candidates. The curation
  proposal is recomputed from current reviews and pinned metadata; it is explicitly
  **metadata-only**, not a repeat of the full preservation-manifest/pose-byte audit.
  V5 remains development-only; V7/V8 and V1/V6 grouping precautions persist.
- No authentication, collaboration server, training, database, bulk certification or
  automatic label correction is introduced. Human visual judgment is still required.

## Verification

Run the existing full environment and frontend checks:

```sh
uv run --locked --extra pose --extra train --extra dashboard pytest --require-no-skips
uv run --locked --extra dashboard ruff check .
npm --prefix web test
npm --prefix web run typecheck
npm --prefix web run build
npm --prefix web run format:check
```

Synthetic tests cover process restart, revision history, source preservation,
idempotency, concurrent stale writers, interrupted publication, audit tampering,
unsafe storage, V9 rejection, exact-boundary/cut validation, CSRF checks, explicit
attestation, frontend save/refresh and conflict recovery. Real repository checks
are read-only: no genuine review case is certified by the implementation agent.

Verified milestone results: **369 pytest tests passed with no skips**, **11 frontend
tests passed**, Ruff, TypeScript, production build and Prettier passed. Pytest reports
one existing upstream Starlette/httpx deprecation warning. A new backend instance
loaded actual V4:497 and V5:262 cases and their exact start frames (PTS-derived times
452431/500 and 179413/1000 seconds respectively), plus the live curation proposal.
The real ledger remained version 1 with zero human reviews. Persistence and revision
writes were exercised only in temporary synthetic workspaces, including a separate
Python process restart. No browser window was opened during this verification.
