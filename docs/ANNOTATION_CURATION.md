# Versioned annotation curation before further Level 1 experiments

The review process and reproducible exclusion proposal build on the accepted
[79-case diagnosis](LEVEL1_LABEL_POSE_DIAGNOSIS.md). The dataset
is **not yet ready for another controlled experiment**: all 18 targeted human reviews
are pending, with no human-confirmed labels or boundaries. This milestone performs
no training and does not activate a revised selection in any existing experiment.
Pose-only XY + validity-gated velocity + mirror remains the reference.

## Separate source, evidence, decisions and proposed selection

1. Original workbooks/media, prepared poses/candidates and approved mask-aware
   policy indexes are immutable inputs. Existing experiments remain unchanged.
2. The accepted 79-case diagnosis stays in `configs/level1_label_pose_review.yaml`.
   Its assistant observations are evidence, not human certification. The existing
   18-case gallery is reused; no new footage, pose extraction or model is required.
3. `configs/annotation_reviews_v1.json` is the initial versioned decision ledger.
   Each recording/source-row ID is bound to a SHA256 of its original prepared
   candidate. Entries have reviewer attribution, timezone-aware review time,
   label/boundary findings, disposition, reasons, evidence and unresolved uncertainty.
   Every initial entry is explicitly attributed to **assistant diagnostic intake**.
4. `configs/level1_curated_proposed.json` pins the ledger, diagnosis, 79-case manifest,
   existing preparation/pose/candidate files and approved policy indexes by SHA256.
   It produces an ignored `curated-selection.json` proposal and class-count report.
   It is a dataset selection configuration, **not a training configuration**.

The ledger tracks the 79 diagnostic cases plus four explicitly **unreviewed
precautionary neighbors**, V10 rows 148/150/151/152. Those four are not additional
visual findings. They complete the same late-uppercut block as disputed rows
146/147 and reviewed rows 149/153, avoiding an unsupported decision to retain an
uninspected interleaved example while holding its neighbors. Their origin, rationale
and uncertainty are explicit. No wrong label is asserted for them.

Initial dispositions: **42 provisional, 41 held**, across 83 entries. Seven held
entries were already excluded; they are not counted again as new losses. There are
**zero human confirmations, acceptances or automatic corrections**.

## Review contract

Label status is `unreviewed`, `ambiguous` or `confirmed`. Boundary status is
`unreviewed`, `ambiguous`, `mixed_action` or `confirmed`. The latter carries explicit
**zero-based decoded frame indices, inclusive end**, not a newly asserted author
convention. A reviewed interval can differ from the source interval, but is stored
only as a separate finding.

Disposition is independent:

- `provisional`: retain only if the unchanged original policy already approved it;
  this is never a clean or human-certified label.
- `hold`: exclude from the proposed training selection while uncertainty remains.
  No guessed label or replacement boundary is required.
- `exclude`: record an exclusion and its reason; a human may make a definitive
  exclusion without claiming a replacement label.
- `accept`: requires human-confirmed label **and** boundaries, native-frame review
  evidence and a stance/lead-rear explanation. An original quality/annotation
  exclusion still takes precedence. Acceptance does not restore an excluded row.

Any confirmed axis requires an explicit manual review, timezone-aware review time and
native frame evidence. New manual records require no personal name; old names remain
in history. Automated observations cannot confirm any axis. A class confirmation additionally requires lead/rear evidence;
"model predicted this" or estimated left/right channels alone do not suffice.
The append command requires `--attest-human-review` for human attribution. This is
an explicit provenance assertion by the person entering the decision, not identity
authentication or proof that their judgment is correct. Blank decision templates
cannot be submitted as confirmations.

Each revision links the previous revision's canonical SHA256. The append command
creates a **new** ledger file/version with its parent file hash, preserving prior
histories. Existing files cannot be overwritten. Pinned configurations reject stale
or manually altered inputs. Local Git preserves the published versions; this is
not a cryptographically authenticated external review service.

Confirmed changed labels/bounds are visible in the report but **never applied** to
existing candidates. Accepted changes block readiness until a separately reviewed
preparation step materializes them; this milestone implements no such rewriting.
Holds can remain unresolved indefinitely. No smoothing, joint filling, quality
threshold change or `no_punch` assignment occurs.

## Proposed losses, computed from actual approved policy selections

The default conservative proposal holds every reviewed ambiguous-family,
ambiguous/mixed-boundary case, all six priority disputes, the existing exclusions,
and the entire V10 late-uppercut block 146–153. It does not extrapolate new labels
to the rest of a recording. Most unreviewed approved examples remain **provisional**.

Counts below are **proposed retained** training examples; changes show original →
proposed. They do not describe files being deleted or current experiments changing.

| Recording | Jab | Cross | Lead hook | Rear hook | Lead uppercut | Rear uppercut | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| V4 | 117 | 39 | 140 | 73 → 70 | 71 → 70 | 87 → 84 | 527 → 520 |
| V6 | 0 | 0 | 64 → 63 | 32 | 125 → 124 | 81 → 78 | 302 → 297 |
| V7 | 8 | 1 | 20 | 1 → 0 | 36 → 31 | 0 | 66 → 60 |
| V8 | 16 | 21 | 54 → 53 | 0 | 21 | 8 → 4 | 120 → 115 |
| V10 | 49 | 36 | 25 | 6 | 4 → 0 | 21 → 17 | 141 → 133 |
| Training total | 190 | 97 | 303 → 301 | 112 → 108 | 257 → 246 | 197 → 183 | **1156 → 1125** |

That is **31 fewer training candidates (2.7%)**. All six classes remain present in
the combined pool. Important source-specific losses are explicit: V7 loses its sole
rear hook; V10 loses all four lead uppercuts; V8 loses half its rear uppercuts.
Do not restore these merely to balance counts.

The narrower 79-case-only scenario would remove 27 training candidates and retain
1,129. The four precautionary neighbors account for the difference. The machine
report exposes `diagnostic_79_only_after` beside the default counts, so the block
hold is not hidden in the total.

**V5 remains development-only with all 583 original eligible examples**, preserving
the frozen benchmark. Rows **432 (LH), 196 (LU), 514 (RU)** are shadow holds, one per
class: a shadow-clean comparison would retain 580, but it is not activated or used
for model selection here. Reported V5 scores remain provisional source-label scores,
not independently certified accuracy. Any future benchmark change must be versioned
and compared on identical examples; do not silently replace the old development set.

V7/V8 remain the same source group; V1/V6 guard remains (V1 is not loaded). Unknown
boxer/session identities remain unknown. V9 is not opened, decoded, hashed or used.
These are recording-group controls, not evidence of boxer independence.

## Minimum human work and readiness gate

Start with the [existing 18-case visual queue](../reports/boxingvi/level1-label-pose-assessment-v2/human-review.html).
Priority order:

| Case | Minimum question | Safe unresolved outcome |
|---|---|---|
| V10:146 / V10:147, compared with anchor V10:44 | Does stance actually change, or does the late block use a different lead/rear convention? | Keep all 146–153 held; do not swap labels. |
| V4:497 | Outgoing uppercut or defensive cover/block? | Hold. |
| V6:613 | Does the window include preceding straight-punch recovery plus a different rising action? | Mixed-action hold; no invented onset. |
| V7:136 | Is there a distinct lead uppercut rather than straight/overhand and recovery? | Ambiguous/mixed-action hold. |
| V8:82 | Which action phase belongs inside the rear-uppercut interval? | Hold boundaries; no global offset. |

Then inspect the original 12 candidate anchors (two per source class):

- Jab: V4:144, V7:8. Cross: V8:51, V10:71.
- Lead hook: V4:343, V6:7. Rear hook: V6:46, V10:142.
- Lead uppercut: V6:264, V8:159. Rear uppercut: V6:266, V10:44.

The minimum operational gate is **18 short case reviews**, not a full-dataset
review. The six disputes may be explicitly held/excluded by the human without
resolving their labels. Each anchor also may be held; there must be **at least one
accepted unchanged eligible anchor per class** among the 12 before readiness clears.
Aim for both per class. If an entire class cannot be verified, remain blocked rather
than certifying it to satisfy a quota. A replacement review protocol would require
an explicitly revised, pinned evidence/configuration version.

Inspect raw motion before the source label/prediction, then active-arm localization,
stance and interval inclusion. Record exactly what was inspected, not all rendered
frames by default. The source's own HUD can disclose exercise labels, so this is not
fully blinded. A possible stance switch is not automatically a label error. Global
camera reflection and author boundary conventions may remain unknown even after
accepting a local example.

These anchors are annotation-quality checks. They are not a new independent test
set: some participate in a future combined training pool, and they were selected
after inspecting model errors. Their diagnostic evaluation is valid only with the
preserved corresponding group-held-out checkpoints. Future model evaluation must
still respect source-group holdouts and development-only V5.

Even after the mechanical gate clears, the result is only
`ready_for_provisional_experiment_review`; training remains disabled and the
majority of labels remain unreviewed. Do not call the whole dataset human-verified.
Before interpreting a future V5 improvement as true six-class recognition, additionally
verify its local stance/phase contract (V5:58 and V5:121 are existing contrast examples),
or explicitly retain that limitation. No new footage inspection is required here.

## Commands and review workflow

No new dependencies are needed beyond the existing pose extra (PyYAML is used to
read the prior diagnosis). Use a fresh ignored output directory on every run:

```sh
uv run --extra pose boxing-motion-ai curate-annotations \
  --config configs/level1_curated_proposed.json \
  --output reports/boxingvi/annotation-curation-replay
```

Exit **1 is expected now**: human reviews/verified class coverage are incomplete.
Exit 2 indicates malformed/stale/missing input; 0 would mean the review gate clears,
not permission or an instruction to train. The report includes per-recording/class
counts, baseline exclusions and reasons, shadow holds, proposed selected row IDs,
unapplied confirmations, pending cases and provenance. Old outputs are not overwritten.

1. Copy the relevant `decision-templates/V4-497.json` (or another queued ID) into a
   new local decision file. Set `source: manual` (a legacy reviewer name is optional),
   timezone-aware `reviewed_at`, actual
   evidence description/frame indices, reasons and uncertainty. The default is an
   unresolved hold, not acceptance. For acceptance, explicitly set both statuses to
   `confirmed`, supply `confirmed_label`, `lead_rear_evidence`, and
   `confirmed_bounds: {"start": ..., "end": ..., "convention": "zero_based_decoded_inclusive"}`.
2. Publish a new ledger version, using the latest ledger for subsequent decisions:

   ```sh
   uv run boxing-motion-ai record-annotation-review \
     --ledger configs/annotation_reviews_v1.json \
     --decision reports/my-V4-497-review.json \
     --output configs/annotation_reviews_v2.json \
     --attest-human-review
   ```

   The template contains the current per-case `supersedes_sha256`; it stays valid
   when other cases are reviewed, but must be regenerated if this same case changes.
   The command prints the new ledger hash. Do not claim human attribution for
   assistant observations or run unfilled templates through the confirmation path.
3. Create a **new** proposed configuration from the existing one, updating only its
   `review_ledger.path` and SHA256 to the published ledger. Leave every other pinned
   input and source/development rule unchanged. Validate by rerunning the curation
   command into another fresh report directory. Keep the prior configuration and
   ledger; never edit the original workbooks/candidates to make the plan pass.

For example, after publishing the new ledger:

```sh
uv run python - <<'PY'
import json
from boxing_motion_ai.data.annotation_review import file_hash, write_new
c = json.load(open('configs/level1_curated_proposed.json'))
p = 'configs/annotation_reviews_v2.json'
c['review_ledger'] = {'path': p, 'sha256': file_hash(p)}
write_new('configs/level1_curated_proposed_v2.json', c)
PY
```

The CLI supports `--help` for both commands. No training loader is silently wired
to this proposal. After human review, a separately approved experiment must consume
and validate the pinned selected-row manifest explicitly.

## Validation and preservation

Synthetic tests cover confirmation requirements, explicit human attribution,
append-only histories, stale pins, canonical case IDs, V9/group guards, reversed or
out-of-video bounds/evidence, original-exclusion precedence, unapplied changed labels,
shadow-only V5 holds, scenario counts, portable reports and deterministic replay.

```sh
uv run --extra pose --extra train pytest
uv run ruff check .
```

The curation run verifies the same **4,896 prior files** before/after report creation,
including source media/workbooks, prepared inputs and preserved experiments. Snapshot
replay compares JSON exactly. Dataset files, decision templates/reports, poses,
images, environments and weights remain excluded from Git. Only the review ledger,
proposed configuration, code, synthetic tests and documentation are versioned.

Validation result: **289 tests passed** (including 16 new synthetic review/curation
cases); Ruff and CLI help checks passed. Current readiness remains blocked, as
expected; the curation command exits 1 rather than reporting a false success.

## Interactive review interface

[Dashboard v0.2](DASHBOARD_V02.md) now appends human decisions through these same
contracts. Its versioned review workspace preserves the original ledger and every
revision; corrections remain separate from prepared datasets. Historical counts
above describe the original proposal, not subsequent dashboard decisions.

## Independent confirmations

[Independent annotation verification](INDEPENDENT_ANNOTATION_VERIFICATION.md) adds
separate punch-type, lead/rear-hand and timing checks without rewriting old revisions
or their hashes. A full six-class label requires both type and hand confirmation;
acceptance requires timing too. Partial human checks remain held and pending, with
each check retaining its actual human revision origin.
