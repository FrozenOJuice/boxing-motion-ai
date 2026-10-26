# Human review validation — 26 October 2026

Subsequent approved preparation and controlled results: [conservative curation study](LEVEL1_CURATION_STUDY.md).
The validation snapshot and gate behavior described below are preserved as history.

The 18 requested reviews are saved correctly. **16 are fully verified unchanged
annotations; two are partial manual holds. Every class has at least one fully
verified, originally eligible anchor.** We can prepare a conservative curated
selection without another human review, but the current automated readiness gate
still blocks training on the two partial cases. No dataset or gate was changed.

## Saved-record validation and backup

Validated dashboard ledger **19**, SHA256
`f5d8ef1e437fa7f50beec40e81b9bb526b1a3422dd2d5b7d4f7426e30f90ce5c`.
The workspace replay validates all 18 contiguous transactions (versions 2–19),
parent/revision hashes, unchanged prior histories, receipt audit bindings,
configuration pins, manual source, timestamps and individual verification targets.
Each reviewed case has its original automated revision followed by one manual
revision. All 54 checkbox values and their selected-axis origins survive replay.
No automated observation was promoted to human evidence.

A lock-protected byte-for-byte backup of all 54 transaction files plus the three
initial ledger/config/catalogue files is stored at:

`reports/boxingvi/human-review-validation-20261026T152916Z/backup/`

The adjacent `backup-manifest.json` records SHA256 for all 57 files. Copies were
verified against their source bytes; originals were checked again after analysis.
This is a local backup, not an off-device disaster-recovery copy. It is excluded
from Git along with the saved reviews. Preserve it before any future preparation.

`validation.json` contains the existing service's proposal; `input-validation.json`
contains validation findings and the original proposal for comparison.
`snapshot-config.json` pins the reviewed snapshot. These are ignored reports,
not activated dataset selections or new human decisions.

All six registered prepared input sets (V4/V5/V6/V7/V8/V10), including pose-file
hashes, were checked against existing pins. All 83 source candidate bindings and
review evidence bounds passed validation. The 274 recorded inspected-frame entries
match the submitted indices and have exact timestamp = PTS × timebase; all 16 timing
confirmations include both endpoints. Receipt source-input hashes match preparation
pins. This validates persisted evidence; it does not independently repeat the
human's visual judgment or freshly decode/hash source videos. The historical broad
preservation manifest was not rerun; V9 was not opened or hashed.

## Actual decisions

Frame intervals below are the stored zero-based decoded, inclusive intervals.
“Yes” means explicitly saved by the human, not inferred from model output.

| Case | Role | Confirmed/target label | Frames | Type | Hand | Timing | Result |
|---|---|---|---|---|---|---|---|
| V4:497 | Dispute | Lead uppercut | 21695–21702 | Yes | Yes | Yes | Accept unchanged |
| V6:613 | Dispute | Rear uppercut | 41090–41104 | Yes | Yes | Yes | Accept unchanged |
| V7:136 | Dispute | Lead uppercut | 15040–15051 | Yes | Yes | Yes | Accept unchanged |
| V8:82 | Dispute | Rear uppercut | 9534–9547 | Yes | Yes | Yes | Accept unchanged |
| V10:146 | Dispute | Rear uppercut | 20729–20737 | Yes | Yes | Yes | Accept unchanged |
| V10:147 | Dispute | Lead uppercut | 20747–20758 | Yes | Yes | Yes | Accept unchanged |
| V4:144 | Anchor | Jab | 10404–10410 | Yes | Yes | Yes | Accept unchanged |
| V7:8 | Anchor | Jab | 4770–4783 | Yes | Yes | Yes | Accept unchanged |
| V8:51 | Anchor | Cross | 7974–7984 | Yes | Yes | Yes | Accept unchanged |
| V10:71 | Anchor | Cross | 5502–5514 | Yes | Yes | Yes | Accept unchanged |
| V4:343 | Anchor | Lead hook | 18192–18201 | Yes | Yes | Yes | Accept unchanged |
| V6:7 | Anchor | Lead hook | 486–496 | Yes | Yes | Yes | Accept unchanged |
| V6:46 | Anchor | Rear hook | 8368–8385 | Yes | Yes | Yes | Accept unchanged |
| V10:142 | Anchor | Rear hook | 20653–20665 | Yes | Yes | Yes | Accept unchanged |
| V6:264 | Anchor | Lead uppercut (unconfirmed target) | 20096–20109 | No | Yes | No | Hold |
| V8:159 | Anchor | Lead uppercut | 16805–16817 | Yes | Yes | Yes | Accept unchanged |
| V6:266 | Anchor | Rear uppercut (unconfirmed target) | 20287–20298 | No | Yes | No | Hold |
| V10:44 | Anchor | Rear uppercut | 4451–4465 | Yes | Yes | Yes | Accept unchanged |

Both partial reviews explicitly note **“No punch is thrown.”** Their full class and
boundaries are unconfirmed; hand-only verification does not override that. Keep
them out of punch training. Do not relabel them `no_punch`, infer replacement
boundaries, or change their saved disposition from hold to exclude.

There are **zero proposed label changes, zero proposed boundary changes, zero new
explicit exclude decisions, and zero new ambiguous/mixed-action decisions** among
these 18. There are two unresolved partial holds. The entire 83-case ledger now has
16 accepts, 37 holds and 30 provisional cases. Of the holds, 35 retain automated
observations, including ambiguous/mixed intervals and precautionary neighbors;
these remain observations, not human judgments.

Verified unchanged eligible anchor support is **2 jab, 2 cross, 2 lead hook,
2 rear hook, 1 lead uppercut and 1 rear uppercut**. Both class semantics and timing
are confirmed for these anchors, including the lead/rear meaning of jab/cross.
This meets the minimum class-coverage requirement; a second uppercut anchor is
not required. These training-pool references are not an independent test set.

## Proposed selection versus existing candidates

The existing planner computes the following **hypothetical**, unactivated selection.
It only considers originally approved mask-aware candidates. No original policy
exclusion is restored, and no candidate file is rewritten.

| Recording | Approved baseline | Old curation proposal | Latest-review proposal | Change vs baseline |
|---|---:|---:|---:|---:|
| V4 | 527 | 520 | 521 | −6 |
| V6 | 302 | 297 | 296 | −6 |
| V7 | 66 | 60 | 61 | −5 |
| V8 | 120 | 115 | 116 | −4 |
| V10 | 141 | 133 | 135 | −6 |
| Training total | **1156** | **1125** | **1129** | **−27** |
| V5 development only | 583 | 583 | 583 | 0 |

| Class | Approved baseline | Old proposal | Latest-review proposal |
|---|---:|---:|---:|
| Jab | 190 | 190 | 190 |
| Cross | 97 | 97 | 97 |
| Lead hook | 303 | 301 | 301 |
| Rear hook | 112 | 108 | 108 |
| Lead uppercut | 257 | 246 | 248 |
| Rear uppercut | 197 | 183 | 185 |

The exact delta versus the old proposal is:

- Six previously proposed holds become eligible *in the planner*: V4:497, V6:613,
  V7:136, V8:82, V10:146 and V10:147, following their explicit full confirmations.
  Each was eligible under the original quality policy. Their human decisions now
  support considering them; **none has actually been restored to a dataset**.
- V6:264 and V6:266 become new holds, removing one lead and one rear uppercut.
- All other selected rows remain identical. Existing quality exclusions still win.
  V10:148–153 remain held; accepting 146/147 does not certify the rest of the block.
- V5 retains its frozen 583 examples, with shadow holds 196/432/514. No benchmark
  labels or membership change. V7/V8 stay grouped, the V1/V6 guard remains, and
  unknown boxer/session identities remain unknown.

If preparation must preserve **every previous proposed hold** pending explicit
inclusion approval, the conservative intersection is **1123** candidates: retain
the old 1125 selection minus the two new V6 holds. Its class counts are
190/97/301/108/245/182 in the table order above. This is the recommended initial
preparation scope under the instruction not to automatically restore exclusions.
The six confirmed disputes can be an explicit later inclusion decision, rather
than a hidden consequence of loading the latest ledger.

## Readiness and next experiment

The unchanged service reports `blocked_pending_review`, solely due to
`targeted_human_review_pending` for V6:264 and V6:266. There are no missing-class
anchor blockers and no accepted corrections requiring materialization. The gate
currently conflates partially verified *held* cases with unfinished review work.
Do not bypass it or claim it already passes.

**Ready to prepare a conservative provisional dataset; not yet ready to launch
training through the existing gate.** The next engineering step should explicitly
allow reviewed unresolved holds to stay excluded when sufficient other anchors
cover every class, with tests and a versioned selection policy. Preserve the partial
confirmations and pending-verification status; do not manufacture full acceptance.
Pin the version-19 ledger and exact selected rows, enforce old exclusions, and
recheck inputs when materializing that new selection. No additional manual review
is necessary to safely leave these two cases held.

Once that preparation is approved and validated, recommend one controlled
**curation-only** comparison: preserved multi-recording pose-only XY + gated
velocity + 50% mirror versus the identical setup on the conservative curated pool.
Use the existing three matched seeds, budget, source/class sampling, quality policy
and architecture; geometry, visibility augmentation and context stay disabled.
Keep V5 development-only and unchanged. Use the existing recording-group holdouts
with no held-out training or checkpoint selection; report supported-class and
six-class F1, per-class recall and mask-only controls. Prioritize rear hooks and
uppercuts and compare identical evaluation rows. Do not claim boxer independence.

Most retained examples remain provisional. The reviews settle these particular
cases, not global stance conventions or all pose errors. The two false-punch notes
make excluding suspect intervals a meaningful controlled test, not a guarantee
that curation will solve generalization. No models were trained in this validation.

## Checks

**154 tests passed**, with one existing upstream Starlette/httpx deprecation
warning; **Ruff passed**. No application code or frontend behavior changed.

```sh
.venv/bin/pytest tests/test_annotation_curation.py tests/test_review_workspace.py \
  tests/test_manual_review_source.py tests/test_independent_verification.py \
  tests/test_dashboard_services.py tests/test_dashboard_api.py -q
.venv/bin/ruff check .
```
