# Read-only dashboard boundary

P1 implemented 2026-10-25. This is a Python metadata service, not an HTTP server,
media server or review editor. P0 is documented in [CI.md](CI.md).

The boundary is: future React client → future FastAPI adapters →
`boxing_motion_ai.services.DashboardService` → explicitly registered metadata and
existing contracts. There is no reason to add FastAPI routes before choosing the
first dashboard screen and its transport/authentication requirements. No framework,
database, worker queue or new ML dependency was added.

## Using the interfaces

```python
from pathlib import Path
from boxing_motion_ai.services import DashboardService

service = DashboardService(Path("/absolute/path/to/boxing-motion-ai"))
inventory = service.inventory()
annotation = service.annotation("V4", 497)
queue = service.review_queue()
proposal = service.curation_status()
metrics = service.model_metrics("velocity-reference", "20261021-xy_velocity-tcn")
response_body = metrics.model_dump(mode="json")
```

Construction accepts a **trusted application root** and optional catalogue path.
Do not let HTTP clients choose either. Invalid/missing catalogue configuration raises
an initialization error; request methods return `QueryResult` envelopes. Methods
accept registered IDs and bounded pagination, not arbitrary paths. No service
method writes files, extracts frames, computes poses or trains/evaluates models.

| Method | Read contract |
|---|---|
| `inventory()` | Six registered preparation statuses, explicit scope, availability, source roles and catalogue hash. Not raw-dataset filesystem discovery. |
| `recording(recording_id)` | Original preparation metadata, unknown boxer/session IDs, source group/guard, opaque pose-array reference marked **unverified**. |
| `annotations(recording_id, offset=0, limit=50)` | Source-order page (limit 1–100), total source rows, complete unchanged prepared annotation records. Includes excluded/reversed rows. |
| `annotation(recording_id, source_row)` | One unchanged candidate and its canonical existing review fingerprint. |
| `statistics(recording_id)` | Counts from pinned candidates and the existing approved policy index; absent selected classes, unselected source rows, original preparation summaries. No policy recalculation. |
| `review_status()` | Existing ledger case IDs, disposition counts and human acceptance count. Does not certify candidate bindings. |
| `review(recording_id, source_row)` | Candidate fingerprint check, complete revision history, latest decision, explicit human-acceptance status. |
| `review_queue()` | Existing six disputes and twelve anchors, each with its own read status and preserved decisions. |
| `review_evidence(recording_id, source_row)` | Existing diagnostic case, native frames, exact PTS, pose evidence/predictions unchanged. Observations are not ground truth. |
| `curation_status()` | Existing proposal snapshot bound to its config/ledger; blocked/ready status remains inside the proposal. No materialization or training permission. |
| `experiments()` / `experiment(experiment_id)` | Registered studies and their unchanged result payloads, including existing fold/seed/model/recording distinctions. |
| `model_metrics(experiment_id, metric_id)` | Registered per-run train/development metrics, ordered confusion matrices and explicit zero-support classes. No recomputation or checkpoint loading. |

The initial catalogue includes the velocity reference (12 per-run metric files),
multi-recording study and RGB feasibility study. Both historical studies retain
their original nested fold/model/recording metrics through `experiment()`. This is
an explicit initial subset, not a claim that older studies have disappeared. Their
files and entrypoints remain unchanged. Register additional supported summaries
when a dashboard screen needs them; do not recursively expose `reports/`.

V4/V5/V6/V7/V8/V10 inventories reuse `CurationConfig` sources. V1/V2/V3 are explicitly
unregistered for this approved full-preparation slice. V9 is reserved and rejected
before metadata reads. V7/V8 remain grouped, V6 carries the V1/V6 guard, V5 remains
development-only, and unknown athlete/session IDs remain null.

## Response and integrity semantics

`dashboard-query-1.0` envelopes contain `status`, `issues`, `data` and
`verification="metadata_only"`:

- `available`: the requested metadata passed the applicable format/hash checks.
  **Not** proof that annotations are correct, pose arrays exist/are unchanged, or
  that a training proposal has human approval.
- `missing`: the registered artifact or requested row/review is absent; data is null.
- `stale`: a pinned hash, candidate fingerprint or proposal provenance disagrees.
- `invalid`: unsafe/unknown ID/path, malformed metadata, unsupported schema or
  violated source/label contract; data is null.
- `incomplete`: a required metadata field is absent, or a composite inventory/queue
  contains unavailable entries. Composite responses retain individual statuses;
  other errors never turn missing evidence into zero counts.

A readable blocked proposal has envelope status `available` and
`data.proposal.status="blocked_pending_review"`. Clients must show both concepts.
Likewise, absent classes are distinct from present classes with zero recall.
Existing six-class F1, supported-class F1 and accuracy remain as stored; no substitute
metric is invented when a source lacks one.

`contracts.py` provides strict versioned Pydantic catalogue/result models.
`dashboard.py` reuses `CurationConfig`, `ReviewLedger`, `latest`, canonical `digest`,
and existing taxonomy/group guards. Original versioned payloads stay nested rather
than being flattened into an invented common training format. Supported preparation
and candidate versions include legacy V5 and current recording 1.1; the explicitly
registered `rgb-legacy-1` adapter reads the existing unversioned RGB summary.

`artifacts.py` reads at most 16 MiB per metadata file, rejects duplicate JSON keys,
nonfinite numbers, symlinks (including parent components), traversal, encoded paths,
nonregular files and non-metadata extensions. POSIX directory-relative no-follow
opens prevent symlink replacement races. macOS/Linux are supported; Windows
portability is not claimed. Paths in payloads are provenance only, never URLs or
permission to serve those files. The service must not become a public static mount.

The catalogue pins small metadata artifacts. Config/policy/candidate hashes are
checked against their existing pins; the curation snapshot checks config and ledger
provenance. The catalogue is a process snapshot: restart after intentionally
versioning a new one. Do not auto-refresh hashes to hide a stale review or regenerate
reports in a request handler. No preservation-manifest scan, pose-byte hash,
checkpoint deserialization or original-workbook/video read occurs here.

Exact native frame indices, rational PTS strings, source cells, confidence fields,
validity masks, exclusions and uncertainties are returned without normalization or
label rewriting. The service does not expose full pose-array slices yet; those
remain opaque references. A future explicit pose/media reader needs its own bounded
exact-frame contract and tests. React should retain rational PTS strings and native
indices as authoritative, deriving rounded seconds only for display. Unlabeled
footage remains unknown, never automatically `no_punch`.

## Verification and readiness

Synthetic tests cover read-only behavior, source/development guards, missing/stale/
incomplete metadata, pagination, reversed intervals, unsupported versions, malformed
JSON, unsafe paths/symlinks, reserved IDs before I/O, masks and large rational PTS,
review attribution, snapshot provenance and absent classes. They run with base
installation dependencies. A fresh interpreter check confirms no NumPy/PyTorch,
pose/video or training-module imports.

Local metadata regression read the six registered preparations and policy indexes,
all 18 queue entries, proposal, diagnostic evidence, three study summaries and all
12 registered reference metric files. No tensors, media, weights or V9 were opened.
The preserved approved selection is 1,156 training / 583 development examples; the
proposal remains 1,125 training / unchanged 583 development, pending human review.
The ledger has 83 cases and zero human acceptances. These are existing results,
not new curation decisions.

Ready for a **read-only** FastAPI/React integration slice. Choose concrete endpoint
response payloads, map typed issue codes to HTTP statuses, and keep heavy work out of
request handlers. Review writes, concurrent revision checks, media serving, job
execution and broader cleanup remain separate future milestones. Do not promote
this boundary into an annotation-editing API without that work.

Final local P1 checks (2026-10-25): **341 passed, zero skips** in the locked full
pose/train environment; **160 passed, 68 expected optional skips** in the locked
base environment. This includes **52 new service tests**. Ruff, CLI help, curation
CLI help, registered-metadata regression and `git diff --check` passed. Registered
metadata hashes were unchanged after queries. No existing training, preparation,
vision or review module was changed by P1. Remote CI has not been run locally.
