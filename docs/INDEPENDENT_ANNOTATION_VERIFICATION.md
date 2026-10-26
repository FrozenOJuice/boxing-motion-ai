# Independent annotation confirmations

The Review form has three independent checkboxes:

- **Punch type is correct.** Movement type/shape has been verified.
- **Lead/rear hand is correct.** The punching arm's relationship to stance has been verified.
- **Start/end timing is correct.** The proposed interval's boundaries have been verified.

Choose Yes to review the original annotation, or Change it to propose a corrected
label/interval. Select only the facts you verified and Save or Save & Next. Notes
are optional for unchanged annotations; the existing short explanation for corrections
still applies. Timing checks need both exact boundary frames to have been displayed.
New punch-type or hand checks require a displayed frame within the punch. No extra
combined attestation or mandatory explanation is added to these paths.

One or two checks save as **Partially verified** and remain held from the proposed
training selection. Three save as **Fully verified**, subject to existing exclusions
and the separate preparation gate for corrected examples. Fully verified does not
restore an excluded example or update prepared data. Unchecking all previously
verified aspects can also be saved, leaving the annotation unverified and held.
Not sure continues to record uncertainty with its existing reasons and personal-review
statement; it never silently asserts any of the three confirmations.

Checkboxes start unchecked for new/automated cases. The form can restore prior human
checks only for the same reviewed label and interval. Restored checks show their
original manual-review provenance, and keep that revision as the evidence origin if carried
forward. Editing the label or interval clears all checks; new verification is required.
Identity preferences never restore confirmations. A legacy human confirmation with
matching explicit label/boundaries can be restored; unscoped legacy assertions are
left unchecked rather than guessed.

## Six-class labels and partial knowledge

All six labels encode lead/rear meaning, including jab (lead straight) and cross
(rear straight). Checking punch type alone does **not** certify the full six-class
label. Neither does checking the hand alone. The existing `confirmed_label` is populated
only when **both** punch type and hand are verified. Timing is independent and can be
verified while neither label aspect is known.

| Punch type | Hand | Timing | Six-class label | Boundaries | Disposition |
| --- | --- | --- | --- | --- | --- |
| Yes | No | No | Unverified | Unverified | Hold |
| No | Yes | No | Unverified | Unverified | Hold |
| No | No | Yes | Unverified | Confirmed | Hold |
| Yes | Yes | No | Confirmed | Unverified | Hold |
| Yes | No | Yes | Unverified | Confirmed | Hold |
| No | Yes | Yes | Unverified | Confirmed | Hold |
| Yes | Yes | Yes | Confirmed | Confirmed | Accept |

A target label stored with verification metadata identifies **what was reviewed**.
It is not a substitute for `confirmed_label` or permission to use a partial case as a
clean training example. Partial cases remain pending in the targeted review queue and
cannot provide human-verified anchor support for curation readiness. The queue includes
a Partially verified filter; the case view and history show the individual checks.

## Versioned storage and compatibility

The dashboard uses a new `verify` request action with a strict boolean selection:

```json
{
  "schema_version": "annotation-verification-1.0",
  "punch_type": true,
  "hand": false,
  "timing": false
}
```

Each new revision's optional `verification` block stores those fields, the reviewed
label/bounds and an `origins` mapping from each verified aspect to its actual human
revision ID. That origin resolves to the manual source, UTC timestamp and native-frame
evidence in the preserved case history; any legacy reviewer name is retained. A carry-forward never credits a different
reviewer with the original assertion. Revoked checks have no active origin; their past
assertions remain in history. The API derives origins from the ledger, never client paths
or claimed reviewer metadata. Inherited frame evidence is reverified against pinned
sources before publishing another version.

The existing ledger stays `annotation-review-1.0`; the added block has its own version.
Legacy revisions and request receipts **omit** the new optional field on serialization.
Their canonical hashes and audit bindings therefore remain unchanged. No migration or
rewrite of existing ledgers is performed. Legacy confirmed label/boundary statuses are
read only as human checks when their actor and original evidence satisfy the existing
contract. Existing fully confirmed API clients remain compatible.

New revisions are published through the same locked, atomic, append-only ledger/config
transactions in `reports/boxingvi/dashboard-reviews-v1/`. Malformed booleans, assistant
confirmation origins, mismatched targets, missing native-frame evidence, stale writes,
known scene-cut crossings for timing and V9 requests are rejected. Notes capture the
selected verification state rather than invented stance observations. Original media,
workbooks, poses, candidates, policy thresholds and historical experiments are untouched.

## Verification

The full suite passes: **422 Python tests** and **36 frontend tests**, plus Ruff,
TypeScript, formatting and production build. Tests exercise all seven nonempty checkbox
combinations against all six classes, partial-to-full transitions, changed targets,
revocation, multiple reviewer origins, old hash preservation, reopening history,
HTTP writes, native frame/timing checks, strict booleans, V9 rejection, independent
checkbox interaction and restoration only from scoped human decisions.

Read-only checks also loaded the existing 83-case real ledger and current curation
proposal. V4:497, V4:144 and V5:262 have no human confirmations and expose three false
flags. No genuine review was saved during verification. Browser visual inspection
remains subject to the previously documented computer-use restriction; no browser
window is opened automatically. After updating, restart the local backend and refresh
the existing dashboard tab so the frontend and API both support `verify`.


## Anonymous manual source

The dashboard no longer requests or remembers reviewer names. New saves assign
`source: manual` on the server, retaining internal `actor_type: human` compatibility.
Automated records have `source: automated` and assistant actor type; they cannot verify
any aspect. The source fields must agree with actor type. Manual is a record of explicit
review input, not authenticated identity or proof of correctness.

New dashboard records omit the optional legacy `reviewer` field. Old records and
receipts keep their names exactly, including old canonical hashes. Missing source on a
legacy record is derived from its original actor for display only. Neither anonymous
source nor a carried-forward check changes the original evidence origin. API clients
cannot supply `source` or `actor_type`; spoofed fields are rejected. Old named request
payloads remain readable/compatible, while the current dashboard sends no name.

No browser name preference is read, and no browser storage is required to save reviews.
Existing named reviews are available in history; the unused former name preference is
left untouched. Updated tests verify anonymous history, separate automated observations,
unchanged legacy names/receipts, strict source validation and all existing verification
guards. Refresh after restarting the local backend to load this contract update.
