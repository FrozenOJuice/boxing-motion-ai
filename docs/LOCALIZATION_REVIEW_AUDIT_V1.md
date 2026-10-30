# Human-review audit: localization batch v1

Audit of saved records at `6227c4c`, 2026-10-30 UTC. **Blocked: do not train or freeze the expanded dataset.**
This document reports existing human assertions; it does not create or correct annotations.

## Finding

V5-motion-101 contains **zero saved punch events and zero uncertain sections**. Its only
revision is an explicit manual completion at **2026-10-30 05:22:01 UTC**, with
`watched_entire=true`, `all_punches_labeled=true`, `no_punches_observed=true` and
`visible_continuous=true`. There are no earlier V5 upserts, edits or removals. The eight
provisional source marks were never saved as human punch events. The UI's zero count accurately
reflects the ledger. The record establishes a no-punch attestation; it cannot establish whether
the user intended that attestation or mistook displayed source marks for saved labels.

The V5 native images at 8142 and 8153 show punching motion, so this empty inventory requires
re-review. That qualitative inspection is not a new verified boundary inventory. Neither
source marks nor these observations are imported as human ground truth.

V6-motion-103 is also an explicit no-punch completion. It conflicts with the independently
human-verified V6:46 anchor. V6-motion-101 also conflicts with the prior V6:7 anchor. These are
substantive contradictions even though all six completion flags and viewing gates validate.

## Saved inventory

All boundaries below are **zero-based native [start, stop)**. The editor's End field is the
**last included frame (`stop - 1`)**. Types are the saved human choices, not classifier outputs
or an independent endorsement of their correctness. No simulated/model predictions were used.

| Interval | Human punches | Uncertain sections | Provisional core suggestions | Saved revisions | Viewing validation |
| --- | ---: | ---: | ---: | ---: | --- |
| V6-motion-101 | 5 | 0 | 3 | 6 | Complete: continuous_playback |
| V6-motion-102 | 4 | 0 | 2 | 5 | Complete: combined |
| V6-motion-103 | 0 | 0 | 3 | 1 | Complete: combined |
| V6-motion-104 | 8 | 0 | 2 | 9 | Complete: continuous_playback |
| V6-motion-105 | 1 | 0 | 4 | 2 | Complete: combined |
| V5-motion-101 | 0 | 0 | 8 | 1 | Complete: combined |

There are **18 saved human punch events, zero uncertain sections, 22 provisional core
suggestions, and 24 saved revisions: 18 upserts and six completions**. All 18 current events
have distinct IDs, both endpoint frames in their upsert inspection evidence, and revalidated
native timing metadata. There are no subsequent edits or removals in these six histories.
No overlapping saved punches were found; touching half-open events remain distinct. All
saved type fields are explicit (seven jabs, ten crosses, one rear hook); none is Type uncertain.
This is an inventory of assertions, not recognition accuracy or proof that every punch was found.

### Exact saved human boundaries

| Interval | Saved type | [start, stop) | Editor first–last included | Source seconds [start, stop) |
| --- | --- | --- | --- | --- |
| V6-motion-101 | jab | [469, 477) | 469–476 | [19.561, 19.895) |
| V6-motion-101 | cross | [477, 483) | 477–482 | [19.895, 20.145) |
| V6-motion-101 | rear_hook | [496, 501) | 496–500 | [20.687, 20.896) |
| V6-motion-101 | jab | [532, 538) | 532–537 | [22.189, 22.439) |
| V6-motion-101 | cross | [558, 564) | 558–563 | [23.273, 23.524) |
| V6-motion-102 | jab | [6453, 6460) | 6453–6459 | [269.144, 269.436) |
| V6-motion-102 | cross | [6461, 6468) | 6461–6467 | [269.477, 269.769) |
| V6-motion-102 | jab | [6529, 6538) | 6529–6537 | [272.314, 272.689) |
| V6-motion-102 | cross | [6545, 6557) | 6545–6556 | [272.981, 273.481) |
| V6-motion-103 | **No saved events** | — | — | — |
| V6-motion-104 | jab | [25045, 25051) | 25045–25050 | [1044.585, 1044.836) |
| V6-motion-104 | cross | [25052, 25059) | 25052–25058 | [1044.877, 1045.169) |
| V6-motion-104 | cross | [25108, 25113) | 25108–25112 | [1047.213, 1047.421) |
| V6-motion-104 | jab | [25121, 25128) | 25121–25127 | [1047.755, 1048.047) |
| V6-motion-104 | cross | [25129, 25136) | 25129–25135 | [1048.089, 1048.381) |
| V6-motion-104 | cross | [25202, 25208) | 25202–25207 | [1051.134, 1051.384) |
| V6-motion-104 | jab | [25215, 25221) | 25215–25220 | [1051.676, 1051.926) |
| V6-motion-104 | cross | [25221, 25229) | 25221–25228 | [1051.926, 1052.260) |
| V6-motion-105 | cross | [30490, 30501) | 30490–30500 | [1271.687, 1272.146) |
| V5-motion-101 | **No saved events** | — | — | — |

The first V6-motion-104 event starts at 25045, before the core's 25050 boundary; it is
left-censored for core supervision. Its full saved bounds must not be clipped or silently
repaired. V6-motion-105's only event starts at its core boundary 30490 and has a censored onset.
All other current event endpoints are inside their review cores and are not marked censored.

### Completion and viewing attestations

Every latest revision is manual, operation `complete`, with `watched_entire=true` and
`all_punches_labeled=true`. V6-motion-103 and V5-motion-101 additionally attest
`no_punches_observed=true`, `visible_continuous=true`. The four nonempty intervals have
`no_punches_observed=false`, `visible_continuous=false`; the latter was not separately attested
on that nonempty UI path and is not a claim that the boxer was invisible.

| Interval | Completion UTC | Exact frames in completion payload | Playback segments | Playback alone complete? | Combined viewing complete? |
| --- | --- | ---: | ---: | --- | --- |
| V6-motion-101 | 04:42:14 | 63 | 10 | True | True |
| V6-motion-102 | 04:46:20 | 58 | 5 | False | True |
| V6-motion-103 | 04:49:51 | 6 | 8 | False | True |
| V6-motion-104 | 04:55:01 | 66 | 15 | True | True |
| V6-motion-105 | 04:56:09 | 19 | 10 | False | True |
| V5-motion-101 | 05:22:01 | 7 | 7 | False | True |

Combined mode unions actual inspected frames with visible real-time playback spans. Thus
`playback_complete=false` is not an incomplete review when `complete=true` under this policy.
All six coverage records bind the expected clip key and native core timing and revalidate.
Coverage establishes what was displayed, not semantic correctness or an exhaustive inventory.
No coverage was added by this audit. The new queue is not automatically a trustworthy dataset
merely because its interface says Completely labeled or Reviewed.

## Contradictions and review priorities

1. **V5-motion-101: explicit empty review.** Eight source suggestions (rows 386–393) are
   [8140,8146), [8146,8160), [8221,8229), [8230,8247), [8307,8315), [8316,8330),
   [8392,8400), [8400,8413). None has a human event. Raw frame inspection supports visible
   punching at 8142/8153; rewatch the entire [8100,8460) core and save actual individual punches
   or uncertain motion. Do not copy eight suggestions mechanically or assume eight is the
   correct exhaustive count. Blindly exporting the current attestation would mark all 360
   core frames as background.
2. **V6-motion-103: hard contradiction.** Its empty review would mark the whole [8280,8472)
   core as background, including the earlier human-verified rear hook V6:46 [8368,8386).
   Prior evidence is the unchanged dashboard ledger v19, origin receipt v15. Re-review that
   anchor and the entire core. Source-only rows 45 [8306,8323) and 47 [8447,8467) are additional
   suggestions, not automatically verified missing events. Native frame 8374 also shows the
   action qualitatively. Do not reinterpret this empty record as Unknown or silently import
   the older anchor to conceal the disagreement.
3. **V6-motion-101: hard timing conflict.** The earlier human-verified lead hook V6:7 is
   [486,497), receipt v16. The new review has a rear hook [496,501), overlapping it for only
   one frame, and no event over [486,496). Its complete inventory therefore contradicts
   ten previously positive frames. The differing lead/rear types also need explicit review;
   proximity alone does not prove the two entries describe the same punch. Recheck roughly
   480–505 and the remainder of the core. Source rows 8 [541,550) and 9 [566,577) have no
   overlapping saved human event. Those source-only discrepancies are review prompts,
   not established counts of missed punches.
4. **V6-motion-105: suspicious incomplete inventory.** Its only event [30490,30501) covers
   the beginning of the core. None overlaps source rows 335–338 at [30509,30532),
   [30535,30558), [30563,30586), [30594,30616). Review later motion/combinations through
   30681 and the preceding context for the censored opening. Do not assume those four
   provisional marks are four verified punches or convert every remaining frame to
   trustworthy background solely from this completion.
5. **V6-motion-102: source-versus-human discrepancy.** Four genuine saved human records exist,
   but neither source hook suggestion [6468,6485), [6566,6589) overlaps them. One merely
   touches the end of the saved cross [6461,6468). Recheck those regions and confirm whether
   the source marks are wrong, events were omitted, or boundary definitions differ. Do not
   discard the four saved events or force their jab/cross labels to match provisional hooks.
6. **V6-motion-104: source-versus-human discrepancy and censoring.** Eight saved events exist.
   Source uppercut suggestions [25094,25109), [25187,25203) overlap saved crosses
   [25108,25113), [25202,25208) for one frame each. Review those trajectories and the full
   movement/recovery definition; source labels cannot overrule human labels automatically.
   Preserve the first event's valid full context bounds and its censored core onset.

The two anchor contradictions are independently supported by previous human evidence. The
remaining source-only differences do not by themselves establish missing-punch counts or
wrong classes. This audit does not choose which conflicting human revision is correct.

## Precisely how to correct in the existing interface

Open **Development → Review → Video intervals → Individual punches · V6 / V5** and select the
relevant interval. No queue reset or new review platform is needed.

- A source marker or its source-list button **only seeks**. It does not create a human event.
  Use **Add punch**, inspect the actual first frame, **Set start here**, inspect the actual
  last included frame, **Set end here**, then **Save punch**. A punch type can remain
  **Type uncertain**; that means the punch is established but its type is unknown.
- Use **Mark uncertain section** for footage where punch existence/separation or visibility
  cannot be established. Save that section; do not label it as a definite punch or background.
- Existing saved events can be corrected with **Edit punch**, then **Save punch**. Adding,
  changing or removing an event appends a revision and resets current completion to false;
  it does not delete the earlier human assertions. Do not delete/recreate the queue or files.
- After finding *every* visible event and uncertain region, rewatch the whole core (using
  context for boundary decisions), check the all-punches/uncertainty attestation and choose
  **Save complete & Next**. Verify the saved human section count is appropriate. For a
  nonempty corrected inventory the new completion has `no_punches_observed=false`. The old
  no-punch revision remains in history. Do not attest no punches simply because you watched
  the video or can see provisional marks. Reaffirm an empty review only if you actually
  establish absence and reconcile contradictory human evidence.

For V6:7 and V6:46, compare both existing human assertions against the footage. If the older
anchor is confirmed, bring the interval inventory into agreement by real inspection; do not
blindly copy its bounds. If the older anchor itself is incorrect, it must receive an explicit
human correction/uncertainty revision with a reason: **Review → Punches → All punches → search
`V6:7` or `V6:46`**. The ordinary punch-review editor offers timing/type corrections and notes.
After such a revision the old pinned anchor/curation audit becomes stale and needs a new
explicit evidence audit/version; it must not be bypassed. If real overlapping punches explain
the disagreement, preserve that finding rather than forcing matching boundaries to satisfy
the adapter's conservative overlap guard; the subsequent audit must address that case explicitly.
If the conflict cannot be resolved,
leave it unresolved and do not train on the disputed span. This audit makes none of those edits.

## Readiness and checks

**Expanded training dataset: not ready. Development reference: not ready.** There are saved V6
positives, but two anchor contradictions and suspicious incomplete inventories block export.
V5 has no saved positive reference and a questionable blanket negative attestation. The original
continuous-v1 training smoke dataset and older artifacts remain untouched; neither the 18 new
records nor the 22 source suggestions have been silently incorporated into a frozen dataset.
V5 remains development-only; V7/V8/V10 motion and V9 are not accessed.

The existing read-only command
`.venv/bin/python -m boxing_motion_ai.data.localization_evidence --check`
**correctly fails closed** with `New review conflicts with existing human anchor; resolve explicitly`.
That is a data contradiction, not a reason to lower validation or repair labels programmatically.
Its first exception is not the complete inventory of issues; this audit examines all six ledgers
individually. No `--snapshot` or training command was run.

Checks replay all 24 revisions, their parent hashes, source/manual flags, endpoint inspection
requirements and completion evidence; verify exact native timing and clip-bound viewing; and
recheck the pre-audit file hashes and review heads afterwards. **43 relevant Python tests pass**
(one existing Starlette/httpx warning). Source-only visual observations are private diagnostic
images, not new annotations. No application or training code was changed. Private full event IDs,
revisions, coverage payloads, immutable review backups and SHA256 inventory are retained under
`reports/boxingvi/localization-review-audit-20261030T052414Z/`. This documentation records a
blocked audit and correction instructions; it is not a dataset snapshot or training approval.
