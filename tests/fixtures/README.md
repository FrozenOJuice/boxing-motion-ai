# Punch motion v2 real fixture

`punch_motion_v2.json` contains actual shoulder-relative torso-scaled wrist/elbow XY traces
from the permitted V6 upload, three reviewed V4 diagnostic contexts and V6-motion-6.
It is derived from existing cached poses; points are rounded to eight decimal places.
`null` is masked/unobservable, not a position estimate. No video, model weights, full source
pose dataset, simulated trajectory or classifier score is included.

Each case records original/derived source hashes, exact timestamps, discontinuity markers
and actual binary activity bounds. `references_for_comparison_only` contains existing human
V4 decisions and is never passed to the segmenter. `expected_candidates` pins development
regression behavior, including misses and a background false proposal, not correct punch labels.

Reproduce with `boxing_motion_ai.detection.instance_diagnostics --fixture-out` as described in
[the v2 diagnostics](../../docs/PUNCH_INSTANCE_SEGMENTATION_V2.md). A complete source inventory
must be prepared first. Use a fresh output path; retained fixtures are not overwritten.
The test suite needs no source videos or checkpoints to replay these traces.
