# Pose input representation: technical assessment

Follow-up: the [relative-arm study](LEVEL1_LIMB_INVESTIGATION.md) measures geometry
coverage and runs an additive, validity-gated partial-observation comparison. Lead
uppercuts improve, but rear punches and V10 remain unresolved. Geometry is optional;
the accepted XY+velocity+mirror reference remains unchanged. The assessment below
records earlier recommendations, not a claim that geometry is still unimplemented.

Approved follow-up: [the velocity study](LEVEL1_VELOCITY_STUDY.md) implements and
evaluates the recommended first intervention. The assessment below records the
earlier proposal; visibility/context, orientation and angle interventions remain
disabled/unimplemented in that comparison.

This is a recommendation, not an implemented feature/model change or a trained
comparison. V4-only mirror augmentation remains the reference; visibility augmentation
and pre-clip context remain optional and disabled. V5 is development-only, V9 reserved.
V6 preparation evidence is documented separately in [V6 preparation](V6_PREPARATION.md).

## What the classifier currently receives

`training/data.py::features`, version `torso-core-2d-1.0`, uses eight core joints:
shoulders, elbows, wrists and hips. X is aspect-corrected; XY is hip-centered and
divided by frame-local shoulder-midpoint-to-hip-midpoint length. Each joint has two
coordinates, visibility, presence and a validity bit, followed by one normalization
validity bit: **41 channels**. All required torso anchors must be valid. Invalid
feature entries are zero sentinels distinguished by masks; original pose records
retain coordinates/confidences/masks unchanged. There is no temporal filling.

Absolute indices and exact timestamps are retained by the loader but are not model
features. The TCN receives native frame steps, without elapsed-time, velocity,
bone-vector or angle channels. Its 15-sample receptive field covers 14 intervals:
approximately 0.584 s in V4/V6 versus 0.467 s in V5. This timing difference does not
establish any annotation-FPS correction. Current mirror augmentation handles an
image reflection with anatomical channel exchange; it does not solve general 3D
view changes, arm occlusion or uncertain lead/rear semantics.

## Assessment of four representation choices

| Choice | Potential value | Main limitation / recommendation |
|---|---|---|
| Current normalized XY | Removes image translation and scalar size; retains useful posture | Signed positions, foreshortening and visibility still depend on viewpoint. Keep as the reference and retain reliable information |
| Orientation normalization | A reliable torso axis can remove in-plane camera/body tilt | Cannot undo out-of-plane yaw. Shoulder axes become unstable when projected shoulder span is tiny. Do not infer stance/facing or rotate every clip from a guessed hand identity |
| Bone directions / joint angles | Explicit flexion and arm configuration; 2D joint angles remove in-plane rotation/translation/scale mathematically | Still perspective-dependent; require all incident joints and nondegenerate bones. Angle-only inputs would lose many rear-class frames. Consider as a later separate ablation, not a replacement for missing joints |
| Time-normalized motion | Causal displacement per exact elapsed second explicitly represents movement rather than native frame speed | Derivatives amplify estimator jitter and changing normalization scale. Require valid endpoints, anchors and continuity; no derivative across cuts/gaps or validity transitions. Does not itself equalize the TCN's physical receptive field |

MediaPipe distinguishes normalized image landmarks (including hip-relative depth)
from estimated world landmarks. Our journal retains the former; its Z coordinate
is not a measured world-space reconstruction. Do not treat it as calibrated 3D or
retroactively invent world coordinates. A future world-pose comparison would need
a new extraction/feature contract and independent quality checks. See the
[official output definition](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).

Bone lengths/directions have precedent as additional skeleton-action information,
but results from other datasets do not establish a gain for BoxingVI. This is
feature motivation, not a recommendation to adopt that paper's architecture. See
[Shi et al., CVPR 2019](https://arxiv.org/abs/1805.07694).

## Local evidence constraining geometry features

Read-only diagnostics use original masks and selected provisional positives; they
are not new classifier inputs. Statistics are medians across clips, conditional on
valid joints. The diagnostic code calculates image-plane angles and speeds only
to assess availability, and never writes transformed pose arrays.

| Class | V4 left/right valid elbow-angle frame fraction, median clip | V5 left/right valid elbow-angle frame fraction, median clip |
|---|---:|---:|
| Rear hook | 100% / 0% | 23.1% / 100% |
| Lead uppercut | 100% / 25% | 100% / 100% |
| Rear uppercut | 100% / 0% | 20% / 100% |

A joint angle needs shoulder, elbow and wrist together. A valid elbow alone is not
enough. V4 rear-hook shoulder-width/torso median is 0.102, versus V5 0.234; rear
uppercut is 0.154 versus 0.319. These support projected geometry differences and
caution against using narrow shoulder spans as a stable facing axis. They do not
establish boxer stance or calibrated camera angle. Existing confidence scores are
reliability cues, not proof of anatomical correctness. Better visible arms in V6
can improve feature availability, but a new source is not proof of generalization.

Full V6 preparation retains 302 provisional examples; median valid elbow-angle
coverage is 100% for each arm in each of its four source classes, although failures
are concentrated in some clips (frame-weighted right-elbow invalidity is 21.7% for
lead uppercuts). V6's rear-hook and rear-uppercut shoulder ratios are 0.597/0.592,
larger than V4/V5. Its median annotated frame-center spans are 0.647/0.792/0.626/0.584 s
for lead hook/rear hook/lead uppercut/rear uppercut, roughly two to three times the
V4/V5 spans. These provisional boundary/phase differences can remain shortcuts
after FPS handling; neither rate normalization nor valid angles certify labels.
The one-arm retention rule also does not guarantee that the punching arm is visible:
V6 row 285's quality sheets show the rising/far arm unreliable while the other arm
passes. New angles or velocities must retain that missingness, not reconstruct a
confident-looking trajectory for it.

## Recommended next controlled experiment — not implemented

**First test validity-gated causal XY velocities per second alongside the existing
normalized XY.** This addresses the measured physical-time mismatch and asks
whether explicit motion helps discriminate hooks/uppercuts without a guessed stance
or aggressive orientation canonicalization. It is a hypothesis, not a promised gain.

Use backward differences of the existing per-frame normalized coordinates divided
by exact PTS differences. Each derivative requires both endpoint joints and torso
anchors to be valid, consecutive decoded frames, and one scene/extraction segment.
These would be apparent image-plane velocities in torso units per second, not
measured glove speed in metres per second.
Mark the first or otherwise invalid derivative missing with its own mask; never
differentiate zero sentinels, interpolate, cross a cut, or assign unknown frames a
class. Native exact timestamps and poses stay immutable. Account for scale jitter
in diagnostics; do not silently smooth it. A complete cut inventory remains a
limitation when certifying continuity, even with independent pose inference.

Freeze the same data selection for both arms of that future experiment. Do not
combine a V6 data expansion with the feature change when attributing gains. Start
with V4 training / V5 development, the same three seeds, mirror probability 0.5,
quality policy, TCN blocks/width/receptive field and training budget. Preserve the
original reference. If added channels require a wider input interface, use matched
input dimensions for a zero-motion control and the real-motion arm, and record the
parameter counts; do not introduce extra temporal blocks or capacity changes.

Include the original mask-only control plus a derivative-validity-only control:
requiring two adjacent reliable observations can itself encode class/length cues.
Report six-class macro-F1, per-class recall, confusion and early/final train/development
gaps, especially lead uppercuts/rear hooks/rear uppercuts. Reject aggregate-only
claims or improvements carried solely by the masks.

This experiment does not remove physical receptive-field differences. A common
**fixed-time** sampling clock is a later independent comparison; avoid warping every
annotated punch to a fixed fraction/length, which uses future boundaries and can
erase real speed variation. Torso-relative bone directions/angles should likewise
be tested separately after their validity coverage is understood. No new feature
extractor, resampler, model architecture or training run is implemented here.
