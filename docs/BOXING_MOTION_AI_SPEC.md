# Boxing Motion AI — Project Specification

**Version:** 1.0 | **Updated:** 2026-10-16 | **Status:** Approved technical direction; dataset verification pending

## 1. Product vision

Build a professional, reproducible computer-vision and machine-learning system that extracts boxing events from recorded video, recognizes punches while they develop, and ultimately summarizes a boxer's actions and tendencies over a round. A future research extension attempts anticipation *before* a punch starts; a separate long-term extension predicts future body poses.

The project is intended as a flagship portfolio demonstration of data engineering, custom temporal model training, causal evaluation, and production-minded software design. **Level 2 is the official core completion gate.** Level 2.5 (analytics) is the preferred post-core expansion. Levels 3 and 4 must never delay completion of Levels 1–2.

### Initial environment and non-goals

- Begin with **one visible boxer**, largely stationary camera, and recorded shadowboxing/bag-training footage. Add match footage and multiple boxer tracking only after the one-boxer pipeline works.
- No requirement to identify punch impact, scoring, landed punches, injury risk, or defensive effectiveness. Do not infer these from punch classification.
- No LLM requirement, microservices, database, managed cloud services, or custom frontend framework for the initial build.
- Pretrained pose estimation is allowed, but **we train and evaluate the temporal punch models ourselves**.
- Never redistribute footage, annotations, or model weights with unclear usage rights. Public GitHub should contain code, documentation, metrics, and media we have permission to share.

## 2. Milestone scope

### Phase 0 — Dataset and project foundation (required; first build task)

**Outcome:** A reproducible, auditable set of footage and annotations suitable for modeling.

- Verify downloaded dataset contents, video metadata, annotation fields, punch classes, temporal boundaries, source identity, availability of contiguous *non-punch* footage, and whether boxer/session splits can be constructed.
- Verify licenses and attribution for footage, annotations, model weights and any portfolio visual examples. Track restrictions rather than assuming GitHub hosting implies permission.
- Create a canonical punch event schema and documented annotation guidelines. Implement validation and a dataset audit report.
- Manually review representative cases across punch classes, boxers, camera viewpoints and negative movements; identify ambiguous onsets and tracking failures.
- Freeze a split plan grouped by boxer identity when reliably available, otherwise by source session/video; no neighboring clips from one recording may cross splits.

**Gate:** We can match the available continuous video to usable labels and establish independent train/validation/test groups. Otherwise, choose a backup and/or collect self-recorded footage before training a model.

### Level 1 — Punch recognition (required)

**User experience:** Upload a video; view a pose-overlay and a timeline of detected punch events labeled by type, with event start/end times and model scores.

- Start with class taxonomy `jab`, `cross`, `lead_hook`, `rear_hook`, `lead_uppercut`, `rear_uppercut` if the verified dataset supports all six; may initially combine hook sides for a baseline if class coverage is poor.
- Add `no_punch` as a **temporal/background state**, not as a mislabeled substitute for excluded punch types. Preserve unknown/ambiguous events for review.
- Establish a trivial baseline, e.g. landmark-derived feature statistics with a logistic regression/random forest model, then an LSTM or TCN clip classifier.
- Train on consistent body pose features. If training from AlphaPose skeletons but deploying MediaPipe, account for domain mismatch by extracting training features with the deployment pose estimator or explicitly evaluating transfer.
- Measure clip-level macro-F1 and per-class confusion, and separately detect event boundaries in full video with event-level precision/recall.

**Gate:** Independent held-out evaluation, reproducible commands, honest failures and a working video-to-events path.

### Level 2 — Early punch recognition (required; core completion)

**User experience:** Show causal, frame-by-frame predictions *while the punch develops*, before the complete action is visible. Indicate the first reliable detection time, current predicted class, and confidence.

- Train a **causal temporal convolutional network (TCN)** over historical normalized poses (joint coordinates, valid keypoint masks, velocities, geometric features and optional stance context). Compare it with the Level 1 baseline and an LSTM.
- Keep all transformations causal: no bidirectional model, future-centered smoothing, full-clip normalization using future frames, future pose interpolation, or lookahead during inference.
- Include untrimmed video intervals with normal guard movement, feints, footwork and defensive motion. An event detector cannot be evaluated solely on presegmented positive clips.
- Use explicit `t_onset`, `t_peak` and `t_end` on a manually verified evaluation subset. The source `start_frame/end_frame` may be provisional, **not automatically an independently validated biomechanical onset**.
- Evaluate percentage of punches correctly identified before peak, first correct detection delay relative to onset, accuracy by fraction of the punch observed, event precision/recall, false positives per minute, and pipeline latency/FPS.
- Require group-separated held-out footage. Freeze test recordings; tune on validation only. Analyze feints, occlusions and stance cases.

**Provisional goals** (not promised model accuracy): first correct detection before `t_peak` on at least half of held-out punches; event-level precision >=80%; no more than one false alarm per minute. Revisit these thresholds after inspecting labels, prevalence, and measurement definitions. Completion requires credible *measured* results, not fabricated thresholds.

### Level 2.5 — Boxer and round analytics (priority expansion after Level 2)

**2.5A: No new action model required**

- Counts and rates: total punches, punches per minute, class distribution, lead versus rear hand where supported, activity by round segment.
- Combinations: detect temporal punch chains for a boxer using a configurable max inter-punch gap, e.g. initially 0.8 s; explicitly document grouping rules, overlaps, interruptions and assumptions.
- Identify most common combinations (double jab, jab-cross, jab-cross-hook), transitions and follow-ups; e.g. `P(next punch | current punch)` on observed detections.
- Provide a round timeline, top combinations, class distribution and an evidence-linked textual summary of observed actions. Use defensible descriptions, not invented judgments.
- Quality evaluation: compare analytics calculated from predictions versus manually annotated events on held-out rounds; report how detection errors propagate to totals and combo counts.

**2.5B: Additional labels and models required**

- Classify defensive head/body movements: slip left/right, duck, roll, backward evasion (subject to unambiguous class definitions and sufficient labeling).
- Optionally estimate footwork, lateral/forward/backward movement, guard changes and stance switches; must handle camera movement and viewpoint ambiguity.
- Create a fighter profile across multiple sessions; label it as *observed tendencies in analyzed footage*, not a universal fighter characteristic.
- Only then consider tracking and attributing actions to **two boxers in a match**. This adds identity association, interacting poses, occlusion, camera motion, and more challenging evaluation.

### Level 3 — Genuine pre-onset punch anticipation (stretch research)

- Predict likelihood of upcoming punch types using **only frames strictly before annotated onset**. Separate evaluation from Level 2.
- Define prediction horizons, e.g. 150/300/500 ms before onset, and include negative periods where no punch follows.
- Compare to simple frequency, prior-punch transition and random/majority baselines. Report class-wise metrics, false alarms and uncertainty; do not market speculative forecasting as deterministic prediction.
- If no improvement appears on unseen fighters, document negative results and stop. Level 2 remains complete.

### Level 4 — Future pose and fist trajectory prediction (optional long term)

- Forecast future skeleton joint coordinates/uncertainty, then derived glove/fist paths. This is a separate sequence-forecasting system with its own metrics and visualization; not a prerequisite for Level 3 or the public demo.

## 3. Dataset decision and constraints

### Primary candidate: BoxingVI

Source: https://github.com/Bikudebug/BoxingVI

The released documentation describes 6,915 punch examples, 20 source videos, six punch classes, start/end frame annotations, and AlphaPose skeletons. However, a 2026-10-02 GitHub issue reports that only 10 videos are present in the linked download folder: https://github.com/Bikudebug/BoxingVI/issues/1 . The downloaded content and permission to reuse videos were **not independently verified** as of this specification. Dataset evidence can be incomplete or internally inconsistent; do not assume the advertised counts are present.

### Secondary candidate: BoxingWeb

Source: https://github.com/gouba2333/BoxingWeb

Its repository documents video, pose and event JSON and offers code to map frame-bounded events to hand/technique. Published work describes 50 annotated rounds for BoxingWeb itself. It contains match footage and will require harder boxer tracking. Availability and licensing of hosted files remain to be validated.

### Small reference/backup: Punch_DL

Source: https://github.com/balezz/Punch_DL

Repository contains sample video, keypoints, and label files with punch frame intervals. Valuable for testing parsers and demonstrations, but repeated scripted sequences make it weak evidence for generalization or pre-onset anticipation.

### Additional data strategy

If no public dataset permits credible evaluation, collect staged footage from consenting participants, with planned variation in stance, tempo, negative movements, feints, camera position and recording session. Privately inspect third-party videos when authorized; use only cleared media in public portfolio material.

## 4. Annotation/data contract

Canonical event fields (proposed, validate against actual source):

- `video_id`: stable source-video identifier
- `boxer_id`: stable identity where known, otherwise unknown; never guess
- `session_id`: provenance and split grouping when known
- `frame_start` / `frame_end`: source annotation bounds
- `frame_onset` / `frame_peak`: **optional manually reviewed** timing labels
- `punch_type`: canonical taxonomy, or `unknown`
- `hand`: lead/rear only if annotated or stance established; don't confuse with left/right
- `annotator`, `source_dataset`, `annotation_version`, `review_status`
- `fps` and video timebase metadata stored once per source video; preserve original frame indexing

Maintain a separate video metadata table with path, resolution, frame count/duration, FPS/timebase, boxer/session identifiers, recording source, download status, license notes and hash where appropriate. Keep unknown metadata explicitly unknown.

All canonical transformations should be tested on synthetic fixtures and spot-checked against real videos. Define overlaps, gaps, feints and `no_punch` labeling rules; unlabeled intervals are **not automatically guaranteed negatives**.

## 5. Technical architecture

```text
Recorded video / (later) webcam
    -> decoder & metadata
    -> boxer detection / identity tracking (initially one boxer)
    -> pretrained pose landmark extractor
    -> aligned causal pose feature sequence
    -> self-trained temporal classifier / detector
    -> event segmentation and class probabilities
    -> punch event store (file-based JSON/Parquet)
    -> visual overlay + timeline + boxer/round analytics
```

**Stack:** Python 3.11; `uv` for dependency locking; PyTorch for training; OpenCV/FFmpeg for video; MediaPipe Pose Landmarker initially for single-boxer inference (compare other estimators such as RTMPose only if pose quality is inadequate); NumPy/Pandas for data; scikit-learn baseline; causal TCN main model; pytest/Ruff for tests; TensorBoard for experiments; Gradio for demo. Use YAML configuration files, simple local artifacts/JSON/Parquet; no external database or web API required initially.

**Pose contract:** document a fixed set and order of joints, coordinates, confidence/missing masks, timestamp units, normalization reference, boxer identity, handedness, and version. If the pose estimator changes, increment the feature schema version and regenerate/train or validate accordingly.

**Deployability:** benchmark decoding+pose estimation+classification+rendering end-to-end on the target machine, not only isolated network forward-pass speed. Webcam inference is optional after the video demo.

## 6. Evaluation design

- Preserve raw source video groups; create **boxer-independent** train/validation/test partitions where practical. Where boxer IDs cannot be verified, enforce session/video separation and disclose the remaining limitation. Never randomly split frames or near-duplicate punch clips.
- Level 1: clip macro-F1, per-class confusion and class support; full-video event precision/recall and boundary quality.
- Level 2: correct punch class versus percent of action already observed; lead time or detection delay relative to `frame_onset` and `frame_peak`; percentage detected before peak; false positives per minute of non-punch footage; precision-recall and end-to-end latency/FPS. Include difficult no-punch and feint intervals.
- Level 2.5: count error, confusion for defense classes, combo extraction accuracy and source-linked examples; do not treat ground truth calculated from model predictions as independent truth.
- Level 3: fixed strictly-pre-onset prediction horizons, precision/recall and calibration/false-alert cost; compare to prior-action and class-frequency baselines.
- Store config, seed, code revision, data/version, label version and aggregate metrics for each experiment. Review actual failure clips.

## 7. Repository layout (grow only as needed)

```text
boxing-motion-ai/
├── configs/
├── data/                       # large/raw data gitignored
├── docs/
│   ├── PROJECT_SPEC.md
│   ├── DATASET_AUDIT.md
│   └── ANNOTATION_GUIDE.md
├── src/boxing_motion_ai/
│   ├── data/                  # dataset adapters, validation, provenance
│   ├── vision/                # decoding, pose, tracking
│   ├── features/              # causal feature extraction
│   ├── models/                # baselines and temporal nets
│   ├── training/
│   ├── evaluation/
│   ├── inference/             # streaming prediction + event creation
│   ├── analytics/             # punch/combo/movement reports
│   └── demo/                  # Gradio
├── tests/
├── scripts/
├── pyproject.toml
├── uv.lock
└── README.md
```

Create directories **when first needed**, not empty placeholders everywhere. Maintain a usable Python CLI for audit/preprocess/train/evaluate/infer and a separate demo entrypoint.

## 8. Delivery sequence (illustrative, 2–3 month core)

1. **Phase 0 (week 1):** Initialize repository, add tests/CI, dataset source registry, metadata/annotation validators; manually inspect real data and permissions. Decide whether BoxingVI is usable.
2. **Level 1 (weeks 2–4):** Video→pose pipeline, normalized features, simple baseline, trained clip classifier, full-video inference and event evaluation.
3. **Level 2 (weeks 5–7):** Causal TCN and early-stage training; hard negatives; onset/peak review; event-level evaluation and latency tests.
4. **Core finish (week 8):** Video-upload demo, overlay, punch timeline, README with reproducible commands, honest metrics and limitations, short cleared demonstration recording.
5. **Level 2.5 (after core):** Punch/combo analytics dashboard, then defensive movement annotations/model and fighter-level profiles. Evaluate genuine match tracking only when ready.
6. **Levels 3–4 (optional):** Formal anticipation and pose forecasting studies, each with separate baselines and metrics.

Weeks are indicative; never claim progress on a phase until acceptance evidence exists.

## 9. First implementation task: Phase 0

Implement **only** the project foundation and dataset-audit CLI. Do not implement pose extraction, training, models, Gradio, or round analysis yet.

Acceptance:

- `uv run pytest` and `uv run ruff check .` pass.
- `uv run boxing-motion-ai --help` provides a working CLI.
- An audit command reads files from a user-specified local dataset directory, reports missing input honestly, validates supported annotations, reports class coverage and video metadata, and writes structured JSON plus human-readable Markdown.
- Synthetic fixtures test valid event boundaries, missing videos, invalid annotations, duplicate IDs/overlapping or suspicious ranges, and incomplete metadata.
- Private raw video and weight files are safely gitignored; sample data and third-party downloads aren't committed.
- README documents setup, obtaining data via its original source, audit usage and its limitations.
- A short `docs/DATASET_AUDIT.md` contains **verified vs unresolved** findings with no fabricated dataset counts.

After Phase 0, inspect the audit report with the project lead and approve the dataset decision before proceeding to Level 1.
