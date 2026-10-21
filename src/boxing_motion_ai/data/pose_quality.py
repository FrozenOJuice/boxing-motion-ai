"""Read-only policy/association diagnostics. No fitting, prediction, imputation or training."""

import csv
import json
from pathlib import Path

from ..vision.contracts import BODY_JOINTS, JOINTS
from ..vision.pipeline import code_provenance, sha256, write_json
from .preparation import CLASS_MAP

CORE = (11, 12, 13, 14, 15, 16, 23, 24)
ANCHORS = (11, 12, 23, 24)
ARMS = ((11, 13, 15), (12, 14, 16))
POLICIES = {
    "strict_body": {"all_joints": list(BODY_JOINTS)},
    "strict_arms": {"all_joints": list(range(11, 17))},
    "core90": {"anchor_frames": 0.95, "one_arm_frames": 0.80, "core_entries": 0.90},
    "mask_aware80": {"anchor_frames": 0.95, "one_arm_frames": 0.80, "core_entries": 0.80},
    "mask_aware70": {"anchor_frames": 0.95, "one_arm_frames": 0.80, "core_entries": 0.70},
    "anchors_only_diagnostic": {"anchor_frames": 0.95, "one_arm_frames": 0.0, "core_entries": 0.0},
}
RECOMMENDED = "mask_aware70"


def policy_metrics(valid, missing):
    import numpy as np

    if not len(valid) or valid.shape[1] != 33 or valid.dtype != bool:
        raise ValueError("Policy metrics require nonempty [T,33] boolean masks")
    if missing.shape != (len(valid),) or missing.dtype != bool:
        raise ValueError("Missing mask must be a boolean frame vector")
    complete_arm = np.stack([valid[:, arm].all(axis=1) for arm in ARMS]).any(axis=0)
    return {
        "anchor_frames": float(valid[:, ANCHORS].all(axis=1).mean()),
        "one_arm_frames": float(complete_arm.mean()),
        "core_entries": float(valid[:, CORE].mean()),
        "missing_frames": float(missing.mean()),
    }


def retain(valid, missing, policy):
    if not len(valid):
        raise ValueError("Empty pose sequences cannot be retained")
    if "all_joints" in policy:
        return bool(valid[:, policy["all_joints"]].all()) and not missing.any()
    metrics = policy_metrics(valid, missing)
    return bool(
        metrics["missing_frames"] <= 0.10
        and all(metrics[key] >= value for key, value in policy.items())
    )


def mutual_information(feature, labels):
    """Plug-in categorical MI in bits, descriptive association only, not prediction accuracy."""
    import numpy as np

    feature, labels = np.asarray(feature), np.asarray(labels)
    if feature.ndim != 1 or labels.shape != feature.shape or not len(feature):
        raise ValueError("MI requires equally sized nonempty categorical vectors")
    _, x = np.unique(feature, return_inverse=True)
    _, y = np.unique(labels, return_inverse=True)
    shape = (int(x.max() + 1), int(y.max() + 1))
    counts = np.bincount(x * shape[1] + y, minlength=shape[0] * shape[1]).reshape(shape)
    p = counts / counts.sum()
    expected = p.sum(axis=1, keepdims=True) @ p.sum(axis=0, keepdims=True)
    used = p > 0
    return float((p[used] * np.log2(p[used] / expected[used])).sum())


def score_validity(xyz, confidence, threshold):
    """Sensitivity mask only; never changes the source mask or estimator detection threshold."""
    import numpy as np

    return (
        np.isfinite(xyz).all(axis=-1)
        & np.isfinite(confidence).all(axis=-1)
        & (xyz[..., :2] >= 0).all(axis=-1)
        & (xyz[..., :2] <= 1).all(axis=-1)
        & (confidence >= threshold).all(axis=-1)
        & (confidence <= 1).all(axis=-1)
    )


def association_report(coverage, labels, permutations=499, seed=20261020):
    import numpy as np

    # Fixed bins, chosen without optimizing a decision rule against source labels.
    bins = np.minimum((coverage * 4).astype(int), 3)
    observed = np.array([mutual_information(bins[:, j], labels) for j in BODY_JOINTS])
    rng = np.random.default_rng(seed)
    null = np.array(
        [
            max(mutual_information(bins[:, j], rng_labels) for j in BODY_JOINTS)
            for rng_labels in (rng.permutation(labels) for _ in range(permutations))
        ]
    )
    # Circular shifts preserve clip-order mask structure and label runs approximately.
    # They are a sensitivity diagnostic, NOT valid independent-session inference.
    shifts = [
        max(mutual_information(bins[:, j], np.roll(labels, k)) for j in BODY_JOINTS)
        for k in range(1, len(labels))
    ]
    entropy = mutual_information(labels, labels)
    return {
        "method": "categorical MI of per-candidate validity coverage bins vs source class",
        "bin_edges": [0, 0.25, 0.5, 0.75, 1],
        "seed": seed,
        "permutations": permutations,
        "class_entropy_bits": entropy,
        "joint_mi_bits": {
            JOINTS[j]: float(value) for j, value in zip(BODY_JOINTS, observed, strict=True)
        },
        "max_joint_mi_bits": float(observed.max()),
        "max_joint_fraction_of_class_entropy": float(observed.max() / entropy) if entropy else 0.0,
        "permutation_max_joint_p_exploratory": float(
            (1 + (null >= observed.max() - 1e-12).sum()) / (permutations + 1)
        ),
        "circular_shift_max_joint_fraction_ge_observed": float(
            np.mean(np.asarray(shifts) >= observed.max() - 1e-12)
        )
        if shifts
        else None,
        "limitation": "One scripted video; dependent clips; no held-out accuracy.",
    }


def _quantiles(values):
    import numpy as np

    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.1, 0.5, 0.9]).tolist() if len(finite) else [None] * 3


def analyze_arrays(arrays, candidates, permutations=499, required_classes=None):
    import numpy as np

    valid, xyz, confidence = arrays["valid_mask"], arrays["coordinates"], arrays["confidence"]
    if not np.array_equal(valid, score_validity(xyz, confidence, 0.5)):
        raise ValueError("Saved masks do not match the fixed source confidence policy (0.5)")
    required = set(CLASS_MAP.values()) if required_classes is None else set(required_classes)
    eligible = [
        c for c in candidates if c["annotation_eligible"] and c.get("quality_assessable", True)
    ]
    if not eligible:
        raise ValueError("No annotation-eligible candidates")
    eligible.sort(key=lambda c: c["source_row"])
    labels = np.array([c["punch_type_provisional"] for c in eligible])
    coverages, decisions = [], []
    for c in eligible:
        a, stop = c["array_slice_half_open"]
        if not 0 <= a < stop <= len(valid) or c["exclusion_reasons"]:
            raise ValueError("Invalid eligible candidate bounds or exclusion flags")
        if (arrays["scene_id"][a:stop] != arrays["scene_id"][a]).any():
            raise ValueError("Eligible candidate crosses a confirmed scene cut")
        v, missing = valid[a:stop], arrays["pose_status"][a:stop] == 2
        coverages.append(v.mean(axis=0))
        decisions.append(
            {
                "source_row": c["source_row"],
                "class": c["punch_type_provisional"],
                "metrics": policy_metrics(v, missing),
                "retained": {name: retain(v, missing, policy) for name, policy in POLICIES.items()},
            }
        )
    coverage = np.asarray(coverages)
    by_class = {}
    threshold_masks = {t: score_validity(xyz, confidence, t) for t in [0.3, 0.5, 0.7]}
    review = []
    for label in CLASS_MAP.values():
        indices = np.flatnonzero(labels == label)
        rows = [eligible[i] for i in indices]
        if not rows:
            if label in required:
                raise ValueError(f"Missing source class {label}; no class may silently disappear")
            by_class[label] = {
                "eligible": 0,
                "event_frame_count": 0,
                "coverage_status": "no_assessed_candidates",
                "retention": {name: 0 for name in POLICIES},
                "strict_body_sensitivity": {str(t): 0 for t in [0.3, 0.5, 0.7]},
                "joints": {
                    JOINTS[j]: {
                        "invalid_frame_fraction": None,
                        "mean_candidate_invalid_fraction": None,
                        "failure_reason_fractions": {},
                        "visibility_q10_q50_q90": [None] * 3,
                        "presence_q10_q50_q90": [None] * 3,
                    }
                    for j in BODY_JOINTS
                },
            }
            continue
        frames = np.concatenate([np.arange(*c["array_slice_half_open"]) for c in rows])
        x, conf = xyz[frames], confidence[frames]
        finite_xyz = np.isfinite(x).all(axis=-1)
        finite_scores = np.isfinite(conf).all(axis=-1)
        failures = {
            "nonfinite_coordinates": ~finite_xyz,
            "nonfinite_scores": ~finite_scores,
            "outside_image": finite_xyz
            & ((x[..., :2] < 0).any(axis=-1) | (x[..., :2] > 1).any(axis=-1)),
            "visibility_below_0.5": np.isfinite(conf[..., 0]) & (conf[..., 0] < 0.5),
            "presence_below_0.5": np.isfinite(conf[..., 1]) & (conf[..., 1] < 0.5),
            "scores_outside_probability_range": finite_scores
            & ((conf < 0).any(axis=-1) | (conf > 1).any(axis=-1)),
        }
        joints = {}
        for j in BODY_JOINTS:
            joints[JOINTS[j]] = {
                "invalid_frame_fraction": float((~valid[frames, j]).mean()),
                "mean_candidate_invalid_fraction": float((1 - coverage[indices, j]).mean()),
                "failure_reason_fractions": {
                    reason: float(values[:, j].mean()) for reason, values in failures.items()
                },
                "visibility_q10_q50_q90": _quantiles(conf[:, j, 0]),
                "presence_q10_q50_q90": _quantiles(conf[:, j, 1]),
            }
        sensitivity = {
            str(t): sum(
                bool(v[c["array_slice_half_open"][0] : c["array_slice_half_open"][1], 11:].all())
                for c in rows
            )
            for t, v in threshold_masks.items()
        }
        by_class[label] = {
            "eligible": len(rows),
            "event_frame_count": len(frames),
            "joints": joints,
            "retention": {
                name: sum(decisions[i]["retained"][name] for i in indices) for name in POLICIES
            },
            "strict_body_sensitivity": sensitivity,
        }
        # Best/median/worst core coverage, deterministic and separate from policy selection.
        ordered = sorted(
            indices.tolist(),
            key=lambda i: (coverage[i, list(CORE)].mean(), eligible[i]["source_row"]),
        )
        for rank, i in [
            ("worst", ordered[0]),
            ("median", ordered[len(ordered) // 2]),
            ("best", ordered[-1]),
        ]:
            c = eligible[i]
            a, stop = c["array_slice_half_open"]
            review.append(
                {
                    "source_row": c["source_row"],
                    "class": label,
                    "rank": rank,
                    "frames": [
                        int(arrays["frame_index"][n])
                        for n in sorted(set([a, (a + stop - 1) // 2, stop - 1]))
                    ],
                }
            )
    # A predeclared interpretable mask cue, not a fitted predictor.
    cue = coverage[:, 13] < 0.5
    cue_counts = {
        label: {
            "left_elbow_under_half_valid": int((cue & (labels == label)).sum()),
            "eligible": int((labels == label).sum()),
        }
        for label in CLASS_MAP.values()
    }
    return {
        "schema_version": "recording-pose-quality-1.1",
        "policies": POLICIES,
        "recommended_policy": RECOMMENDED,
        "source_mask_threshold": 0.5,
        "joint_groups": {
            "core": list(CORE),
            "anchors": list(ANCHORS),
            "arms": [list(arm) for arm in ARMS],
        },
        "policy_missing_frame_limit": 0.10,
        "policy_scope": "Original 0.5 masks; class-independent retention. No accuracy validation.",
        "by_class": by_class,
        "candidate_decisions": decisions,
        "visual_review_selection": review,
        "mask_class_association": association_report(coverage, labels, permutations),
        "predeclared_mask_cue_counts": cue_counts,
        "causation_limit": "Mechanical mask causes only. Physical causes need visual review.",
        "threshold_sensitivity_limit": "Diagnostic masks only; sources unchanged.",
    }


def assess_quality(dataset, output, permutations=499):
    import numpy as np

    dataset, output = Path(dataset), Path(output)
    if output.exists() or output.resolve().is_relative_to(dataset.resolve()):
        raise ValueError("Use a new report directory outside the prepared dataset")
    if any(p.name == "raw" for p in output.resolve().parents):
        raise ValueError("Reports must stay outside raw data")
    if not 1 <= permutations <= 5000:
        raise ValueError("Permutation count must be between 1 and 5000")
    manifest = json.loads((dataset / "preparation.json").read_text())
    if manifest["status"] != "complete" or sha256(dataset / "poses.npz") != manifest["npz_sha256"]:
        raise ValueError("Prepared dataset is incomplete or its pose array hash changed")
    candidates = [
        json.loads(line) for line in (dataset / "candidates.jsonl").read_text().splitlines()
    ]
    with np.load(dataset / "poses.npz", allow_pickle=False) as arrays:
        result = analyze_arrays(arrays, candidates, permutations, required_classes=())
    result["source_classes_present"] = sorted(set(manifest.get("class_map", CLASS_MAP).values()))
    result["source_classes_absent"] = sorted(
        set(CLASS_MAP.values()) - set(result["source_classes_present"])
    )
    result["classes_without_assessed_candidates"] = [
        k for k, v in result["by_class"].items() if not v["eligible"]
    ]
    result["video_id"] = manifest.get("video_id", "V5")
    result["recording_id"] = manifest.get("recording_id", result["video_id"])
    result["boxer_id"] = manifest.get("boxer_id")
    result["scope"] = manifest.get("scope", "full_video")
    result["inputs"] = {
        name: sha256(dataset / name)
        for name in ["preparation.json", "poses.npz", "candidates.jsonl"]
    }
    result["code"] = code_provenance()
    result["code"]["sha256"]["src/boxing_motion_ai/data/pose_quality.py"] = sha256(__file__)
    result["split_status"] = manifest["split_status"]
    result["annotation_uncertainties"] = manifest["mapping_uncertainty"]
    output.mkdir(parents=True)
    write_json(output / "quality.json", result)
    with (output / "joint-failure-matrix.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", *[JOINTS[j] for j in BODY_JOINTS]])
        for label, values in result["by_class"].items():
            writer.writerow(
                [
                    label,
                    *[values["joints"][JOINTS[j]]["invalid_frame_fraction"] for j in BODY_JOINTS],
                ]
            )
    write_json(
        output / "policy-index.json",
        {
            "schema_version": "recording-mask-policy-index-1.1",
            "video_id": result["video_id"],
            "recording_id": result["recording_id"],
            "boxer_id": result["boxer_id"],
            "inputs": result["inputs"],
            "policy": RECOMMENDED,
            "definition": POLICIES[RECOMMENDED],
            "max_missing_frame_fraction": 0.10,
            "source_mask_threshold": 0.5,
            "joint_groups": result["joint_groups"],
            "selected_source_rows": [
                c["source_row"] for c in result["candidate_decisions"] if c["retained"][RECOMMENDED]
            ],
            "all_policy_indices": {
                name: [
                    c["source_row"] for c in result["candidate_decisions"] if c["retained"][name]
                ]
                for name in POLICIES
            },
            "note": "Immutable poses/scores/masks. No filling/training.",
        },
    )
    lines = [
        f"# {manifest.get('video_id', 'V5')} partial-pose policy comparison",
        "",
        "All counts use annotation-eligible candidates and original masks at 0.5. No training.",
        "",
        "| Class | Eligible | " + " | ".join(POLICIES) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in POLICIES) + " |",
    ]
    for label, values in result["by_class"].items():
        lines.append(
            f"| {label} | {values['eligible']} | "
            + " | ".join(str(values["retention"][name]) for name in POLICIES)
            + " |"
        )
    lines += [
        "",
        "See quality.json for per-joint reasons, sensitivity, policy definitions and descriptive",
        "mask/class association. A high retention count is not proof of accurate coordinates.",
        "policy-index.json selects existing rows; original masks/data stay unchanged.",
        "",
    ]
    (output / "policies.md").write_text("\n".join(lines))
    return result
