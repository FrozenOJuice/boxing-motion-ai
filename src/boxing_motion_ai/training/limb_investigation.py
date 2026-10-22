"""Read-only feature-coverage investigation; no model, optimizer or fitting branch."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ..vision.pipeline import sha256
from .data import CLASSES, PoseClips, exact_times
from .limbs import GROUPS, LIMB_POLICY, MASK_COLUMNS, limb_features
from .velocity import VelocityClips

RECORDINGS = ("V4", "V5", "V6", "V7", "V8", "V10")


def quantiles(values):
    a = np.asarray(values)
    return (
        dict(zip(("p05", "median", "p95"), np.quantile(a, [0.05, 0.5, 0.95]).tolist(), strict=True))
        if len(a)
        else None
    )


def describe_clips(clips):
    """Aggregate selected clips, preserving source rows and side-specific availability."""
    if not clips:
        return {"clips": 0, "frames": 0, "coverage": None, "by_side": None}
    masks = np.concatenate([c["masks"] for c in clips])
    sides = []
    for side in range(2):
        rows = [c["sides"][side] for c in clips]
        sides.append(
            {
                "clips_any_elbow_angle": sum(r["angle_frames"] > 0 for r in rows),
                "clips_majority_elbow_angle": sum(r["majority_angle"] for r in rows),
                "clips_majority_shoulder_wrist": sum(r["majority_wrist"] for r in rows),
                "elbow_angle_degrees": quantiles([v for r in rows for v in r["angles"]]),
                "upper_arm_length_torso_units": quantiles(
                    [v for r in rows for v in r["upper_lengths"]]
                ),
                "forearm_length_torso_units": quantiles(
                    [v for r in rows for v in r["fore_lengths"]]
                ),
                "short_upper_arm_frames": sum(r["short_upper"] for r in rows),
                "short_forearm_frames": sum(r["short_fore"] for r in rows),
                "angle_jump_opportunities": sum(r["angle_pairs"] for r in rows),
                "large_angle_jumps": sum(r["large_jumps"] for r in rows),
            }
        )
    return {
        "clips": len(clips),
        "frames": len(masks),
        "coverage": masks.mean(0).tolist(),
        "either_arm_coverage": masks.any(1).mean(0).tolist(),
        "both_arm_coverage": masks.all(1).mean(0).tolist(),
        "shoulder_width_torso_units": quantiles([v for c in clips for v in c["shoulder_width"]]),
        "by_side": sides,
    }


def clip_measurements(c, settings, gap):
    x = c["x"]
    geometry = limb_features(x).reshape(-1, 2, 17).numpy()
    masks = geometry[..., list(MASK_COLUMNS)].astype(bool)
    joint = x[:, :40].reshape(-1, 8, 5).numpy()
    times = exact_times(c)
    raw = c["raw"]
    consecutive = np.array(
        [
            raw["frame_index"][i] == raw["frame_index"][i - 1] + 1
            and raw["scene_id"][i] == raw["scene_id"][i - 1]
            and raw["segment_id"][i] == raw["segment_id"][i - 1]
            and raw["pose_status"][i] != 2
            and raw["pose_status"][i - 1] != 2
            and 0 < times[i] - times[i - 1] <= gap
            for i in range(1, len(x))
        ],
        dtype=bool,
    )
    sides = []
    for side in range(2):
        upper = geometry[:, side, 3:5]
        fore = geometry[:, side, 6:8]
        angle = np.degrees(np.arccos(geometry[:, side, 9].clip(-1, 1)))
        angle_valid = masks[:, side, 3]
        pairs = angle_valid[1:] & angle_valid[:-1] & consecutive
        sides.append(
            {
                "angle_frames": int(angle_valid.sum()),
                "majority_angle": bool(angle_valid.mean() >= settings["majority_frame_fraction"]),
                "majority_wrist": bool(
                    masks[:, side, 0].mean() >= settings["majority_frame_fraction"]
                ),
                "angles": angle[angle_valid].tolist(),
                "upper_lengths": np.linalg.norm(upper, axis=-1)[masks[:, side, 1]].tolist(),
                "fore_lengths": np.linalg.norm(fore, axis=-1)[masks[:, side, 2]].tolist(),
                "short_upper": int(
                    (
                        (
                            np.linalg.norm(upper, axis=-1)
                            < settings["short_projected_segment_torso_units"]
                        )
                        & masks[:, side, 1]
                    ).sum()
                ),
                "short_fore": int(
                    (
                        (
                            np.linalg.norm(fore, axis=-1)
                            < settings["short_projected_segment_torso_units"]
                        )
                        & masks[:, side, 2]
                    ).sum()
                ),
                "angle_pairs": int(pairs.sum()),
                "large_jumps": int(
                    (
                        (np.abs(np.diff(angle)) > settings["large_consecutive_angle_jump_degrees"])
                        & pairs
                    ).sum()
                ),
            }
        )
    return {
        "source_row": c["source_row"],
        "class": CLASSES[c["label"]],
        "masks": masks,
        "sides": sides,
        "shoulder_width": np.linalg.norm(joint[:, 0, :2] - joint[:, 1, :2], axis=-1).tolist(),
    }


def load_settings(path):
    settings = yaml.safe_load(Path(path).read_text())
    expected = {
        "schema_version": "limb-investigation-1.0",
        "reference_registry": "configs/level1_reference.yaml",
        "base_config": "configs/level1_initial.yaml",
        "transfer_config": "configs/level1_velocity_transfer.yaml",
        "short_projected_segment_torso_units": 0.05,
        "majority_frame_fraction": 0.5,
        "large_consecutive_angle_jump_degrees": 30.0,
        "recordings": list(RECORDINGS),
    }
    if settings != expected:
        raise ValueError("Use the frozen read-only six-recording investigation; V9 reserved")
    return settings


def write_report(output, result):
    lines = [
        "# Candidate arm-geometry coverage",
        "",
        result["decision"],
        "",
        "Coverage is source-validity availability, not verified anatomical correctness.",
        "",
        "| Recording | Clips | Wrist-relative L/R | Elbow cosine L/R | Either elbow |",
        "|---|---:|---:|---:|---:|",
    ]
    for v, d in result["recordings"].items():
        a = d["all"]
        coverage = a["coverage"]
        lines.append(
            f"| {v} | {a['clips']} | {coverage[0][0]:.2%} / {coverage[1][0]:.2%} | "
            f"{coverage[0][3]:.2%} / {coverage[1][3]:.2%} | "
            f"{a['either_arm_coverage'][3]:.2%} |"
        )
    for v, d in result["recordings"].items():
        lines += [
            "",
            f"## {v}",
            "",
            "| Class | Clips | Elbow coverage L/R | Majority-angle clips L/R | "
            "Majority-wrist clips L/R |",
            "|---|---:|---:|---:|---:|",
        ]
        for c, a in d["by_class"].items():
            if not a["clips"]:
                lines.append(f"| {c} | 0 | no eligible examples | — | — |")
                continue
            left, right = a["by_side"]
            lines.append(
                f"| {c} | {a['clips']} | {a['coverage'][0][3]:.2%} / "
                f"{a['coverage'][1][3]:.2%} | {left['clips_majority_elbow_angle']} / "
                f"{right['clips_majority_elbow_angle']} | "
                f"{left['clips_majority_shoulder_wrist']} / "
                f"{right['clips_majority_shoulder_wrist']} |"
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def run(config_path, output, progress=print):
    from fractions import Fraction

    from .experiment import load_config, seed_all, write_json
    from .velocity_transfer import load_transfer_config, source_support

    config_path, output = Path(config_path), Path(output)
    settings = load_settings(config_path)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Choose a fresh output below reports/")
    base = load_config(Path(settings["base_config"]))
    transfer = load_transfer_config(Path(settings["transfer_config"]))
    specs = [base["train"], base["development"], *transfer["recordings"]]
    if [s["recording"] for s in specs] != list(RECORDINGS):
        raise ValueError("Unexpected source scope")
    prior_root = Path("reports/boxingvi/level1-velocity-transfer-v1")
    preserved = json.loads((prior_root / "preserved-inputs.json").read_text())
    preserved.update({str(p): sha256(p) for p in prior_root.rglob("*") if p.is_file()})
    if any("/V9/" in p or "/V9." in p for p in preserved):
        raise ValueError("Reserved inputs outside scope")
    if any(sha256(p) != digest for p, digest in preserved.items()):
        raise ValueError("Previous artifact changed")
    seed_all(0, 2)
    result = {
        "schema_version": "limb-investigation-result-1.0",
        "training_performed": False,
        "policy": LIMB_POLICY,
        "groups": GROUPS,
        "settings": settings,
        "reference_registry_sha256": sha256(settings["reference_registry"]),
        "recordings": {},
        "decision": "Read-only investigation; no training stage enabled. "
        "Inspect class- and side-specific coverage before approving any feature experiment.",
    }
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(config_path.read_text())
    write_json(output / "preserved-inputs.json", preserved)
    for spec in specs:
        v = spec["recording"]
        diagnostic = v not in ("V4", "V5")
        ds = VelocityClips(PoseClips(spec, v, diagnostic=diagnostic), diagnostic=diagnostic)
        gap = Fraction(ds.provenance["velocity_policy"]["max_gap_seconds_exact"])
        clips = [clip_measurements(c, settings, gap) for c in ds.clips]
        result["recordings"][v] = {
            "all": describe_clips(clips),
            "by_class": {c: describe_clips([x for x in clips if x["class"] == c]) for c in CLASSES},
            "clip_coverage": [
                {
                    "source_row": c["source_row"],
                    "class": c["class"],
                    "frames": len(c["masks"]),
                    "coverage": c["masks"].mean(0).tolist(),
                }
                for c in clips
            ],
            "provenance": ds.provenance,
            "source_support": source_support(ds),
            "source_group": "original V4 training"
            if v == "V4"
            else ("V5 development" if v == "V5" else spec["source_group_id"]),
            "related_recordings_guard": spec.get("related_recordings_guard", [v]),
        }
        progress(f"{v}: {len(ds)} eligible clips measured; no masks/selection changed")
    sources = [
        Path(__file__),
        *(Path(__file__).with_name(name) for name in ("limbs.py", "data.py", "velocity.py")),
        config_path,
        Path(settings["base_config"]),
        Path(settings["transfer_config"]),
        Path(settings["reference_registry"]),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    result["implementation_hashes"] = {}
    for source in sources:
        relative = source.resolve().relative_to(Path.cwd().resolve())
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        result["implementation_hashes"][relative.as_posix()] = sha256(source)
    write_json(output / "summary.json", result)
    write_report(output, result)
    if any(sha256(p) != digest for p, digest in preserved.items()):
        raise ValueError("Previous artifact changed")
    write_json(output / "preservation.json", {"verified_files": len(preserved), "changed": []})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
