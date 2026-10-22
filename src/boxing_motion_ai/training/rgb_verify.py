"""Replay frozen pilot checkpoints, cached RGB samples and matched training decisions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..vision.pipeline import sha256
from .experiment import seed_all, write_json
from .models import CausalTCN
from .multi_recording import ALL, load_pool, settings
from .multi_study import evaluate_record, verify_manifest
from .rgb import POLICY, VisualClips, crop_box, encoder, sampling_plan, views
from .rgb_review import decode_requested
from .rgb_study import PRIOR, SEED


def input_quality(ds, indices, boxes, size):
    """Report crop/sampling limitations, without changing data or selecting a policy."""
    lookup = {int(i): b for i, b in zip(indices, boxes, strict=True)}
    counts = {}
    for c in ds.clips:
        row = counts.setdefault(
            c["label"],
            dict(
                clips=0,
                unique_rgb_samples=[],
                native_frames=[],
                reliable_arm_points=0,
                arm_slots=0,
                reliable_arm_outside_crop=0,
                reliable_arm_in_hud_mask=0,
            ),
        )
        selected, _ = sampling_plan(c)
        row["clips"] += 1
        row["unique_rgb_samples"].append(len(selected))
        row["native_frames"].append(len(c["x"]))
        for j in selected:
            box = lookup[int(c["raw"]["frame_index"][j])]
            xy = c["raw"]["coordinates"][j, 11:17, :2]
            valid = c["raw"]["valid_mask"][j, 11:17]
            pixel = xy * np.array(size)
            outside = (
                (pixel[:, 0] < box[0])
                | (pixel[:, 0] >= box[2])
                | (pixel[:, 1] < box[1])
                | (pixel[:, 1] >= box[3])
            )
            masked = np.zeros(6, bool)
            for a, b, d, e in POLICY["hud_rectangles_normalized"][c["recording_id"]]:
                masked |= (xy[:, 0] >= a) & (xy[:, 0] <= d) & (xy[:, 1] >= b) & (xy[:, 1] <= e)
            row["arm_slots"] += 6
            row["reliable_arm_points"] += int(valid.sum())
            row["reliable_arm_outside_crop"] += int((valid & outside).sum())
            row["reliable_arm_in_hud_mask"] += int((valid & masked).sum())
    from .data import CLASSES

    return {CLASSES[k]: v for k, v in counts.items()}


def run(output, cache_path):
    output, cache_path = Path(output), Path(cache_path)
    if any(not p.resolve().is_relative_to(Path("reports").resolve()) for p in (output, cache_path)):
        raise ValueError("RGB artifacts must remain below ignored reports/")
    _, base, specs = settings("configs/level1_multi_recording.yaml")
    seed_all(SEED, base["threads"])
    pose = load_pool(specs)
    import yaml

    config = yaml.safe_load((output / "study.yaml").read_text())
    device = config["encoder_device"]
    model, transform, _ = encoder(device)
    checked_frames, checked_predictions, checked_epochs = 0, 0, 0
    maximum_feature_error = 0.0
    quality = {}
    previews = Image.new("RGB", (224 * 5, 224 * 6), (32, 32, 32))
    for row, v in enumerate(ALL):
        ds = pose[v]["xy_velocity"]
        manifest = json.loads((cache_path / v / "manifest.json").read_text())
        if sha256(cache_path / v / "features.npz") != manifest["feature_sha256"]:
            raise ValueError("Feature cache changed")
        with np.load(cache_path / v / "features.npz", allow_pickle=False) as z:
            indices, features, boxes = z["frame_index"], z["features"], z["boxes"]
        for c in ds.clips:
            _, mapping = sampling_plan(c)
            if not set(mapping) <= set(indices):
                raise ValueError("Sampling plan references uncached source frame")
        # One exact decoded source sample per recording, through the frozen encoder again.
        c = ds[0]
        i = int(c["raw"]["frame_index"][0])
        pos = int(np.where(indices == i)[0][0])
        from fractions import Fraction

        target = Fraction(manifest["exact_pts"][pos])
        source = json.loads(
            Path(f"data/processed/boxingvi/{v}/pose-full-image-v1/manifest.json").read_text()
        )
        if source["inputs"]["video"]["filename"] != f"{v}.mkv":
            raise ValueError("Unexpected source filename; V9 remains reserved")
        video = Path("data/raw/boxingvi/RGB_videos") / f"{v}.mkv"
        _, image, _ = next(decode_requested(video, {i: target}))
        quality[v] = input_quality(ds, indices, boxes, image.size)
        box, _ = crop_box(c["raw"]["coordinates"][0], c["raw"]["valid_mask"][0], image.size)
        variants = views(image, box, v)
        previews.paste(image.resize((224, 224)), (0, row * 224))
        for col, im in enumerate(variants):
            previews.paste(im.resize((224, 224)), ((col + 1) * 224, row * 224))
        with torch.inference_mode():
            replay = model(torch.stack([transform(im) for im in variants]).to(device)).cpu().numpy()
        maximum_feature_error = max(
            maximum_feature_error, float(np.max(np.abs(replay - features[pos])))
        )
        np.testing.assert_allclose(replay, features[pos], rtol=1e-4, atol=1e-4)
        checked_frames += 1
    previews.save(output / "reproducibility-previews.jpg")
    write_json(output / "input-quality.json", quality)
    summary = json.loads((output / "summary.json").read_text())
    for case, data in summary["cases"].items():
        root = output / case
        expected_audit = json.loads(
            (PRIOR / case / f"seed-{SEED}" / "xy_velocity/tcn/sampling-audit.json").read_text()
        )
        for mode in ("rgb", "hybrid", "background"):
            ds = {v: VisualClips(pose[v]["xy_velocity"], cache_path / v, mode) for v in ALL}
            dest = root / mode
            audit = json.loads((dest / "sampling-audit.json").read_text())
            checked_epochs += len(audit)
            if audit != expected_audit:
                raise ValueError("Paired source draws or mirror decisions changed")
            state = torch.load(dest / "best.pt", weights_only=True)
            if state["selection_recordings"] != ["V5"]:
                raise ValueError("Checkpoint selection changed")
            head = CausalTCN(state["input_channels"], base["hidden"], base["dropout"])
            head.load_state_dict(state["model_state"])
            head.eval()
            for v in ["V5", *data["fold"]["heldout"], *data["fold"]["train"]]:
                path = dest / f"{v}-predictions.json"
                previous = json.loads(path.read_text())
                temporary = output / "replay-predictions.json"
                _, detail = evaluate_record(head, ds[v], "x", temporary)
                temporary.unlink()
                if detail != previous:
                    raise ValueError("Frozen RGB prediction replay differs")
                checked_predictions += 1
    preserved = json.loads((output / "preserved-inputs.json").read_text())
    verify_manifest(preserved)
    result = dict(
        cache_frames_reencoded=checked_frames,
        frozen_prediction_sets=checked_predictions,
        prior_files_unchanged=len(preserved),
        paired_audit_epochs=checked_epochs,
        max_feature_replay_atol=1e-4,
        maximum_observed_feature_error=maximum_feature_error,
        v9_accessed=False,
    )
    write_json(output / "reproducibility.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    print(run(a.output, a.cache))
