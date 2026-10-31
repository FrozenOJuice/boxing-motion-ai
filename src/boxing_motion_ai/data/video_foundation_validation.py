"""Read-only provenance/timing validation for the bounded video-foundation artifacts."""

import argparse
import hashlib
import json
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

from ..analysis.common import sha, write_json
from ..detection.localization_decoder_study import BASE, CACHE, cases
from ..services.dashboard import require
from ..vision.full_body import CHANNELS, VERSION, encode
from ..vision.observation_study import arrays, fresh, read
from ..vision.video_foundation import PIXEL_VERSION
from .blue_localization import validate as human_validation
from .video_foundation import CONFIG, TRAIN, admitted


def eligibility(clips, snapshot):
    """Extra exclusion-only weak-label overlay; never reinstate the 18 omitted human events."""
    result = []
    for c in clips:
        start, stop = c["source_frames"]
        reasons = []
        for interval in snapshot["intervals"]:
            if interval["provenance"]["recording"] != c["recording"]:
                continue
            for e in interval["excluded_events"]:
                if start < e["stop"] and e["start"] < stop:
                    reasons.append("excluded_manual_event_overlap")
            for e in interval["events"]:
                if (
                    [start, stop] == [e["start"], e["stop"]]
                    and not e.get("type_uncertain", False)
                    and c["clip_class"] != e["punch_type"]
                ):
                    reasons.append("exact_human_class_conflicts_with_provisional_class")
        result.append(
            dict(
                id=c["id"],
                weak_training_eligible=not reasons,
                reasons=sorted(set(reasons)),
                localization_loss=False,
                background_loss=False,
            )
        )
    return result


def verify(dataset, features, output):
    cfg = read(CONFIG)
    data = read(dataset / "manifest.json")
    video = read(features / "manifest.json")
    require(
        data["config_sha256"] == sha(CONFIG)
        and data["channels"] == CHANNELS
        and data["feature_version"] == VERSION
        and set(data["recordings"]) == set(TRAIN),
        "foundation_dataset_binding_changed",
    )
    for p, h in data["code_sha256"].items():
        require(sha(Path(p)) == h, "foundation_producer_code_changed")
    protocol = read(features / "protocol.json")
    require(
        protocol["config"] == cfg
        and video["pixel_version"] == PIXEL_VERSION
        and video["fullbody_version"] == VERSION
        and video["channels"] == CHANNELS,
        "foundation_video_contract_changed",
    )
    for p, h in protocol["code_sha256"].items():
        require(sha(Path(p)) == h, "foundation_video_producer_changed")
    weights = cfg["pretrained_encoder"]
    require(
        video["weights"] == weights and sha(Path(weights["path"])) == weights["sha256"],
        "foundation_encoder_changed",
    )
    current = human_validation(Path.cwd())
    require(
        data["localization_pin"] == read(cfg["localization_pin"])
        and data["localization_summary"] == current["summary"],
        "foundation_human_data_changed",
    )
    curated = read(dataset / "approved-source-receipt/selection.json")
    require(
        sha(Path(curated["review_ledger"]["path"])) == curated["review_ledger"]["sha256"],
        "foundation_review_changed",
    )
    seen = set()
    checked = 0
    coverage = {}
    for v, r in data["recordings"].items():
        root = Path(r["provenance"]["dataset"])
        for name, h in r["provenance"]["hashes"].items():
            require(sha(root / name) == h, "foundation_source_changed")
        require(
            sha(Path(f"data/raw/boxingvi/RGB_videos/{v}.mkv")) == r["video_sha256"],
            "foundation_video_source_changed",
        )
        f = dataset / v / "fullbody.npz"
        require(sha(f) == r["feature_sha256"], "foundation_body_cache_changed")
        a = arrays(f)
        raw = arrays(root / "poses.npz")
        source_rows = {
            r["source_row"]: r
            for r in [
                json.loads(line) for line in (root / "candidates.jsonl").read_text().splitlines()
            ]
        }
        indices = a["frame_index"]
        require(
            a["features"].shape == (len(indices), 209)
            and np.isfinite(a["features"]).all()
            and np.all(np.diff(indices) > 0),
            "foundation_body_shape",
        )
        for k in ("pts", "time_base_num", "time_base_den"):
            require(np.array_equal(a[k], raw[k][indices]), "foundation_body_source_clock_changed")
        clips = [c for c in data["weak_clips"] if c["recording"] == v]
        require(
            {int(c["id"].split(":")[1]) for c in clips} == set(curated["selected"][v]),
            "foundation_curation_membership_changed",
        )
        for c in clips:
            require(
                c["id"] not in seen
                and admitted(v, "weak_classification")
                and not any(
                    c[k]
                    for k in (
                        "class_verified",
                        "instance_boundaries_verified",
                        "localization_loss",
                        "background_loss",
                    )
                ),
                "foundation_weak_truth_promoted",
            )
            seen.add(c["id"])
            start, stop = c["source_frames"]
            source = source_rows[int(c["id"].split(":")[1])]
            require(
                c["clip_class"] == source["punch_type_provisional"]
                and c["source_frames"] == source["array_slice_half_open"]
                and c["annotation_sha256"]
                == hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
                "foundation_weak_annotation_changed",
            )
            require(
                np.array_equal(indices[c["feature_rows"]], np.arange(start, stop)),
                "foundation_weak_frame_join_changed",
            )

            def clock(i, raw=raw):
                return str(
                    Q(
                        int(raw["pts"][i]) * int(raw["time_base_num"][i]),
                        int(raw["time_base_den"][i]),
                    )
                )

            require(
                c["timestamps"] == [clock(start), clock(stop)], "foundation_weak_time_join_changed"
            )
            checked += 1
        coverage[v] = dict(
            normalization_valid_fraction=float(a["normalization_valid"].mean()),
            usable_joint_fractions=a["features"][:, 4:176:8].mean(0).tolist(),
        )
    require(len(seen) == len(data["weak_clips"]), "foundation_unregistered_weak_source")
    windows = 0
    source_cases = cases("training")
    require(set(video["cases"]) == set(cfg["feature_cases"]), "foundation_video_case_scope_changed")
    for name, r in video["cases"].items():
        require(r["recording"] in TRAIN, "foundation_reserved_video_features")
        p = features / name / "features.npz"
        require(sha(p) == r["feature_sha256"], "foundation_pretrained_cache_changed")
        a = arrays(p)
        n = len(a["pts"])
        idx = a["source_frame_index"]
        good = a["video_valid"]
        case = source_cases[name]
        for key in ("pts", "time_base_num", "time_base_den"):
            require(np.array_equal(a[key], case["arrays"][key]), "foundation_video_clock_changed")
        offset = case["provenance"]["source_start"]
        require(
            np.array_equal(idx, np.arange(offset, offset + n)),
            "foundation_video_source_map_changed",
        )
        rawpath = (
            CACHE / name / "poses.npz"
            if (CACHE / name / "poses.npz").exists()
            else BASE / name / "poses.npz"
        )
        require(sha(rawpath) == r["poses_sha256"], "foundation_context_poses_changed")
        raw = arrays(rawpath)
        body, normal, reset = encode(
            raw, case["video"]["width"] / case["video"]["height"], case["arrays"]["reset"]
        )
        require(
            np.array_equal(body, a["fullbody"])
            and np.array_equal(normal, a["normalization_valid"])
            and np.array_equal(reset, a["reset"]),
            "foundation_body_recompute_changed",
        )
        require(
            Q(r["source_stop_timestamp_exact"]) == Q(case["video"]["stop_timestamp_exact"]),
            "foundation_context_stop_changed",
        )
        require(
            a["fullbody"].shape == (n, 209)
            and a["video"].shape == (n, 512)
            and np.isfinite(a["video"]).all()
            and np.all(np.diff(idx) == 1),
            "foundation_fused_shape_or_clock",
        )
        require(
            not a["video"][~good].any() and (a["video_source_windows"][~good] == -1).all(),
            "foundation_unavailable_context_filled",
        )
        for i in np.flatnonzero(good):
            start, stop = a["video_source_windows"][i]
            require(start == idx[i] - 7 and stop == idx[i] + 9, "foundation_window_source_shift")
            u, z = i - 7, i + 9
            require(
                0 <= u < z <= n and not a["reset"][u + 1 : z].any(),
                "foundation_window_crosses_source_reset",
            )
            windows += 1
    out = fresh(output)
    selection = eligibility(data["weak_clips"], current)
    write_json(
        out / "eligibility.json",
        dict(
            version="weak-label-exclusion-overlay-1.0",
            dataset_sha256=sha(dataset / "manifest.json"),
            new_event_labels=False,
            rows=selection,
        ),
    )
    write_json(
        out / "verification.json",
        dict(
            weak_clips_checked=checked,
            eligible_weak_clips=sum(c["weak_training_eligible"] for c in selection),
            additional_weak_exclusions=sum(not c["weak_training_eligible"] for c in selection),
            native_video_windows_checked=windows,
            no_new_instance_truth=True,
            training=False,
            body_coverage=coverage,
            dataset_sha256=sha(dataset / "manifest.json"),
            features_sha256=sha(features / "manifest.json"),
            weights_sha256=weights["sha256"],
            rights_for_new_datasets_verified=False,
            validation_source_sha256=sha(Path(__file__)),
        ),
    )
    # Retain exact final producer source alongside hashes; earlier preflights stay untouched.
    for p in {*data["code_sha256"], *protocol["code_sha256"]}:
        path = Path(p)
        target = out / "source-snapshot" / path.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
    print("verified", checked, "weak clip joins and", windows, "native windows", flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    verify(a.dataset, a.features, a.output)
