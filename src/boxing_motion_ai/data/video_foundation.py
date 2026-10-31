"""Task-specific adapter over approved curation and the immutable blue localization snapshot."""

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

from ..analysis.common import sha, write_json
from ..services.dashboard import require
from ..training.data import CLASSES, PoseClips, exact_times
from ..vision.full_body import CHANNELS, VERSION, encode
from ..vision.observation_study import arrays, fresh, read
from .blue_localization import validate
from .curated_subset import prepare as curated_receipt
from .interval_pilot import GROUPS

CONFIG = Path("configs/video_foundation_v1.json")
TRAIN = ("V4", "V6")


def admitted(recording, role):
    """This milestone cannot acquire/unhold a source or inherit legacy evaluation training roles."""
    return recording in TRAIN and role in ("weak_classification", "existing_localization")


def safe_weak_rows(clips, selected):
    result = [c for c in clips if c["source_row"] in selected]
    require({c["source_row"] for c in result} == set(selected), "foundation_curation_not_in_pool")
    require(len(result) == len(selected), "foundation_duplicate_rows")
    for c in result:
        require(admitted(c["recording_id"], "weak_classification"), "foundation_forbidden_role")
        require(not c["annotation"]["exclusion_reasons"], "foundation_excluded_row")
    return result


def prepare(output):
    cfg = read(CONFIG)
    require(
        cfg["training_recordings"] == list(TRAIN)
        and cfg["new_recordings_admitted"] == []
        and cfg["excluded_manual_events"] == 18,
        "foundation_role_or_subset_change",
    )
    snapshot = validate(Path.cwd())  # Existing human-save, conflict and source-group validation.
    out = fresh(output)
    approval = read(cfg["curation_approval"])
    reviewed = read(approval["reviewed_config"]["path"])
    receipt = curated_receipt(cfg["curation_approval"], out / "approved-source-receipt")
    entries, recordings, video_hashes = [], {}, set()
    # The receipt audits legacy metadata; only these two sources may supply feature tensors.
    for v in TRAIN:
        spec = reviewed["sources"][v]
        ds = PoseClips(
            dict(
                dataset=spec["dataset"],
                policy=spec["policy"]["path"],
                pose_run=f"data/processed/boxingvi/{v}/pose-full-image-v1",
            ),
            v,
            multi_recording=True,
        )
        clips = safe_weak_rows(ds.clips, receipt["selected"][v])
        source = Path(f"data/raw/boxingvi/RGB_videos/{v}.mkv")
        digest = sha(source)
        require(
            digest == ds.provenance["source_input_hashes"]["video_sha256"],
            "foundation_video_changed",
        )
        require(digest not in video_hashes, "foundation_duplicate_source")
        video_hashes.add(digest)
        raw = arrays(Path(spec["dataset"]) / "poses.npz")
        width, height = ds.provenance["source_dimensions"]
        source_resets = np.zeros(len(raw["frame_index"]), bool)
        preparation = read(Path(spec["dataset"]) / "preparation.json")
        for cut in preparation.get("confirmed_scene_cuts", []):
            require(type(cut) is int and 0 <= cut < len(source_resets), "foundation_invalid_cut")
            source_resets[cut] = True
        x, normal, reset = encode(raw, width / height, source_resets)
        indices = sorted({int(i) for c in clips for i in c["raw"]["frame_index"]})
        frames = np.array(indices, np.int64)
        dest = out / v
        dest.mkdir()
        np.savez_compressed(
            dest / "fullbody.npz",
            features=x[frames],
            normalization_valid=normal[frames],
            reset=reset[frames],
            **{k: raw[k][frames] for k in ("frame_index", "pts", "time_base_num", "time_base_den")},
        )
        lookup = {frame: j for j, frame in enumerate(indices)}
        for c in clips:
            a = c["raw"]
            times = exact_times(c)
            start, stop = c["annotation"]["array_slice_half_open"]
            # Exact rational stop; never derive it from nominal FPS.
            end = (
                str(
                    Q(
                        int(raw["pts"][stop]) * int(raw["time_base_num"][stop]),
                        int(raw["time_base_den"][stop]),
                    )
                )
                if stop < len(raw["pts"])
                else None
            )
            require(end is not None, "foundation_stop_clock_unavailable")
            entries.append(
                dict(
                    id=f"{v}:{c['source_row']}",
                    recording=v,
                    source_group=GROUPS[v],
                    source_frames=[start, stop],
                    timestamps=[str(times[0]), end],
                    feature_rows=[lookup[int(i)] for i in a["frame_index"]],
                    clip_class=CLASSES[c["label"]],
                    class_verified=False,
                    instance_boundaries_verified=False,
                    localization_loss=False,
                    background_loss=False,
                    label_source="original_provisional_workbook",
                    annotation_sha256=hashlib.sha256(
                        json.dumps(c["annotation"], sort_keys=True).encode()
                    ).hexdigest(),
                )
            )
        counts = Counter(e["clip_class"] for e in entries if e["recording"] == v)
        valid = raw["valid_mask"][frames]
        recordings[v] = dict(
            clips=len(clips),
            class_counts=dict(counts),
            unique_frames=len(frames),
            feature_sha256=sha(dest / "fullbody.npz"),
            video_sha256=digest,
            source_pose_sha256=sha(Path(spec["dataset"]) / "poses.npz"),
            provenance=ds.provenance,
            related_recordings_guard=GROUPS[v],
            valid_joint_fractions=valid.mean(0).tolist(),
            identity_known=False,
            rights="existing_user_authorized_local_research_only",
        )
    write_json(
        out / "manifest.json",
        dict(
            version="video-foundation-dataset-1.0",
            config_sha256=sha(CONFIG),
            feature_version=VERSION,
            channels=CHANNELS,
            recordings=recordings,
            weak_clips=entries,
            localization_pin=read(cfg["localization_pin"]),
            localization_summary=snapshot["summary"],
            localization_changed=False,
            weak_clips_are_not_new_instance_references=True,
            new_recordings=0,
            roles=dict(
                V4="training",
                V6="training",
                V5="development_only",
                V7="evaluation_only",
                V8="evaluation_only",
                V10="evaluation_only",
                V1="alignment_hold",
                V2="alignment_hold",
                V3="alignment_hold",
                V9="inaccessible",
            ),
            conservative_publisher_guards=[["V1", "V6"], ["V2", "V3", "V4"], ["V7", "V8"]],
            untouched_reserved=["V5", "V7", "V8", "V9", "V10"],
            unresolved_alignment_holds=["V1", "V2", "V3"],
            training_performed=False,
            code_sha256={
                str(p): sha(p)
                for p in [Path(__file__), Path("src/boxing_motion_ai/vision/full_body.py")]
            },
        ),
    )
    print({v: recordings[v]["class_counts"] for v in TRAIN}, flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args().output)
