"""Native feature extension for the frozen blue-only experiment; production is untouched."""

import argparse
import hashlib
from fractions import Fraction as Q
from pathlib import Path
from time import perf_counter

import av
import numpy as np
from PIL import Image, ImageDraw

from ..analysis.activity_contract import activity_pin
from ..analysis.common import sha, write_json
from ..data.blue_localization import PIN, targets, validate
from ..data.interval_pilot import ROLES
from ..data.verified_background import workspace_at
from ..services.dashboard import require
from ..services.interval_review import interval_timing
from ..training.rgb import GRAY
from ..training.rgb import POLICY as RGB_POLICY
from ..vision.observation_study import arrays, fresh, read
from .data import encode_source
from .decoder import DecoderConfig
from .localization_data import (
    POLICY,
    fixed_regions,
    load_cache,
    motion_pair,
    redacted,
    source_resets,
)
from .upload import load_model, predict_arrays

VERSION = "native-blue-localization-features-1.1"
# Fixed source-pixel QA masks: flags, wall posters, left gym sign. No labels/predictions.
DEV_IDENTIFIERS = [[0, 0, 1, 0.16], [0.55, 0.04, 1, 0.22], [0, 0.22, 0.29, 0.43]]
OLD_CACHE = Path("reports/boxingvi/punch-localization-features-v1-2")
CONFIG = Path("configs/punch_localization_blue_v1.json")


def development_redacted(image, box, torso):
    """Same native-motion preprocessing; only the pre-existing V5 HUD rectangles differ."""
    crop = image.convert("RGB").crop(box)
    support = Image.new("L", crop.size, 255)
    pen, valid = ImageDraw.Draw(crop), ImageDraw.Draw(support)
    w, h = image.size
    pixel = Image.new("L", image.size, 255).crop(box)
    valid.bitmap((0, 0), Image.eval(pixel, lambda x: 255 - x), fill=0)
    rectangles = [
        tuple(
            int(t * d)
            for t, d in zip(
                (max(0, a - 0.01), max(0, b - 0.01), min(1, c + 0.01), min(1, d + 0.01)),
                (w, h, w, h),
                strict=True,
            )
        )
        for a, b, c, d in (RGB_POLICY["hud_rectangles_normalized"]["V5"] + DEV_IDENTIFIERS)
    ]
    for a, b, c, d in rectangles + torso:
        rectangle = (a - box[0], b - box[1], c - box[0], d - box[1])
        pen.rectangle(rectangle, fill=GRAY)
        valid.rectangle(rectangle, fill=0)
    crop.paste(GRAY, mask=Image.eval(pixel, lambda x: 255 - x))
    n = POLICY["image_size"]
    return (
        np.asarray(crop.resize((n, n), Image.Resampling.BILINEAR)),
        np.asarray(support.resize((n, n), Image.Resampling.NEAREST)) > 0,
    )


def prepare(output, preflight):
    import cv2

    cv2.setNumThreads(2)
    root = Path.cwd()
    dataset = validate(root)
    out = fresh(output)
    protected = read(preflight / "protected-before.json")
    protected.update(read(".analysis-local/localization-evidence-preservation.json"))
    for p, h in protected.items():
        require("v9" not in p.lower() and sha(Path(p)) == h, "blue_protected_artifact_changed")
    write_json(out / "protected-before.json", protected)
    config = read(CONFIG)
    require(config["dataset"] == read(PIN), "blue_experiment_dataset_changed")
    # Verify the inherited cache and retain it by immutable reference, not by rewriting it.
    load_cache(OLD_CACHE, read("configs/punch_localization_smoke_v1.json"))
    ws = workspace_at(root, "localization-v1")
    binary = load_model(root, activity_pin(root))
    decoder = DecoderConfig.model_validate(read("configs/decoder_v1.json"))
    receipts = {}
    for interval in dataset["intervals"]:
        began = perf_counter()
        p, name = interval["provenance"], interval["id"]
        v = p["recording"]
        require(v in ("V4", "V5", "V6") and p["role"] == ROLES[v], "blue_feature_source_role")
        spec = ws.base._recording(v)
        prep = ws.base._preparation(v, spec)
        require(
            spec.inputs == p["source_inputs"]
            and prep["inputs"]["video_sha256"] == p["video_sha256"],
            "blue_feature_source_changed",
        )
        # The shared loader verifies the full pose hash and contiguous native frame sequence.
        with ws.frames.lock:
            full = ws.frames._load(v, spec, prep)
        start, stop = p["context_start"], p["context_end"] + 1
        interval_timing(full, start, stop, prep.get("confirmed_scene_cuts", []))
        source_pose = arrays(root / spec.dataset / "poses.npz")
        raw = {k: a[start:stop].copy() for k, a in source_pose.items()}
        raw["frame_index"] = np.arange(stop - start, dtype=np.int64)
        times = [
            Q(int(raw["pts"][i]) * int(raw["time_base_num"][i]), int(raw["time_base_den"][i]))
            for i in range(stop - start)
        ]
        end = Q(
            int(full["pts"][stop]) * int(full["time_base_num"][stop]),
            int(full["time_base_den"][stop]),
        )
        source_media = Path(f"data/raw/boxingvi/RGB_videos/{v}.mkv")
        with ws.base.store.open_regular(str(source_media)) as stream:
            ws.frames._verify(stream, p["video_sha256"])
        with av.open(str(source_media)) as container:
            track = container.streams.video[0]
            track.codec_context.thread_count = 2
            video = dict(
                width=track.width,
                height=track.height,
                frame_count=len(times),
                frames=[
                    dict(
                        frame_index=i,
                        pts=int(raw["pts"][i]),
                        time_base_num=int(raw["time_base_num"][i]),
                        time_base_den=int(raw["time_base_den"][i]),
                        timestamp_exact=str(t),
                        media_time=float(t - times[0]),
                    )
                    for i, t in enumerate(times)
                ],
                stop_timestamp_exact=str(end),
                duration_seconds=float(end - times[0]),
                confirmed_scene_cuts=[],
            )
            _, reset = source_resets(raw, video)
            pose, observable, _, _ = encode_source(raw, video["width"] / video["height"], [])
            pose = pose.numpy()
            pose[~observable] = 0
            box, torso = fixed_regions(raw, (track.width, track.height))
            container.seek(int(times[0] / track.time_base), stream=track, backward=True)
            visual, pixels, preview, previous, previous_mask = [], [], [], None, None
            for frame in container.decode(track):
                require(frame.pts is not None and not frame.is_corrupt, "blue_invalid_source_frame")
                actual = frame.pts * frame.time_base
                i = len(visual)
                if actual < times[i]:
                    continue
                require(actual == times[i], "blue_source_pts_mismatch")
                image = frame.to_image()
                pixels.append(
                    hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()
                )
                rgb, support = (
                    development_redacted(image, box, torso)
                    if v == "V5"
                    else redacted(image, box, torso, v)
                )
                row = (
                    motion_pair(previous, rgb, previous_mask, support, times[i] - times[i - 1])
                    if i and not reset[i] and observable[i - 1] and observable[i]
                    else np.zeros(97, np.float32)
                )
                visual.append(row)
                if i in (0, len(times) // 2, len(times) - 1):
                    preview.append(Image.fromarray(rgb))
                previous, previous_mask = rgb, support
                if len(visual) == len(times):
                    break
        require(len(visual) == len(times), "blue_incomplete_native_decode")
        y, mask, censored = targets(
            interval, video, start, Q(config["training"]["boundary_radius_seconds"])
        )
        dest = out / name
        dest.mkdir()
        np.savez_compressed(
            dest / "features.npz",
            pose=pose,
            visual=np.array(visual),
            observable=observable,
            reset=reset,
            targets=y,
            supervision_mask=mask,
            source_frame_index=np.arange(start, stop),
            **{k: raw[k] for k in ("frame_index", "pts", "time_base_num", "time_base_den")},
        )
        np.savez_compressed(dest / "poses.npz", **raw)
        references = [
            dict(
                event_id=e["event_id"],
                type=e["punch_type"],
                start_frame=e["start"] - start,
                stop_frame_exclusive=e["stop"] - start,
            )
            for e in interval["events"]
            if e["reference_eligible"]
        ]
        provenance = dict(
            kind="reviewed_source",
            recording=v,
            role=p["role"],
            source_start=start,
            review_start=p["start"],
            review_stop=p["stop"],
            exhaustive=False,
            selected_events=len(interval["events"]),
            scorable_events=len(references),
        )
        activity = predict_arrays(binary, raw, video, name, decoder)
        for stem, obj in (
            ("video", video),
            ("references", references),
            ("provenance", provenance),
            ("activity", activity),
        ):
            write_json(dest / f"{stem}.json", obj)
        canvas = Image.new("RGB", (160 * len(preview), 160))
        for j, im in enumerate(preview):
            canvas.paste(im, (160 * j, 0))
        canvas.save(dest / "redaction-preview.png")
        pose_origin = Path(f"data/processed/boxingvi/{v}/pose-full-image-v1/manifest.json")
        original = read(pose_origin)
        require(
            original["inputs"]["model"]["sha256"]
            == read("configs/assisted_level1_v1.json")["pose_model"]["sha256"],
            "blue_pose_model_changed",
        )
        receipt = dict(
            case=name,
            train_eligible=v in ("V4", "V6"),
            source_media=str(source_media),
            source_video_sha256=p["video_sha256"],
            source_inputs=spec.inputs,
            source_pixels_sha256=pixels,
            fixed_crop=box,
            identity_redaction=torso,
            pose_origin=dict(path=str(pose_origin), sha256=sha(pose_origin)),
            dataset=read(PIN),
            interval_id=name,
            censoring=censored,
            frames=len(times),
            supervised_frames=mask.sum(0).tolist(),
            model_observable_frames=int((observable & (np.array(visual)[:, -1] == 1)).sum()),
            metadata_sha256={
                s: sha(dest / f"{s}.json")
                for s in ("video", "references", "provenance", "activity")
            },
            features_sha256=sha(dest / "features.npz"),
            poses_sha256=sha(dest / "poses.npz"),
            seconds=perf_counter() - began,
        )
        write_json(dest / "receipt.json", receipt)
        receipts[name] = sha(dest / "receipt.json")
        print(
            name, len(times), receipt["supervised_frames"], round(receipt["seconds"], 2), flush=True
        )
    write_json(
        out / "manifest.json",
        dict(
            version=VERSION,
            dataset=read(PIN),
            cases=receipts,
            inherited_cache=dict(path=str(OLD_CACHE), sha256=sha(OLD_CACHE / "manifest.json")),
            policy=POLICY,
            development_hud=RGB_POLICY["hud_rectangles_normalized"]["V5"],
            development_identifiers=DEV_IDENTIFIERS,
            config_sha256=sha(CONFIG),
            code_sha256={
                str(p): sha(p)
                for p in [
                    Path(__file__),
                    Path("src/boxing_motion_ai/data/blue_localization.py"),
                    Path("src/boxing_motion_ai/detection/localization_data.py"),
                ]
            },
        ),
    )


def load(path):
    path = Path(path)
    manifest = read(path / "manifest.json")
    require(
        manifest["version"] == VERSION
        and manifest["dataset"] == read(PIN)
        and manifest["config_sha256"] == sha(CONFIG),
        "blue_cache_binding_changed",
    )
    dataset = validate(Path.cwd())
    require(set(manifest["cases"]) == {i["id"] for i in dataset["intervals"]}, "blue_cache_scope")
    require(
        manifest["policy"] == POLICY
        and manifest["development_hud"] == RGB_POLICY["hud_rectangles_normalized"]["V5"]
        and manifest["development_identifiers"] == DEV_IDENTIFIERS,
        "blue_feature_policy_changed",
    )
    for p, h in manifest["code_sha256"].items():
        require(sha(Path(p)) == h, "blue_feature_code_changed")
    require(
        sha(OLD_CACHE / "manifest.json") == manifest["inherited_cache"]["sha256"],
        "blue_old_cache_changed",
    )
    cases, _ = load_cache(OLD_CACHE, read("configs/punch_localization_smoke_v1.json"))
    for i in dataset["intervals"]:
        directory = path / i["id"]
        r = read(directory / "receipt.json")
        require(
            sha(directory / "receipt.json") == manifest["cases"][i["id"]], "blue_receipt_changed"
        )
        p = i["provenance"]
        require(
            r["train_eligible"] == (p["recording"] in ("V4", "V6"))
            and p["role"] == ROLES[p["recording"]],
            "blue_development_leakage",
        )
        require(
            sha(Path(r["source_media"])) == r["source_video_sha256"]
            and r["source_video_sha256"] == p["video_sha256"],
            "blue_source_changed",
        )
        for stem, h in r["metadata_sha256"].items():
            require(sha(directory / f"{stem}.json") == h, "blue_metadata_changed")
        require(
            sha(directory / "features.npz") == r["features_sha256"]
            and sha(directory / "poses.npz") == r["poses_sha256"],
            "blue_feature_bytes_changed",
        )
        a = arrays(directory / "features.npz")
        video = read(directory / "video.json")
        y, mask, _ = targets(
            i, video, p["context_start"], Q(read(CONFIG)["training"]["boundary_radius_seconds"])
        )
        require(
            np.array_equal(y, a["targets"]) and np.array_equal(mask, a["supervision_mask"]),
            "blue_cached_labels_changed",
        )
        require(
            a["pose"].shape == (r["frames"], 65)
            and a["visual"].shape == (r["frames"], 97)
            and np.isfinite(a["pose"]).all()
            and np.isfinite(a["visual"]).all(),
            "blue_features_invalid",
        )
        source_resets(arrays(directory / "poses.npz"), video)
        cases[i["id"]] = dict(
            arrays=a,
            video=video,
            receipt=r,
            references=read(directory / "references.json"),
            provenance=read(directory / "provenance.json"),
        )
    return cases, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output, args.preflight)
