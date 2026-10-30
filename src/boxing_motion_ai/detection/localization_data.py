"""Native RGB-motion cache and adapter over the existing frozen continuous manifest.

No trainable visual encoder, joint filling, static RGB appearance or label-guided crops.
The accepted poses and all previous caches are read-only.
"""

import argparse
import shutil
from fractions import Fraction as Q
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from ..analysis.common import sha, write_json
from ..data.continuous_dataset import validate
from ..data.interval_pilot import GROUPS
from ..training.rgb import GRAY, crop_box
from ..training.rgb import POLICY as RGB_POLICY
from ..vision.observation_study import BASE, CASES, arrays, fresh, read, study_manifest
from .data import encode_source

VERSION = "native-rgb-motion-masked-pose-1.2"
PIXELS = Path("reports/boxingvi/pose-observation-study-v1-complete")
TRAIN = {f"{v}-motion-{i}" for v in ("V4", "V6") for i in range(1, 7)} | {
    f"V4-bg-{i}" for i in range(1, 4)
}
POLICY = dict(
    version=VERSION,
    image_size=160,
    grid=4,
    visual_channels=97,
    pose_channels=65,
    rgb="adjacent-frame absolute RGB and signed/RMS grayscale differences only; no static color",
    flow="Farneback .5/2/15/2/5/1.1/0; subtract median unmasked flow; crop widths per second",
    scale="tanh of differences/5 and normalized flow/2; no fitted normalization",
    crop="fixed median of existing 4.5-torso square boxes, whole-case offline, no label inputs",
    redaction="existing fixed HUD rectangles expanded 1 percent; V4 bag logo rectangle; "
    "fixed whole-case shirt/lower-body unions; head/guard retained",
    source_identifier_rectangles={"V4": [[0.24, 0.28, 0.28, 0.46]], "V6": []},
    masks="pairwise pixel support; scalar motion-valid bit; no raw crop/identity metadata",
    timing="every decoded native frame; exact adjacent PTS; no resampling or holding",
)


def authorized(manifest):
    intervals = manifest["intervals"]
    if {i.get("id") for i in intervals} != TRAIN or len(intervals) != len(TRAIN):
        raise ValueError("Unexpected or held continuous training interval")
    for i in intervals:
        p = i["provenance"]
        if (
            p["recording"] not in ("V4", "V6")
            or not i["exhaustive"]
            or p["role"] != "train_candidate"
            or p["source_group_guard"] != GROUPS[p["recording"]]
        ):
            raise ValueError("Localization source-role/group/exhaustiveness restriction")
    if sum(len(i["events"]) for i in intervals) != 11:
        raise ValueError("Frozen 11-event supervision required")
    return {i["id"]: i for i in intervals}


def fixed_regions(raw, size):
    good = raw["valid_mask"][:, [11, 12, 23, 24]].all(1) & (raw["pose_status"] != 2)
    if not good.any():
        return (0, 0, max(size), max(size)), [(0, 0, *size)]
    boxes = [
        crop_box(c, m, size)[0]
        for c, m in zip(raw["coordinates"][good], raw["valid_mask"][good], strict=True)
    ]
    box = tuple(np.rint(np.median(boxes, axis=0)).astype(int).tolist())
    anchors = raw["coordinates"][good][:, [11, 12, 23, 24], :2] * np.array(size)
    shoulder, hip = anchors[:, :2].mean(1), anchors[:, 2:].mean(1)
    length = np.linalg.norm(shoulder - hip, axis=1)
    # Static unions remove shirt/shorts text, not the face/guard workspace. No moving edges.
    torso = (
        int(np.min(anchors[..., 0]) - 0.05 * np.median(length)),
        int(np.min(shoulder[:, 1])),
        int(np.max(anchors[..., 0]) + 0.05 * np.median(length)),
        int(np.max(hip[:, 1] + 0.10 * length)),
    )
    lower_body = (0, int(np.min(hip[:, 1] - 0.05 * length)), size[0], size[1])
    return box, [torso, lower_body]


def redacted(image, box, torso, recording):
    if recording not in ("V4", "V6"):
        raise ValueError("Only V4/V6 visual sources authorized")
    crop = image.convert("RGB").crop(box)
    support = Image.new("L", crop.size, 255)
    pen, valid = ImageDraw.Draw(crop), ImageDraw.Draw(support)
    w, h = image.size
    normalized = (
        RGB_POLICY["hud_rectangles_normalized"][recording]
        + POLICY["source_identifier_rectangles"][recording]
    )
    rectangles = [
        tuple(
            int(t * d)
            for t, d in zip(
                (max(0, a - 0.01), max(0, b - 0.01), min(1, c + 0.01), min(1, d + 0.01)),
                (w, h, w, h),
                strict=True,
            )
        )
        for a, b, c, d in normalized
    ]
    rectangles += [torso] if len(torso) == 4 and np.isscalar(torso[0]) else torso
    # Outside-source padding must not contribute flow or black-vs-gray edge differences.
    pixel = Image.new("L", image.size, 255).crop(box)
    valid.bitmap((0, 0), Image.eval(pixel, lambda x: 255 - x), fill=0)
    for a, b, c, d in rectangles:
        rectangle = (a - box[0], b - box[1], c - box[0], d - box[1])
        pen.rectangle(rectangle, fill=GRAY)
        valid.rectangle(rectangle, fill=0)
    # Make padding identical to source-redaction gray; it is masked out in any case.
    crop.paste(GRAY, mask=Image.eval(pixel, lambda x: 255 - x))
    n = POLICY["image_size"]
    return (
        np.asarray(crop.resize((n, n), Image.Resampling.BILINEAR)),
        np.asarray(support.resize((n, n), Image.Resampling.NEAREST)) > 0,
    )


def motion_pair(previous, current, previous_mask, current_mask, elapsed):
    """Native adjacent-frame image motion, not recovered anatomical coordinates."""
    import cv2

    if elapsed <= 0:
        raise ValueError("Positive exact adjacent elapsed time required")
    valid = previous_mask & current_mask
    # Erode away redaction boundaries so artificial mask edges cannot become evidence.
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if not valid.any():
        return np.zeros(97, np.float32)
    if np.array_equal(previous, current):
        # Exact zero motion, rather than numerical optical-flow residuals of static texture.
        return np.array([*([0.0] * 96), 1.0], np.float32)
    a, b = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (previous, current))
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 2, 15, 2, 5, 1.1, 0)
    flow -= np.median(flow[valid], axis=0)
    dt = float(elapsed)
    rgb = np.abs(current.astype(float) - previous.astype(float)).mean(2) / 255 / dt
    gray = (b.astype(float) - a.astype(float)) / 255 / dt
    u, v = flow[..., 0] / len(a) / dt, flow[..., 1] / len(a) / dt
    values = []
    for rr in np.array_split(np.arange(len(a)), 4):
        for cc in np.array_split(np.arange(len(a)), 4):
            region = np.ix_(rr, cc)
            mask = valid[region]
            if not mask.any():
                values.extend([0.0] * 6)
                continue
            get = lambda x, region=region, mask=mask: x[region][mask]  # noqa: E731
            values.extend(
                np.tanh(
                    [
                        get(rgb).mean() / 5,
                        get(gray).mean() / 5,
                        np.sqrt((get(gray) ** 2).mean()) / 5,
                        get(u).mean() / 2,
                        get(v).mean() / 2,
                        np.hypot(get(u), get(v)).mean() / 2,
                    ]
                )
            )
    return np.array([*values, 1.0], np.float32)


def source_resets(raw, video):
    times = [Q(r["timestamp_exact"]) for r in video["frames"]]
    if len(times) != len(raw["pts"]) or any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("Invalid complete source clock")
    gap = (
        2 * median(b - a for a, b in zip(times, times[1:], strict=False))
        if len(times) > 1
        else Q(1, 10)
    )
    reset = np.ones(len(times), bool)
    for i in range(len(times)):
        if video["frames"][i]["frame_index"] != i:
            raise ValueError("Source-frame index mismatch")
        if (
            Q(int(raw["pts"][i]) * int(raw["time_base_num"][i]), int(raw["time_base_den"][i]))
            != times[i]
        ):
            raise ValueError("Pose/video source clock mismatch")
        if i == 0:
            continue
        reset[i] = (
            times[i] - times[i - 1] > gap
            or i in video.get("confirmed_scene_cuts", [])
            or any(raw[k][i] != raw[k][i - 1] for k in ("scene_id", "segment_id"))
        )
    return times, reset


def supervision(interval, video, offset, radius):
    """Activity union and boundary evidence, with separate Unknown/censoring masks."""
    n = len(video["frames"])
    y, mask = np.zeros((n, 3), np.float32), np.zeros((n, 3), bool)
    if interval is None:
        return y, mask, []
    p = interval["provenance"]
    if (
        not interval["exhaustive"]
        or p["recording"] not in ("V4", "V6")
        or p["role"] != "train_candidate"
    ):
        raise ValueError("Unreviewed or wrong-role supervision")
    a, z = p["start"] - offset, p["stop"] - offset
    if not 0 <= a < z <= n:
        raise ValueError("Exhaustive core outside source cache")
    times = [Q(r["timestamp_exact"]) for r in video["frames"]]
    stop = Q(video["stop_timestamp_exact"])
    mask[a:z] = True
    seen = np.zeros(n, bool)
    for run in interval["supervision_runs"]:
        u, v = run["start"] - offset, run["stop"] - offset
        if not a <= u < v <= z or seen[u:v].any() or run["activity"] not in (0, 1):
            raise ValueError("Conflicting or invalid activity supervision")
        seen[u:v] = True
        y[u:v, 0] = run["activity"]
    if not seen[a:z].all():
        raise ValueError("Incomplete exhaustive supervision")
    core_stop = times[z] if z < n else stop
    # Boundary absence immediately outside a reviewed domain is Unknown.
    for i in range(a, z):
        if times[i] - times[a] < radius or core_stop - times[i] <= radius:
            mask[i, 1:] = False
    censoring = []
    for event in interval["events"]:
        u, v = event["start"] - offset, event["stop"] - offset
        if not (0 <= u < v <= n):
            raise ValueError("Known event bounds exceed context")
        left, right = u <= a, v >= z
        censoring.append(dict(event_id=event["event_id"], left_censored=left, right_censored=right))
        for channel, point, censored in ((1, u, left), (2, v - 1, right)):
            if censored:
                continue
            for i in range(a, z):
                distance = abs(times[i] - times[point])
                if distance < radius:
                    y[i, channel] = max(y[i, channel], float(1 - distance / radius))
    return y, mask, censoring


def prepare(output):
    import cv2

    cv2.setNumThreads(2)
    root = Path.cwd()
    manifest = validate(root)  # Existing human-review/source/backup validator, no new framework.
    intervals = authorized(manifest)
    pixel_manifest = study_manifest(PIXELS)
    out = fresh(output)
    protected = read(PIXELS / "protected-before.json")
    for p, h in protected.items():
        if "v9" in p.lower() or sha(Path(p)) != h:
            raise ValueError("Protected artifact changed or forbidden path")
    # Preserve the completed pose study as well as all original protected artifacts.
    for p in PIXELS.rglob("*"):
        if p.is_file():
            protected[str(p)] = sha(p)
    for preflight in (
        Path("reports/boxingvi/punch-localization-features-v1"),
        Path("reports/boxingvi/punch-localization-features-v1-1"),
    ):
        if preflight.exists():
            for p in preflight.rglob("*"):
                if p.is_file():
                    protected[str(p)] = sha(p)
    write_json(out / "protected-before.json", protected)
    config = read("configs/punch_localization_smoke_v1.json")
    dataset_pin = read("configs/continuous_dataset_v1.json")
    if config["dataset_sha256"] != dataset_pin["sha256"]:
        raise ValueError("Continuous dataset binding changed")
    entries = {}
    for name in CASES:
        began = perf_counter()
        case, base = PIXELS / name, BASE / name
        identity = read(case / "input.json")
        if sha(case / "input.json") != pixel_manifest["cases"][name]:
            raise ValueError("Pixel source identity changed")
        for filename, digest in identity["baseline_sha256"].items():
            if sha(base / filename) != digest:
                raise ValueError("Baseline input changed")
        if sha(Path(identity["source_media"])) != identity["source_sha256"]:
            raise ValueError("Original source video changed")
        raw, video = arrays(base / "poses.npz"), read(base / "video.json")
        times, reset = source_resets(raw, video)
        pose, observable, _, _ = encode_source(
            raw, video["width"] / video["height"], video.get("confirmed_scene_cuts", [])
        )
        pose = pose.numpy()
        pose[~observable] = 0
        box, torso = fixed_regions(raw, (video["width"], video["height"]))
        recording = "V6" if name.startswith("V6") else "V4"
        visual, preview, old, old_mask = [], [], None, None
        previous_observed = False
        for i in range(len(times)):
            path = case / "frames" / f"{i:04d}.png"
            if sha(path) != identity["pixel_sha256"][path.name]:
                raise ValueError("Native source pixels changed")
            with Image.open(path) as image:
                rgb, support = redacted(image, box, torso, recording)
            row = (
                motion_pair(old, rgb, old_mask, support, times[i] - times[i - 1])
                if i and not reset[i] and previous_observed and observable[i]
                else np.zeros(97, np.float32)
            )
            visual.append(row)
            if i in {0, len(times) // 2, len(times) - 1}:
                preview.append(Image.fromarray(rgb))
            old, old_mask, previous_observed = rgb, support, observable[i]
        dest = out / name
        dest.mkdir()
        prov = read(base / "provenance.json")
        interval = intervals.get(name)
        offset = prov.get("source_start", 0)
        if interval is not None and interval["provenance"]["recording"] != recording:
            raise ValueError("Cross-source supervision")
        y, mask, censored = supervision(
            interval, video, offset, Q(config["boundary_radius_seconds"])
        )
        # Targets remain separate from observation availability; missing joints are not negatives.
        np.savez_compressed(
            dest / "features.npz",
            pose=pose,
            visual=np.array(visual),
            observable=observable,
            reset=reset,
            targets=y,
            supervision_mask=mask,
            **{k: raw[k] for k in ("frame_index", "pts", "time_base_num", "time_base_den")},
        )
        for stem in ("video", "references", "provenance", "activity"):
            shutil.copy2(base / f"{stem}.json", dest / f"{stem}.json")
        canvas = Image.new("RGB", (160 * len(preview), 160))
        for j, im in enumerate(preview):
            canvas.paste(im, (j * 160, 0))
        canvas.save(dest / "redaction-preview.png")
        receipt = dict(
            case=name,
            train_eligible=interval is not None,
            input_sha256=sha(case / "input.json"),
            source=identity,
            fixed_crop=box,
            identity_redaction=torso,
            censoring=censored,
            features_sha256=sha(dest / "features.npz"),
            metadata_sha256={
                s: sha(dest / f"{s}.json")
                for s in ("video", "references", "provenance", "activity")
            },
            frames=len(times),
            seconds=perf_counter() - began,
            supervised_frames=mask.sum(0).tolist(),
            observable_frames=int(observable.sum()),
            motion_frames=int(np.array(visual)[:, -1].sum()),
        )
        write_json(dest / "receipt.json", receipt)
        entries[name] = sha(dest / "receipt.json")
        print(name, receipt["frames"], round(receipt["seconds"], 2), flush=True)
    write_json(
        out / "manifest.json",
        dict(
            version=VERSION,
            policy=POLICY,
            cases=entries,
            dataset=dataset_pin,
            config_sha256=sha(Path("configs/punch_localization_smoke_v1.json")),
            code_sha256={
                str(p): sha(p)
                for p in [
                    Path(__file__),
                    Path("src/boxing_motion_ai/detection/data.py"),
                    Path("src/boxing_motion_ai/training/velocity.py"),
                    Path("src/boxing_motion_ai/training/rgb.py"),
                    Path("src/boxing_motion_ai/training/data.py"),
                    Path("src/boxing_motion_ai/vision/contracts.py"),
                ]
            },
            versions=dict(numpy=np.__version__, opencv=cv2.__version__),
            authorization=config["purpose"],
            production_changed=False,
        ),
    )


def load_cache(path, config):
    path = Path(path)
    if not path.resolve().is_relative_to(Path.cwd() / "reports"):
        raise ValueError("Use ignored local cache")
    manifest = read(path / "manifest.json")
    if (
        manifest["version"] != VERSION
        or manifest["policy"] != POLICY
        or set(manifest["cases"]) != set(CASES)
        or manifest["dataset"] != read("configs/continuous_dataset_v1.json")
        or manifest["dataset"]["sha256"] != config["dataset_sha256"]
    ):
        raise ValueError("Cache dataset/policy/source restrictions changed")
    for p, h in manifest["code_sha256"].items():
        if sha(Path(p)) != h:
            raise ValueError("Feature implementation changed; use a fresh cache")
    if manifest["config_sha256"] != sha(Path("configs/punch_localization_smoke_v1.json")):
        raise ValueError("Frozen experiment config changed")
    cases = {}
    binding = manifest["dataset"]
    if sha(Path(binding["path"])) != binding["sha256"]:
        raise ValueError("Frozen labels changed")
    intervals = authorized(read(binding["path"]))
    verified_sources = set()
    for name, h in manifest["cases"].items():
        directory = path / name
        r = read(directory / "receipt.json")
        if sha(directory / "receipt.json") != h or r["train_eligible"] != (name in TRAIN):
            raise ValueError("Stale/cross-role case receipt")
        if sha(directory / "features.npz") != r["features_sha256"]:
            raise ValueError("Feature cache changed")
        source = r["source"]
        media = source["source_media"]
        if media not in verified_sources:
            if sha(Path(media)) != source["source_sha256"]:
                raise ValueError("Original feature source changed")
            verified_sources.add(media)
        for filename, digest in source["baseline_sha256"].items():
            if sha(BASE / name / filename) != digest:
                raise ValueError("Original pose/label source changed")
        for s, digest in r["metadata_sha256"].items():
            if sha(directory / f"{s}.json") != digest:
                raise ValueError("Label/source metadata changed")
        a = arrays(directory / "features.npz")
        video = read(directory / "video.json")
        provenance = read(directory / "provenance.json")
        y, mask, _ = supervision(
            intervals.get(name),
            video,
            provenance.get("source_start", 0),
            Q(config["boundary_radius_seconds"]),
        )
        if not np.array_equal(y, a["targets"]) or not np.array_equal(mask, a["supervision_mask"]):
            raise ValueError("Cached supervision differs from frozen human references")
        n = r["frames"]
        if (
            a["pose"].shape != (n, 65)
            or a["visual"].shape != (n, 97)
            or not np.isfinite(a["pose"]).all()
            or not np.isfinite(a["visual"]).all()
            or a["targets"].shape != (n, 3)
            or a["supervision_mask"].shape != (n, 3)
        ):
            raise ValueError("Invalid feature/supervision arrays")
        cases[name] = dict(
            arrays=a,
            receipt=r,
            video=video,
            references=read(directory / "references.json"),
            provenance=provenance,
        )
    return cases, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
