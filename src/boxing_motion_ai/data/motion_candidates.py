"""Deterministic activity-stratified proposals; never classify actions or certify absence."""

from collections import Counter
from pathlib import Path

import numpy as np

from ..api.media import FrameService
from ..data.annotation_review import file_hash
from ..services.artifacts import ServiceError
from ..services.dashboard import DashboardService
from ..services.interval_review import IntervalWorkspace, interval_timing
from ..services.review_workspace import encode, sha
from .interval_pilot import GROUPS, ROLES

STRATA = (
    "quiet_motion",
    "quiet_motion",
    "lower_body_motion",
    "lower_body_motion",
    "moderate_arm_motion",
    "mixed_motion",
)
HINTS = {
    "quiet_motion": "Less overall movement",
    "lower_body_motion": "More lower-body movement",
    "moderate_arm_motion": "Moderate arm movement",
    "mixed_motion": "Mixed movement",
}
ARMS = [13, 14, 15, 16]
ANCHORS = [11, 12, 23, 24]


def motion_metrics(a, start, stop, aspect):
    """Mask-gated image-plane torso units/second, from native elapsed PTS. No imputation."""
    valid = a["valid_mask"][start:stop]
    if (
        not valid[:, ANCHORS].all()
        or np.min(valid[:, ARMS].mean(axis=0)) < 0.9
        or np.min(valid[:, [27, 28]].mean(axis=0)) < 0.9
    ):
        return None
    # Reject long occlusion runs rather than letting missing punching arms seem quiet.
    bad = ~valid[:, ARMS].all(axis=1)
    times = a["pts"][start:stop] * a["time_base_num"][start:stop] / a["time_base_den"][start:stop]
    dt = np.diff(times)
    if not np.all((dt > 0) & (dt <= 0.1)) or len(times) < 2:
        return None
    run = 0.0
    for i, missing in enumerate(bad):
        run = run + dt[min(i, len(dt) - 1)] if missing else 0.0
        if run > 0.15:
            return None
    xy = a["coordinates"][start:stop, :, :2] * [aspect, 1]
    if not np.isfinite(xy[:, ANCHORS]).all():
        return None
    hips = xy[:, [23, 24]].mean(axis=1)
    shoulders = xy[:, [11, 12]].mean(axis=1)
    scale = np.linalg.norm(shoulders - hips, axis=1)
    if np.min(scale) < 1e-4:
        return None
    local = (xy - hips[:, None]) / scale[:, None, None]

    def speed(points, joints):
        mask = valid[:-1, joints] & valid[1:, joints]
        values = np.linalg.norm(np.diff(points[:, joints], axis=0), axis=-1) / dt[:, None]
        values = values[mask & np.isfinite(values)]
        return values

    arm = speed(local, ARMS)
    feet = speed(local, [27, 28])
    if len(arm) < 0.8 * (stop - start - 1) * len(ARMS) or np.max(arm) > 12:
        return None
    hip = np.linalg.norm(np.diff(hips, axis=0), axis=-1) / ((scale[:-1] + scale[1:]) / 2) / dt
    return dict(
        arm_p50=float(np.median(arm)),
        arm_p90=float(np.quantile(arm, 0.9)),
        arm_peak=float(np.max(arm)),
        feet_p50=float(np.median(feet)) if len(feet) else None,
        feet_coverage=float(valid[:, [27, 28]].mean()),
        hip_p50=float(np.median(hip)),
        arm_coverage_min=float(np.min(valid[:, ARMS].mean(axis=0))),
        anchor_coverage=1.0,
        units="projected_torso_lengths_per_second",
    )


def select_diverse(windows, duration):
    """Six separated proposals, two quiet/two lower-body/one arm/one mixed, no labels."""
    if len(windows) < 6:
        return []
    arm = np.array([w["metrics"]["arm_p90"] for w in windows])
    # All ranks depend only on training recordings; no development/evaluation is inspected.
    median, upper = np.quantile(arm, [0.5, 0.7])
    chosen = []
    thirds = Counter()
    for stratum in STRATA:
        pool = []
        for w in windows:
            m = w["metrics"]
            if m["arm_p90"] > upper:
                continue
            lower = (
                m["feet_p50"] if m["feet_p50"] is not None and m["feet_coverage"] >= 0.9 else None
            )
            if stratum == "quiet_motion":
                score = -(m["arm_p90"] + m["hip_p50"] + (lower or 0))
            elif stratum == "lower_body_motion":
                if lower is None or m["arm_p90"] > median:
                    continue
                score = (lower + m["hip_p50"]) / (1 + m["arm_p90"])
            elif stratum == "moderate_arm_motion":
                if not np.quantile(arm, 0.25) <= m["arm_p90"] <= upper:
                    continue
                score = -abs(m["arm_p90"] - median)
            else:
                score = m["hip_p50"] + (lower or 0) + 0.25 * m["arm_p50"]
            pool.append((score, w))
        for score, w in sorted(
            pool,
            key=lambda pair: (
                thirds[min(2, int(pair[1]["start_seconds"] / (duration / 3)))],
                -pair[0],
                pair[1]["start"],
            ),
        ):
            third = min(2, int(w["start_seconds"] / (duration / 3)))
            if any(abs(w["start_seconds"] - x["start_seconds"]) < 30 for x in chosen):
                continue
            chosen.append(w | {"stratum": stratum, "rank_score": float(score)})
            thirds[third] += 1
            break
        else:
            break
    return chosen


def prepare(root):
    import av

    root = Path(root)
    output = root / "reports/boxingvi/interval-review-motion-v2"
    config = root / "configs/interval_review_motion_v2.json"
    if output.exists():
        raise ValueError("Preserve the existing motion-pilot manifest")
    base = DashboardService(root)
    frames = FrameService(base)
    old = IntervalWorkspace(base, frames).snapshot()
    pins = {
        "configs/interval_review_pilot_v1.json": file_hash(
            root / "configs/interval_review_pilot_v1.json"
        )
    }
    original_ref, _ = base.store.json("configs/interval_review_pilot_v1.json")
    pins[original_ref["path"]] = original_ref["sha256"]
    for directory in (
        "reports/boxingvi/interval-reviews-v1",
        "reports/boxingvi/dashboard-reviews-v1",
    ):
        for p in (root / directory).rglob("*.json"):
            pins[str(p.relative_to(root))] = file_hash(p)
    entries = []
    analysis = {}
    originals = old["manifest"]["entries"]
    for v in ("V4", "V6"):
        spec = base._recording(v)
        prep = base._preparation(v, spec)
        a = frames._load(v, spec, prep)
        rows = base._rows(v, spec)
        times = a["pts"] * a["time_base_num"] / a["time_base_den"]
        with base.store.open_regular(f"data/raw/boxingvi/RGB_videos/{v}.mkv") as stream:
            frames._verify(stream, prep["inputs"]["video_sha256"])
            with av.open(stream) as container:
                track = container.streams.video[0]
                aspect = track.width / track.height
        forbidden = np.zeros(len(times), bool)
        # Plausible bounds, including reversed/excluded rows, form exclusion envelopes only.
        for row in rows:
            x, y = row.get("raw_start"), row.get("raw_end")
            if (
                isinstance(x, (int, float))
                and isinstance(y, (int, float))
                and np.isfinite([x, y]).all()
                and x == int(x)
                and y == int(y)
            ):
                lo, hi = sorted((int(x), int(y)))
                if 0 <= lo <= hi < len(times):
                    forbidden[
                        np.searchsorted(times, times[lo] - 1) : np.searchsorted(
                            times, times[hi] + 1, side="right"
                        )
                    ] = True
        for e in originals:
            if e["recording"] == v:
                forbidden[
                    np.searchsorted(times, times[e["context_start"]] - 2) : np.searchsorted(
                        times, times[e["context_end"]] + 2, side="right"
                    )
                ] = True
        windows = []
        rejected = Counter()
        for second in np.arange(times[0] + 2, times[-1] - 6, 1):
            x = int(np.searchsorted(times, second))
            y = int(np.searchsorted(times, second + 4))
            if forbidden[x:y].any():
                rejected["source_or_original_queue_guard"] += 1
                continue
            lo = int(np.searchsorted(times, times[x] - 1))
            hi = int(np.searchsorted(times, times[y] + 1))
            try:
                # Require context itself continuous and independent of forbidden source/old windows.
                if forbidden[lo:hi].any():
                    rejected["context_conflict"] += 1
                    continue
                interval_timing(a, lo, hi, prep.get("confirmed_scene_cuts", []))
            except ServiceError:
                rejected["continuity"] += 1
                continue
            metrics = motion_metrics(a, lo, hi, aspect)
            if metrics is None:
                rejected["pose_or_spike_gate"] += 1
                continue
            windows.append(
                dict(
                    start=x,
                    stop=y,
                    context_start=lo,
                    context_end=hi - 1,
                    start_seconds=float(times[x]),
                    metrics=metrics,
                )
            )
        chosen = select_diverse(windows, float(times[-1] - times[0]))
        if len(chosen) != 6:
            raise ValueError(
                f"{v}: only {len(chosen)} reliable diverse proposals; do not silently relax gates"
            )
        analysis[v] = dict(
            eligible_windows=len(windows),
            rejected=dict(rejected),
            selected=chosen,
            aspect=aspect,
            arm_p90_quantiles=np.quantile(
                [w["metrics"]["arm_p90"] for w in windows], [0.25, 0.5, 0.7]
            ).tolist(),
        )
        for i, w in enumerate(chosen, 1):
            x, y = w["start"], w["stop"]
            entries.append(
                dict(
                    id=f"{v}-motion-{i}",
                    recording=v,
                    role="train_candidate",
                    purpose="background_candidate",
                    start=x,
                    stop=y,
                    context_start=w["context_start"],
                    context_end=w["context_end"],
                    frame_count=len(times),
                    timing=interval_timing(a, x, y),
                    source_inputs=spec.inputs,
                    video_sha256=prep["inputs"]["video_sha256"],
                    source_group_guard=GROUPS[v],
                    boxer_id=None,
                    session_id=None,
                    selection="mask_gated_activity_ranking_not_action_classification",
                    selection_stratum=w["stratum"],
                    discovery_hint=HINTS[w["stratum"]],
                    selection_metrics=w["metrics"],
                    state="unknown",
                    endpoint_pts={
                        k: dict(
                            pts=int(a["pts"][j]),
                            time_base_num=int(a["time_base_num"][j]),
                            time_base_den=int(a["time_base_den"][j]),
                        )
                        for k, j in (("start", x), ("stop_exclusive", y))
                    },
                )
            )
        pins.update({f"{spec.dataset}/{k}": h for k, h in spec.inputs.items()})
        pins[f"data/raw/boxingvi/RGB_videos/{v}.mkv"] = prep["inputs"]["video_sha256"]
    manifest = dict(
        schema_version="interval-pilot-1.0",
        pilot_id="motion-v2",
        roles=ROLES,
        entries=entries,
        source="automated",
        human_decisions=0,
        training_enabled=False,
        original_pilot=old["head"],
        motion_policy_version="motion-proposal-1.0",
        disclosure=(
            "Strata describe pose statistics only; "
            "guards, footwork, defenses and no-punch labels remain unverified"
        ),
    )
    raw = encode(manifest)
    ref = encode(
        dict(path="reports/boxingvi/interval-review-motion-v2/manifest.json", sha256=sha(raw))
    )
    if config.exists() and config.read_bytes() != ref:
        raise ValueError("Existing motion-pilot pin differs")
    for path, h in pins.items():
        if file_hash(root / path) != h:
            raise ValueError(f"Protected input changed: {path}")
    output.mkdir(parents=True)
    (output / "manifest.json").write_bytes(raw)
    (output / "analysis.json").write_bytes(encode(analysis))
    (output / "preserved-inputs.json").write_bytes(encode(pins))
    if not config.exists():
        config.write_bytes(ref)
    print("12 unknown activity-ranked proposals; zero human decisions, no negative labels")


if __name__ == "__main__":
    prepare(Path.cwd())
