"""Prepare a bounded unverified review queue. Never creates human decisions or negatives."""

from fractions import Fraction
from pathlib import Path

import numpy as np

from ..api.media import FrameService
from ..services.artifacts import ServiceError
from ..services.dashboard import DashboardService
from ..services.interval_review import interval_timing
from ..services.review_workspace import encode, sha

ROLES = {
    "V4": "train_candidate",
    "V6": "train_candidate",
    "V5": "development",
    "V7": "evaluation_reserved",
    "V8": "evaluation_reserved",
    "V10": "evaluation_reserved",
}
GROUPS = {
    "V4": ["V4"],
    "V5": ["V5"],
    "V6": ["V1", "V6"],
    "V7": ["V7", "V8"],
    "V8": ["V7", "V8"],
    "V10": ["V10"],
}


def prepare(root):
    root = Path(root)
    output = root / "reports/boxingvi/interval-review-pilot-v1"
    config = root / "configs/interval_review_pilot_v1.json"
    if output.exists():
        raise ValueError("Pilot already exists; preserve its immutable manifest")
    base = DashboardService(root)
    frames = FrameService(base)
    entries = []
    for v, role in ROLES.items():
        spec = base._recording(v)
        prep = base._preparation(v, spec)
        a = frames._load(v, spec, prep)
        t = a["pts"] * a["time_base_num"] / a["time_base_den"]
        rows = base._rows(v, spec)
        occupied = np.zeros(len(t), bool)
        for row in rows:
            bounds = row.get("source_interval_half_open")
            if bounds and 0 <= bounds[0] < bounds[1] <= len(t):
                occupied[bounds[0] : bounds[1]] = True
        gaps, _ = base.store.json(f"{spec.dataset}/unknown-intervals.json")
        used = []

        def add(
            start,
            stop,
            purpose,
            reason,
            *,
            a=a,
            prep=prep,
            t=t,
            used=used,
            v=v,
            role=role,
            spec=spec,
        ):
            try:
                timing = interval_timing(a, start, stop, prep.get("confirmed_scene_cuts", []))
            except (ValueError, ServiceError):
                return False
            if any(not (t[stop] + 1 <= t[x] or t[y] + 1 <= t[start]) for x, y in used):
                return False
            kind = "bg" if purpose == "background_candidate" else "eval"
            number = 1 + sum(e["recording"] == v and e["purpose"] == purpose for e in entries)
            left = max(0, int(np.searchsorted(t, t[start] - 1)))
            right = min(len(t) - 1, int(np.searchsorted(t, t[stop] + 1)))
            # Keep context inside the same known scene/segment; reviewed interval is unchanged.
            for i in range(start - 1, left - 1, -1):
                if any(
                    a[k][i] != a[k][start] for k in ("scene_id", "segment_id")
                ) or i + 1 in prep.get("confirmed_scene_cuts", []):
                    left = i + 1
                    break
            for i in range(stop, right + 1):
                if any(a[k][i] != a[k][start] for k in ("scene_id", "segment_id")) or i in prep.get(
                    "confirmed_scene_cuts", []
                ):
                    right = i - 1
                    break
            entries.append(
                dict(
                    id=f"{v}-{kind}-{number}",
                    recording=v,
                    role=role,
                    purpose=purpose,
                    start=start,
                    stop=stop,
                    context_start=left,
                    context_end=right,
                    frame_count=len(t),
                    timing=timing,
                    source_inputs=spec.inputs,
                    video_sha256=prep["inputs"]["video_sha256"],
                    source_group_guard=GROUPS[v],
                    boxer_id=None,
                    session_id=None,
                    selection=reason,
                    state="unknown",
                    endpoint_pts={
                        name: dict(
                            pts=int(a["pts"][i]),
                            time_base_num=int(a["time_base_num"][i]),
                            time_base_den=int(a["time_base_den"][i]),
                        )
                        for name, i in (("start", start), ("stop_exclusive", stop))
                    },
                )
            )
            used.append((start, stop))
            return True

        candidates = []
        for gap in gaps["intervals"]:
            lo, hi = gap["frame_start"], gap["frame_stop_exclusive"]
            if lo <= 0 or hi >= len(t) or t[hi] - t[lo] < 5:
                continue
            middle = (t[lo] + t[hi]) / 2
            x, y = int(np.searchsorted(t, middle - 2)), int(np.searchsorted(t, middle + 2))
            candidates.append((float(t[hi] - t[lo]), x, y))
        # Deterministic long-gap discovery; no movement class or absence is inferred.
        count = 0
        for _, x, y in sorted(candidates, reverse=True):
            if add(x, y, "background_candidate", "center_of_source_annotation_gap_unverified"):
                count += 1
            if count == 5:
                break
        if role == "evaluation_reserved":
            blocks = []
            for second in np.arange(1, t[-1] - 31, 1):
                x, y = int(np.searchsorted(t, second)), int(np.searchsorted(t, second + 29.9))
                if y < len(t):
                    blocks.append((float(occupied[x:y].mean()), x, y))
            for reverse, reason in (
                (True, "higher_source_annotation_coverage"),
                (False, "lower_source_annotation_coverage"),
            ):
                for _, x, y in sorted(blocks, reverse=reverse):
                    if add(x, y, "evaluation_block", reason + "_not_exhaustive_truth"):
                        break
                else:
                    raise ValueError(f"No safe evaluation block for {v}")
    manifest = dict(
        schema_version="interval-pilot-1.0",
        roles=ROLES,
        entries=entries,
        source="automated",
        human_decisions=0,
        training_enabled=False,
        evaluation_policy=(
            "V7/V8 and V10 reserved as whole groups for this detector pilot; V5 dev only"
        ),
        annotation_policy=(
            "All candidates unknown until explicit interval review; "
            "punch-present requires later event labeling"
        ),
    )
    duration = sum(
        Fraction(e["timing"]["stop_exclusive"]) - Fraction(e["timing"]["start"])
        for e in entries
        if e["purpose"] == "evaluation_block"
    )
    assert duration <= 360
    raw = encode(manifest)
    reference = encode(
        dict(path="reports/boxingvi/interval-review-pilot-v1/manifest.json", sha256=sha(raw))
    )
    if config.exists() and config.read_bytes() != reference:
        raise ValueError("Existing pilot configuration differs; do not overwrite it")
    output.mkdir(parents=True)
    (output / "manifest.json").write_bytes(raw)
    if not config.exists():
        config.write_bytes(reference)
    print(
        f"{len(entries)} unverified intervals; "
        f"{float(duration):.3f} evaluation seconds; no reviews saved"
    )


if __name__ == "__main__":
    prepare(Path.cwd())
