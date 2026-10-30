"""One source/event-balanced objective over unchanged human targets and observation masks."""

from fractions import Fraction as Q

import numpy as np
import torch
from torch.nn import functional as F

from ..services.dashboard import require
from .localization_model import batch, model_observable, sequences

SOURCES = ("V4", "V6")
HEADS = ("activity", "onset", "end")


def events(case, interval=None):
    """Return known native boundaries, including partially supervised events, never suggestions."""
    offset = case["provenance"].get("source_start", 0)
    censor = {c["event_id"]: c for c in case["receipt"]["censoring"]}
    if interval is None:
        result = [dict(e) for e in case["references"]]
    else:
        result = [
            dict(
                event_id=e["event_id"],
                start_frame=e["start"] - offset,
                stop_frame_exclusive=e["stop"] - offset,
            )
            for e in interval["events"]
        ]
    require(len({e["event_id"] for e in result}) == len(result), "duplicate_balanced_event")
    for e in result:
        c = censor[e["event_id"]]
        e["onset_available"] = c.get("onset_available", not c.get("left_censored", False))
        e["end_available"] = c.get("end_available", not c.get("right_censored", False))
    return result


def ownership(case, known_events, radius):
    """Attribute existing positive boundary mass to original max-triangle contributors only."""
    a = case["arrays"]
    times = [Q(f["timestamp_exact"]) for f in case["video"]["frames"]]
    n = len(times)
    valid = a["supervision_mask"] & model_observable(a)[:, None]
    y = a["targets"]
    require(y.shape == valid.shape == (n, 3), "balanced_target_shape")
    require(
        np.isfinite(y[valid]).all() and ((y[valid] >= 0) & (y[valid] <= 1)).all(),
        "invalid_balanced_targets",
    )
    output = {}
    for head, name in ((1, "onset"), (2, "end")):
        contributors = np.zeros((len(known_events), n), bool)
        for j, e in enumerate(known_events):
            start, stop = e["start_frame"], e["stop_frame_exclusive"]
            require(0 <= start < stop <= n, "balanced_event_outside_context")
            if not e[f"{name}_available"]:
                continue
            point = start if head == 1 else stop - 1
            for i, t in enumerate(times):
                if case["provenance"].get("exhaustive") is False and not start <= i < stop:
                    continue
                distance = abs(t - times[point])
                if distance >= radius:
                    continue
                triangle = float(1 - distance / radius)
                contributors[j, i] = (
                    valid[i, head] and y[i, head] > 0 and abs(float(y[i, head]) - triangle) < 1e-6
                )
        count = contributors.sum(0)
        require(
            np.all(count[valid[:, head] & (y[:, head] > 0)] > 0),
            "unattributed_existing_boundary_target",
        )
        share = contributors / np.maximum(count, 1)
        output[head] = share * np.where(valid[:, head], y[:, head], 0)
    return output


def prepare(cases, mode, intervals, radius=Q(1, 12)):
    """Build loss coefficients without modifying, widening or filling any original mask."""
    eligible = {n: c for n, c in cases.items() if c["receipt"]["train_eligible"]}
    for c in eligible.values():
        require(c["provenance"]["recording"] in SOURCES, "balanced_training_role_leakage")
    seqs = sequences(eligible, mode, training=True)
    x, y, mask, present = batch(seqs)
    source = np.array([eligible[s["case"]]["provenance"]["recording"] for s in seqs])
    require(set(source) == set(SOURCES), "balanced_sources_missing")
    positive = np.zeros(tuple(y.shape), np.float64)
    negative = np.zeros_like(positive)
    event_rows, masses = [], []
    for name, case in eligible.items():
        known = events(case, intervals.get(name))
        owned = ownership(case, known, radius)
        for head in (1, 2):
            for e, values in zip(known, owned[head], strict=True):
                tensor = np.zeros(tuple(y.shape[:2]), np.float64)
                for j, s in enumerate(seqs):
                    if s["case"] == name:
                        tensor[j, : s["stop"] - s["start"]] = values[s["start"] : s["stop"]]
                mass = float(tensor.sum())
                masses.append(
                    dict(
                        case=name,
                        event_id=e["event_id"],
                        head=HEADS[head],
                        observed_positive_mass=mass,
                        available=e[f"{HEADS[head]}_available"],
                        source=case["provenance"]["recording"],
                    )
                )
                if mass:
                    event_rows.append((case["provenance"]["recording"], head, tensor / mass))
    target, valid = y.numpy(), mask.numpy()
    for s in SOURCES:
        sel = source == s
        for h in range(3):
            if h == 0:
                values = np.where(valid[sel, :, h], target[sel, :, h], 0)
                require(values.sum() > 0, "balanced_positive_source_head_missing")
                positive[sel, :, h] = values / values.sum()
            else:
                rows = [t for src, head, t in event_rows if src == s and head == h]
                require(bool(rows), "balanced_boundary_source_head_missing")
                positive[:, :, h] += sum(rows) / len(rows)
            values = np.where(valid[sel, :, h], 1 - target[sel, :, h], 0)
            require(values.sum() > 0, "balanced_negative_source_head_missing")
            negative[sel, :, h] = values / values.sum()
    # Equal sources, heads and BCE sign terms. All labels remain the cached soft targets.
    positive /= len(SOURCES) * len(HEADS) * 2
    negative /= len(SOURCES) * len(HEADS) * 2
    require(not (positive[~valid].any() or negative[~valid].any()), "balanced_mask_widened")
    summary = {}
    raw_positive = (y * mask).sum((0, 1))
    old_weight = ((mask.sum((0, 1)) - raw_positive) / raw_positive).clamp(max=40)
    old_denominator = (((1 - y) + old_weight * y) * mask).sum((0, 1))
    for s in SOURCES:
        sel = source == s
        summary[s] = dict(
            positive_coefficient_mass=positive[sel].sum((0, 1)).tolist(),
            negative_coefficient_mass=negative[sel].sum((0, 1)).tolist(),
            eligible_boundary_events=[
                sum(src == s and h == head for src, h, _ in event_rows) for head in (1, 2)
            ],
            baseline_supervised_frames=mask[source == s].sum((0, 1)).tolist(),
            baseline_positive_target_mass=(y[source == s] * mask[source == s]).sum((0, 1)).tolist(),
            baseline_normalization_share=(
                (((1 - y[source == s]) + old_weight * y[source == s]) * mask[source == s]).sum(
                    (0, 1)
                )
                / old_denominator
            ).tolist(),
        )
    for row in masses:
        h = HEADS.index(row["head"])
        row["baseline_positive_coefficient_mass"] = float(
            old_weight[h] * row["observed_positive_mass"] / old_denominator[h] / 3
        )
        row["balanced_positive_coefficient_mass"] = (
            1 / (12 * summary[row["source"]]["eligible_boundary_events"][h - 1])
            if row["observed_positive_mass"]
            else 0
        )
    return dict(
        x=x,
        y=y,
        mask=mask,
        present=present,
        seqs=seqs,
        sources=source,
        positive=torch.from_numpy(positive).float(),
        negative=torch.from_numpy(negative).float(),
        audit=dict(
            source_coefficients=summary,
            event_mass=masses,
            baseline_positive_weights=old_weight.tolist(),
            supervised_frames=mask.sum((0, 1)).tolist(),
        ),
    )


def terms(logits, prepared):
    pos, neg = prepared["positive"], prepared["negative"]
    require(logits.shape == pos.shape == neg.shape, "balanced_logit_shape")
    # Unknown logits contribute neither value nor gradient, including NaNs there.
    z = torch.where(pos + neg > 0, logits, 0)
    return pos * F.softplus(-z) + neg * F.softplus(z)


def loss(logits, prepared):
    return terms(logits, prepared).sum()
