"""Explicit multi-recording pool, grouped folds and training-only sampling/shortcuts."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .data import CLASSES, PoseClips, exact_times
from .experiment import load_config
from .limbs import LimbClips
from .velocity import VelocityClips
from .velocity_transfer import load_transfer_config, source_support

TRAIN = ("V4", "V6", "V7", "V8", "V10")
ALL = (*TRAIN, "V5")
GROUPS = {
    "V4": "fightcamp-unresolved",
    "V5": "self-improvement-boxer-unresolved",
    "V6": "natebowerfitness-unresolved",
    "V7": "precision-striking-unresolved",
    "V8": "precision-striking-unresolved",
    "V10": "utah-lee-unresolved",
}
GUARDS = {
    v: (["V1", "V6"] if v == "V6" else ["V7", "V8"] if v in ("V7", "V8") else [v]) for v in ALL
}
SEEDS = (20261021, 20261022, 20261023)
ARMS = ("xy_velocity", "xy_velocity_limb")
MODELS = ("tcn", "tcn_mask_only")


def settings(path):
    c = yaml.safe_load(Path(path).read_text())
    expected = {
        "schema_version": "multi-recording-study-1.0",
        "base_config": "configs/level1_initial.yaml",
        "transfer_config": "configs/level1_velocity_transfer.yaml",
        "training_recordings": list(TRAIN),
        "development_recordings": ["V5"],
        "seeds": list(SEEDS),
        "logo_pilot_seed": SEEDS[0],
        "representations": list(ARMS),
        "mirror_probability": 0.5,
        "visibility_probability": 0.0,
        "context_seconds": 0.0,
        "sampler": "class_uniform_group_sqrt_recording_sqrt",
        "sampling_replacement": True,
        "samples_per_epoch": "training_pool_size",
        "loss_class_weights": "uniform_to_avoid_double_balancing",
        "logo_selection": "V5_only_never_heldout_group",
        "duration_frame_bins": [8, 15, 24],
        "duration_second_bins": [0.3, 0.6, 1.0],
    }
    if c != expected:
        raise ValueError("Frozen multi-recording protocol required; V5 dev-only, V9 reserved")
    base = load_config(Path(c["base_config"]))
    if any(
        base[k] != val
        for k, val in dict(
            epochs=40,
            batch_size=32,
            learning_rate=0.001,
            weight_decay=0.001,
            hidden=32,
            dropout=0.1,
            threads=2,
            limit_per_class=None,
        ).items()
    ):
        raise ValueError("Original architecture and training budget must remain unchanged")
    transfer = load_transfer_config(Path(c["transfer_config"]))
    specs = {
        s["recording"]: s for s in [base["train"], *transfer["recordings"], base["development"]]
    }
    return c, base, specs


def validate_fold(train, development, heldout):
    if not train or len(set(train)) != len(train) or not set(train) <= set(TRAIN):
        raise ValueError("Only approved V4/V6/V7/V8/V10 training; V5 dev-only, V9 reserved")
    if list(development) != ["V5"] or not set(heldout) <= set(TRAIN):
        raise ValueError("V5 development and non-V9 guarded diagnostic holdouts required")
    sets = [{GROUPS[v] for v in part} for part in (train, development, heldout)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Source group crosses training/development/held-out boundaries")
    for part in (train, heldout):
        if bool("V7" in part) != bool("V8" in part):
            raise ValueError("V7/V8 must remain together")
    if heldout and set(train) | set(heldout) != set(TRAIN):
        raise ValueError("LOGO must partition the approved training pool")


def folds():
    result = []
    for v in ("V4", "V6", "V7", "V10"):
        held = [r for r in TRAIN if GROUPS[r] == GROUPS[v]]
        train = [r for r in TRAIN if r not in held]
        validate_fold(train, ["V5"], held)
        result.append(
            dict(name="hold-" + "-".join(held), train=train, development=["V5"], heldout=held)
        )
    return result


class Pool(Dataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.clips = [c for ds in self.datasets for c in ds.clips]
        keys = [(c["recording_id"], c["source_row"]) for c in self.clips]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Empty or duplicate recording/source-row pool")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return self.clips[i]


def sample_probabilities(clips):
    """Uniform class; sqrt group counts; sqrt recording counts; uniform within cell.

    No held-out/development examples or class frequencies are consulted. Square-root
    allocation reduces dominant sources without giving singleton cells full group mass.
    Use unit class loss weights to avoid balancing classes twice.
    """
    counts = Counter((c["label"], GROUPS[c["recording_id"]], c["recording_id"]) for c in clips)
    if any(not any(key[0] == label for key in counts) for label in range(6)):
        raise ValueError("Training pool must support all six classes")
    group_n = Counter()
    for (label, group, _), n in counts.items():
        group_n[label, group] += n
    class_den = Counter()
    for (label, _), n in group_n.items():
        class_den[label] += np.sqrt(n)
    record_den = Counter()
    for (label, group, _), n in counts.items():
        record_den[label, group] += np.sqrt(n)
    p = []
    for c in clips:
        label, record = c["label"], c["recording_id"]
        g = GROUPS[record]
        p.append(
            (1 / 6)
            * np.sqrt(group_n[label, g])
            / class_den[label]
            * np.sqrt(counts[label, g, record])
            / record_den[label, g]
            / counts[label, g, record]
        )
    p = np.asarray(p, dtype=np.float64)
    if not np.isclose(p.sum(), 1):
        raise ValueError("Invalid sampling probabilities")
    return torch.from_numpy(p)


def sampling_summary(clips, p):
    result = {
        "class_mass": {},
        "group_mass": {},
        "recording_mass": {},
        "cells": [],
        "replacement": True,
        "epoch_draws": len(clips),
        "uniform_class_loss_weights": [1] * 6,
    }
    for field, getter in (
        ("class_mass", lambda c: CLASSES[c["label"]]),
        ("group_mass", lambda c: GROUPS[c["recording_id"]]),
        ("recording_mass", lambda c: c["recording_id"]),
    ):
        a = defaultdict(float)
        for c, w in zip(clips, p.tolist(), strict=True):
            a[getter(c)] += w
        result[field] = dict(a)
    for label, record in sorted({(c["label"], c["recording_id"]) for c in clips}):
        indices = [
            i for i, c in enumerate(clips) if c["label"] == label and c["recording_id"] == record
        ]
        result["cells"].append(
            dict(
                label=CLASSES[label],
                recording=record,
                n=len(indices),
                expected_epoch_draws=float(p[indices].sum()) * len(clips),
                max_expected_draws_per_example=float(p[indices].max()) * len(clips),
            )
        )
    return result


def load_pool(specs, limit=None):
    """Load fixed approved inputs without changing labels, masks or transforms."""
    cache = {}
    for v in ALL:
        base = PoseClips(specs[v], v, limit, multi_recording=True)
        vel = VelocityClips(base, multi_recording=True)
        cache[v] = {ARMS[0]: vel, ARMS[1]: LimbClips(vel)}
    return cache


def duration_key(clip, cfg):
    times = exact_times(clip)
    span = float(times[-1] - times[0])
    return (
        int(np.digitize(len(times), cfg["duration_frame_bins"])),
        int(np.digitize(span, cfg["duration_second_bins"])),
    )


def duration_control(train, p, evaluation, cfg):
    """Training-only fixed-bin lookup diagnostic, not a new recognition architecture."""
    counts = np.ones((4, 4, 6), dtype=float)  # One pseudo-count in units of epoch exposure.
    for c, w in zip(train, p.tolist(), strict=True):
        counts[*duration_key(c, cfg), c["label"]] += w * len(train)
    predictions = []
    for c in evaluation:
        a = counts[duration_key(c, cfg)]
        prob = a / a.sum()
        predictions.append(
            dict(
                recording_id=c["recording_id"],
                source_row=c["source_row"],
                target=CLASSES[c["label"]],
                prediction=CLASSES[int(prob.argmax())],
                probabilities=prob.tolist(),
            )
        )
    return predictions, counts.tolist()


def audit_pool(cache):
    result = {}
    for v, variants in cache.items():
        ds = variants[ARMS[0]]
        rows = [
            json.loads(s)
            for s in (Path(ds.provenance["dataset"]) / "candidates.jsonl").read_text().splitlines()
        ]
        mapping = defaultdict(set)
        for row in rows:
            if row["annotation_eligible"]:
                literal = row["source_class"].strip().lower().replace(" ", "_")
                if literal != row["punch_type_provisional"]:
                    raise ValueError("Source-label inconsistency requires review before training")
                mapping[row["source_class"]].add(row["punch_type_provisional"])
        if any(len(labels) != 1 for labels in mapping.values()):
            raise ValueError("Contradictory literal source-class mapping")
        by_class = {}
        for i, label in enumerate(CLASSES):
            clips = [c for c in ds.clips if c["label"] == i]
            spans = [float(exact_times(c)[-1] - exact_times(c)[0]) for c in clips]
            by_class[label] = dict(
                n=len(clips),
                median_frames=float(np.median([len(c["x"]) for c in clips])) if clips else None,
                median_pts_span_seconds=float(np.median(spans)) if clips else None,
            )
        result[v] = dict(
            source_support=source_support(ds),
            by_class=by_class,
            literal_class_map={k: sorted(v) for k, v in mapping.items()},
            provenance=ds.provenance,
            source_group_id=GROUPS[v],
            related_recordings_guard=GUARDS[v],
        )
    return result
