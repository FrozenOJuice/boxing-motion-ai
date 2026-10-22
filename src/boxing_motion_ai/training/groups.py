"""Frozen recording groups for expansion; unknown people remain unknown."""

from collections import Counter
from pathlib import Path

from torch.utils.data import Dataset

from ..vision.pipeline import sha256
from .data import CLASSES, PoseClips

GROUPS = {
    "V4": "fightcamp-unresolved",
    "V5": "self-improvement-boxer-unresolved",
    "V7": "precision-striking-unresolved",
    "V8": "precision-striking-unresolved",
}


def source_groups(specs):
    """Keep the legacy gate immutable; V10 is a separate, explicitly allowed study."""
    ids = [s.get("recording") for s in specs]
    if ids == ["V4", "V7", "V8"]:
        return GROUPS
    if ids == ["V4", "V10"]:
        return {"V4": GROUPS["V4"], "V5": GROUPS["V5"], "V10": "utah-lee-unresolved"}
    raise ValueError("Only V4/V7/V8 or V4/V10 expansion; V5 development, V9 reserved")


def validate_split(specs, development):
    if not isinstance(specs, list) or not all(isinstance(s, dict) for s in specs):
        raise ValueError(
            "Expansion requires ordered V4/V7/V8 training, V5 development; V9 reserved"
        )
    groups = source_groups(specs)
    if development.get("recording") != "V5":
        raise ValueError("V5 must remain development")
    fields = {"recording", "dataset", "pose_run", "policy"}
    if any(set(s) != fields for s in [*specs, development]):
        raise ValueError("Unexpected recording specification")
    train_groups = {groups[s["recording"]] for s in specs}
    if groups[development["recording"]] in train_groups:
        raise ValueError("Source group crosses splits")
    paths = [Path(s["dataset"]).resolve() for s in [*specs, development]]
    if len(set(paths)) != len(paths):
        raise ValueError("Recording datasets must be distinct")


class GroupedClips(Dataset):
    def __init__(self, specs, limit_per_class=None):
        groups = source_groups(specs)
        self.datasets = [
            PoseClips(s, s["recording"], limit_per_class, expansion=True) for s in specs
        ]
        self.clips = [c for ds in self.datasets for c in ds.clips]
        for c in self.clips:
            c["source_group_id"] = groups[c["recording_id"]]
        counts = Counter(c["label"] for c in self.clips)
        if any(counts[i] == 0 for i in range(6)):
            raise ValueError("Combined training must retain all six classes")
        self.provenance = {
            "schema_version": "grouped-training-1.0",
            "recordings": [ds.provenance for ds in self.datasets],
            "source_groups": {s["recording"]: groups[s["recording"]] for s in specs},
            "identity_status": "boxer/session unknown; source groups are conservative guards",
            "class_counts": {name: counts[i] for i, name in enumerate(CLASSES)},
            "examples": [
                {"recording_id": c["recording_id"], "source_row": c["source_row"]}
                for c in self.clips
            ],
        }

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return self.clips[i]


def verify_unchanged(dataset):
    members = dataset.datasets if isinstance(dataset, GroupedClips) else [dataset]
    for ds in members:
        for name, digest in ds.provenance["hashes"].items():
            if sha256(Path(ds.provenance["dataset"]) / name) != digest:
                raise ValueError("Dataset changed during experiment")
