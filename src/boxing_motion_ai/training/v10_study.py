"""Controlled V4 versus V4/V10 study, reusing the frozen mirror-only runner."""

import argparse
import json
from pathlib import Path

import yaml

from ..vision.pipeline import sha256
from .data import FEATURE_VERSION, PoseClips
from .expansion_study import SEEDS, run_study
from .experiment import load_config, write_json
from .groups import GroupedClips, source_groups, validate_split


def proposal_config(path, gate):
    """Translate the accepted proposal; reject split/settings changes before data access."""
    proposal = yaml.safe_load(Path(path).read_text())
    if [s.get("recording") for s in proposal["train"]] != ["V4", "V10"]:
        raise ValueError("V4/V10 training only; V5 development, V9 reserved; no V7/V8")
    if proposal["development"].get("recording") != "V5":
        raise ValueError("V5 must remain development")
    fields = ("recording", "dataset", "policy", "pose_run")
    train = [{k: s[k] for k in fields} for s in proposal["train"]]
    dev = {k: proposal["development"][k] for k in fields}
    validate_split(train, dev)
    base = load_config(Path(proposal["base_config"]))
    fixed = {
        "schema_version": "level1-v10-proposal-1.0",
        "seeds": SEEDS,
        "model": "existing_causal_tcn",
        "feature_version": FEATURE_VERSION,
        "quality_policy": "mask_aware70",
        "mirror_probability": 0.5,
        "visibility_augmentation": False,
        "pre_clip_context": False,
        "controls": ["tcn", "tcn_mask_only"],
        "reserved_recordings": ["V9"],
    }
    fixed.update(
        {
            k: base[k]
            for k in [
                "epochs",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "hidden",
                "dropout",
                "threads",
            ]
        }
    )
    if any(proposal.get(k) != v for k, v in fixed.items()) or dev != base["development"]:
        raise ValueError("Proposal changes frozen architecture, budget, features or split")
    if train[0] != base["train"]:
        raise ValueError("V4 reference dataset changed")
    groups = source_groups(train)
    for s in [*proposal["train"], proposal["development"]]:
        if (
            s["source_group_id"] != groups[s["recording"]]
            or s["boxer_id"] is not None
            or s["session_id"] is not None
        ):
            raise ValueError("Source guard/unknown identity contract changed")
    return proposal, {
        "schema_version": "expansion-study-1.0",
        "base_config": proposal["base_config"],
        "reference": proposal["reference"],
        "seeds": SEEDS,
        "train": train,
        "gate": str(gate),
    }


def freeze_gate(proposal_path, output):
    output = Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh gate under reports/")
    proposal, study = proposal_config(proposal_path, output)
    readiness_path = Path(proposal["readiness"])
    snapshot = proposal["dataset_snapshot"]
    if sha256(readiness_path) != snapshot["readiness_sha256"]:
        raise ValueError("Accepted V10 readiness changed")
    ready = json.loads(readiness_path.read_text())
    if (
        ready["schema_version"] != "v10-preparation-readiness-1.0"
        or ready["provisional_preparation_ready"] is not True
        or ready["source_group_id"] != "utah-lee-unresolved"
    ):
        raise ValueError("V10 preparation is not provisionally ready")
    hashes = {
        str(proposal_path): sha256(proposal_path),
        str(readiness_path): sha256(readiness_path),
    }
    for name, digest in ready["hashes"].items():
        if sha256(name) != digest:
            raise ValueError("Accepted preparation provenance changed")
        hashes[name] = digest
    for key in ("video", "workbook"):
        name = ready["source_inputs"][f"{key}_path"]
        digest = ready["source_inputs"][f"{key}_sha256"]
        if sha256(name) != digest:
            raise ValueError("Original V10 source changed")
        hashes[name] = digest
    train = GroupedClips(study["train"])
    development = PoseClips(proposal["development"], "V5")
    v10 = train.datasets[1]
    if (
        v10.provenance["class_counts"] != snapshot["V10_retained"]
        or train.provenance["class_counts"] != snapshot["V4_plus_V10_counts"]
        or len(train) != snapshot["V4_plus_V10_total"]
        or v10.provenance["policy_sha256"] != snapshot["V10_policy_index_sha256"]
    ):
        raise ValueError("Accepted V10 selection/counts changed")
    excluded = {9, 10, 123, *range(130, 138)}
    if set(v10.provenance["source_rows"]) & excluded:
        raise ValueError("Quarantined V10 annotations cannot be restored")
    if set(v10.provenance["source_rows"]) != {
        n for c in ready["class_counts"].values() for n in c["retained_source_rows"]
    }:
        raise ValueError("V10 retained row identities changed")
    for ds, spec in zip(
        [*train.datasets, development], [*study["train"], proposal["development"]], strict=True
    ):
        for name, digest in ds.provenance["hashes"].items():
            hashes[str(Path(spec["dataset"]) / name)] = digest
        for name, digest in ds.provenance["pose_source_hashes"].items():
            hashes[str(Path(spec["pose_run"]) / name)] = digest
        hashes[spec["policy"]] = ds.provenance["policy_sha256"]
    prior = {
        str(p): sha256(p)
        for root in sorted(Path("reports/boxingvi").glob("level1-*"))
        for p in root.rglob("*")
        if p.is_file()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.parent / "preserved-prior-experiments.json", prior)
    result = {
        "schema_version": "v10-training-gate-1.0",
        "training_ready": True,
        "scope": "User-approved provisional V4/V10 positive-clip experiment only",
        "training_specs": study["train"],
        "development_spec": proposal["development"],
        "source_groups": source_groups(study["train"]),
        "hashes": hashes,
        "train": train.provenance,
        "development": development.provenance,
        "excluded_source_rows": sorted(excluded),
        "uncertainties": ready["uncertainties"],
    }
    write_json(output, result)
    return result


def check_v10_gate(path, specs):
    if [s.get("recording") for s in specs] != ["V4", "V10"]:
        raise ValueError("V4/V10 gate only; V5 development and V9 reserved")
    gate = json.loads(Path(path).read_text())
    if (
        gate["schema_version"] != "v10-training-gate-1.0"
        or gate["training_ready"] is not True
        or gate["training_specs"] != specs
        or gate["source_groups"] != source_groups(specs)
        or gate["development_spec"]["recording"] != "V5"
    ):
        raise ValueError("Frozen V10 training gate mismatch")
    for name, digest in gate["hashes"].items():
        if sha256(name) != digest:
            raise ValueError(f"Frozen V10 input changed: {name}")
    return gate


def run(proposal_path, gate, output, smoke=False):
    proposal, study = proposal_config(proposal_path, gate)
    check_v10_gate(gate, study["train"])
    output = Path(output)
    # Runnable config is generated beside the fresh run, preserving the starting proposal.
    config = output.with_name(output.name + "-study.yaml")
    if config.exists() or output.exists():
        raise ValueError("Use a fresh study and output path")
    if not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Study outputs must be under reports/")
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(study, sort_keys=False))
    summary = run_study(config, output, smoke)
    (output / "accepted-proposal.yaml").write_text(Path(proposal_path).read_text())
    prior = json.loads((Path(gate).parent / "preserved-prior-experiments.json").read_text())
    if any(sha256(p) != h for p, h in prior.items()):
        raise ValueError("Prior experiment artifacts changed")
    write_json(
        output / "prior-preservation-check.json",
        {
            "verified_files": len(prior),
            "all_unchanged": True,
            "proposal_sha256": sha256(proposal_path),
            "readiness_sha256": proposal["dataset_snapshot"]["readiness_sha256"],
        },
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=Path("configs/level1_v10_proposed.yaml"))
    parser.add_argument(
        "--gate", type=Path, default=Path("reports/boxingvi/level1-v10-data-v1/training-gate.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-gate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.freeze_gate:
        freeze_gate(args.proposal, args.output)
    else:
        run(args.proposal, args.gate, args.output, args.smoke)
