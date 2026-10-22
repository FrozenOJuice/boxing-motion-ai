"""Synthetic V10 split/gate/runner checks; no actual source footage or V9."""

import json
from pathlib import Path

import pytest


def test_v10_grouped_loader_keeps_reference_and_identity_contract(tmp_path):
    pytest.importorskip("torch")
    from test_expansion import synthetic_expansion

    from boxing_motion_ai.training.groups import GROUPS, GroupedClips, source_groups, validate_split

    specs = [synthetic_expansion(tmp_path, v) for v in ["V4", "V10"]]
    dev = synthetic_expansion(tmp_path, "V5")
    validate_split(specs, dev)
    ds = GroupedClips(specs)
    assert len(ds) == 12
    assert {c["recording_id"] for c in ds.clips} == {"V4", "V10"}
    assert all(c["boxer_id"] is None and c["session_id"] is None for c in ds.clips)
    assert source_groups(specs)["V10"] == "utah-lee-unresolved"
    assert "V10" not in GROUPS  # historical gates retain their exact original groups
    assert ds.provenance["class_counts"]["rear_uppercut"] == 2


@pytest.mark.parametrize(
    "ids", [["V4", "V5"], ["V4", "V9"], ["V4", "V7", "V8", "V10"], ["V10", "V4"]]
)
def test_unapproved_combinations_rejected_before_input_reads(ids, monkeypatch):
    pytest.importorskip("torch")
    from boxing_motion_ai.training.groups import GroupedClips

    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("Input opened"))
    with pytest.raises(ValueError, match="V9 reserved"):
        GroupedClips([{"recording": v} for v in ids])


def test_v10_gate_preserves_frozen_inputs_and_rejects_changed_selection(tmp_path):
    pytest.importorskip("torch")
    from boxing_motion_ai.training.groups import source_groups
    from boxing_motion_ai.training.v10_study import check_v10_gate
    from boxing_motion_ai.vision.pipeline import sha256

    specs = [{"recording": v} for v in ["V4", "V10"]]
    frozen = tmp_path / "synthetic-policy.json"
    frozen.write_text('{"selected_source_rows":[2,3]}')
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": "v10-training-gate-1.0",
                "training_ready": True,
                "training_specs": specs,
                "development_spec": {"recording": "V5"},
                "source_groups": source_groups(specs),
                "hashes": {str(frozen): sha256(frozen)},
            }
        )
    )
    assert check_v10_gate(gate, specs)["training_ready"]
    frozen.write_text('{"selected_source_rows":[2,3,123]}')
    with pytest.raises(ValueError, match="input changed"):
        check_v10_gate(gate, specs)


def test_accepted_proposal_translates_without_changing_budget_or_reference(tmp_path):
    pytest.importorskip("torch")
    import yaml

    from boxing_motion_ai.training.v10_study import proposal_config

    path = Path("configs/level1_v10_proposed.yaml")
    original = path.read_bytes()
    proposal, config = proposal_config(path, "reports/synthetic/gate.json")
    assert [s["recording"] for s in config["train"]] == ["V4", "V10"]
    assert config["seeds"] == [20261021, 20261022, 20261023]
    assert proposal["epochs"] == 40 and proposal["mirror_probability"] == 0.5
    assert not proposal["visibility_augmentation"] and not proposal["pre_clip_context"]
    assert path.read_bytes() == original
    proposal["train"][1]["recording"] = "V9"
    bad = tmp_path / "proposal.yaml"
    bad.write_text(yaml.safe_dump(proposal))
    with pytest.raises(ValueError, match="V9 reserved"):
        proposal_config(bad, "reports/synthetic/gate.json")


def test_v10_training_smoke_uses_only_approved_recordings(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import yaml
    from test_expansion import synthetic_expansion

    from boxing_motion_ai.training.experiment import run_experiment
    from boxing_motion_ai.training.groups import source_groups

    specs = [synthetic_expansion(tmp_path, v) for v in ["V4", "V10"]]
    dev = synthetic_expansion(tmp_path, "V5")
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": "v10-training-gate-1.0",
                "training_ready": True,
                "training_specs": specs,
                "development_spec": dev,
                "source_groups": source_groups(specs),
                "hashes": {},
            }
        )
    )
    config = yaml.safe_load(Path("configs/level1_initial.yaml").read_text())
    config.update(
        schema_version="level1-experiment-1.4",
        seed=7,
        epochs=1,
        hidden=4,
        threads=1,
        train=specs,
        development=dev,
        mirror_probability=0.5,
        expansion_gate=str(gate),
        runs=["tcn", "tcn_mask_only"],
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "reports/synthetic-v10"
    run_experiment(path, out, progress=lambda _: None)
    for model in ["tcn", "tcn_mask_only"]:
        rows = json.loads((out / model / "train_predictions.json").read_text())
        assert {r["recording_id"] for r in rows} == {"V4", "V10"}
        rows = json.loads((out / model / "development_predictions.json").read_text())
        assert {r["recording_id"] for r in rows} == {"V5"}
    assert (out / "tcn/augmentation.json").read_bytes() == (
        out / "tcn_mask_only/augmentation.json"
    ).read_bytes()
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["context_policy"]["seconds"] == 0
    assert provenance["visibility_augmentation"]["probability"] == 0
