"""Synthetic read-only services: no original media, pose tensors or checkpoints."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

from boxing_motion_ai.data.annotation_review import LABELS, RECORDINGS, digest
from boxing_motion_ai.data.curation import GROUPS, GUARDS
from boxing_motion_ai.services import DashboardService
from boxing_motion_ai.services.artifacts import MetadataStore, ServiceError


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return pin(root, name)


def pin(root, name):
    return {"path": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path.resolve()
    sources = {}
    rows = {}
    for v in RECORDINGS:
        directory = f"data/processed/boxingvi/{v}/candidates-full-v1"
        rows[v] = [
            dict(
                schema_version="v5-candidate-1.0" if v == "V5" else "recording-candidate-1.1",
                video_id=v,
                source_row=i + 1,
                punch_type_provisional=label,
                raw_start=100 + i,
                raw_end=103 + i,
                candidate_start_index=100 + i,
                candidate_end_index=103 + i,
                validity_mask=[True, False],
                confidence=[0.9, 0.1],
                exact_pts="9007199254740993/30000",
                exclusion_reasons=[],
            )
            for i, label in enumerate(LABELS)
        ]
        save(
            root,
            f"{directory}/preparation.json",
            dict(
                schema_version="v5-preparation-1.0" if v == "V5" else "recording-preparation-1.1",
                video_id=v,
                source_frame_count=200,
                npz_sha256="a" * 64,
            ),
        )
        (root / directory / "candidates.jsonl").write_text("\n".join(map(json.dumps, rows[v])))
        inputs = {
            name: pin(root, f"{directory}/{name}")["sha256"]
            for name in ("preparation.json", "candidates.jsonl")
        }
        inputs["poses.npz"] = "a" * 64  # Deliberately absent: metadata queries must not open it.
        policy = save(
            root,
            f"reports/{v}/policy.json",
            dict(
                schema_version="recording-mask-policy-index-1.1",
                video_id=v,
                policy="mask_aware70",
                source_mask_threshold=0.5,
                inputs=inputs,
                selected_source_rows=[1, 2, 3, 4, 5],
            ),
        )
        sources[v] = dict(
            dataset=directory,
            policy=policy,
            inputs=inputs,
            source_group=GROUPS[v],
            related_recordings_guard=GUARDS[v],
            role="development" if v == "V5" else "training",
        )
    evidence = save(
        root,
        "reports/cases.json",
        [
            dict(
                id="V4:1",
                frames=[100],
                exact_pts=["9007199254740993/30000"],
                validity_mask=[True, False],
            )
        ],
    )
    diagnostic = save(root, "configs/diagnostic.json", {})
    revision = dict(
        revision_id="initial",
        supersedes_sha256=None,
        actor_type="assistant",
        reviewer="Synthetic reviewer",
        reviewed_at="2026-10-25T10:00:00Z",
        disposition="hold",
        label_status="ambiguous",
        boundary_status="mixed_action",
        reasons=["Needs human review"],
        unresolved_uncertainty=["Unverified stance"],
        evidence=[
            dict(
                kind="diagnostic_observation",
                reference="sheet.jpg",
                description="Synthetic observation",
                frame_indices=[100],
            )
        ],
    )
    ledger = save(
        root,
        "configs/review.json",
        dict(
            schema_version="annotation-review-1.0",
            ledger_version=1,
            parent_ledger_sha256=None,
            diagnostic_sha256=diagnostic["sha256"],
            cases_sha256=evidence["sha256"],
            cases={
                "V4:1": dict(
                    source_candidate_sha256=digest(rows["V4"][0]),
                    origin="diagnostic_79",
                    revisions=[revision],
                )
            },
        ),
    )
    config = dict(
        schema_version="annotation-curation-plan-1.0",
        status="proposal_no_training",
        review_ledger=ledger,
        diagnostic=diagnostic,
        cases=evidence,
        preservation_manifest=dict(path="reports/never-open.json", sha256="b" * 64),
        sources=sources,
        unreviewed_policy="retain_previously_approved_as_provisional",
        development_policy="unchanged_frozen_benchmark_with_shadow_holds",
        training_representation="xy_velocity",
        mirror_probability=0.5,
        visibility_probability=0.0,
        context_seconds=0.0,
        priority_disputes=["V4:1"],
        verification_anchors=[],
    )
    config_ref = save(root, "configs/curation.json", config)
    proposal = save(
        root,
        "reports/proposal.json",
        dict(
            schema_version="curated-selection-proposal-1.0",
            ledger_version=1,
            status="blocked_pending_review",
            training_enabled=False,
            provenance=dict(config_sha256=config_ref["sha256"], ledger_sha256=ledger["sha256"]),
        ),
    )
    summary = save(
        root,
        "reports/summary.json",
        dict(schema_version="velocity-study-result-1.0", seeds=[1], by_model={}),
    )
    scores = dict(
        accuracy=1.0,
        macro_f1=1 / 6,
        labels=list(LABELS),
        confusion_matrix=[[1 if i == j == 0 else 0 for j in range(6)] for i in range(6)],
        per_class={
            label: dict(
                support=int(i == 0), precision=float(i == 0), recall=float(i == 0), f1=float(i == 0)
            )
            for i, label in enumerate(LABELS)
        },
    )
    metrics = save(root, "reports/metrics.json", dict(train=scores, development=scores))
    catalogue = dict(
        schema_version="dashboard-catalogue-1.0",
        curation=config_ref,
        proposal=proposal,
        experiments={
            "reference": dict(
                summary=summary,
                format="velocity-study-result-1.0",
                status="reference",
                description="Synthetic",
                metrics={"seed-1": metrics},
            )
        },
    )
    save(root, "configs/dashboard_catalogue.json", catalogue)
    return root, config, catalogue, rows


def publish(workspace):
    root, config, catalogue, _ = workspace
    catalogue["curation"] = save(root, "configs/curation.json", config)
    save(root, "configs/dashboard_catalogue.json", catalogue)
    return DashboardService(root)


def change_rows(workspace, v):
    root, config, _, rows = workspace
    path = f"{config['sources'][v]['dataset']}/candidates.jsonl"
    (root / path).write_text("\n".join(map(json.dumps, rows[v])))
    config["sources"][v]["inputs"]["candidates.jsonl"] = pin(root, path)["sha256"]
    return publish(workspace)


def test_inventory_and_statistics_are_read_only_explicit_and_lightweight(workspace, monkeypatch):
    root, _, _, rows = workspace
    service = DashboardService(root)
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.chdir(root.parent)  # No dependence on CWD.
    inventory = service.inventory()
    assert inventory.status == "available"
    assert inventory.data["available_recordings"] == 6
    assert inventory.data["records"]["V5"]["data"]["role"] == "development"
    assert inventory.data["records"]["V6"]["data"]["related_recordings_guard"] == ["V1", "V6"]
    assert inventory.data["records"]["V7"]["data"]["source_group"] == GROUPS["V8"]
    assert inventory.data["records"]["V4"]["data"]["boxer_id"] is None
    stats = service.statistics("V4")
    assert stats.status == "available" and stats.data["approved_provisional_examples"] == 5
    assert stats.data["absent_approved_classes"] == ["rear_uppercut"]
    assert stats.data["source_pose_bytes_verified"] is False
    assert service.annotations("V4", offset=2, limit=2).data["items"] == rows["V4"][2:4]
    assert service.annotation("V4", 1).data["source"] == rows["V4"][0]
    assert service.annotation("V4", 99).status == "missing"
    assert service.annotations("V4", offset=100).data["items"] == []
    assert before == {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def test_review_decisions_evidence_and_blocked_snapshot_remain_separate(workspace):
    service = DashboardService(workspace[0])
    result = service.review("V4", 1)
    assert result.status == "available" and result.data["human_accepted"] is False
    assert result.data["latest"]["boundary_status"] == "mixed_action"
    assert service.review_queue().data["entries"][0]["priority"] == "dispute"
    evidence = service.review_evidence("V4", 1).data
    assert evidence["case"]["exact_pts"] == ["9007199254740993/30000"]
    assert evidence["case"]["validity_mask"] == [True, False]
    assert evidence["diagnostic_is_ground_truth"] is False
    assert service.review("V4", 2).status == "missing"
    snapshot = service.curation_status()
    assert snapshot.data["proposal"]["status"] == "blocked_pending_review"
    assert snapshot.data["preservation_manifest_reverified"] is False


def test_stale_review_candidate_and_curation_provenance(workspace):
    workspace[3]["V4"][0]["raw_start"] = 99
    service = change_rows(workspace, "V4")
    assert service.review("V4", 1).status == "stale"
    assert service.curation_status().issues == ["stale_curation_snapshot"]


@pytest.mark.parametrize("recording", ["V9", "v9", "../V4", "V1", "/V4", "V4/../../V9"])
def test_unknown_or_reserved_ids_rejected_before_io(workspace, monkeypatch, recording):
    service = DashboardService(workspace[0])

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid ID must not read metadata")

    monkeypatch.setattr(service.store, "read", forbidden)
    for method in (service.recording, service.annotations, service.statistics):
        assert method(recording).status == "invalid"
    assert service.annotation(recording, 1).status == "invalid"
    assert service.review(recording, 1).status == "invalid"
    assert service.review_evidence(recording, 1).status == "invalid"


@pytest.mark.parametrize(
    "path",
    [
        "../x.json",
        "/tmp/x.json",
        "V9/a.json",
        "v9.xlsx.json",
        "foo/V9-x.json",
        "x\\a.json",
        "x:foo.json",
        "x//a.json",
        "x/./a.json",
        "%2e%2e/x.json",
        "x.pt",
        "x.npz",
    ],
)
def test_unsafe_paths_fail_before_open(tmp_path, monkeypatch, path):
    store = MetadataStore(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Unsafe path reached filesystem")

    monkeypatch.setattr(os, "open", forbidden)
    with pytest.raises(ServiceError):
        store.read(path)


@pytest.mark.parametrize("directory", [False, True])
def test_symlinks_never_followed(tmp_path, directory):
    root = tmp_path.resolve()
    reserved = root / "V9"
    reserved.mkdir()
    (reserved / "hidden.json").write_text("{}")  # Synthetic reserved fixture only.
    link = root / "link"
    link.symlink_to(reserved if directory else reserved / "hidden.json")
    if not directory:
        link.rename(root / "link.json")
    with pytest.raises(ServiceError, match="unsafe_or_unreadable"):
        MetadataStore(root).read("link/hidden.json" if directory else "link.json")


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', "{bad"])
def test_malformed_json_is_a_safe_error(workspace, raw):
    root, _, catalogue, _ = workspace
    (root / "reports/summary.json").write_text(raw)
    catalogue["experiments"]["reference"]["summary"] = pin(root, "reports/summary.json")
    result = publish(workspace).experiment("reference")
    assert result.status == "invalid" and result.data is None


def test_missing_stale_and_incomplete_are_not_zero_counts(workspace):
    root, config, _, _ = workspace
    path = f"{config['sources']['V4']['dataset']}/preparation.json"
    (root / path).unlink()
    service = DashboardService(root)
    assert service.recording("V4").status == "missing"
    assert service.statistics("V4").data is None
    assert service.inventory().status == "incomplete"
    save(root, path, {})
    assert service.recording("V4").status == "stale"
    config["sources"]["V4"]["inputs"]["preparation.json"] = save(
        root,
        path,
        dict(video_id="V4", schema_version="recording-preparation-1.1", npz_sha256="a" * 64),
    )["sha256"]
    assert publish(workspace).recording("V4").status == "incomplete"


@pytest.mark.parametrize("change", ["duplicate", "wrong_recording", "wrong_schema", "missing"])
def test_candidate_contract_errors(workspace, change):
    row = workspace[3]["V4"][0]
    if change == "duplicate":
        workspace[3]["V4"].append(row.copy())
    elif change == "wrong_recording":
        row["video_id"] = "V9"
    elif change == "wrong_schema":
        row["schema_version"] = "future-9"
    else:
        del row["raw_start"]
    result = change_rows(workspace, "V4").annotations("V4")
    assert result.status in ("invalid", "incomplete") and result.data is None


def test_reversed_excluded_annotation_is_not_repaired(workspace):
    row = workspace[3]["V4"][0]
    row.update(raw_start=200, raw_end=100, exclusion_reasons=["reversed_interval"])
    result = change_rows(workspace, "V4").annotation("V4", 1)
    assert result.data["source"] == row


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"limit": True}])
def test_pagination_validation(workspace, kwargs):
    assert DashboardService(workspace[0]).annotations("V4", **kwargs).status == "invalid"


def test_metrics_preserve_values_and_distinguish_absent_classes(workspace):
    service = DashboardService(workspace[0])
    result = service.model_metrics("reference", "seed-1")
    assert result.status == "available"
    assert result.data["absent_classes"]["development"] == list(LABELS[1:])
    assert result.data["metrics"]["development"]["macro_f1"] == 1 / 6
    assert result.data["metrics_recomputed"] is False
    assert service.experiment("reference").status == "available"
    assert service.experiments().data["items"]["reference"]["metric_ids"] == ["seed-1"]
    assert service.experiment("../../V9").status == "invalid"
    assert service.model_metrics("reference", "../weights.pt").status == "invalid"


def test_catalogue_pins_and_invalid_metrics(workspace):
    root, _, catalogue, _ = workspace
    save(root, "reports/metrics.json", {})
    assert DashboardService(root).model_metrics("reference", "seed-1").status == "stale"
    catalogue["experiments"]["reference"]["metrics"]["seed-1"] = pin(root, "reports/metrics.json")
    assert publish(workspace).model_metrics("reference", "seed-1").status == "invalid"


def test_imports_do_not_load_optional_compute_packages():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from boxing_motion_ai.services import DashboardService
assert not {'torch', 'numpy', 'PIL', 'av', 'mediapipe', 'torchvision'} & sys.modules.keys()
assert not any(k.startswith('boxing_motion_ai.training') for k in sys.modules)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_review_status_is_not_certification(workspace):
    service = DashboardService(workspace[0])
    status = service.review_status()
    assert status.data["dispositions"] == {"hold": 1}
    assert status.data["human_accepted"] == 0
    assert status.data["source_candidate_fingerprints_verified"] is False
    workspace[3]["V4"][0]["raw_start"] = 999
    service = change_rows(workspace, "V4")
    assert service.review_queue().status == "incomplete"


@pytest.mark.parametrize("violation", ["role", "group", "guard"])
def test_existing_source_group_and_development_guards_are_reused(workspace, violation):
    source = workspace[1]["sources"]["V5" if violation == "role" else "V7"]
    if violation == "role":
        source["role"] = "training"
    elif violation == "group":
        source["source_group"] = "guessed-boxer"
    else:
        source["related_recordings_guard"] = ["V7"]
    assert publish(workspace).recording("V4").status == "invalid"


def test_assistant_cannot_confirm_a_label(workspace):
    root, config, _, _ = workspace
    ledger = json.loads((root / "configs/review.json").read_text())
    revision = ledger["cases"]["V4:1"]["revisions"][0]
    revision.update(
        label_status="confirmed", confirmed_label="jab", lead_rear_evidence="Assistant guess"
    )
    config["review_ledger"] = save(root, "configs/review.json", ledger)
    assert publish(workspace).review_status().status == "invalid"


def test_metadata_size_limit_and_nonregular_files(tmp_path, monkeypatch):
    from boxing_motion_ai.services import artifacts

    root = tmp_path.resolve()
    monkeypatch.setattr(artifacts, "MAX_BYTES", 8)
    (root / "large.json").write_text(" " * 9)
    with pytest.raises(ServiceError, match="not_bounded_regular_metadata"):
        MetadataStore(root).read("large.json")
    os.mkfifo(root / "pipe.json")
    with pytest.raises(ServiceError, match="not_bounded_regular_metadata"):
        MetadataStore(root).read("pipe.json")


@pytest.mark.parametrize("format_name", ["multi-recording-result-1.0", "rgb-legacy-1"])
def test_other_registered_historical_summary_formats(workspace, format_name):
    root, _, catalogue, _ = workspace
    value = {"cases": {}, "smoke": False}
    value.update({"seed": 1} if format_name == "rgb-legacy-1" else {"schema_version": format_name})
    spec = catalogue["experiments"]["reference"]
    spec["format"] = format_name
    spec["summary"] = save(root, "reports/summary.json", value)
    service = publish(workspace)
    assert service.experiment("reference").data["results"] == value
    value["schema_version"] = "unsupported-version"
    spec["summary"] = save(root, "reports/summary.json", value)
    assert publish(workspace).experiment("reference").status == "invalid"


@pytest.mark.parametrize("problem", ["accuracy", "support", "matrix", "labels"])
def test_metric_contract_mismatches_are_reported(workspace, problem):
    root, _, catalogue, _ = workspace
    value = json.loads((root / "reports/metrics.json").read_text())
    metric = value["development"]
    if problem == "accuracy":
        metric["accuracy"] = 2.0
    elif problem == "support":
        metric["per_class"]["jab"]["support"] = 5
    elif problem == "matrix":
        metric["confusion_matrix"] = [[1]]
    else:
        metric["labels"] = list(reversed(LABELS))
    catalogue["experiments"]["reference"]["metrics"]["seed-1"] = save(
        root, "reports/metrics.json", value
    )
    assert publish(workspace).model_metrics("reference", "seed-1").status == "invalid"
