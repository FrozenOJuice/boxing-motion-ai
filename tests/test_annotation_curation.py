"""Synthetic annotation-review histories and exclusion plans; no real dataset needed."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from typer.testing import CliRunner

from boxing_motion_ai.cli import app
from boxing_motion_ai.data.annotation_review import (
    ReviewCase,
    ReviewLedger,
    Revision,
    append_review,
    digest,
    file_hash,
    load_ledger,
    write_new,
)
from boxing_motion_ai.data.curation import (
    GROUPS,
    GUARDS,
    LABELS,
    RECORDINGS,
    build_plan,
    load_config,
    read_sources,
    run_curation,
    validate_bindings,
)


def revision(*, actor="assistant", disposition="hold", label=None, bounds=None):
    return dict(
        revision_id="review-1",
        supersedes_sha256=None,
        actor_type=actor,
        reviewer="Synthetic test reviewer",
        reviewed_at="2026-10-22T10:00:00Z",
        disposition=disposition,
        label_status="confirmed" if label else "ambiguous",
        confirmed_label=label,
        lead_rear_evidence="Observed stance in native frames" if label else None,
        boundary_status="confirmed" if bounds else "mixed_action",
        confirmed_bounds=bounds,
        reasons=["ambiguous_phase"],
        evidence=[
            dict(
                kind="human_frame_review" if actor == "human" else "diagnostic_observation",
                reference="synthetic-sheet.jpg",
                description="Synthetic test evidence",
                frame_indices=[3, 4],
            )
        ],
        unresolved_uncertainty=["Origin convention unverified"],
    )


def bounds(start=3, end=4):
    return dict(start=start, end=end, convention="zero_based_decoded_inclusive")


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def pin(path):
    return dict(path=str(path), sha256=file_hash(path))


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sources = {}
    for v in RECORDINGS:
        root = Path(f"data/processed/boxingvi/{v}/candidates-full-v1")
        rows = [
            dict(
                source_row=i,
                video_id=v,
                punch_type_provisional=LABELS[(i - 1) % 6],
                candidate_start_index=i * 3,
                candidate_end_index=i * 3 + 1,
                annotation_eligible=i != 7,
                exclusion_reasons=["suspect_overlap"] if i == 7 else [],
            )
            for i in range(1, 8)
        ]
        # Include the legacy V5 video_id-only candidate schema.
        if v != "V5":
            for r in rows:
                r["recording_id"] = v
        root.mkdir(parents=True)
        (root / "candidates.jsonl").write_text("\n".join(map(json.dumps, rows)))
        (root / "poses.npz").write_bytes(b"synthetic pose bytes; never decoded in curation")
        save(
            root / "preparation.json",
            dict(video_id=v, source_frame_count=100, npz_sha256=file_hash(root / "poses.npz")),
        )
        inputs = {
            name: file_hash(root / name)
            for name in ("preparation.json", "poses.npz", "candidates.jsonl")
        }
        policy = Path(f"reports/policies/{v}/policy-index.json")
        save(
            policy,
            dict(
                video_id=v,
                schema_version="recording-mask-policy-index-1.1",
                policy="mask_aware70",
                source_mask_threshold=0.5,
                inputs=inputs,
                selected_source_rows=list(range(1, 7)),
            ),
        )
        sources[v] = dict(
            dataset=str(root),
            policy=pin(policy),
            inputs=inputs,
            source_group=GROUPS[v],
            related_recordings_guard=GUARDS[v],
            role="development" if v == "V5" else "training",
        )
    cases = [{"id": k} for k in ("V4:1", "V5:1", "V6:6")]
    case_path = Path("reports/diagnostic/cases.json")
    save(case_path, cases)
    diagnostic = Path("configs/diagnostic.json")  # JSON is also valid YAML.
    save(
        diagnostic,
        dict(
            assessments={c["id"]: {} for c in cases},
            priority_disputes=["V4:1"],
            candidate_evaluation_subset=["V6:6"],
        ),
    )
    entries = {}
    for c in cases:
        v, row = c["id"].split(":")
        original = json.loads(
            (Path(sources[v]["dataset"]) / "candidates.jsonl")
            .read_text()
            .splitlines()[int(row) - 1]
        )
        entries[c["id"]] = dict(
            source_candidate_sha256=digest(original),
            origin="diagnostic_79",
            revisions=[revision(disposition="provisional" if v == "V6" else "hold")],
        )
    ledger_path = Path("configs/review-v1.json")
    ledger = ReviewLedger.model_validate(
        dict(
            schema_version="annotation-review-1.0",
            ledger_version=1,
            parent_ledger_sha256=None,
            diagnostic_sha256=file_hash(diagnostic),
            cases_sha256=file_hash(case_path),
            cases=entries,
        )
    )
    save(ledger_path, ledger.model_dump(mode="json"))
    protected = Path("reports/diagnostic/protected.json")
    save(protected, {})
    config = dict(
        schema_version="annotation-curation-plan-1.0",
        status="proposal_no_training",
        review_ledger=pin(ledger_path),
        diagnostic=pin(diagnostic),
        cases=pin(case_path),
        preservation_manifest=pin(protected),
        sources=sources,
        unreviewed_policy="retain_previously_approved_as_provisional",
        development_policy="unchanged_frozen_benchmark_with_shadow_holds",
        training_representation="xy_velocity",
        mirror_probability=0.5,
        visibility_probability=0.0,
        context_seconds=0.0,
        priority_disputes=["V4:1"],
        verification_anchors=["V6:6"],
    )
    config_path = Path("configs/proposed.json")
    save(config_path, config)
    return config_path, ledger_path


@pytest.mark.parametrize(
    "change",
    [
        dict(actor_type="assistant"),
        dict(lead_rear_evidence=None),
        dict(
            evidence=[
                dict(
                    kind="diagnostic_observation",
                    reference="note",
                    description="prediction",
                    frame_indices=[3],
                )
            ]
        ),
        dict(
            evidence=[
                dict(
                    kind="human_frame_review",
                    reference="sheet",
                    description="uncertain",
                    frame_indices=[],
                )
            ]
        ),
        dict(reviewed_at="2026-10-22T10:00:00"),
    ],
)
def test_confirmation_needs_human_stance_native_frames_and_review_time(change):
    r = revision(actor="human", disposition="accept", label="jab", bounds=bounds())
    r.update(change)
    with pytest.raises(ValueError):
        Revision.model_validate(r)


def test_ambiguous_mixed_and_excluded_decisions_need_no_forced_label():
    r = Revision.model_validate(revision())
    assert r.disposition == "hold" and r.boundary_status == "mixed_action"
    assert r.confirmed_bounds is r.confirmed_label is None
    excluded = revision(actor="human", disposition="exclude")
    excluded["unresolved_uncertainty"] = []
    assert Revision.model_validate(excluded).disposition == "exclude"
    with pytest.raises(ValueError, match="Acceptance"):
        Revision.model_validate(revision(actor="human", disposition="accept"))
    invalid = revision()
    invalid["confirmed_label"] = "no_punch"
    with pytest.raises(ValueError):
        Revision.model_validate(invalid)
    with pytest.raises(ValueError, match="reversed"):
        Revision.model_validate(revision(actor="human", bounds=bounds(5, 4)))


def test_append_is_new_version_requires_attestation_and_preserves_parent(study):
    _, ledger_path = study
    before = file_hash(ledger_path)
    old = load_ledger(ledger_path)
    r = revision(actor="human")
    r.update(
        revision_id="human-2",
        supersedes_sha256=digest(old.cases["V4:1"].revisions[-1].model_dump(mode="json")),
    )
    save("reports/decision.json", dict(case_id="V4:1", revision=r))
    output = Path("configs/review-v2.json")
    with pytest.raises(ValueError, match="attest"):
        append_review(ledger_path, "reports/decision.json", output)
    updated = append_review(ledger_path, "reports/decision.json", output, attest_human=True)
    assert updated.ledger_version == 2 and updated.parent_ledger_sha256 == before
    assert file_hash(ledger_path) == before
    assert len(updated.cases["V4:1"].revisions) == 2
    with pytest.raises(FileExistsError):
        append_review(ledger_path, "reports/decision.json", output, attest_human=True)
    with pytest.raises(ValueError, match="never original data"):
        write_new("data/raw/review.json", {})


def test_history_rejects_rewrite_stale_parent_duplicate_ids_and_aliases():
    a = revision()
    b = revision()
    b.update(revision_id="review-2", supersedes_sha256="0" * 64)
    with pytest.raises(ValueError, match="hash-linked"):
        ReviewCase(source_candidate_sha256="0" * 64, origin="diagnostic_79", revisions=[a, b])
    b["supersedes_sha256"] = digest(Revision.model_validate(a).model_dump(mode="json"))
    b["revision_id"] = a["revision_id"]
    with pytest.raises(ValueError, match="hash-linked"):
        ReviewCase(source_candidate_sha256="0" * 64, origin="diagnostic_79", revisions=[a, b])
    for key in ("V9:1", "V4:01"):
        with pytest.raises(ValueError, match="identities"):
            ReviewLedger(
                schema_version="annotation-review-1.0",
                ledger_version=1,
                parent_ledger_sha256=None,
                diagnostic_sha256="0" * 64,
                cases_sha256="0" * 64,
                cases={
                    key: dict(
                        source_candidate_sha256="0" * 64, origin="diagnostic_79", revisions=[]
                    )
                },
            )


def test_proposal_excludes_training_holds_but_preserves_v5_and_unknowns(study):
    cfg_path, ledger_path = study
    cfg = load_config(cfg_path)
    ledger = load_ledger(ledger_path)
    sources = read_sources(cfg)
    result = build_plan(cfg, ledger, sources)
    assert result["records"]["V4"]["selected_source_rows"] == [2, 3, 4, 5, 6]
    assert result["records"]["V5"]["selected_source_rows"] == list(range(1, 7))
    assert result["records"]["V5"]["development_shadow_hold_rows"] == [1]
    assert result["records"]["V4"]["by_class"]["jab"]["removed"] == 1
    assert result["human_accepted_cases"] == []
    assert not result["training_enabled"] and result["status"] == "blocked_pending_review"
    assert all(
        r["boxer_id"] is None and r["session_id"] is None for r in result["records"].values()
    )


def test_original_exclusions_never_restore_and_confirmed_changes_never_rewrite(study):
    cfg_path, ledger_path = study
    cfg = load_config(cfg_path)
    sources, ledger = read_sources(cfg), load_ledger(ledger_path)
    accepted = revision(actor="human", disposition="accept", label="jab", bounds=bounds(21, 22))
    ledger.cases["V6:7"] = ReviewCase(
        source_candidate_sha256=digest(sources["V6"]["rows"][7]),
        origin="precautionary_neighbor",
        revisions=[accepted],
    )
    changed = revision(actor="human", disposition="accept", label="cross", bounds=bounds())
    ledger.cases["V4:1"] = ReviewCase(
        source_candidate_sha256=digest(sources["V4"]["rows"][1]),
        origin="diagnostic_79",
        revisions=[changed],
    )
    before = deepcopy(sources)
    result = build_plan(cfg, ledger, sources)
    assert 7 not in result["records"]["V6"]["selected_source_rows"]
    assert result["confirmed_changes_not_applied"] == ["V4:1"]
    assert "confirmed_changes_need_separate_preparation" in result["blockers"]
    assert sources == before


def test_human_can_defer_disputes_while_confirming_anchors(study):
    cfg_path, ledger_path = study
    cfg = load_config(cfg_path)
    ledger, sources = load_ledger(ledger_path), read_sources(cfg)
    ledger.cases["V4:1"].revisions = [Revision.model_validate(revision(actor="human"))]
    ledger.cases["V6:6"].revisions = [
        Revision.model_validate(
            revision(
                actor="human", disposition="accept", label="rear_uppercut", bounds=bounds(18, 19)
            )
        )
    ]
    result = build_plan(cfg, ledger, sources)
    assert result["status"] == "blocked_pending_review"  # No verified anchors for five classes yet.
    assert result["pending_human_queue"] == [] and not result["training_enabled"]
    assert 1 not in result["records"]["V4"]["selected_source_rows"]
    # A reviewer can keep the dispute held while confirming one anchor per class.
    cfg.verification_anchors = [f"V6:{i}" for i in range(1, 7)]
    for i, label in enumerate(LABELS, 1):
        ledger.cases[f"V6:{i}"] = ReviewCase(
            source_candidate_sha256=digest(sources["V6"]["rows"][i]),
            origin="diagnostic_79",
            revisions=[
                revision(
                    actor="human",
                    disposition="accept",
                    label=label,
                    bounds=bounds(i * 3, i * 3 + 1),
                )
            ],
        )
    result = build_plan(cfg, ledger, sources)
    assert result["status"] == "ready_for_provisional_experiment_review"
    assert result["human_verified_anchor_support"] == dict.fromkeys(LABELS, 1)
    assert not result["training_enabled"]


def test_proposal_is_reproducible_portable_immutable_and_cli_honestly_blocks(study):
    cfg_path, ledger_path = study
    # Pin every synthetic input to catch accidental modifications by report writing.
    before = {
        p: file_hash(p)
        for d in ("configs", "data", "reports")
        for p in Path(d).rglob("*")
        if p.is_file()
    }
    result = run_curation(cfg_path, "reports/plan-1")
    run_curation(cfg_path, "reports/plan-2")
    assert (
        Path("reports/plan-1/curated-selection.json").read_bytes()
        == Path("reports/plan-2/curated-selection.json").read_bytes()
    )
    assert before == {p: file_hash(p) for p in before}
    template = json.loads(Path("reports/plan-1/decision-templates/V4-1.json").read_text())
    with pytest.raises(ValueError):  # Blank reviewer/time cannot become certification.
        Revision.model_validate(template["revision"])
    assert result["pending_human_queue"] == ["V4:1", "V6:6"]
    runner = CliRunner()
    response = runner.invoke(
        app, ["curate-annotations", "--config", str(cfg_path), "--output", "reports/plan-cli"]
    )
    assert response.exit_code == 1 and "blocked_pending_review" in response.output
    assert "training is disabled" in response.output
    refusal = runner.invoke(
        app, ["curate-annotations", "--config", str(cfg_path), "--output", "data/raw/new-plan"]
    )
    assert refusal.exit_code == 2 and not Path("data/raw/new-plan").exists()


def test_rejects_stale_sources_config_scope_and_out_of_video_review(study):
    cfg_path, ledger_path = study
    cfg, ledger = load_config(cfg_path), load_ledger(ledger_path)
    sources = read_sources(cfg)
    case_data = json.loads(cfg.cases.read_bytes())
    diagnostic = json.loads(cfg.diagnostic.read_bytes())
    validate_bindings(cfg, ledger, case_data, diagnostic, sources)
    ledger.cases["V4:1"].revisions[0].evidence[0].frame_indices = [100]
    with pytest.raises(ValueError, match="video length"):
        validate_bindings(cfg, ledger, case_data, diagnostic, sources)
    ledger.cases["V4:1"].revisions = [
        Revision.model_validate(revision(actor="human", bounds=bounds(3, 100)))
    ]
    with pytest.raises(ValueError, match="boundaries exceed"):
        validate_bindings(cfg, ledger, case_data, diagnostic, sources)
    Path(cfg.sources["V4"].dataset, "poses.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        read_sources(cfg)
    data = json.loads(cfg_path.read_text())
    data["sources"]["V9"] = data["sources"]["V4"]
    save("configs/invalid.json", data)
    with pytest.raises(ValueError, match="V9 reserved"):
        load_config("configs/invalid.json")
    data["sources"].pop("V9")
    data["sources"]["V7"]["source_group"] = "different"
    save("configs/invalid.json", data)
    with pytest.raises(ValueError, match="guards"):
        load_config("configs/invalid.json")


def test_precautionary_neighbors_are_separate_from_reviewed_case_scenario(study):
    cfg_path, ledger_path = study
    cfg, ledger = load_config(cfg_path), load_ledger(ledger_path)
    sources = read_sources(cfg)
    original = sources["V10"]["rows"][3]
    ledger.cases["V10:3"] = ReviewCase(
        source_candidate_sha256=digest(original),
        origin="precautionary_neighbor",
        revisions=[revision()],
    )
    result = build_plan(cfg, ledger, sources)
    counts = result["records"]["V10"]["by_class"]["lead_hook"]
    assert counts == dict(before=1, after=0, removed=1, diagnostic_79_only_after=1)
    assert result["human_accepted_cases"] == []


def test_case_fingerprints_and_pinned_ledgers_prevent_silent_changes(study):
    cfg_path, ledger_path = study
    cfg, ledger = load_config(cfg_path), load_ledger(ledger_path)
    sources = read_sources(cfg)
    cases, diagnostic = json.loads(cfg.cases.read_bytes()), json.loads(cfg.diagnostic.read_bytes())
    ledger.cases["V4:1"].source_candidate_sha256 = "0" * 64
    with pytest.raises(ValueError, match="candidate changed"):
        validate_bindings(cfg, ledger, cases, diagnostic, sources)
    save(ledger_path, ledger.model_dump(mode="json"))
    with pytest.raises(ValueError, match="Stale pinned input"):
        run_curation(cfg_path, "reports/stale-plan")
    assert not Path("reports/stale-plan").exists()


def test_cli_requires_explicit_human_attribution_and_leaves_data_unchanged(study):
    _, ledger_path = study
    ledger = load_ledger(ledger_path)
    r = revision(actor="human")
    r.update(
        revision_id="human-2",
        supersedes_sha256=digest(ledger.cases["V4:1"].revisions[-1].model_dump(mode="json")),
    )
    save("reports/decision.json", dict(case_id="V4:1", revision=r))
    args = [
        "record-annotation-review",
        "--ledger",
        str(ledger_path),
        "--decision",
        "reports/decision.json",
        "--output",
        "configs/review-v2.json",
    ]
    runner = CliRunner()
    assert runner.invoke(app, args).exit_code == 2
    assert not Path("configs/review-v2.json").exists()
    assert runner.invoke(app, [*args, "--attest-human-review"]).exit_code == 0
    assert load_ledger(ledger_path).ledger_version == 1
    assert load_ledger("configs/review-v2.json").ledger_version == 2
