"""Synthetic pose phases, exact-time proposals, quality gates and local job integrity."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_assisted_inference import classifier as classifier  # noqa: F401
from test_dashboard_services import workspace as workspace  # noqa: F401
from test_detector import source
from test_uploaded_activity import prepared

np = pytest.importorskip("numpy")
from boxing_motion_ai.analysis.automatic import classify_candidates  # noqa: E402
from boxing_motion_ai.detection.instances import Policy, disjoint, segment  # noqa: E402

ROOT = Path(__file__).parents[1]
POLICY = Policy.model_validate(json.loads((ROOT / "configs/punch_instances_v1.json").read_text()))


def sequence(cycles=(("left", 10, 30),), n=100):
    raw = source(n)
    raw["pts"] = np.arange(n) * 20 + 7000
    raw["time_base_den"][:] = 1000
    # .3 torso height; aspect 4/3. Wrist motion is shoulder-relative torso units.
    for arm, shoulder, elbow, wrist, x in [("left", 11, 13, 15, 0.4), ("right", 12, 14, 16, 0.6)]:
        raw["coordinates"][:, shoulder, 0] = x
        raw["coordinates"][:, elbow, :2] = [x, 0.3]
        raw["coordinates"][:, wrist, :2] = [x, 0.22]
        for side, a, z in cycles:
            if side == arm:
                for i in range(a, min(z, n)):
                    t = (i - a) / (z - a)
                    excursion = 2 * t if t <= 0.5 else 2 * (1 - t)
                    raw["coordinates"][i, wrist, 0] += excursion * 0.3 / (4 / 3)
                    raw["coordinates"][i, elbow, 0] += excursion * 0.15 / (4 / 3)
                    raw["coordinates"][i, elbow, 1] -= excursion * 0.09
    video = prepared(raw)
    frames = [
        dict(
            frame_index=i,
            timestamp_exact=f["timestamp_exact"],
            observable=True,
            discontinuity_before=i == 0,
        )
        for i, f in enumerate(video["frames"])
    ]
    activity = dict(
        frames=frames,
        bouts=[dict(bout_id="one-combination-bout", start_frame=0, stop_frame_exclusive=n)],
        effective_reset_gap_seconds_exact="1/25",
    )
    return raw, video, activity


def test_one_isolated_cycle_has_explicit_phases_and_exact_observed_boundaries():
    raw, video, activity = sequence()
    result = segment(raw, video, activity, POLICY)
    assert len(result["candidates"]) == 1
    p = result["candidates"][0]
    assert (p["start_frame"], p["peak_frame"], p["stop_frame_exclusive"]) == (10, 20, 31)
    assert p["evidence_through_frame"] >= 34
    assert p["excursion_torso"] == pytest.approx(1)
    classified = classify_candidates(
        result,
        type(
            "Unavailable", (), {"classify": lambda *a: dict(status="unavailable", reason="test")}
        )(),
        raw,
        video,
    )
    p = classified["candidates"][0]
    assert p["start_timestamp_exact"] == "36/5"
    assert p["stop_timestamp_exact"] == "381/50"
    assert p["source_start"]["pts"] == raw["pts"][10]
    assert p["classification"]["status"] == "unavailable"
    assert p["verified"] is False


@pytest.mark.parametrize(
    "cycles", [(("left", 10, 30), ("right", 40, 60)), (("left", 10, 30), ("left", 40, 60))]
)
def test_combination_can_have_two_complete_instances_in_one_activity_bout(cycles):
    raw, video, activity = sequence(cycles)
    result = segment(raw, video, activity, POLICY)
    assert len(activity["bouts"]) == 1 and len(result["candidates"]) == 2
    assert {p["activity_bout_id"] for p in result["candidates"]} == {"one-combination-bout"}
    assert [p["peak_frame"] for p in result["candidates"]] == [20, 50]
    assert result == segment(raw, video, activity, POLICY)


def test_no_motion_or_no_activity_does_not_get_classified():
    raw, video, activity = sequence(())
    assert not segment(raw, video, activity, POLICY)["candidates"]
    raw, video, activity = sequence()
    activity["bouts"] = []
    assert not segment(raw, video, activity, POLICY)["candidates"]


@pytest.mark.parametrize(
    "kind", ["missing", "arm_missing", "gap", "scene", "segment", "cut", "reset"]
)
def test_never_fills_missing_pose_or_crosses_continuity_boundaries(kind):
    raw, video, activity = sequence()
    if kind in ("missing", "arm_missing"):
        j = 11 if kind == "missing" else 13
        raw["coordinates"][20, j] = np.nan
        raw["valid_mask"][20, j] = False
        if kind == "missing":
            activity["frames"][20]["observable"] = False
    elif kind == "gap":
        raw["pts"][20:] += 1000
        video = prepared(raw)
        for i, f in enumerate(video["frames"]):
            activity["frames"][i]["timestamp_exact"] = f["timestamp_exact"]
    elif kind in ("scene", "segment"):
        raw[kind + "_id"][20:] = 1
    elif kind == "cut":
        video["confirmed_scene_cuts"] = [20]
    else:
        activity["frames"][20]["discontinuity_before"] = True
    result = segment(raw, video, activity, POLICY)
    assert not result["candidates"] and result["unresolved_activity"]


def test_incomplete_outbound_motion_is_unresolved_not_one_punch_per_bout():
    raw, video, activity = sequence((("left", 10, 70),), n=40)
    result = segment(raw, video, activity, POLICY)
    assert not result["candidates"] and result["unresolved_activity"]


def test_overlap_clusters_are_not_trimmed_into_fake_individual_clips():
    raw, video, activity = sequence((("left", 10, 40), ("right", 20, 50)))
    result = segment(raw, video, activity, POLICY)
    assert not result["candidates"]
    assert len([p for p in result["rejected_proposals"] if "overlapping" in p["reason"]]) == 2
    assert result["unresolved_activity"][0]["stop_frame_exclusive"] == len(raw["pts"])


def test_duplicate_and_chained_overlap_proposals_have_deterministic_safe_resolution():
    def p(a, z, arm="left"):
        return dict(start_frame=a, stop_frame_exclusive=z, arm=arm, peak_frame=a + 1)

    a = p(0, 10)
    assert disjoint([a, deepcopy(a)]) == ([a], [])
    accepted, rejected = disjoint([p(20, 30), p(8, 15), a, p(14, 19)])
    assert accepted == [p(20, 30)] and len(rejected) == 3


def test_classifier_original_quality_gates_and_six_scores_are_preserved(classifier):
    _, model = classifier
    raw, video, activity = sequence()
    result = classify_candidates(segment(raw, video, activity, POLICY), model, raw, video)
    prediction = result["candidates"][0]["classification"]
    assert prediction["status"] == "available" and len(prediction["scores"]) == 6
    assert "inference_seconds" not in prediction
    again = classify_candidates(segment(raw, video, activity, POLICY), model, raw, video)
    assert again == result
    raw["coordinates"][10:31, 11:25] = np.nan
    raw["valid_mask"][10:31, 11:25] = False
    unavailable = classify_candidates(
        dict(
            candidates=[dict(start_frame=10, stop_frame_exclusive=31, evidence_through_frame=35)],
            unresolved_activity=[],
            rejected_proposals=[],
        ),
        model,
        raw,
        video,
    )
    assert unavailable["candidates"][0]["classification"]["status"] == "unavailable"
    assert "scores" not in unavailable["candidates"][0]["classification"]


def test_one_classification_failure_does_not_invent_scores_or_hide_other_candidates():
    raw, video, activity = sequence((("left", 10, 30), ("right", 40, 60)))

    class Failed:
        def classify(self, *args):
            raise RuntimeError("test inference failure")

    result = classify_candidates(segment(raw, video, activity, POLICY), Failed(), raw, video)
    assert len(result["candidates"]) == 2
    assert all(
        p["classification"]["status"] == "unavailable" and "scores" not in p["classification"]
        for p in result["candidates"]
    )


def test_pause_at_extension_is_not_a_guard_boundary_or_two_artificial_half_punches():
    raw, video, activity = sequence(())
    # Outbound 10..20, held extension 20..30, return 30..40.
    for i in range(10, 41):
        excursion = min((i - 10) / 10, 1) if i <= 30 else (40 - i) / 10
        raw["coordinates"][i, 15, 0] += excursion * 0.3 / (4 / 3)
    r = segment(raw, video, activity, POLICY)
    assert len(r["candidates"]) == 1
    assert r["candidates"][0]["start_frame"] == 10
    assert r["candidates"][0]["stop_frame_exclusive"] == 41


def test_multi_peak_motion_without_a_stable_return_is_not_an_inverted_arm_cycle():
    raw, video, activity = sequence((("left", 10, 30), ("left", 30, 50)))
    # Hold both peaks; the brief middle return never establishes a quiet guard.
    for a, z in [(20, 25), (40, 45)]:
        raw["coordinates"][a:z, 15, :2] = raw["coordinates"][a, 15, :2]
        raw["coordinates"][a:z, 13, :2] = raw["coordinates"][a, 13, :2]
    r = segment(raw, video, activity, POLICY)
    assert not r["candidates"] and r["unresolved_activity"]


def test_auto_jobs_reuse_activity_cache_preserve_manual_outputs_and_retry(tmp_path, monkeypatch):
    from test_uploaded_activity import ready

    from boxing_motion_ai.analysis.common import write_json
    from boxing_motion_ai.analysis.jobs import AnalysisError, AnalysisJobs

    jobs = AnalysisJobs(tmp_path)
    key, directory = ready(jobs)
    task = jobs.automatic
    fingerprint = ["a" * 64]
    monkeypatch.setattr(
        task, "identity", lambda k: (fingerprint[0], {"activity_input_fingerprint": "b" * 64})
    )
    monkeypatch.setattr(jobs, "_activity_identity", lambda k: "b" * 64)
    submissions, workers = [], []
    monkeypatch.setattr(jobs.executor, "submit", lambda *a: submissions.append(a))
    manual = directory / "prediction-0-2.json"
    manual.write_text('{"manual":"preserved"}')

    def run(key, module, *args, **kwargs):
        workers.append(module)
        if module.endswith("detection.upload"):
            write_json(
                directory / f"activity-{'b' * 64}.json",
                dict(
                    schema_version="uploaded-activity-detection-1.0",
                    video_id=key,
                    input_fingerprint="b" * 64,
                    bouts=[],
                ),
            )
        else:
            state = jobs.activity(key)
            write_json(
                directory / f"automatic-{fingerprint[0]}.json",
                dict(
                    schema_version="automatic-punch-candidates-2.0",
                    video_id=key,
                    input_fingerprint=fingerprint[0],
                    activity_input_fingerprint="b" * 64,
                    activity_result_sha256=state["result_sha256"],
                    candidates=[],
                ),
            )

    monkeypatch.setattr(jobs, "run", run)
    state = task.start(key)
    assert task.start(key) == state and len(submissions) == 1
    for action in [
        lambda: jobs.remove(key),
        lambda: jobs.scan(key),
        lambda: jobs.classify(key, 0, 2),
    ]:
        with pytest.raises(AnalysisError):
            action()
    task.process(key, fingerprint[0], state["request_id"])
    assert task.status(key)["status"] == "ready"
    assert workers == ["boxing_motion_ai.detection.upload", "boxing_motion_ai.analysis.automatic"]
    assert task.start(key)["status"] == "ready" and len(submissions) == 1
    assert manual.read_text() == '{"manual":"preserved"}'
    fingerprint[0] = "c" * 64
    assert task.status(key)["status"] == "failed"
    state = task.start(key)
    monkeypatch.setattr(
        jobs,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AnalysisError("Synthetic classifier worker failure")),
    )
    task.process(key, fingerprint[0], state["request_id"])
    assert "classifier worker failure" in task.status(key)["message"]
    monkeypatch.setattr(jobs, "run", run)
    state = task.start(key)
    task.process(key, fingerprint[0], state["request_id"])
    assert task.status(key)["status"] == "ready"
    assert (
        workers.count("boxing_motion_ai.detection.upload") == 1
    )  # Never repeats cached activity/poses.
    (directory / f"activity-{'b' * 64}.json").write_text("changed")
    assert task.status(key)["status"] == "failed"
    (directory / "automatic-state.json").write_text("invalid json")
    assert task.status(key)["status"] == "failed"


def test_auto_api_is_local_and_upload_scoped(workspace, monkeypatch):
    from fastapi.testclient import TestClient
    from test_uploaded_activity import ready

    from boxing_motion_ai.api.app import create_app

    app = create_app(workspace[0])
    jobs = app.state.analysis
    key, _ = ready(jobs)
    monkeypatch.setattr(jobs.automatic, "identity", lambda k: ("a" * 64, {}))
    monkeypatch.setattr(jobs.executor, "submit", lambda *a: None)
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    path = f"/api/analyze/videos/{key}/automatic"
    assert client.get(path).json()["status"] == "not_started"
    assert client.post(path, json={}).status_code == 403
    h = {"origin": "http://127.0.0.1:8000", "x-review-client": "dashboard-v0.2"}
    assert client.post(path, json={}, headers=h).status_code == 202
    assert client.get("/api/analyze/videos/V9/automatic").status_code == 404
    assert client.post("/api/analyze/videos/V9/automatic", json={}, headers=h).status_code == 404
    jobs.automatic.running.clear()
    assert "interrupted" in client.get(path).json()["message"]


def test_rigid_body_translation_does_not_become_a_wrist_excursion():
    raw, video, activity = sequence(())
    offsets = np.sin(np.linspace(0, 7, len(raw["pts"]))) * 0.12
    raw["coordinates"][:, :, 0] += offsets[:, None]
    assert not segment(raw, video, activity, POLICY)["candidates"]


def test_automatic_fingerprint_binds_classifier_and_policy(tmp_path):
    from unittest.mock import patch

    from boxing_motion_ai.analysis import automatic_contract as contract
    from boxing_motion_ai.analysis.common import write_json

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/assisted_level1_v1.json").write_text("{}")
    (tmp_path / contract.POLICY).write_bytes((ROOT / contract.POLICY).read_bytes())
    base = ("b" * 64, {"source": "synthetic"}, {}, {}, {})
    binding = dict(
        checkpoint=dict(path="checkpoint.pt", sha256="1" * 64),
        provenance=dict(path="provenance.json", sha256="2" * 64),
    )
    with (
        patch.object(contract, "activity_inputs", return_value=base),
        patch.object(contract, "pin", return_value=binding),
    ):
        first = contract.inputs(tmp_path, tmp_path, "a")[0]
        assert first == contract.inputs(tmp_path, tmp_path, "a")[0]
        policy = json.loads((tmp_path / contract.POLICY).read_text())
        policy["minimum_prominence_torso"] = 0.5
        write_json(tmp_path / contract.POLICY, policy)
        assert contract.inputs(tmp_path, tmp_path, "a")[0] != first
        second = contract.inputs(tmp_path, tmp_path, "a")[0]
        binding["checkpoint"]["sha256"] = "3" * 64
        assert contract.inputs(tmp_path, tmp_path, "a")[0] != second


def test_automatic_worker_rejects_stale_request_before_model_loading(monkeypatch, tmp_path):
    from boxing_motion_ai.analysis import automatic

    monkeypatch.setattr(automatic, "inputs", lambda *a: ("new", {}, {}, {}, {}))
    with pytest.raises(ValueError, match="changed while queued"):
        automatic.analyze(tmp_path, tmp_path, "a", "old", "hash")


def test_retryable_classifier_failures_can_be_retried_without_reupload(tmp_path, monkeypatch):
    from test_uploaded_activity import ready

    from boxing_motion_ai.analysis.jobs import AnalysisJobs

    jobs = AnalysisJobs(tmp_path)
    key, _ = ready(jobs)
    task = jobs.automatic
    monkeypatch.setattr(task, "identity", lambda k: ("a" * 64, {}))
    monkeypatch.setattr(
        task,
        "status",
        lambda k: dict(
            status="ready",
            result=dict(
                candidates=[dict(classification=dict(status="unavailable", retryable=True))]
            ),
        ),
    )
    calls = []
    monkeypatch.setattr(jobs.executor, "submit", lambda *a: calls.append(a))
    assert task.start(key)["status"] == "processing" and len(calls) == 1


def test_automatic_worker_classifies_eligible_candidates_and_writes_deterministic_record(
    classifier, monkeypatch, tmp_path
):
    from boxing_motion_ai.analysis import automatic

    root, _ = classifier
    raw, video, activity = sequence((("left", 10, 30), ("right", 40, 60)))
    activity["unobservable_intervals"] = []
    directory = tmp_path / "synthetic-upload"
    directory.mkdir()
    np.savez_compressed(directory / "poses.npz", **raw)
    identity = dict(activity_input_fingerprint="b" * 64)
    policy = POLICY.model_dump(mode="json")
    monkeypatch.setattr(automatic, "inputs", lambda *a: ("a" * 64, identity, video, {}, policy))
    monkeypatch.setattr(automatic, "activity_record", lambda *a: activity)
    result = automatic.analyze(root, directory, "synthetic", "a" * 64, "activity-hash")
    assert len(result["candidates"]) == 2
    assert all(len(p["classification"]["scores"]) == 6 for p in result["candidates"])
    path = directory / f"automatic-{'a' * 64}.json"
    original = path.read_bytes()
    assert result == automatic.analyze(root, directory, "synthetic", "a" * 64, "activity-hash")
    assert path.read_bytes() == original
    assert result["human_reviewed"] is False
    (root / "checkpoint.pt").unlink()  # This test's synthetic checkpoint only.
    with pytest.raises(ValueError, match="missing"):
        automatic.analyze(root, directory, "synthetic", "a" * 64, "activity-hash")
