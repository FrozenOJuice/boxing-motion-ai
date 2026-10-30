"""Loss contracts established without V5 feature tensors or model predictions."""

from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import pytest
import torch

from boxing_motion_ai.detection import localization_balanced_loss as objective
from boxing_motion_ai.detection.localization_data import supervision


def case(recording, spans=((10, 20),), n=60):
    video = dict(
        frames=[dict(timestamp_exact=str(Q(i, 25)), frame_index=i) for i in range(n)],
        stop_timestamp_exact=str(Q(n, 25)),
    )
    events = [dict(event_id=f"{recording}-{j}", start=a, stop=z) for j, (a, z) in enumerate(spans)]
    interval = dict(
        exhaustive=True,
        provenance=dict(recording=recording, role="train_candidate", start=0, stop=n),
        events=events,
        supervision_runs=[
            dict(start=i, stop=i + 1, activity=int(any(a <= i < z for a, z in spans)))
            for i in range(n)
        ],
    )
    y, mask, censor = supervision(interval, video, 0, Q(1, 12))
    a = dict(
        pose=np.zeros((n, 65), np.float32),
        visual=np.zeros((n, 97), np.float32),
        observable=np.ones(n, bool),
        reset=np.zeros(n, bool),
        targets=y,
        supervision_mask=mask,
    )
    a["visual"][:, -1] = 1
    a["reset"][0] = True
    return dict(
        arrays=a,
        video=video,
        references=[
            dict(event_id=e["event_id"], start_frame=e["start"], stop_frame_exclusive=e["stop"])
            for e in events
        ],
        receipt=dict(train_eligible=True, censoring=censor),
        provenance=dict(recording=recording, source_start=0),
    )


def data():
    return {"V4-case": case("V4"), "V6-case": case("V6", ((10, 16), (30, 48)))}


def test_each_source_head_sign_has_fixed_coefficient_mass_and_constant_loss():
    p = objective.prepare(data(), "fused", {})
    for r in p["audit"]["source_coefficients"].values():
        assert r["positive_coefficient_mass"] == pytest.approx([1 / 12] * 3)
        assert r["negative_coefficient_mass"] == pytest.approx([1 / 12] * 3)
    assert float(objective.loss(torch.zeros_like(p["y"]), p)) == pytest.approx(np.log(2))


def test_balanced_logit_gradients_equal_the_declared_sign_decomposition():
    p = objective.prepare(data(), "fused", {})
    z = torch.linspace(-4, 4, p["y"].numel()).reshape(p["y"].shape).requires_grad_(True)
    objective.loss(z, p).backward()
    expected = p["negative"] * z.detach().sigmoid() - p["positive"] * (1 - z.detach().sigmoid())
    assert torch.allclose(z.grad, expected, atol=1e-8)


def test_baseline_diagnostic_terms_reproduce_the_original_loss():
    from boxing_motion_ai.detection.localization_balanced_experiment import old_terms
    from boxing_motion_ai.detection.localization_model import loss as old_loss

    p = objective.prepare(data(), "fused", {})
    y, mask = p["y"], p["mask"]
    mass = (y * mask).sum((0, 1))
    w = ((mask.sum((0, 1)) - mass) / mass).clamp(max=40)
    z = torch.linspace(-4, 4, y.numel()).reshape(y.shape)
    assert float(old_terms(z, p).sum()) == pytest.approx(float(old_loss(z, y, mask, w)))


def test_boundary_positive_events_have_equal_weight_despite_event_length():
    p = objective.prepare(data(), "fused", {})
    # Same source, disjoint neighborhoods, different duration: both endpoints have equal mass.
    for h in (1, 2):
        pos = p["positive"][1, :, h]
        assert float(pos[:25].sum()) == pytest.approx(1 / 24)
        assert float(pos[25:].sum()) == pytest.approx(1 / 24)


def test_duplicate_source_sequence_does_not_change_total_source_contribution():
    d = data()
    p = objective.prepare(d, "fused", {})
    d["V4-duplicate"] = deepcopy(d["V4-case"])
    q = objective.prepare(d, "fused", {})
    assert p["audit"]["source_coefficients"]["V4"]["positive_coefficient_mass"] == pytest.approx(
        q["audit"]["source_coefficients"]["V4"]["positive_coefficient_mass"]
    )


def test_unknown_and_missing_observations_have_no_loss_or_gradient_and_no_mask_mutation():
    d = data()
    d["V4-case"]["arrays"]["supervision_mask"][22:30] = False
    d["V4-case"]["arrays"]["observable"][35:40] = False
    before = deepcopy(d)
    p = objective.prepare(d, "fused", {})
    inactive = p["positive"] + p["negative"] == 0
    z = torch.zeros_like(p["y"])
    z[inactive] = float("nan")
    z.requires_grad_(True)
    value = objective.loss(z, p)
    assert torch.isfinite(value)
    value.backward()
    assert (z.grad[inactive] == 0).all()
    for name in d:
        for k in d[name]["arrays"]:
            assert np.array_equal(d[name]["arrays"][k], before[name]["arrays"][k])


def test_censored_boundary_is_not_reintroduced_as_an_event_positive():
    d = data()
    d["V4-case"] = case("V4", ((0, 20), (30, 45)))
    p = objective.prepare(d, "fused", {})
    rows = p["audit"]["event_mass"]
    row = next(r for r in rows if r["event_id"] == "V4-0" and r["head"] == "onset")
    assert row["observed_positive_mass"] == 0 and not row["available"]


def test_sparse_conflict_mask_and_unavailable_ending_are_preserved():
    from boxing_motion_ai.data.blue_localization import targets

    d = data()
    c = d["V6-case"]
    interval = dict(
        events=[
            dict(event_id="partial", start=10, stop=20, reference_eligible=False),
            dict(event_id="full", start=30, stop=45, reference_eligible=True),
        ],
        excluded_events=[dict(start=19, stop=23)],
    )
    y, mask, censored = targets(interval, c["video"], 0, Q(1, 12))
    c["arrays"]["targets"], c["arrays"]["supervision_mask"] = y, mask
    c["receipt"]["censoring"] = censored
    c["provenance"]["exhaustive"] = False
    d["V6-background"] = case("V6", spans=())  # Existing verified negatives, not sparse gaps.
    p = objective.prepare(d, "fused", {"V6-case": interval})
    assert not (p["positive"][1, 19:23] + p["negative"][1, 19:23]).any()
    assert not (p["positive"][1, 24:30] + p["negative"][1, 24:30]).any()
    row = next(
        r for r in p["audit"]["event_mass"] if r["event_id"] == "partial" and r["head"] == "end"
    )
    assert row["observed_positive_mass"] == 0 and not row["available"]


def test_development_changes_cannot_affect_loss_inputs_or_weights():
    d = data()
    p = objective.prepare(d, "fused", {})
    d["V5-dev"] = case("V6")
    d["V5-dev"]["provenance"]["recording"] = "V5"
    d["V5-dev"]["receipt"]["train_eligible"] = False
    d["V5-dev"]["arrays"]["targets"][:] = float("nan")
    q = objective.prepare(d, "fused", {})
    for key in ("x", "y", "mask", "present", "positive", "negative"):
        assert torch.equal(p[key], q[key])
    d["V5-dev"]["receipt"]["train_eligible"] = True
    with pytest.raises(ValueError, match="role_leakage"):
        objective.prepare(d, "fused", {})


def test_unattributed_positive_target_fails_closed():
    d = data()
    d["V4-case"]["arrays"]["targets"][50, 1] = 1
    with pytest.raises(ValueError, match="unattributed"):
        objective.prepare(d, "fused", {})


def test_boundary_ties_share_target_mass_without_double_counting():
    c = case("V4", ((10, 20), (10, 20)))
    known = objective.events(c)
    own = objective.ownership(c, known, Q(1, 12))
    for h in (1, 2):
        assert np.array_equal(own[h][0], own[h][1])
        assert own[h].sum() == pytest.approx(c["arrays"]["targets"][:, h].sum())


def test_missing_source_sign_is_rejected_without_inventing_negatives():
    d = data()
    c = d["V6-case"]
    c["arrays"]["supervision_mask"][:] &= c["arrays"]["targets"][:, :1] > 0
    with pytest.raises(ValueError, match="negative_source_head_missing"):
        objective.prepare(d, "fused", {})


def test_development_cannot_load_features_before_frozen_artifacts_validate(tmp_path, monkeypatch):
    from boxing_motion_ai.detection import localization_balanced_experiment as experiment

    accessed = []
    monkeypatch.setattr(experiment, "cases", lambda phase: accessed.append(phase))
    training = tmp_path / "training"
    training.mkdir()
    (training / "freeze.json").write_text('{"code_sha256":{}}')
    with pytest.raises(ValueError, match="changed_after_freeze"):
        experiment.development(training, tmp_path / "output")
    assert not accessed


def test_permitted_real_training_fixture_coefficients_and_masks():
    cache = Path("reports/boxingvi/blue-localization-features-v1-redacted/manifest.json")
    if not cache.exists():
        pytest.skip("private permitted training fixtures unavailable")
    from boxing_motion_ai.detection.localization_balanced_experiment import training_data

    d, intervals = training_data()
    assert not any(n.startswith("V5") for n in d)
    p = objective.prepare(d, "fused", intervals)
    assert p["audit"]["supervised_frames"] == [1688, 1630, 1630]
    assert p["audit"]["source_coefficients"]["V6"]["eligible_boundary_events"] == [13, 11]
    for r in p["audit"]["source_coefficients"].values():
        assert r["positive_coefficient_mass"] == pytest.approx([1 / 12] * 3)
        assert r["negative_coefficient_mass"] == pytest.approx([1 / 12] * 3)
