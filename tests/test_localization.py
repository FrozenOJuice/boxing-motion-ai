"""Native motion/supervision and offline instance contracts; no training media required."""

from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from boxing_motion_ai.detection.localization_data import (
    TRAIN,
    authorized,
    motion_pair,
    redacted,
    source_resets,
    supervision,
)
from boxing_motion_ai.detection.localization_decode import decode
from boxing_motion_ai.detection.localization_model import Localizer, fit, loss, model_input, predict
from boxing_motion_ai.vision.observation_study import read

CONFIG = read(Path(__file__).parents[1] / "configs/punch_localization_smoke_v1.json")


def video(n=50):
    times = [Q(i, 25) for i in range(n)]
    return dict(
        width=160,
        height=160,
        frame_count=n,
        duration_seconds=float(Q(n, 25)),
        stop_timestamp_exact=str(Q(n, 25)),
        frames=[
            dict(
                frame_index=i,
                pts=i,
                time_base_num=1,
                time_base_den=25,
                timestamp_exact=str(t),
                media_time=float(t),
            )
            for i, t in enumerate(times)
        ],
    )


def interval():
    return dict(
        exhaustive=True,
        provenance=dict(start=5, stop=45, recording="V4", role="train_candidate"),
        supervision_runs=[
            dict(start=5, stop=10, activity=0),
            dict(start=10, stop=20, activity=1),
            dict(start=20, stop=45, activity=0),
        ],
        events=[dict(start=10, stop=20, event_id="a")],
    )


def test_boundary_targets_use_reviewed_onset_and_last_included_frame_not_impact():
    y, mask, c = supervision(interval(), video(), 0, Q(1, 12))
    assert y[10, 1] == y[19, 2] == 1
    assert np.all(y[10:20, 0] == 1) and y[20, 0] == 0
    assert not mask[:5].any() and not mask[45:].any()
    assert not mask[5, 1] and not mask[44, 2]
    assert c == [dict(event_id="a", left_censored=False, right_censored=False)]
    u, m, _ = supervision(None, video(), 0, Q(1, 12))
    assert not m.any() and not u.any()


def test_censored_events_and_unknown_context_never_manufacture_edge_targets():
    i = interval()
    i["events"] = [dict(start=5, stop=48, event_id="partial")]
    i["supervision_runs"] = [dict(start=5, stop=45, activity=1)]
    y, m, c = supervision(i, video(), 0, Q(1, 12))
    assert c[0]["left_censored"] and c[0]["right_censored"]
    assert not y[:, 1:].any() and not m[45:].any()
    assert y[44, 0] == 1
    i["exhaustive"] = False
    with pytest.raises(ValueError):
        supervision(i, video(), 0, Q(1, 12))


def test_boundary_overlap_targets_are_not_given_fake_arm_or_class_assignments():
    i = interval()
    i["events"].append(dict(start=16, stop=25, event_id="overlap"))
    i["supervision_runs"] = [
        dict(start=5, stop=10, activity=0),
        dict(start=10, stop=25, activity=1),
        dict(start=25, stop=45, activity=0),
    ]
    y, _, _ = supervision(i, video(), 0, Q(1, 12))
    assert y[10, 1] == y[16, 1] == y[19, 2] == y[24, 2] == 1
    assert y.shape[1] == 3


def test_supervision_conflict_incomplete_and_role_guards():
    i = interval()
    i["supervision_runs"].append(dict(start=12, stop=14, activity=0))
    with pytest.raises(ValueError, match="Conflicting"):
        supervision(i, video(), 0, Q(1, 12))
    i = interval()
    i["supervision_runs"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        supervision(i, video(), 0, Q(1, 12))
    i = interval()
    i["provenance"]["recording"] = "V7"
    with pytest.raises(ValueError):
        supervision(i, video(), 0, Q(1, 12))
    with pytest.raises(ValueError):
        authorized(dict(intervals=[i]))
    assert len(TRAIN) == 15 and "V4-bg-4" not in TRAIN and "V4-bg-5" not in TRAIN


def test_exact_clock_and_reset_reject_pose_mismatch():
    v = video()
    raw = {
        k: np.array([r[k] for r in v["frames"]]) for k in ["pts", "time_base_num", "time_base_den"]
    }
    raw.update(scene_id=np.zeros(50, int), segment_id=np.zeros(50, int))
    v["confirmed_scene_cuts"] = [12]
    raw["segment_id"][30:] = 1
    _, reset = source_resets(raw, v)
    assert np.flatnonzero(reset).tolist() == [0, 12, 30]
    raw["pts"][0] = 1
    with pytest.raises(ValueError, match="mismatch"):
        source_resets(raw, v)


def test_static_frames_do_not_expose_colors_texture_or_source_identifiers():
    mask = np.ones((160, 160), bool)
    a = np.zeros((160, 160, 3), np.uint8)
    a[20:40, 20:40] = 255
    b = np.full_like(a, 100)
    x = motion_pair(a, a, mask, mask, Q(1, 25))
    y = motion_pair(b, b, mask, mask, Q(1, 25))
    np.testing.assert_allclose(x, y, atol=1e-7)
    assert not x[:-1].any() and x[-1] == 1
    moving = np.roll(a, 8, axis=1)
    assert np.linalg.norm(motion_pair(a, moving, mask, mask, Q(1, 25))[:-1]) > 0.01
    assert not motion_pair(a, moving, np.zeros_like(mask), mask, Q(1, 25)).any()


def test_hud_and_face_torso_changes_cannot_enter_motion_features():
    a = Image.new("RGB", (400, 400), "gray")
    b = a.copy()
    d = ImageDraw.Draw(b)
    d.rectangle((0, 0, 79, 139), fill="white")  # V4 fixed HUD rectangle.
    d.rectangle((160, 60, 240, 320), fill="black")  # Fixed identity/torso mask.
    x, m = redacted(a, (0, 0, 400, 400), (160, 60, 240, 320), "V4")
    y, n = redacted(b, (0, 0, 400, 400), (160, 60, 240, 320), "V4")
    np.testing.assert_array_equal(x, y)
    assert not motion_pair(x, y, m, n, Q(1, 25))[:-1].any()
    with pytest.raises(ValueError):
        redacted(a, (0, 0, 400, 400), (0, 0, 1, 1), "V9")


def fake_case():
    rng = np.random.default_rng(12)
    n = 50
    y, m, _ = supervision(interval(), video(), 0, Q(1, 12))
    result = dict(
        arrays=dict(
            pose=rng.normal(size=(n, 65)).astype(np.float32),
            visual=rng.normal(size=(n, 97)).astype(np.float32),
            targets=y,
            supervision_mask=m,
            observable=np.ones(n, bool),
            reset=np.array([True] + [False] * (n - 1)),
        ),
        receipt=dict(train_eligible=True),
    )

    result["arrays"]["visual"][:, -1] = 1
    result["arrays"]["visual"][0, -1] = 0
    return result


def test_inputs_ignore_all_targets_references_ids_and_timestamps():
    a = fake_case()
    before = model_input(a["arrays"], "fused")
    b = deepcopy(a)
    b["arrays"]["targets"][:] = 999
    b["references"] = [dict(type="invented")]
    b["arrays"]["pts"] = np.arange(50) * 100
    assert torch.equal(before, model_input(b["arrays"], "fused"))
    assert not model_input(a["arrays"], "pose")[:, 65:].any()
    assert not model_input(a["arrays"], "rgb_motion")[:, :65].any()
    assert before.shape == (50, 162)


def test_padding_does_not_change_real_predictions_and_context_is_bounded():
    torch.manual_seed(1)
    m = Localizer().eval()
    x = torch.randn(1, 30, 162)
    pad = torch.cat((x, torch.randn(1, 20, 162)), 1)
    present = torch.arange(50)[None] < 30
    with torch.no_grad():
        assert torch.allclose(m(x), m(pad, present)[:, :30], atol=1e-6)
        altered = x.clone()
        altered[:, 24:] += 100
        assert torch.equal(m(x)[:, :17], m(altered)[:, :17])  # Seven future native samples only.


def test_unknown_labels_and_padding_have_no_loss():
    logits = torch.randn(1, 50, 3)
    y = torch.zeros_like(logits)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[:, 10:30] = True
    initial = loss(logits, y, mask, torch.ones(3))
    y[~mask] = float("nan")
    assert torch.equal(initial, loss(logits, y, mask, torch.ones(3)))


def test_training_repeatability_missing_observations_and_reset_isolation():
    c = fake_case()
    c["arrays"]["observable"][25:28] = False
    c["arrays"]["reset"][35] = True
    config = {**CONFIG, "epochs": 3}
    cases = {"synthetic": c}
    a, da = fit(cases, "fused", config)
    b, db = fit(cases, "fused", config)
    assert da == db and all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    p = predict(a, c, "fused")
    assert np.isnan(p[25:28]).all()
    altered = deepcopy(c)
    altered["arrays"]["pose"][35:] += 100
    np.testing.assert_array_equal(p[:35], predict(a, altered, "fused")[:35])


def score_sequence(starts=(10, 26), ends=(21, 37)):
    p = np.zeros((50, 3), np.float32)
    p[8:40, 0] = 0.9
    for i in starts:
        p[i, 1] = 0.9
    for i in ends:
        p[i, 2] = 0.9
    return p


def test_decoder_separates_consecutive_punches_without_activity_valley():
    p = score_sequence()
    o = np.ones(50, bool)
    r = np.zeros(50, bool)
    r[0] = True
    out = decode(p, o, r, video(), CONFIG["decoder"])
    assert [(a["start_frame"], a["stop_frame_exclusive"]) for a in out["candidates"]] == [
        (10, 22),
        (26, 38),
    ]
    assert out == decode(p, o, r, video(), CONFIG["decoder"])
    assert out["candidates"][0]["start_timestamp_exact"] == "2/5"
    assert not any("class" in k for k in out["candidates"][0])


def test_decoder_preserves_overlap_censoring_and_unobservable_motion():
    p = score_sequence(starts=(10, 16), ends=(30,))
    o = np.ones(50, bool)
    r = np.zeros(50, bool)
    r[0] = True
    a = decode(p, o, r, video(), CONFIG["decoder"])
    assert not a["candidates"]
    assert any(x.get("overlap_ambiguous") for x in a["unresolved"])
    p = score_sequence(starts=(10,), ends=(37,))
    o[24:28] = False
    p[24:28] = np.nan
    a = decode(p, o, r, video(), CONFIG["decoder"])
    assert not a["candidates"] and a["unobservable"]
    assert any(x.get("right_censored") for x in a["unresolved"])
    o[:] = True
    p = score_sequence(starts=(0,), ends=(49,))
    a = decode(p, o, r, video(), CONFIG["decoder"])
    assert not a["candidates"]


def test_decoder_no_punch_plateaus_gaps_and_duplicate_peaks():
    o = np.ones(50, bool)
    r = np.zeros(50, bool)
    r[0] = True
    assert not decode(np.zeros((50, 3)), o, r, video(), CONFIG["decoder"])["candidates"]
    p = score_sequence()
    p[11, 1] = 0.9
    assert len(decode(p, o, r, video(), CONFIG["decoder"])["candidates"]) == 2
    r[18] = True
    a = decode(p, o, r, video(), CONFIG["decoder"])
    assert len(a["candidates"]) == 1 and a["interruptions"]


def test_full_inventory_guards_group_roles_and_frozen_experiment_budget():
    from pydantic import ValidationError

    from boxing_motion_ai.data.interval_pilot import GROUPS
    from boxing_motion_ai.detection.localization_smoke import Experiment

    entries = []
    for j, name in enumerate(sorted(TRAIN)):
        item = interval()
        item["id"] = name
        v = name[:2]
        item["provenance"].update(recording=v, source_group_guard=GROUPS[v])
        item["events"] = [dict(event_id=str(k)) for k in range(11)] if j == 0 else []
        entries.append(item)
    assert len(authorized(dict(intervals=entries))) == 15
    wrong = deepcopy(entries)
    wrong[0]["provenance"]["role"] = "evaluation_reserved"
    with pytest.raises(ValueError, match="restriction"):
        authorized(dict(intervals=wrong))
    wrong = deepcopy(entries)
    wrong[-1]["provenance"]["source_group_guard"] = ["V6"]
    with pytest.raises(ValueError, match="restriction"):
        authorized(dict(intervals=wrong))
    assert Experiment.model_validate(CONFIG).epochs == 120
    for changes in [
        {"epochs": 121},
        {"seed": 1},
        {"decoder": {**CONFIG["decoder"], "boundary_threshold": 0.4}},
        {"class_head": 6},
    ]:
        with pytest.raises(ValidationError):
            Experiment.model_validate({**CONFIG, **changes})


def test_missing_wrist_stays_masked_without_changing_activity_supervision():
    from test_automatic_punches import sequence

    from boxing_motion_ai.detection.data import encode_source

    raw, v, _ = sequence()
    raw["confidence"][10:20, 15] = 0
    raw["valid_mask"][10:20, 15] = False
    x, obs, _, _ = encode_source(raw, v["width"] / v["height"])
    assert obs[10:20].all() and not x[10:20, 20:25].any()
    assert not x[10:20, 53:56].any()  # Wrist velocity and derivative validity remain absent.
    y, m, _ = supervision(interval(), video(100), 0, Q(1, 12))
    assert y[10:20, 0].all() and m[10:20, 0].all()  # Annotation isn't rewritten as background.


def test_missing_native_motion_pair_is_abstention_for_every_input_mode():
    from boxing_motion_ai.detection.localization_model import model_observable, sequences

    c = fake_case()
    c["arrays"]["visual"][16, -1] = 0
    assert not model_observable(c["arrays"])[[0, 16]].any()
    for mode in ("pose", "rgb_motion", "fused"):
        s = sequences({"case": c}, mode, training=True)
        assert [(r["start"], r["stop"]) for r in s] == [(1, 16), (17, 50)]


def test_decoder_collapses_one_plateau_instead_of_periodic_fake_boundary_peaks():
    p = score_sequence(starts=(), ends=())
    p[10:16, 1] = 0.9
    p[25:31, 2] = 0.9
    out = decode(p, np.ones(50, bool), np.zeros(50, bool), video(), CONFIG["decoder"])
    assert out["boundary_peaks"][0]["onsets"] == [10]
    assert out["boundary_peaks"][0]["ends"] == [25]
    assert len(out["candidates"]) == 1
