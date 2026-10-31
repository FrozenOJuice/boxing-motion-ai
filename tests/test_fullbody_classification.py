"""Matched initialization, clip isolation and fail-closed RGB gate primitives."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image
from test_video_foundation import raw

from boxing_motion_ai.training.experiment import seed_all
from boxing_motion_ai.training.fullbody_classification import initialize, state_hash
from boxing_motion_ai.training.fullbody_classification_data import (
    DERIVATIVES,
    MAP65,
    MASKS,
    View,
    clip_body,
    reflected_raw,
    source_spec,
)
from boxing_motion_ai.vision.person_rgb_study import preserve


def test_semantic_mapping_preserves_shared_initial_function_and_rng():
    cfg = dict(hidden=32, dropout=0.1)
    seed_all(42, 2)
    core = initialize("core65", cfg)
    state = torch.get_rng_state()
    seed_all(42, 2)
    body = initialize("body209", cfg)
    assert torch.equal(state, torch.get_rng_state())
    x = torch.randn(3, 12, 209)
    mask = torch.ones(3, 12, dtype=torch.bool)
    core.eval()
    body.eval()
    assert torch.allclose(core(x[:, :, MAP65], mask), body(x, mask), atol=1e-6)
    assert state_hash(core) == state_hash(initialize_again(cfg))


def initialize_again(cfg):
    seed_all(42, 2)
    return initialize("core65", cfg)


def test_physical_reflection_is_involution_and_never_mutates_raw():
    r = raw(8)
    r["valid_mask"][3, 15] = False
    original = deepcopy(r)
    twice = reflected_raw(reflected_raw(r))
    for k in r:
        assert np.array_equal(r[k], original[k], equal_nan=True)
        assert np.allclose(r[k], twice[k], equal_nan=True, atol=1e-7)


def test_clip_derivatives_reset_and_mask_ablation_has_no_continuous_evidence():
    r = raw(8)
    x, ref, normal = clip_body({"raw": r}, 1, [])
    assert not x[0, DERIVATIVES].any()
    c = dict(x=x[:, MAP65], core65_reflected=x[:, MAP65], body209=x, body209_reflected=ref)
    masked = View([c], "mask209")[0]["x"]
    keep = torch.zeros(209, dtype=torch.bool)
    keep[MASKS] = True
    assert not masked[:, ~keep].any() and torch.equal(masked[:, keep], x[:, keep])
    assert normal.all() and x[1, DERIVATIVES].any()


@pytest.mark.parametrize("recording", ["V1", "V7", "V8", "V9", "V10"])
def test_forbidden_recordings_rejected_before_file_access(recording):
    with pytest.raises(Exception, match="classifier_forbidden_recording"):
        source_spec(recording)


def test_person_matte_does_not_inpaint_and_ocr_removal_overrides_person():
    image = Image.fromarray(np.full((40, 50, 3), 240, np.uint8))
    probability = np.zeros((40, 50), np.float32)
    probability[10:30, 10:30] = 1
    out, retained = preserve(image, probability, [(16, 16, 20, 20, "LOGO")], dilation=0)
    arr = np.asarray(out)
    assert not retained[18, 18] and (arr[18, 18] == 128).all()
    assert retained[12, 12] and (arr[12, 12] == 240).all()
    assert not retained[0, 0] and (arr[0, 0] == 128).all()
    assert (np.asarray(image) == 240).all()
    with pytest.raises(Exception, match="rgb_matte_shape"):
        preserve(image, np.ones((5, 5)), [])


def test_small_fit_is_exactly_repeatable_and_modes_share_sampling():
    from boxing_motion_ai.training.fullbody_classification import fit
    from boxing_motion_ai.training.fullbody_classification_data import clip_identity

    clips = []
    for label in range(6):
        r = raw(8)
        r["coordinates"][:, 15, 0] += 0.002 * label
        x, ref, normal = clip_body({"raw": r}, 1, [])
        clips.append(
            dict(
                raw=r,
                x=x[:, MAP65],
                mask_x=torch.ones(8, 16),
                label=label,
                recording_id="V4",
                source_row=label,
                body209=x,
                body209_reflected=ref,
                core65_reflected=ref[:, MAP65],
                body_normalization=normal,
            )
        )
    cfg = dict(
        hidden=32,
        dropout=0.1,
        threads=2,
        learning_rate=0.001,
        weight_decay=0.001,
        epochs=2,
        batch_size=3,
        mirror_probability=0.5,
        gradient_clip=1.0,
    )
    a, report, rows = fit(clips, "core65", 42, cfg)
    b, repeat, repeated = fit(clips, "core65", 42, cfg)
    _, body, _ = fit(clips, "body209", 42, cfg)
    assert report == repeat and rows == repeated and state_hash(a) == state_hash(b)
    assert report["sampling_mirror_sha256"] == body["sampling_mirror_sha256"]
    assert clip_identity(clips[0])["native_frames"] == [0, 8]
    assert clip_identity(clips[0])["first_pts_exact"] == "0"
    assert all("softmax_scores" in r and "probabilities" not in r for r in rows)


def test_development_rejects_tampered_freeze_before_loading_v5(tmp_path, monkeypatch):
    import json

    import boxing_motion_ai.training.fullbody_classification as experiment

    evidence = tmp_path / "checkpoint"
    evidence.write_text("changed")
    frozen = dict(code_sha256={}, input_sha256={}, training_files_sha256={str(evidence): "wrong"})
    (tmp_path / "training-freeze.json").write_text(json.dumps(frozen))

    def forbidden():
        raise AssertionError("V5 must not be read after failed integrity check")

    monkeypatch.setattr(experiment, "development_views", forbidden)
    with pytest.raises(Exception, match="classifier_frozen_artifact_changed"):
        experiment.development(tmp_path)
