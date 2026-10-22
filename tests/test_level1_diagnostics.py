"""Frozen diagnostic controls, not new fitting or corrective preprocessing."""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from boxing_motion_ai.training.diagnostics import (  # noqa: E402
    decode_exact_frames,
    length_control,
    perturb,
    predictions,
    runtime_checks,
)
from boxing_motion_ai.training.models import CausalTCN, StaticBaseline  # noqa: E402


def test_perturbations_are_copies_and_preserve_joint_semantics_explicitly():
    x = torch.arange(3 * 41, dtype=torch.float32).reshape(3, 41)
    original = x.clone()
    reflected = perturb(x, "reflect_x", False)
    torch.testing.assert_close(reflected[:, :40:5], -x[:, :40:5])
    torch.testing.assert_close(reflected[:, 1:40:5], x[:, 1:40:5])
    assert torch.equal(perturb(x, "reverse", False), x.flip(0))
    swapped = perturb(x, "swap_sides", False)
    torch.testing.assert_close(swapped[:, :5], x[:, 5:10])
    torch.testing.assert_close(swapped[:, -1], x[:, -1])
    torch.testing.assert_close(x, original)
    mask = torch.zeros(3, 8)
    assert perturb(mask, "constant_validity", True).eq(1).all()
    with pytest.raises(ValueError):
        perturb(x, "constant_validity", False)
    with pytest.raises(ValueError):
        perturb(x, "invented", False)


def test_identical_tokens_expose_causal_boundary_length_but_not_static_length():
    torch.manual_seed(13)
    torch.set_num_threads(1)
    static = length_control(StaticBaseline(8).eval(), 8)
    temporal = length_control(CausalTCN(8, hidden=8).eval(), 8)
    assert static["max_absolute_logit_difference"] < 1e-6
    assert temporal["max_absolute_logit_difference"] > 1e-4
    assert len(temporal["logits"]) == 30


def test_runtime_padding_and_causal_checks_use_existing_model():
    torch.manual_seed(4)
    model = CausalTCN(41, hidden=8).eval()
    dataset = [{"x": torch.randn(7, 41)}]
    result = runtime_checks(model, dataset)
    assert result["right_padding_logit_max_error"] < 1e-6
    assert result["future_perturbation_prefix_max_error"] == 0


def test_frozen_prediction_controls_do_not_mutate_examples():
    from types import SimpleNamespace

    clips = [
        dict(x=torch.randn(n, 41), mask_x=torch.ones(n, 8), label=i, source_row=i + 2)
        for i, n in enumerate((4, 9))
    ]
    copies = [c["x"].clone() for c in clips]
    ds = SimpleNamespace(clips=clips)
    model = StaticBaseline(41).eval()
    a = predictions(model, ds)
    b = predictions(model, ds, kind="shuffle", batch_size=1)
    np.testing.assert_allclose(a, b, atol=1e-6)
    for c, saved in zip(clips, copies, strict=True):
        torch.testing.assert_close(c["x"], saved)


def test_visual_seek_requires_exact_pts_and_resumes_forward_decode(monkeypatch):
    from fractions import Fraction
    from types import SimpleNamespace

    class Frame:
        time_base = Fraction(1, 1000)
        is_corrupt = False

        def __init__(self, pts):
            self.pts = pts

        def reformat(self, **kwargs):
            return self

        def to_ndarray(self, **kwargs):
            return np.zeros((216, 384, 3), dtype=np.uint8)

    class Container:
        streams = SimpleNamespace(
            video=[
                SimpleNamespace(
                    time_base=Fraction(1, 1000), codec_context=SimpleNamespace(thread_count=0)
                )
            ]
        )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def seek(self, *args, **kwargs):
            pass

        def decode(self, stream):
            return iter([Frame(0), Frame(33), Frame(67), Frame(100)])

    av = pytest.importorskip("av")

    monkeypatch.setattr(av, "open", lambda _: Container())
    result = decode_exact_frames("synthetic.mkv", {Fraction(33, 1000): 1, Fraction(67, 1000): 2})
    assert set(result) == {1, 2}
    with pytest.raises(ValueError, match="Exact source PTS not found"):
        decode_exact_frames("synthetic.mkv", {Fraction(34, 1000): 1})
