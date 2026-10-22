"""Exact-time boundaries and punch-only pooling, with unchanged TCN architecture."""

from fractions import Fraction

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from boxing_motion_ai.training.context import (  # noqa: E402
    context_collate,
    context_start,
    exact_timeline,
    run_starts,
)
from boxing_motion_ai.training.models import CausalTCN  # noqa: E402


def inventory(n=40, fps=30):
    return {
        "frame_index": np.arange(n),
        "pts": np.arange(n),
        "time_base_num": np.ones(n, dtype=np.int64),
        "time_base_den": np.full(n, fps),
        "scene_id": np.zeros(n, dtype=np.int64),
        "segment_id": np.zeros(n, dtype=np.int64),
        "pose_status": np.zeros(n, dtype=np.uint8),
    }


@pytest.mark.parametrize("fps,count", [(24, 15), (30, 19)])
def test_time_context_accounts_for_rate_without_rounding_fps_into_indices(fps, count):
    a = inventory(fps=fps)
    times = exact_timeline(a)
    starts, _ = run_starts(a, times)
    lo = context_start(times, 30, starts[30], 0.65)
    assert 30 - lo == count
    assert times[lo] >= times[30] - Fraction(13, 20)
    assert times[lo - 1] < times[30] - Fraction(13, 20)


def test_exact_lower_bound_is_inclusive_and_context_stops_before_punch():
    a = inventory(n=10, fps=20)
    t = exact_timeline(a)
    assert context_start(t, 8, 0, 0.25) == 3
    assert t[3] == t[8] - Fraction(1, 4)
    assert context_start(t, 8, 0, 0) == 8


@pytest.mark.parametrize(
    "boundary", ["cut", "scene", "segment", "index_gap", "time_gap", "time_base"]
)
def test_context_stops_at_every_known_boundary(boundary):
    a = inventory(n=10)
    cuts = []
    if boundary == "cut":
        cuts = [5]
    elif boundary == "scene":
        a["scene_id"][5:] = 1
    elif boundary == "segment":
        a["segment_id"][5:] = 1
    elif boundary == "index_gap":
        a["frame_index"][5:] += 1
    elif boundary == "time_gap":
        a["pts"][5:] += 30
    elif boundary == "time_base":
        a["time_base_den"][5:] = 15
    t = exact_timeline(a)
    starts, reasons = run_starts(a, t, cuts)
    assert starts[4] == 0 and starts[5] == 5
    assert context_start(t, 7, starts[7], 0.65) >= 5
    assert reasons[5]


def test_missing_pose_detection_is_not_missing_extraction():
    a = inventory(n=10)
    a["pose_status"][5] = 2
    starts, reasons = run_starts(a, exact_timeline(a))
    assert not starts.any()
    assert list(reasons) == [0]


def test_context_is_encoded_but_only_punch_frames_are_pooled_and_padding_is_ignored():
    torch.manual_seed(5)
    model = CausalTCN(8, hidden=8, dropout=0).eval()
    x = torch.randn(1, 23, 8)
    mask = torch.arange(23)[None] >= 15
    with torch.no_grad():
        expected = model.head(model.encode(x)[:, 15:].mean(1))
        torch.testing.assert_close(model(x, mask), expected)
        extended = torch.cat((x, torch.full((1, 10, 8), 100.0)), 1)
        extended_mask = torch.cat((mask, torch.zeros(1, 10, dtype=torch.bool)), 1)
        torch.testing.assert_close(model(x, mask), model(extended, extended_mask))
        changed = x.clone()
        changed[:, 14] += 10
        assert not torch.allclose(model(x, mask), model(changed, mask))
        changed = x.clone()
        changed[:, 15:] += 10
        torch.testing.assert_close(model.encode(x)[:, :15], model.encode(changed)[:, :15])


def test_full_history_removes_constant_token_length_boundary_signal():
    torch.manual_seed(11)
    model = CausalTCN(8, hidden=8, dropout=0).eval()
    warmed, cold = [], []
    with torch.no_grad():
        for n in range(3, 26):
            warmed.append(model(torch.ones(1, 15 + n, 8), torch.arange(15 + n)[None] >= 15))
            cold.append(model(torch.ones(1, n, 8), torch.ones(1, n, dtype=torch.bool)))
    assert max((x - warmed[0]).abs().max() for x in warmed).item() < 1e-6
    assert max((x - cold[0]).abs().max() for x in cold).item() > 1e-4


def test_context_collation_preserves_pooling_offsets_and_never_pools_right_padding():
    clips = [
        dict(
            x=torch.ones(n, 41),
            mask_x=torch.ones(n, 8),
            label=i,
            source_row=i,
            pool_mask=torch.arange(n) >= offset,
        )
        for i, (n, offset) in enumerate(((8, 5), (12, 7)))
    ]
    b = context_collate(clips)
    assert b["lengths"].tolist() == [8, 12]
    assert b["pool_mask"].sum(1).tolist() == [3, 5]
    assert not (b["pool_mask"] & ~b["time_mask"]).any()
