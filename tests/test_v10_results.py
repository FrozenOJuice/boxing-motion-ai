"""Known synthetic seed changes test paired analysis, not model performance."""

from copy import deepcopy

import pytest


def test_paired_results_and_mean_confusion_do_not_inflate_support():
    pytest.importorskip("torch")
    import numpy as np

    from boxing_motion_ai.training.data import CLASSES
    from boxing_motion_ai.training.v10_results import compare_runs

    metric = {
        "accuracy": 0.5,
        "macro_f1": 0.5,
        "per_class": {c: {"precision": 0.6, "recall": 0.5, "f1": 0.55} for c in CLASSES},
        "confusion_matrix": np.eye(6, dtype=int).tolist(),
    }
    record = {
        "selected_epoch": 2,
        "development": metric,
        "train": {"macro_f1": 0.8},
        "last_epoch": {"train": {"macro_f1": 0.9}, "development": {"macro_f1": 0.4}},
    }
    runs = {}
    for seed, gain in enumerate([0.1, -0.2, 0.3], 1):
        runs[str(seed)] = {
            arm: {m: deepcopy(record) for m in ["tcn", "tcn_mask_only"]}
            for arm in ["v4_only", "expanded"]
        }
        runs[str(seed)]["expanded"]["tcn"]["development"]["macro_f1"] += gain
        runs[str(seed)]["expanded"]["tcn_mask_only"]["development"]["macro_f1"] += 0.05
    before = deepcopy(runs)
    r = compare_runs(runs, [1, 2, 3])
    full = r["by_model"]["tcn"]
    assert full["paired_macro_f1_change"]["values"] == pytest.approx([0.1, -0.2, 0.3])
    assert full["paired_macro_f1_change"]["sample_std"] == pytest.approx(
        np.std([0.1, -0.2, 0.3], ddof=1)
    )
    assert full["positive_macro_f1_seeds"] == 2
    assert r["full_gain_minus_mask_gain"]["values"] == pytest.approx([0.05, -0.25, 0.25])
    assert full["expanded"]["mean_confusion_matrix"] == np.eye(6).tolist()
    assert full["expanded"]["final_train_dev_gap"]["mean"] == pytest.approx(0.5)
    assert full["expanded"]["per_class"]["rear_uppercut"]["precision"]["mean"] == 0.6
    assert runs == before
