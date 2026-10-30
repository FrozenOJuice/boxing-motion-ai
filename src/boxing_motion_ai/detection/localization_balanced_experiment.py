"""One fixed-budget balanced-loss study; frozen training artifacts gate V5 inference."""

import argparse
import platform
import random
from datetime import UTC, datetime
from fractions import Fraction as Q
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from ..analysis.common import sha, write_json
from ..data.blue_localization import PIN, targets, validate
from ..services.dashboard import require
from ..vision.observation_study import fresh, read
from . import localization_balanced_loss as balanced
from .localization_blue_experiment import config as baseline_config
from .localization_blue_experiment import record, summarize
from .localization_decode import decode as decode_v1
from .localization_decode_v2 import decode as decode_v2
from .localization_decoder_study import CACHE, RUN, cases, diagnose
from .localization_model import MODES, Localizer, model_observable, predict
from .localization_smoke import state_hash

CONFIG = Path("configs/punch_localization_balanced_v1.json")
FILES = [
    Path(__file__),
    Path(balanced.__file__),
    CONFIG,
    Path("src/boxing_motion_ai/detection/localization_model.py"),
    Path("src/boxing_motion_ai/detection/localization_decode.py"),
    Path("src/boxing_motion_ai/detection/localization_decode_v2.py"),
    Path("src/boxing_motion_ai/detection/localization_decoder_study.py"),
    Path("src/boxing_motion_ai/detection/localization_blue_experiment.py"),
    Path("tests/test_localization_balanced.py"),
    Path("configs/localization_decoder_v2.json"),
    Path("configs/punch_localization_blue_v1.json"),
]


def binding():
    return {str(p): sha(p) for p in FILES}


def protocol():
    cfg = read(CONFIG)
    require(
        cfg["version"] == "source-event-balanced-localization-1.0"
        and cfg["sources"] == list(balanced.SOURCES)
        and Path(cfg["feature_cache"]) == CACHE
        and Path(cfg["baseline_run"]) == RUN
        and cfg["baseline_config"] == "configs/punch_localization_blue_v1.json",
        "balanced_protocol_changed",
    )
    return dict(loss=cfg, baseline=baseline_config(), code_sha256=binding())


def training_data():
    dataset = validate(Path.cwd())
    data = cases("training")  # Does not open V5 tensors, features or predictions.
    intervals = {
        i["id"]: i for i in dataset["intervals"] if i["provenance"]["recording"] in balanced.SOURCES
    }
    for name, interval in intervals.items():
        case = data[name]
        y, mask, _ = targets(interval, case["video"], case["provenance"]["source_start"], Q(1, 12))
        require(
            np.array_equal(y, case["arrays"]["targets"])
            and np.array_equal(mask, case["arrays"]["supervision_mask"]),
            "balanced_cached_labels_changed",
        )
    return data, intervals


def old_terms(logits, p):
    y, mask = p["y"], p["mask"]
    y = torch.where(mask, y, 0)
    positive = (y * mask).sum((0, 1))
    weights = ((mask.sum((0, 1)) - positive) / positive).clamp(max=40)
    denominator = (((1 - y) + weights * y) * mask).sum((0, 1))
    return (
        F.binary_cross_entropy_with_logits(logits, y, pos_weight=weights, reduction="none")
        * mask
        / denominator
        / 3
    )


def gradients(model, p, alternative):
    """Deterministic eval-mode parameter gradients, not dropout-training gradient estimates."""
    model.eval()
    z = model(p["x"], p["present"])
    values = balanced.terms(z, p) if alternative else old_terms(z, p)
    vectors, rows = {}, {}
    for s in balanced.SOURCES:
        sel = torch.from_numpy(p["sources"] == s)
        for h, head in enumerate(balanced.HEADS):
            v = values[sel, :, h].sum()
            g = torch.autograd.grad(v, model.parameters(), retain_graph=True)
            g = torch.cat([t.flatten() for t in g])
            vectors[s, h] = g
            rows[f"{s}-{head}"] = dict(loss=float(v.detach()), gradient_norm=float(g.norm()))
    heads = [vectors["V4", h] + vectors["V6", h] for h in range(3)]
    prob = z.detach().sigmoid()
    if alternative:
        pos, neg = p["positive"], p["negative"]
    else:
        y, mask = p["y"], p["mask"]
        positive = (y * mask).sum((0, 1))
        w = ((mask.sum((0, 1)) - positive) / positive).clamp(max=40)
        den = (((1 - y) + w * y) * mask).sum((0, 1))
        pos, neg = w * y * mask / den / 3, (1 - y) * mask / den / 3
    return dict(
        source_head=rows,
        loss=float(values.sum().detach()),
        head_gradient_norms=[float(g.norm()) for g in heads],
        source_cosines=[
            float(F.cosine_similarity(vectors["V4", h], vectors["V6", h], dim=0)) for h in range(3)
        ],
        head_cosines={
            f"{a}-{b}": float(F.cosine_similarity(heads[a], heads[b], dim=0))
            for a, b in ((0, 1), (0, 2), (1, 2))
        },
        positive_logit_gradient_sum=(pos * (1 - prob)).sum((0, 1)).tolist(),
        negative_logit_gradient_sum=(neg * prob).sum((0, 1)).tolist(),
    )


def fit(data, mode, cfg, intervals):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    p = balanced.prepare(data, mode, intervals, Q(cfg["boundary_radius_seconds"]))
    model = Localizer(cfg["hidden"], cfg["dropout"])
    initial = {
        name: gradients(model, p, alternative)
        for name, alternative in (("baseline_objective", False), ("balanced_objective", True))
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    curves = []
    for epoch in range(cfg["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        value = balanced.loss(model(p["x"], p["present"]), p)
        require(bool(torch.isfinite(value)), "nonfinite_balanced_loss")
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        model.eval()
        with torch.no_grad():
            z = model(p["x"], p["present"])
            t = balanced.terms(z, p)
            by_source = {
                s: t[torch.from_numpy(p["sources"] == s)].sum((0, 1)).tolist()
                for s in balanced.SOURCES
            }
            curves.append(
                dict(
                    epoch=epoch + 1,
                    balanced_loss=float(t.sum()),
                    baseline_loss=float(old_terms(z, p).sum()),
                    source_head_losses=by_source,
                    gradient_norm=float(norm),
                )
            )
    final = {
        name: gradients(model, p, alternative)
        for name, alternative in (("baseline_objective", False), ("balanced_objective", True))
    }
    return model.eval(), dict(
        initial=initial,
        final=final,
        curves=curves,
        coefficients=p["audit"],
        batch_shape=list(p["x"].shape),
    )


def evaluate(data, models, out, cfg):
    summary, evidence, hashes = {}, {}, {}
    policy2 = read("configs/localization_decoder_v2.json")
    for mode in MODES:
        summary[mode], evidence[mode] = {}, {}
        for name, model in models[mode].items():
            rows1, rows2 = [], []
            evidence[mode][name] = {}
            directory = out / mode / name
            directory.mkdir(parents=True)
            for case_name, case in data.items():
                scores = predict(model, case, mode)
                if name == "baseline":
                    saved = RUN / mode / f"{case_name}-scores.npz"
                    hashes[str(saved)] = sha(saved)
                    with np.load(saved, allow_pickle=False) as z:
                        require(
                            np.array_equal(scores, z["scores"], equal_nan=True),
                            "balanced_baseline_prediction_changed",
                        )
                a = case["arrays"]
                args = scores, model_observable(a), a["reset"], case["video"]
                r1, r2 = decode_v1(*args, cfg["decoder"]), decode_v2(*args, policy2)
                require(r1["boundary_peaks"] == r2["boundary_peaks"], "balanced_peaks_changed")
                one, two = record(case_name, r1, case), record(case_name, r2, case)
                rows1.append(one)
                rows2.append(two)
                evidence[mode][name][case_name] = diagnose(scores, case, one, two, policy2)
                np.savez_compressed(directory / f"{case_name}-scores.npz", scores=scores)
            write_json(directory / "decoder-v1.json", rows1)
            write_json(directory / "decoder-v2.json", rows2)
            summary[mode][name] = dict(v1=summarize(rows1), v2=summarize(rows2))
    write_json(out / "summary.json", summary)
    write_json(out / "evidence.json", evidence)
    return hashes


def restore(path, cfg):
    m = Localizer(cfg["hidden"], cfg["dropout"])
    m.load_state_dict(torch.load(path, weights_only=True, map_location="cpu")["state_dict"])
    return m.eval()


def train(output):
    spec = protocol()
    cfg = spec["baseline"]["training"]
    data, intervals = training_data()
    out = fresh(output)
    write_json(out / "protocol.json", spec)
    models, verification, baseline_gradients = {}, {}, {}
    for mode in MODES:
        checkpoint = RUN / mode / "checkpoint.pt"
        require(
            sha(checkpoint) == read(RUN / mode / "verification.json")["checkpoint_sha256"],
            "historical_balanced_checkpoint_changed",
        )
        old = restore(checkpoint, cfg)
        p = balanced.prepare(data, mode, intervals)
        baseline_gradients[mode] = {
            key: gradients(old, p, alternative)
            for key, alternative in (("baseline_objective", False), ("balanced_objective", True))
        }
        began = perf_counter()
        model, diagnostic = fit(data, mode, cfg, intervals)
        seconds = perf_counter() - began
        dest = out / mode
        dest.mkdir()
        torch.save(
            dict(
                state_dict=model.state_dict(),
                mode=mode,
                protocol=spec,
                dataset=read(PIN),
                model_version="offline-punch-localizer-1.0",
            ),
            dest / "checkpoint.pt",
        )
        write_json(dest / "fit.json", diagnostic)
        repeated, repeat = fit(data, mode, cfg, intervals)
        require(
            state_hash(model) == state_hash(repeated) and diagnostic == repeat,
            "balanced_training_not_deterministic",
        )
        restored = restore(dest / "checkpoint.pt", cfg)
        for case in data.values():
            scores = predict(model, case, mode)
            require(
                np.array_equal(scores, predict(repeated, case, mode), equal_nan=True)
                and np.array_equal(scores, predict(restored, case, mode), equal_nan=True),
                "balanced_repeat_predictions_changed",
            )
        verification[mode] = dict(
            state_sha256=state_hash(model),
            checkpoint_sha256=sha(dest / "checkpoint.pt"),
            baseline_checkpoint_sha256=sha(checkpoint),
            deterministic_fit=True,
            reload_exact=True,
            training_seconds=seconds,
        )
        models[mode] = dict(baseline=old, balanced=model)
        print(mode, "training/repeat/reload verified", flush=True)
    write_json(out / "baseline-gradients.json", baseline_gradients)
    saved_scores = evaluate(data, models, out, cfg)
    write_json(out / "verification.json", verification)
    write_json(
        out / "provenance.json",
        dict(
            created_at=datetime.now(UTC).isoformat(),
            phase="training",
            cases=list(data),
            code_sha256=binding(),
            dataset=read(PIN),
            saved_baseline_scores=saved_scores,
            feature_cache_sha256=sha(CACHE / "manifest.json"),
            runtime=dict(
                python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__
            ),
            new_architecture=False,
            promotion=False,
            v5_inference=False,
        ),
    )
    for p in FILES:
        target = out / "source-snapshot" / p.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(p.read_bytes())
    # All resulting checkpoints, training predictions, policies, code and reports frozen before V5.
    write_json(
        out / "freeze.json",
        dict(
            created_at=datetime.now(UTC).isoformat(),
            code_sha256=binding(),
            artifacts={str(p): sha(p) for p in out.rglob("*") if p.is_file()},
            scope=(
                "one objective, training V4/V6 only; no V5 feature/model evaluation before freeze"
            ),
        ),
    )
    return out


def development(training, output):
    frozen = read(training / "freeze.json")
    require(frozen["code_sha256"] == binding(), "balanced_code_changed_after_freeze")
    require(
        all(sha(Path(p)) == h for p, h in frozen["artifacts"].items()),
        "balanced_frozen_training_artifacts_changed",
    )
    spec = read(training / "protocol.json")
    cfg = spec["baseline"]["training"]
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    data = cases("development")  # Deliberately after checkpoint/policy/code verification.
    out = fresh(output)
    models = {
        mode: dict(
            baseline=restore(RUN / mode / "checkpoint.pt", cfg),
            balanced=restore(training / mode / "checkpoint.pt", cfg),
        )
        for mode in MODES
    }
    hashes = evaluate(data, models, out, cfg)
    write_json(
        out / "provenance.json",
        dict(
            created_at=datetime.now(UTC).isoformat(),
            phase="development",
            freeze_sha256=sha(training / "freeze.json"),
            saved_baseline_scores=hashes,
            cases=list(data),
            tuning=False,
            precision=None,
            false_positive_rate=None,
        ),
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("training", "development"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    args = parser.parse_args()
    if args.phase == "training":
        train(args.output)
    else:
        require(args.training is not None, "balanced_training_freeze_required")
        development(args.training, args.output)
