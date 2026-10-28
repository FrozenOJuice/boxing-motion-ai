"""Fixed-budget V4-only engineering fit and reviewed-frame inference. No model selection."""

import hashlib
import json
import platform
import random
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import Field
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data.annotation_review import StrictModel
from ..services.artifacts import MetadataStore, safe_relative
from ..services.contracts import Artifact
from ..training.augmentation import mirror_features
from ..vision.pipeline import code_provenance, sha256
from .data import HELD, ReviewedWindows, collate
from .data import VERSION as FEATURE_VERSION
from .model import VERSION, BinaryDetector


class Config(StrictModel):
    schema_version: Literal["binary-detector-smoke-1.0"]
    dataset: Artifact
    training_recording: Literal["V4"]
    held_intervals: list[str]
    seed: int
    epochs: int = Field(ge=1, le=100)
    core_frames: int = Field(ge=1, le=128)
    batch_size: int = Field(ge=1, le=32)
    learning_rate: float = Field(gt=0, le=0.01)
    hidden: Literal[32]
    dropout: Literal[0.1]
    mirror_probability: Literal[0.5]
    threads: Literal[2]
    decision_threshold: Literal[0.5]
    purpose: Literal["user_approved_v4_training_only_smoke_no_model_selection"]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fresh(root, output):
    safe_relative(output)
    path = Path(root) / output
    if not output.startswith("reports/") or path.exists():
        raise ValueError("Choose a new ignored reports/ directory")
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError("Unsafe output path")
    path.mkdir(parents=True, exist_ok=False)
    return path


def verify_binding(root, config):
    current, _ = MetadataStore(root).json("configs/continuous_dataset_v1.json")
    if config.dataset.model_dump(mode="json") != current:
        raise ValueError("Checkpoint/config must bind the accepted continuous dataset v1")
    MetadataStore(root).json(config.dataset.path, config.dataset.sha256)


def load(root, config_path):
    value, _ = MetadataStore(root).json(config_path)
    config = Config.model_validate(value)
    if set(config.held_intervals) != HELD:
        raise ValueError("Both cardio intervals must stay held")
    verify_binding(root, config)
    dataset = ReviewedWindows(root, "V4", config.core_frames)
    allowed = {f"V4-motion-{i}" for i in range(1, 7)} | {f"V4-bg-{i}" for i in range(1, 4)}
    if {i["id"] for i in dataset.intervals} != allowed:
        raise ValueError("Unexpected V4 smoke pool")
    if min(dataset.diagnostics["supervised_observable_frames"]) <= 0:
        raise ValueError("Both activities need observable supervision")
    return config, dataset


def masked_loss(logits, labels, mask, pos_weight):
    if not mask.any():
        raise ValueError("Empty supervised batch")
    loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none", pos_weight=logits.new_tensor(pos_weight)
    )
    weights = torch.where(labels > 0, logits.new_tensor(pos_weight), 1)
    return loss[mask].sum() / weights[mask].sum()


@torch.no_grad()
def fit_diagnostics(model, loader, weight, threshold=0.5):
    model.eval()
    numerator = denominator = 0.0
    confusion = torch.zeros((2, 2), dtype=torch.long)
    for x, y, mask in loader:
        logits = model(x)
        w = torch.where(y > 0, weight, 1.0)
        numerator += float(
            (
                F.binary_cross_entropy_with_logits(
                    logits, y, reduction="none", pos_weight=x.new_tensor(weight)
                )
                * mask
            ).sum()
        )
        denominator += float(w[mask].sum())
        predicted = (logits.sigmoid() >= threshold).long()
        confusion += torch.bincount((y[mask].long() * 2 + predicted[mask]), minlength=4).reshape(
            2, 2
        )
    return dict(
        weighted_loss=numerator / denominator,
        confusion=confusion.tolist(),
        interpretation="training_fit_only_not_generalization",
    )


def run(root, config_path, output):
    config, dataset = load(root, config_path)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    model = BinaryDetector(hidden=config.hidden, dropout=config.dropout)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=0,
    )
    evaluation = DataLoader(
        dataset, batch_size=config.batch_size, collate_fn=collate, num_workers=0
    )
    mirror_rng = torch.Generator().manual_seed(config.seed + 1)
    counts = dataset.diagnostics["supervised_observable_frames"]
    weight = counts[0] / counts[1]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    initial = fit_diagnostics(model, evaluation, weight)
    curves = []
    gradients = []
    updates = 0
    for epoch in range(config.epochs):
        model.train()
        for x, y, mask in loader:
            selected = torch.rand(len(x), generator=mirror_rng) < config.mirror_probability
            x[selected] = mirror_features(x[selected])
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(x), y, mask, weight)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite detector loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            gradients.append(float(norm))
            optimizer.step()
            updates += 1
        curves.append(dict(epoch=epoch + 1, **fit_diagnostics(model, evaluation, weight)))
    if not (max(gradients) > 0 and curves[-1]["weighted_loss"] < initial["weighted_loss"]):
        raise ValueError("Smoke failed: no gradient or no reduction in fixed training-fit loss")
    path = fresh(root, output)
    code = code_provenance()
    repo_root = Path(__file__).resolve().parents[3]
    code["sha256"].update(
        {
            str(p.relative_to(repo_root)): sha256(p)
            for p in [
                *Path(__file__).parent.glob("*.py"),
                Path(root) / "src/boxing_motion_ai/training/data.py",
                Path(root) / "src/boxing_motion_ai/training/models.py",
                Path(root) / "src/boxing_motion_ai/training/velocity.py",
                Path(root) / "src/boxing_motion_ai/training/augmentation.py",
            ]
        }
    )
    checkpoint = dict(
        version=VERSION,
        feature_version=FEATURE_VERSION,
        config=config.model_dump(mode="json"),
        state_dict=model.state_dict(),
        dataset_sha256=config.dataset.sha256,
    )
    torch.save(checkpoint, path / "checkpoint.pt")
    checkpoint_hash = sha256(path / "checkpoint.pt")
    (path / "checkpoint.sha256").write_text(checkpoint_hash + "\n")
    restored = BinaryDetector()
    restored.load_state_dict(
        torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)["state_dict"]
    )
    restored.eval()
    with torch.no_grad():
        sample = collate([dataset[0]])[0]
        reload_error = float((restored(sample) - model(sample)).abs().max())
    if reload_error != 0:
        raise ValueError("Checkpoint reload changed predictions")
    summary = dict(
        schema_version="binary-detector-smoke-result-1.0",
        purpose=config.purpose,
        training_recordings=["V4"],
        validation_recordings=[],
        test_recordings=[],
        v6_optimization_frames=0,
        epochs=config.epochs,
        updates=updates,
        initial=initial,
        final=curves[-1],
        max_gradient_norm=max(gradients),
        parameter_change_l2=float(
            sum((before[k] - v).square().sum() for k, v in model.state_dict().items()).sqrt()
        ),
        checkpoint_sha256=checkpoint_hash,
        checkpoint_reload_max_error=reload_error,
        parameters=sum(p.numel() for p in model.parameters()),
        positive_loss_weight=weight,
        optimizer="AdamW, weight_decay=0.0001, gradient_norm_clip=1",
        runtime=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            numpy=np.__version__,
            device="cpu",
            threads=config.threads,
        ),
        code=code,
        dataset=dataset.diagnostics,
        feature_sha256=hashlib.sha256(dataset.x.numpy().tobytes()).hexdigest(),
        metrics_scope="training fit only; no validation, event evaluation or threshold selection",
    )
    write(path / "config.json", config.model_dump(mode="json"))
    write(path / "curves.json", curves)
    write(path / "summary.json", summary)
    return summary


@torch.no_grad()
def predict(root, run_path, output, recording="V4"):
    safe_relative(run_path)
    if not run_path.startswith("reports/"):
        raise ValueError("Use a local smoke checkpoint under reports/")
    run_dir = Path(root) / run_path
    store = MetadataStore(root)
    with store.open_regular(f"{run_path}/checkpoint.sha256") as receipt:
        expected = receipt.read(100).decode().strip()
    with store.open_regular(f"{run_path}/checkpoint.pt") as checkpoint:
        if hashlib.file_digest(checkpoint, "sha256").hexdigest() != expected:
            raise ValueError("Checkpoint hash mismatch")
        checkpoint.seek(0)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["version"] != VERSION or saved.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Unsupported detector checkpoint")
    config = Config.model_validate(saved["config"])
    if saved["dataset_sha256"] != config.dataset.sha256:
        raise ValueError("Checkpoint dataset binding mismatch")
    if config.training_recording != "V4" or set(config.held_intervals) != HELD:
        raise ValueError("Invalid checkpoint scope")
    verify_binding(root, config)
    data = ReviewedWindows(root, recording, config.core_frames)
    torch.set_num_threads(config.threads)
    model = BinaryDetector()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    by_frame = {}
    stream_error = 0.0
    for window in data.windows:
        logits = model(window["x"][None])[0]
        state = None
        for j, frame in enumerate(window["x"]):
            probability, state = model.step(frame, state)
            stream_error = max(stream_error, abs(float(probability) - float(logits[j].sigmoid())))
            if window["loss_mask"][j]:
                by_frame[int(window["frames"][j])] = float(probability)
    if stream_error > 1e-5:
        raise ValueError("Streaming/window disagreement")
    path = fresh(root, output)
    lines = []
    for interval in data.intervals:
        p = interval["provenance"]
        for n in range(p["start"], p["stop"]):
            pts, num, den = (int(data.raw[k][n]) for k in ("pts", "time_base_num", "time_base_den"))
            probability = by_frame.get(n)
            lines.append(
                dict(
                    recording=recording,
                    interval_id=interval["id"],
                    frame_index=n,
                    pts=pts,
                    time_base_num=num,
                    time_base_den=den,
                    timestamp_exact=str(Fraction(pts * num, den)),
                    probability_punch=probability,
                    observable=probability is not None,
                    decision=None
                    if probability is None
                    else int(probability >= config.decision_threshold),
                    reviewed_target=int(data.labels[interval["id"]][n - p["start"]]),
                )
            )
    (path / "frames.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in lines)
    )
    result = dict(
        recording=recording,
        rows=len(lines),
        unobservable_rows=sum(not r["observable"] for r in lines),
        checkpoint_sha256=sha256(run_dir / "checkpoint.pt"),
        dataset_sha256=config.dataset.sha256,
        streaming_window_max_error=stream_error,
        threshold=config.decision_threshold,
        scope="reviewed footage only; training-fit outputs, not generalization evaluation",
    )
    write(path / "receipt.json", result)
    return result
