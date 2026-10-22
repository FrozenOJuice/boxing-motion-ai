"""Replay training-only draws and reload selected multi-recording checkpoints."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..vision.pipeline import sha256
from .data import collate
from .experiment import evaluate, seed_all, write_json
from .models import CausalTCN
from .multi_recording import (
    ARMS,
    MODELS,
    Pool,
    load_pool,
    sample_probabilities,
    settings,
    validate_fold,
)
from .multi_study import prediction_metrics, verify_manifest


def verify_sampling(clips, probabilities, seed, audit, batch_size=32):
    sample_rng = torch.Generator().manual_seed(seed + 3000003)
    mirror_rng = torch.Generator().manual_seed(seed + 1000003)
    for epoch in audit:
        indices = torch.multinomial(
            probabilities, len(clips), replacement=True, generator=sample_rng
        ).tolist()
        mirrors = []
        for start in range(0, len(indices), batch_size):
            mirrors.extend(
                (
                    torch.rand(min(batch_size, len(indices) - start), generator=mirror_rng) < 0.5
                ).tolist()
            )
        if epoch["indices"] != indices or epoch["mirrored"] != mirrors:
            raise ValueError("Sampling/mirror replay differs")
    return len(audit)


def verify(output):
    output = Path(output)
    cfg, base, specs = settings(output / "study.yaml")
    result = json.loads((output / "summary.json").read_text())
    seed_all(0, base["threads"])
    cache = load_pool(specs, 2 if result["smoke"] else None)
    count, sets, epochs = 0, 0, 0
    for case, d in result["cases"].items():
        fold = d["fold"]
        validate_fold(fold["train"], ["V5"], fold["heldout"])
        pool = Pool([cache[v][ARMS[0]] for v in fold["train"]])
        p = sample_probabilities(pool.clips)
        expected_ids = [[c["recording_id"], c["source_row"]] for c in pool.clips]
        for seed in d["seeds"]:
            for arm in ARMS:
                for name in MODELS:
                    root = output / case / f"seed-{seed}" / arm / name
                    state = torch.load(root / "best.pt", weights_only=True)
                    provenance = json.loads((root / "provenance.json").read_text())
                    if (
                        state["training_recordings"] != fold["train"]
                        or state["selection_recordings"] != ["V5"]
                        or state["development_recordings"] != ["V5"]
                        or provenance["training_examples"] != expected_ids
                        or state["provenance_sha256"] != sha256(root / "provenance.json")
                        or sha256(root / "best.pt")
                        != d["training"][str(seed)][arm][name]["checkpoint_sha256"]
                    ):
                        raise ValueError("Checkpoint split or provenance mismatch")
                    audit = json.loads((root / "sampling-audit.json").read_text())
                    epochs += verify_sampling(pool.clips, p, seed, audit)
                    channels = (
                        (65 if arm == ARMS[0] else 99)
                        if name == "tcn"
                        else (16 if arm == ARMS[0] else 28)
                    )
                    model = CausalTCN(channels, base["hidden"], base["dropout"])
                    model.load_state_dict(state["model_state"])
                    model.eval()
                    for video in ["V5", *fold["heldout"]]:
                        metric, detail = evaluate(
                            model,
                            DataLoader(cache[video][arm], batch_size=32, collate_fn=collate),
                            "x" if name == "tcn" else "mask_x",
                            torch.ones(6),
                        )
                        saved = json.loads((root / f"{video}-predictions.json").read_text())
                        if (
                            detail != saved
                            or metric["confusion_matrix"]
                            != d["runs"][str(seed)][arm][video][name]["confusion_matrix"]
                        ):
                            raise ValueError("Checkpoint predictions do not reproduce exactly")
                        sets += 1
                    # Recompute every saved metric, including in-sample source diagnostics.
                    for video, ms in d["runs"][str(seed)][arm].items():
                        filename = (
                            "heldout-group-predictions.json"
                            if video == "heldout_group"
                            else f"{video}-predictions.json"
                        )
                        saved = json.loads((root / filename).read_text())
                        fresh = prediction_metrics(saved)
                        for field in (
                            "confusion_matrix",
                            "accuracy",
                            "macro_f1",
                            "supported_macro_f1",
                        ):
                            if fresh[field] != ms[name][field]:
                                raise ValueError("Saved metrics differ")
                    count += 1
    manifest = json.loads((output / "preserved-inputs.json").read_text())
    verify_manifest(manifest)
    proof = dict(
        checkpoints_reloaded=count,
        prediction_sets_reproduced_exactly=sets,
        sampler_mirror_epochs_replayed=epochs,
        prior_files_verified=len(manifest),
        heldout_groups_used_for_training_or_selection=False,
    )
    write_json(output / "reproducibility.json", proof)
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(verify(parser.parse_args().output))
