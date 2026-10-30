"""Read-only experiment interpretation and source-aligned plots; never selects a model."""

import argparse
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

from ..analysis.common import sha, write_json
from ..vision.observation_study import arrays, fresh, read
from .localization_model import MODES


def report(run, cache, output):
    import matplotlib.pyplot as plt

    run, cache = Path(run), Path(cache)
    provenance = read(run / "provenance.json")
    if sha(cache / "manifest.json") != provenance["cache_manifest_sha256"]:
        raise ValueError("Cross-experiment feature cache")
    out = fresh(output)
    summaries, receipts = read(run / "summary.json"), {}
    for mode in MODES:
        if sha(run / mode / "checkpoint.pt") != summaries[mode]["checkpoint_sha256"]:
            raise ValueError("Experiment checkpoint changed")
        records = read(run / mode / "records.json")
        reasons, unique_frames, seconds, unknown_frames = Counter(), 0, Q(0), 0
        errors = []
        for r in records:
            video = read(cache / r["case"] / "video.json")
            n = len(video["frames"])
            times = [Q(f["timestamp_exact"]) for f in video["frames"]] + [
                Q(video["stop_timestamp_exact"])
            ]
            mask = np.zeros(n, bool)
            for span in r["result"]["unresolved"]:
                mask[span["start_frame"] : span["stop_frame_exclusive"]] = True
                reasons[span["reason"]] += 1
            unique_frames += int(mask.sum())
            seconds += sum((times[i + 1] - times[i]) for i in range(n) if mask[i])
            unknown_frames += sum(
                s["stop_frame_exclusive"] - s["start_frame"] for s in r["result"]["unobservable"]
            )
            if r["comparison"]:
                errors.extend(r["comparison"]["matched_boundary_errors"])
        receipts[mode] = dict(
            unresolved_reasons=dict(reasons),
            unresolved_unique_frames=unique_frames,
            unresolved_unique_seconds_exact=str(seconds),
            unobservable_frames=unknown_frames,
            matched_boundary_errors=errors,
            scope="sum of diagnostic streams, with overlapping recordings; "
            "not independent exposure",
        )
    write_json(out / "details.json", receipts)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 120})
    for name in ("V4-bg-1", "V4-bg-2", "V4-bg-3", "V6-upload"):
        video = read(cache / name / "video.json")
        origin = read(cache / name / "provenance.json")
        offset = origin.get("source_start", 0)
        x = np.arange(len(video["frames"])) + offset
        refs = read(cache / name / "references.json")
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 7), layout="constrained")
        for ax, mode in zip(axes, MODES, strict=True):
            p = arrays(run / mode / f"{name}-scores.npz")["scores"]
            for k, (label, color) in enumerate(
                zip(("activity", "onset", "end"), ("#c47b17", "#087e8b", "#b22e6f"), strict=True)
            ):
                ax.plot(x, p[:, k], label=label, color=color, lw=1.2)
            for r in refs:
                ax.axvspan(
                    r["start_frame"] + offset,
                    r["stop_frame_exclusive"] + offset,
                    color="#808080",
                    alpha=0.12,
                )
            if origin["kind"] == "reviewed_source":
                ax.axvspan(
                    x[0],
                    origin["review_start"],
                    facecolor="none",
                    edgecolor="#cccccc",
                    hatch="///",
                    lw=0,
                )
                ax.axvspan(
                    origin["review_stop"],
                    x[-1],
                    facecolor="none",
                    edgecolor="#cccccc",
                    hatch="///",
                    lw=0,
                )
            result = next(r for r in read(run / mode / "records.json") if r["case"] == name)[
                "result"
            ]
            for candidate in result["candidates"]:
                ax.plot(
                    [candidate["start_frame"] + offset, candidate["stop_frame_exclusive"] + offset],
                    [-0.08, -0.08],
                    color="#218046",
                    lw=4,
                    solid_capstyle="butt",
                )
            ax.axhline(0.5, color="#999999", lw=0.7, ls="--")
            ax.set(ylim=(-0.15, 1.05), ylabel=mode)
            ax.grid(alpha=0.15)
        axes[0].legend(loc="upper right", ncols=3)
        axes[-1].set_xlabel("Original source frame" if offset else "Upload frame")
        fig.suptitle(
            f"{name}: gray=human span; green=candidate; hatched=unreviewed context\n"
            "Training-source diagnostic, not generalization; all scores uncalibrated"
        )
        fig.savefig(out / f"{name}-boundaries.png")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for mode in MODES:
        fit = read(run / mode / "fit.json")
        ax.plot(
            [0] + [r["epoch"] for r in fit["curves"]],
            [fit["initial_loss"]] + [r["training_loss"] for r in fit["curves"]],
            label=mode,
        )
    ax.set(
        xlabel="Fixed full-batch update",
        ylabel="Weighted training loss",
        yscale="log",
        title="Training fit only — no validation or epoch selection",
    )
    ax.legend()
    fig.savefig(out / "training-loss.png")
    plt.close(fig)
    write_json(
        out / "artifact-hashes.json",
        {
            str(p.relative_to(run)): sha(p)
            for p in run.rglob("*")
            if p.is_file() and out not in p.parents
        },
    )
    write_json(
        out / "report-provenance.json",
        dict(
            code_sha256=sha(Path(__file__)),
            cache_manifest_sha256=provenance["cache_manifest_sha256"],
            no_training=True,
            upload_references="none; all upload predictions are unverified",
        ),
    )
    return receipts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.cache, args.output)
