"""Small local dataset audit and pose-extraction CLI."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from .audit import run_audit, write_report

app = typer.Typer(
    no_args_is_help=True,
    help="Boxing Motion AI: local dataset auditing and per-recording pose extraction.",
)


class Dataset(StrEnum):
    BOXINGVI = "boxingvi"


@app.callback()
def main():
    """Inspect local data. Never download footage or train models."""


@app.command()
def audit(
    dataset: Annotated[Dataset, typer.Option(help="Dataset adapter to use.")],
    root: Annotated[Path, typer.Option(help="Local dataset root; may be missing or empty.")],
    output: Annotated[Path, typer.Option(help="Report directory outside the dataset root.")],
    mapping: Annotated[
        Path | None,
        typer.Option(
            help="Verified XLSX layout/class/video mapping JSON; see docs/ANNOTATION_GUIDE.md."
        ),
    ] = None,
):
    """Write audit.json and audit.md. Exit 1 while inputs or manual checks are incomplete."""
    if output.resolve().is_relative_to(root.resolve()):
        raise typer.BadParameter("--output must be outside --root to keep discovery reproducible")
    try:
        report = run_audit(root, mapping)
        write_report(report, output)
    except OSError as exc:
        typer.echo(f"Cannot read inputs or write reports: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        f"{dataset.value}: {report['status']}; "
        f"{report['counts']['xlsx']} workbook(s), {report['counts']['videos']} video(s), "
        f"{report['counts']['parsed_events']} parsed event row(s)."
    )
    typer.echo(f"Reports: {output / 'audit.json'} and {output / 'audit.md'}")
    for diagnostic in report["issues"]:
        if diagnostic["severity"] == "error":
            typer.echo(
                f"{diagnostic['code']}: {diagnostic['location']}: {diagnostic['message']}", err=True
            )
    typer.echo("Phase 0 gate remains unverified; review diagnostics and manual checks.")
    raise typer.Exit(1)


@app.command("extract-poses")
def extract_poses(
    video: Annotated[
        Path, typer.Option(help="Local V1–V10 video; never downloaded by this command.")
    ],
    workbook: Annotated[Path, typer.Option(help="Original matching workbook; read only.")],
    model: Annotated[Path, typer.Option(help="Local MediaPipe Pose Landmarker .task model.")],
    output: Annotated[Path, typer.Option(help="New output directory outside original dataset.")],
    resume: Annotated[
        bool, typer.Option(help="Reuse validated completed IMAGE frames after interruption.")
    ] = False,
    recording: Annotated[
        Path | None, typer.Option(help="Explicit inspected recording YAML. Required beyond V5.")
    ] = None,
    ranges: Annotated[
        list[str] | None,
        typer.Option(
            "--range",
            help="Repeat START:STOP decoded-index ranges; stop exclusive. "
            "Omit for all frames. Annotation end convention remains unknown.",
        ),
    ] = None,
    config: Annotated[
        Path | None, typer.Option(help="YAML settings; explicit CLI values override.")
    ] = None,
    max_width: Annotated[
        int | None, typer.Option(min=64, help="Inference resize width; no frame dropping.")
    ] = None,
    confidence: Annotated[
        float | None, typer.Option(min=0, max=1, help="Pose and valid-joint threshold.")
    ] = None,
    overlay_every: Annotated[
        int | None, typer.Option(min=1, help="Overlay interval within each range.")
    ] = None,
    overlay_limit: Annotated[
        int | None, typer.Option(min=0, help="Maximum review JPEGs; zero disables.")
    ] = None,
):
    """Extract poses using explicit recording settings; no training."""
    from .vision.config import load_config
    from .vision.pipeline import extract_recording, frame_ranges
    from .vision.recording import load_recording, v5_recording
    from .vision.sparse_decoder import choose_decoder

    try:
        settings = load_config(
            config,
            ranges=ranges,
            max_width=max_width,
            confidence=confidence,
            overlay_every=overlay_every,
            overlay_limit=overlay_limit,
        )
        if not all(p.is_file() for p in (video, workbook, model)):
            raise ValueError("Video, workbook and model must all exist")
        if recording is None and video.stem.upper() != "V5":
            raise ValueError("--recording is required beyond V5; no timing/layout inheritance")
        manifest = extract_recording(
            video,
            workbook,
            model,
            output,
            frame_ranges(settings.ranges),
            settings.max_width,
            settings.confidence,
            settings.overlay_every,
            settings.overlay_limit,
            progress=lambda n: typer.echo(f"Extracted {n} frames…"),
            pose_mode=settings.pose_mode,
            scene_cuts=settings.scene_cuts,
            overlay_frames=settings.overlay_frames,
            recording=load_recording(recording) if recording else v5_recording(),
            resume=resume,
            decoder=choose_decoder(output, frame_ranges(settings.ranges)),
        )
    except ImportError as exc:
        typer.echo("Pose dependencies missing: run uv sync --extra pose --locked", err=True)
        raise typer.Exit(2) from exc
    except Exception as exc:
        typer.echo(f"Pose extraction failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"{manifest['scope']}: {manifest['processed_frame_count']} frames; "
        f"{manifest['status_counts']}. Outputs: {output}"
    )
    typer.echo("Annotation mapping remains provisional. Review skeletons before further work.")


@app.command("prepare-v5")
def prepare_v5_command(
    pose_run: Annotated[Path, typer.Option(help="Completed full-video IMAGE pose run directory.")],
    workbook: Annotated[Path, typer.Option(help="Original V5.xlsx; hash must match extraction.")],
    output: Annotated[Path, typer.Option(help="New generated dataset directory outside raw data.")],
):
    """Pack raw poses and provisional punch candidates; exclude suspect rows, never train."""
    from .data.preparation import prepare_v5

    try:
        report = prepare_v5(pose_run, workbook, output)
    except ImportError as exc:
        typer.echo("Install optional pose dependencies: uv sync --locked --extra pose", err=True)
        raise typer.Exit(2) from exc
    except Exception as exc:
        typer.echo(f"Preparation failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Prepared {report['annotation_row_count']} source rows; reports: {output}")
    typer.echo("V5-only provisional candidates; no independent split or training performed.")


@app.command("assess-pose-quality")
def assess_pose_quality(
    dataset: Annotated[Path, typer.Option(help="Immutable prepared recording dataset directory.")],
    output: Annotated[Path, typer.Option(help="New read-only-analysis report directory.")],
    video: Annotated[
        Path | None, typer.Option(help="Optional original video for diagnostic sheets.")
    ] = None,
    permutations: Annotated[
        int, typer.Option(min=1, max=5000, help="Label permutations; no model fitting.")
    ] = 499,
):
    """Compare partial-pose policies and mask/class association without training."""
    from .data.pose_quality import assess_quality

    try:
        result = assess_quality(dataset, output, permutations)
        if video is not None:
            from .data.quality_review import render_quality_review

            render_quality_review(dataset, video, output)
    except Exception as exc:
        typer.echo(f"Quality assessment failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Compared {len(result['policies'])} policies across six classes. Reports: {output}")
    typer.echo("Descriptive diagnostics only; no classifier trained or original data modified.")


@app.command("prepare-recording")
def prepare_recording_command(
    pose_run: Annotated[Path, typer.Option(help="Completed IMAGE pose run.")],
    workbook: Annotated[Path, typer.Option(help="Original matching workbook, read only.")],
    output: Annotated[Path, typer.Option(help="New generated directory outside raw data.")],
    allow_sample: Annotated[
        bool, typer.Option(help="Allow partial extraction; unextracted is unassessed.")
    ] = False,
):
    """Prepare one recording, preserving source indices, identities and sample coverage."""
    from .data.preparation import prepare_recording

    try:
        report = prepare_recording(pose_run, workbook, output, allow_sample=allow_sample)
    except Exception as exc:
        typer.echo(f"Preparation failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"{report['video_id']}: {report['extracted_frame_count']} extracted frames; {output}"
    )


@app.command("plan-pose-sample")
def plan_pose_sample(
    workbook: Annotated[Path, typer.Option(help="Read-only original workbook.")],
    recording: Annotated[Path, typer.Option(help="Explicit inspected recording YAML.")],
    output: Annotated[Path, typer.Option(help="New generated sample-plan directory.")],
    per_class: Annotated[int, typer.Option(min=2, help="Time-spread samples per class.")] = 5,
):
    """Select a reproducible sample before seeing estimator confidence or masks."""
    from .vision.recording import load_recording
    from .vision.sampling import plan_sample

    try:
        plan = plan_sample(workbook, load_recording(recording), output, per_class)
    except Exception as exc:
        typer.echo(f"Sample planning failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Selected {len(plan['selected_rows'])} rows; config: {output / 'pose-config.json'}")


@app.command("compare-pose-quality")
def compare_pose_quality(
    dataset: Annotated[list[Path], typer.Option(help="Repeat for distinct recording datasets.")],
    output: Annotated[Path, typer.Option(help="New comparison report directory.")],
    per_class: Annotated[int, typer.Option(min=2, help="Time-spread candidates per class.")] = 5,
):
    """Compare fixed-policy retention and joint visibility using balanced samples."""
    from .data.comparison import compare_recordings

    try:
        result = compare_recordings(dataset, output, per_class)
    except Exception as exc:
        typer.echo(f"Comparison failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Compared {len(result['recordings'])} recordings; {output / 'comparison.md'}")


@app.command("validate-pose-samples")
def validate_pose_samples(
    root: Annotated[Path, typer.Option(help="Existing BoxingVI raw root; read only.")],
    processed: Annotated[Path, typer.Option(help="Ignored processed-data root, outside raw data.")],
    output: Annotated[Path, typer.Option(help="Ignored batch report root; rerun to resume.")],
    model: Annotated[Path, typer.Option(help="Existing local MediaPipe model; never downloaded.")],
    recording: Annotated[list[Path], typer.Option(help="Repeat explicit recording YAML paths.")],
    per_class: Annotated[
        int, typer.Option(min=2, help="Time-spread samples per available class.")
    ] = 5,
):
    """Resume a batch of IMAGE samples, prepare candidates and assess unchanged masks."""
    from .data.validation import validate_samples

    try:
        state = validate_samples(root, processed, output, model, recording, per_class, typer.echo)
    except Exception as exc:
        typer.echo(f"Sample validation failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    if any(e["status"] != "complete" for e in state["recordings"].values()):
        raise typer.Exit(1)


@app.command("diagnose-alignment")
def diagnose_alignment_command(
    root: Annotated[Path, typer.Option(help="Existing raw BoxingVI root.")],
    evidence: Annotated[
        Path, typer.Option(help="Prior source manifest and exact ffprobe frame inventory.")
    ],
    output: Annotated[Path, typer.Option(help="New ignored diagnostic report directory.")],
    recording: Annotated[list[Path], typer.Option(help="Repeat inspected recording YAML paths.")],
    plans: Annotated[Path | None, typer.Option(help="Optional existing sample plans root.")] = None,
):
    """Compare raw-index and rate hypotheses against cached, hash-matched source timing."""
    from .data.alignment import diagnose_alignment

    try:
        result = diagnose_alignment(root, evidence, output, recording, plans)
    except Exception as exc:
        typer.echo(f"Alignment diagnostics failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{len(result['recordings'])} recordings; no automatic corrections: {output}")


@app.command("summarize-datasets")
def summarize_datasets_command(
    dataset: Annotated[list[Path], typer.Option(help="Repeat prepared recording directories.")],
    quality: Annotated[list[Path], typer.Option(help="Repeat corresponding quality.json reports.")],
    alignment: Annotated[Path, typer.Option(help="Hash-matched dataset-wide alignment JSON.")],
    output: Annotated[Path, typer.Option(help="New ignored summary directory.")],
    review: Annotated[
        Path | None, typer.Option(help="Explicit provisional visual-review decisions JSON.")
    ] = None,
):
    """Summarize source counts, sample coverage and readiness without inferring alignment."""
    from .data.summary import summarize_datasets

    try:
        result = summarize_datasets(dataset, quality, alignment, output, review)
    except Exception as exc:
        typer.echo(f"Dataset summary failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{len(result['recordings'])} recordings; no training: {output / 'dataset.md'}")


@app.command("review-pose-sample")
def review_pose_sample(
    pose_run: Annotated[Path, typer.Option(help="Completed sampled pose extraction.")],
    plan: Annotated[Path, typer.Option(help="Matching sample-plan directory.")],
    output: Annotated[Path, typer.Option(help="New ignored visual-review directory.")],
):
    """Make per-class sheets from existing overlays, without decoding again."""
    from .data.sample_review import render_sample_review

    try:
        render_sample_review(pose_run, plan, output)
    except Exception as exc:
        typer.echo(f"Visual review failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Review index: {output / 'review.md'}")


@app.command('train-level1')
def train_level1(
    config: Annotated[Path, typer.Option(help='Versioned V4-train / V5-development YAML.')],
    output: Annotated[Path, typer.Option(help='New ignored directory below reports/.')],
):
    """Train static/causal TCN baselines and matched mask-only ablations (CPU)."""
    try:
        from .training.experiment import run_experiment

        run_experiment(config, output, typer.echo)
    except ImportError as exc:
        typer.echo('Training dependencies missing: uv sync --extra train --extra pose', err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        typer.echo(f'Experiment failed: {type(exc).__name__}: {exc}', err=True)
        raise typer.Exit(1) from exc
