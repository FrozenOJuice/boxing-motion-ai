"""Small local dataset audit and pose-extraction CLI."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from .audit import run_audit, write_report

app = typer.Typer(
    no_args_is_help=True, help="Boxing Motion AI: local dataset auditing and V5 pose extraction."
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
    video: Annotated[Path, typer.Option(help="Local V5 video; never downloaded by this command.")],
    workbook: Annotated[Path, typer.Option(help="Original V5.xlsx; read only.")],
    model: Annotated[Path, typer.Option(help="Local MediaPipe Pose Landmarker .task model.")],
    output: Annotated[Path, typer.Option(help="New output directory outside original dataset.")],
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
    """Extract V5 poses with provisional raw-index annotation provenance; no training."""
    from .vision.config import load_config
    from .vision.pipeline import extract_v5, frame_ranges

    try:
        settings = load_config(
            config,
            ranges=ranges,
            max_width=max_width,
            confidence=confidence,
            overlay_every=overlay_every,
            overlay_limit=overlay_limit,
        )
        manifest = extract_v5(
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
    dataset: Annotated[Path, typer.Option(help="Immutable prepared V5 dataset directory.")],
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
