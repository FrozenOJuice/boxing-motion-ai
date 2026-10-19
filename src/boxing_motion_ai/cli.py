"""Small, local-only Phase 0 CLI."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from .audit import run_audit, write_report

app = typer.Typer(no_args_is_help=True, help="Boxing Motion AI: local dataset audit (Phase 0).")


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
