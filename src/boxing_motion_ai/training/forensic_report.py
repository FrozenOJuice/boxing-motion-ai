"""Summarize a diagnostic visual review, without changing labels or eligibility."""

import argparse
import html
import json
import os
from collections import Counter
from pathlib import Path

import yaml

from ..vision.pipeline import sha256
from .data import CLASSES
from .experiment import write_json
from .forensic_review import ARMS, pose_stats
from .multi_recording import ALL
from .multi_study import verify_manifest

EVIDENCE = {"consistent", "ambiguous", "visibly_incorrect"}
POSE = {"aligned", "masked_failure", "ambiguous", "confident_misplacement"}


def validate_review(cases, config):
    """Reject incomplete assessments or any attempt to apply diagnostic corrections."""
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids) or set(ids) != set(config["assessments"]):
        raise ValueError("Assessment IDs must match unique selected cases exactly")
    if (
        config["schema_version"] != "label-pose-review-1.0"
        or config["status"] != "assistant_visual_review_not_human_ground_truth"
        or not config["diagnostic_only"]
        or config["annotation_overrides"]
        or config["new_exclusions"]
        or config["human_review_status"] != "pending"
    ):
        raise ValueError("This report cannot correct annotations or certify human ground truth")
    for c in cases:
        if c["recording_id"] not in ALL or c["expected"] not in CLASSES:
            raise ValueError("Reserved recording or unknown class")
        if c["id"] != f"{c['recording_id']}:{c['source_row']}":
            raise ValueError("Case identity mismatch")
        a = config["assessments"][c["id"]]
        if any(a[k] not in EVIDENCE for k in ("family", "side", "boundary")):
            raise ValueError("Invalid evidence assessment")
        if a["pose"] not in POSE or not a["observation"].strip():
            raise ValueError("Pose mode and independent observation required")
    selected = {c["id"]: c for c in cases}
    for key in ("candidate_evaluation_subset", "priority_disputes"):
        queue = config[key]
        if len(set(queue)) != len(queue) or not set(queue) <= set(ids):
            raise ValueError("Human queue must contain unique existing cases")
        if key == "candidate_evaluation_subset" and any(
            not selected[i]["approved_candidate"] or selected[i]["recording_id"] == "V5"
            for i in queue
        ):
            raise ValueError("Candidate evaluation subset requires eligible non-development cases")


def summarize(cases, assessments):
    def counts(rows):
        return {
            axis: dict(Counter(assessments[c["id"]][axis] for c in rows))
            for axis in ("family", "side", "boundary", "pose")
        }

    approved = [c for c in cases if c["approved_candidate"]]
    clear = [
        c
        for c in approved
        if all(assessments[c["id"]][k] == "consistent" for k in ("family", "boundary"))
        and assessments[c["id"]]["pose"] == "aligned"
    ]
    family = {"jab": "straight", "cross": "straight"}
    family.update({k: k.split("_", 1)[1] for k in CLASSES[2:]})
    comparison = Counter()
    for c in approved:
        p = c["prediction"]
        if p is None or p["target"] != c["expected"] or p["prediction"] not in CLASSES:
            raise ValueError("Approved case requires matching frozen prediction")
        label, predicted = c["expected"], p["prediction"]
        comparison[
            "source_label_agreement"
            if label == predicted
            else "within_family_side_disagreement"
            if family[label] == family[predicted]
            else "different_family"
        ] += 1
    return dict(
        cases=len(cases),
        approved=len(approved),
        held=len(cases) - len(approved),
        assessments=counts(cases),
        by_recording={
            v: counts([c for c in cases if c["recording_id"] == v])
            for v in ALL
            if any(c["recording_id"] == v for c in cases)
        },
        source_label_comparison=dict(comparison),
        clear_family_boundary_pose_cases=len(clear),
        clear_cases_with_source_label_disagreement=[
            c["id"] for c in clear if c["prediction"]["prediction"] != c["expected"]
        ],
        interpretation="Purposive diagnostic counts, not error prevalence or verified accuracy",
        human_verified_cases=0,
        corrections=[],
        new_exclusions=[],
        training_performed=False,
        v9_accessed=False,
    )


def write_review(cases_path, config_path, output, *, dense_path=None):
    """Write only into a fresh reports directory; preserve existing review and datasets."""
    import numpy as np

    cases_path, config_path, output = map(Path, (cases_path, config_path, output))
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh ignored reports destination")
    config = yaml.safe_load(config_path.read_text())
    if sha256(cases_path) != config["case_manifest_sha256"]:
        raise ValueError("Case manifest changed since visual assessment")
    cases = json.loads(cases_path.read_text())
    validate_review(cases, config)
    dense_pages = []
    if dense_path is not None:
        dense_path = Path(dense_path)
        dense_pages = json.loads(dense_path.read_text())
        if any(p["id"] not in {c["id"] for c in cases} for p in dense_pages):
            raise ValueError("Dense pages must reference selected cases")
    protected = json.loads((cases_path.parent / "protected.json").read_text())
    if any("/V9/" in p or "/V9." in p for p in protected):
        raise ValueError("V9 reserved")
    verify_manifest(protected)
    output.mkdir(parents=True)
    result = summarize(cases, config["assessments"])
    scores = {}
    for v in ALL:
        rows = [c for c in cases if c["recording_id"] == v]
        if not rows:
            continue
        with np.load(f"data/processed/boxingvi/{v}/candidates-full-v1/poses.npz") as z:
            # No coordinates are normalized, filled, or rewritten here.
            a = {k: z[k] for k in ("confidence", "valid_mask")}
        for c in rows:
            start, end = map(int, c["raw_bounds"])
            scores[c["id"]] = pose_stats(a, start, end, include_failures=True)
    write_json(output / "confidence-audit.json", scores)
    write_json(output / "summary.json", result)
    pending = []
    for purpose in ("priority_disputes", "candidate_evaluation_subset"):
        for case_id in config[purpose]:
            pending.append(
                dict(
                    id=case_id,
                    purpose=purpose,
                    status="pending_human_review",
                    observed_family=None,
                    observed_anatomical_arm=None,
                    stance=None,
                    lead_rear=None,
                    boundary_evidence=None,
                    pose_localization_evidence=None,
                    decision=None,
                    reviewer=None,
                )
            )
    write_json(output / "human-review.json", pending)
    lookup = {c["id"]: c for c in cases}
    queue = [
        "<!doctype html><meta charset='utf-8'><title>Pending human review</title>",
        "<style>body{font:16px sans-serif;max-width:1150px;margin:auto}img{width:100%}</style>",
        "<h1>18-case proposed human review</h1>",
        "<p>Pending human certification. Inspect movement before revealing hypotheses. "
        "Source HUDs may disclose exercise names. No label/eligibility changes.</p>",
    ]
    for item in pending:
        c = lookup[item["id"]]
        sheet = Path(os.path.relpath(cases_path.parent / c["sheet"], output)).as_posix()
        queue += [
            f"<h2>{c['id']} — {item['purpose']}</h2>",
            f"<img loading='lazy' src='{html.escape(sheet, quote=True)}'>",
        ]
        for page in dense_pages:
            if page["id"] == c["id"]:
                link = Path(os.path.relpath(dense_path.parent / page["sheet"], output)).as_posix()
                queue.append(
                    f"<p><a href='{html.escape(link, quote=True)}'>"
                    f"Native frames {page['frames'][0]}–{page['frames'][-1]}</a></p>"
                )
        hypotheses = dict(
            source_label=c["expected"],
            frozen_prediction=c["prediction"],
            assistant_observation=config["assessments"][c["id"]],
        )
        queue.append(
            "<details><summary>Reveal hypotheses, not ground truth</summary><pre>"
            + html.escape(json.dumps(hypotheses, indent=2))
            + "</pre></details>"
        )
    (output / "human-review.html").write_text("\n".join(queue))
    lines = [
        "# Diagnostic visual review (assistant assessment; human certification pending)",
        "",
        "Source labels and model outputs are hypotheses. No annotations or eligibility changed.",
        "Side means stance-relative lead/rear; an estimated L/R channel is not stance evidence.",
        "Boundary 'visibly_incorrect' means visibly mixed phases for a clean single-action clip;",
        "it does not establish the authors' onset/end rules or a replacement interval.",
        "Pose masked_failure means a visibility/confidence failure in either arm, not necessarily",
        "the punching arm. No confirmed high-confidence anatomical misplacement was established.",
        "",
        "Inspect raw footage first, then skeleton, then reveal source annotation/prediction.",
        "The source's own overlays can disclose exercise labels; review is not fully blinded.",
        "",
        "| Case | Family | Lead/rear | Boundary | Pose | Independent visual observation |",
        "|---|---|---|---|---|---|",
    ]
    for c in cases:
        a = config["assessments"][c["id"]]
        link = Path(os.path.relpath(cases_path.parent / "index.html", output)).as_posix()
        cells = [a[k] for k in ("family", "side", "boundary", "pose", "observation")]
        lines.append(
            f"| [{c['id']}]({link}#{c['id']}) | "
            + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in cells)
            + " |"
        )
    (output / "REVIEW.md").write_text("\n".join(lines) + "\n")
    verify_manifest(protected)
    write_json(
        output / "provenance.json",
        dict(
            cases_sha256=sha256(cases_path),
            assessments_sha256=sha256(config_path),
            preserved_files=len(protected),
            changed=[],
            arm_joints=ARMS,
        ),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--assessments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dense", type=Path, help="Optional dense.json for native-frame page links"
    )
    args = parser.parse_args()
    print(write_review(args.cases, args.assessments, args.output, dense_path=args.dense))
