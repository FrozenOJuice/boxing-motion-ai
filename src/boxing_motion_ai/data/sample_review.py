"""Arrange already extracted source overlays for class/time review; no decoding or inference."""

import json
from pathlib import Path

from ..vision.pipeline import sha256, write_json
from .preparation import CLASS_MAP


def render_sample_review(pose_run, plan, output):
    from PIL import Image, ImageDraw, ImageFont

    pose_run, plan, output = map(Path, (pose_run, plan, output))
    if output.exists() or any(p.name == "raw" for p in output.resolve().parents):
        raise ValueError("Use a new visual review directory outside raw data")
    manifest = json.loads((pose_run / "manifest.json").read_text())
    selected = json.loads((plan / "sample-plan.json").read_text())
    if (
        manifest["status"] != "complete"
        or selected["workbook_sha256"] != manifest["inputs"]["workbook"]["sha256"]
        or selected["recording"] != manifest["recording_config"]
    ):
        raise ValueError("Visual sample inputs disagree or extraction is incomplete")
    available = {Path(p).name for p in manifest["overlay_files"]}
    output.mkdir(parents=True)
    evidence = {
        "video_id": manifest["video_id"],
        "pose_manifest_sha256": sha256(pose_run / "manifest.json"),
        "sample_plan_sha256": sha256(plan / "sample-plan.json"),
        "classes": {},
    }
    font = ImageFont.load_default(size=19)
    for label in CLASS_MAP.values():
        rows = [
            r
            for r in selected["selected_rows"]
            if selected["recording"]["class_map"].get(r["source_class"]) == label
        ]
        rows.sort(key=lambda r: r["candidate_start_index"])
        if not rows:
            evidence["classes"][label] = {"selected_events": 0, "review_rows": [], "sheet": None}
            continue
        chosen = [rows[i] for i in sorted({0, len(rows) // 2, len(rows) - 1})]
        sheet = Image.new("RGB", (1920, len(chosen) * 480), "#101827")
        draw = ImageDraw.Draw(sheet)
        used = []
        for line, row in enumerate(chosen):
            a, b = row["candidate_start_index"], row["candidate_end_index"]
            b -= int(selected["recording"]["end_assumption"] == "exclusive")
            indices = sorted({a, (a + b) // 2, b})
            flags = row["flags"][3:]
            draw.text(
                (10, line * 480 + 7),
                f"{manifest['video_id']} expected {label} / source row {row['source_row']} / "
                f"{'FLAGGED diagnostic only' if flags else 'PROVISIONAL, unverified'}",
                font=font,
                fill="#f5ce80",
            )
            for col, n in enumerate(indices):
                name = f"frame_{n:06d}.jpg"
                if name not in available:
                    raise ValueError(f"Expected sample overlay is absent: {name}")
                with Image.open(pose_run / "overlays" / name) as image:
                    image.thumbnail((640, 434))
                    sheet.paste(image, (640 * col, line * 480 + 40))
            used.append(
                {
                    "source_row": row["source_row"],
                    "source_class": row["source_class"],
                    "frames": indices,
                    "flags": row["flags"],
                }
            )
        name = f"{label}.jpg"
        sheet.save(output / name, quality=92)
        evidence["classes"][label] = {
            "selected_events": len(rows),
            "review_rows": used,
            "sheet": name,
        }
    write_json(output / "review.json", evidence)
    lines = [
        f"# {manifest['video_id']} class/time pose review",
        "",
        "Original masks unchanged. Expected classes and frame mappings remain provisional. "
        "Flagged events are diagnostic only. Three source indices left-to-right: start/middle/end.",
        "",
    ]
    for label, values in evidence["classes"].items():
        lines.append(
            f"- [{label}]({values['sheet']})"
            if values["sheet"]
            else f"- {label}: no source sample; not invented"
        )
    (output / "review.md").write_text("\n".join(lines) + "\n")
    return evidence
