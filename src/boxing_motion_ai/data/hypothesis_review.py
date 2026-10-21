"""Exact source-frame comparison of declared timing hypotheses; no pose inference."""

import json
from fractions import Fraction
from pathlib import Path

from ..vision.pipeline import sha256, write_json
from ..vision.sparse_decoder import decode_selected
from .alignment import nearest_index


def render_hypotheses(root, evidence, diagnostics, video_id, rows, output, decoder=decode_selected):
    from PIL import Image, ImageDraw, ImageFont

    root, evidence, output = map(Path, (root, evidence, output))
    if output.exists() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Use new visual output outside source inputs")
    diag = json.loads(Path(diagnostics).read_text())["recordings"][video_id]
    source = root / "RGB_videos" / f"{video_id}.mkv"
    if sha256(source) != diag["inputs"]["video_sha256"]:
        raise ValueError("Video does not match the timing evidence")
    times = [
        Fraction(f["pts_time"])
        for f in json.loads((evidence / f"{video_id}.ffprobe.json").read_text())["frames"]
    ]
    anchors = {r["source_row"]: r for r in diag["anchors"]}
    cases = []
    for row in rows:
        anchor = anchors[row]
        a, b = map(int, anchor["raw_bounds"])
        for hypothesis, rate in [
            ("raw decoded indices", None),
            ("time at exactly 24 FPS", Fraction(24)),
            ("time at exactly 30 FPS", Fraction(30)),
        ]:
            indices = [a, (a + b) // 2, b]
            if rate is not None:
                targets = [Fraction(n, 1) / rate for n in indices]
                if any(not times[0] <= t <= times[-1] for t in targets):
                    raise ValueError("Requested timing hypothesis extends beyond source")
                indices = [nearest_index(times, t) for t in targets]
            if any(not 0 <= n < len(times) for n in indices):
                raise ValueError("Requested raw hypothesis extends beyond source")
            cases.append(
                {
                    "source_row": row,
                    "expected_class": anchor["source_class"],
                    "hypothesis": hypothesis,
                    "indices": indices,
                }
            )
    wanted = sorted({n for c in cases for n in c["indices"]})
    ranges = []
    for n in wanted:
        if ranges and ranges[-1][1] == n:
            ranges[-1][1] += 1
        else:
            ranges.append([n, n + 1])
    output.mkdir(parents=True)
    frames = output / "frames"
    frames.mkdir()
    observed = []
    font = ImageFont.load_default(size=17)
    iterator = decoder(source, 960, ranges=ranges)
    try:
        for frame in iterator:
            if frame.rgb is None:
                continue
            if frame.time != times[frame.index]:
                raise ValueError("Decoded frame PTS differs from cached timing inventory")
            image = Image.fromarray(frame.rgb)
            image.thumbnail((640, 360))
            canvas = Image.new("RGB", (640, 400), "#101827")
            canvas.paste(image, (0, 40))
            ImageDraw.Draw(canvas).text(
                (8, 8),
                f"{video_id} source n={frame.index} PTS {float(frame.time):.3f}s",
                font=font,
                fill="white",
            )
            canvas.save(frames / f"{frame.index}.jpg", quality=92)
            observed.append(
                {"frame_index": frame.index, "pts": frame.pts, "time_base": str(frame.time_base)}
            )
    finally:
        iterator.close()
    if {r["frame_index"] for r in observed} != set(wanted):
        raise ValueError("Not all hypothesis frames were decoded")
    for row in rows:
        selected = [c for c in cases if c["source_row"] == row]
        sheet = Image.new("RGB", (1920, 3 * 450), "#101827")
        draw = ImageDraw.Draw(sheet)
        for line, case in enumerate(selected):
            draw.text(
                (8, line * 450 + 8),
                f"row {row} expected {case['expected_class']} / {case['hypothesis']} / UNVERIFIED",
                font=font,
                fill="#f5ce80",
            )
            for col, n in enumerate(case["indices"]):
                with Image.open(frames / f"{n}.jpg") as image:
                    sheet.paste(image, (640 * col, line * 450 + 42))
        sheet.save(output / f"row-{row}.jpg", quality=93)
    result = {
        "video_id": video_id,
        "video_sha256": sha256(source),
        "cases": cases,
        "decoded_frames": observed,
        "limitation": "Timing alternatives only; no verified mapping or automatic correction",
    }
    write_json(output / "hypotheses.json", result)
    (output / "review.md").write_text(
        f"# {video_id} timing hypothesis review\n\n" + "No mapping is certified by these stills. "
        "Each sheet compares raw, 24-FPS time, and 30-FPS time.\n\n"
        + "\n".join(f"- [Source row {n}](row-{n}.jpg)" for n in rows)
        + "\n"
    )
    return result
