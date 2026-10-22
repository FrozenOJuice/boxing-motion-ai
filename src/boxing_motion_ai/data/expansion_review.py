"""Read-only V7/V8 expansion gate. Hypotheses are never annotation corrections.

Run with ``python -m boxing_motion_ai.data.expansion_review --output reports/...``.
Only these two recordings are opened; pose inference and training are not invoked.
"""

import argparse
import json
from collections import Counter
from fractions import Fraction
from html import escape
from pathlib import Path

from ..vision.annotations import read_annotations
from ..vision.pipeline import sha256, write_json
from ..vision.recording import load_recording
from .alignment import nearest_index
from .preparation import GLOBAL_UNCERTAINTIES, SOURCE_LABELS

RECORDINGS = ("V7", "V8")
SOURCE_GROUP = "precision-striking-unresolved"


def summarize_rows(rows, frame_count, class_map):
    """Preserve flags, including shared endpoints and literal label whitespace."""
    inventory, lookup = [], {r["source_row"]: r for r in rows}
    pairs = sorted(
        {
            tuple(sorted((r["source_row"], n)))
            for r in rows
            for n in r.get("overlapping_source_rows", [])
        }
    )
    overlaps = []
    for a, b in pairs:
        left, right = lookup[a], lookup[b]
        span = min(left["raw_end"], right["raw_end"]) - max(left["raw_start"], right["raw_start"])
        overlaps.append(
            {
                "source_rows": [a, b],
                "end_minus_start_overlap": span,
                "kind": "shared_endpoint" if span == 0 else "strict_overlap",
            }
        )
    for row in rows:
        flags = sorted(set(row["flags"]) - GLOBAL_UNCERTAINTIES)
        a, b = row["candidate_start_index"], row["candidate_end_index"]
        if a is None or b is None or not 0 <= a <= b < frame_count:
            flags.append("invalid_or_out_of_video_bounds")
        label = class_map.get(row["source_class"])
        if label is None:
            flags.append("unmapped_literal_class")
        inventory.append(
            {
                **row,
                "punch_type_provisional": label,
                "structural_exclusion_reasons": sorted(set(flags)),
                "structurally_eligible": not flags,
                "alignment_review_status": "pending",
                "training_approved": False,
            }
        )
    counts = {}
    for label in SOURCE_LABELS:
        selected = [r for r in inventory if r["punch_type_provisional"] == label]
        counts[label] = {
            "source": len(selected),
            "structurally_eligible": sum(r["structurally_eligible"] for r in selected),
            "training_approved": 0,
        }
    return {
        "rows": inventory,
        "by_class": counts,
        "overlap_pairs": overlaps,
        "literal_class_counts": dict(Counter(r["source_class"] for r in rows)),
        "unmapped_rows": [
            r["source_row"] for r in inventory if r["punch_type_provisional"] is None
        ],
    }


def timing_hypotheses(row, times):
    """Nearest actual PTS, with explicit out-of-range status (never clipped as valid)."""
    a, b = int(row["raw_start"]), int(row["raw_end"])
    result = {}
    for name, start, end in [("raw_zero_based", a, b), ("raw_one_based", a - 1, b - 1)]:
        ok = 0 <= start <= end < len(times)
        result[name] = {
            "indices_inclusive_hypothesis": [start, end],
            "in_bounds": ok,
            "start_pts": str(times[start]) if ok else None,
        }
    for rate in (Fraction(30), Fraction(30000, 1001)):
        start, end = Fraction(a, 1) / rate, Fraction(b, 1) / rate
        ok = times[0] <= start <= end <= times[-1]
        indices = [nearest_index(times, start), nearest_index(times, end)] if ok else None
        result[f"annotation_clock_{rate}"] = {
            "indices_inclusive_hypothesis": indices,
            "in_bounds": ok,
            "target_start_seconds": str(start),
            "start_pts": str(times[indices[0]]) if ok else None,
        }
    return result


def select_review_rows(rows, class_map):
    """Early/middle/late of every available literal class, plus overlap examples."""
    selected = {}
    for label in SOURCE_LABELS:
        group = sorted(
            [r for r in rows if class_map.get(r["source_class"]) == label],
            key=lambda r: (r["raw_start"], r["source_row"]),
        )
        for i in sorted({0, len(group) // 2, len(group) - 1}) if group else []:
            selected[group[i]["source_row"]] = group[i]
    # Include a shared endpoint and a strict overlap, without choosing a winning label.
    for kind in ("shared", "strict"):
        for row in rows:
            others = row.get("overlapping_source_rows", [])
            lookup = {r["source_row"]: r for r in rows}
            match = [
                n
                for n in others
                if (
                    min(row["raw_end"], lookup[n]["raw_end"])
                    == max(row["raw_start"], lookup[n]["raw_start"])
                )
                == (kind == "shared")
            ]
            if match:
                for r in [row, lookup[match[0]]]:
                    selected[r["source_row"]] = r
                break
    # Fully nested labels cannot be resolved merely by changing end inclusion.
    nested = []
    lookup = {r["source_row"]: r for r in rows}
    for row in rows:
        for other_id in row.get("overlapping_source_rows", []):
            other = lookup[other_id]
            if other["raw_start"] <= row["raw_start"] <= row["raw_end"] <= other["raw_end"] and (
                row["raw_start"],
                row["raw_end"],
            ) != (other["raw_start"], other["raw_end"]):
                nested.append((row["raw_end"] - row["raw_start"], row["source_row"], other_id))
    if nested:
        _, a, b = min(nested)
        selected[a], selected[b] = lookup[a], lookup[b]
    return sorted(selected.values(), key=lambda r: r["source_row"])


def decode_review(video, times, anchors, output):
    """Seek as acceleration, identify frames only by matching exact inventory PTS."""
    import av
    from PIL import Image, ImageDraw, ImageFont

    required = set()
    for row in anchors:
        bounds = [
            h["indices_inclusive_hypothesis"] for h in row["hypotheses"].values() if h["in_bounds"]
        ]
        lo, hi = (
            max(0, min(b[0] for b in bounds) - 8),
            min(len(times), max(b[1] for b in bounds) + 9),
        )
        row["review_indices"] = list(range(lo, hi))
        required.update(row["review_indices"])
    images, previous, decoder = {}, None, None
    output.mkdir(parents=True)
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 4
        for index in sorted(required):
            target = times[index]
            if previous is None or target - previous > 1:
                container.seek(int(target / stream.time_base), stream=stream, backward=True)
                decoder = container.decode(stream)
            for frame in decoder:
                if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                    raise ValueError("Missing/corrupt source timing")
                actual = frame.pts * frame.time_base
                if actual < target:
                    continue
                if actual != target:
                    raise ValueError(f"PTS mismatch at decoded inventory index {index}")
                resized = frame.reformat(width=384, height=216)
                image = Image.fromarray(resized.to_ndarray(format="rgb24"))
                image.save(output / f"{index}.jpg", quality=88)
                images[index] = image
                previous = target
                break
            else:
                raise ValueError(f"EOF before expected frame {index}")
    font = ImageFont.load_default(size=17)
    for row in anchors:
        sheet = Image.new("RGB", (384 * 5, 280 * 2), "#101827")
        draw = ImageDraw.Draw(sheet)
        for line, key in enumerate(("raw_zero_based", "annotation_clock_30")):
            a, b = row["hypotheses"][key]["indices_inclusive_hypothesis"]
            indices = [a - 5, a, (a + b) // 2, b, b + 5]
            for col, n in enumerate(indices):
                if n in images:
                    sheet.paste(images[n], (384 * col, 280 * line + 60))
                draw.text(
                    (384 * col + 3, 280 * line + 27),
                    f"frame {n}, PTS {float(times[n]):.3f}s",
                    fill="white",
                    font=font,
                )
            draw.text(
                (3, 280 * line + 3),
                f"row {row['source_row']} {row['source_class']}: {key} [{a},{b}]",
                fill="#ffda80",
                font=font,
            )
        sheet.save(output.parent / f"row-{row['source_row']}.jpg", quality=91)
    return len(required)


def write_review_html(directory, anchors, times):
    payload = []
    for row in anchors:
        payload.append(
            {
                **row,
                "frame_times": [
                    {"index": i, "seconds": float(times[i]), "exact_pts_seconds": str(times[i])}
                    for i in row["review_indices"]
                ],
            }
        )
    data = json.dumps(payload).replace("<", "\\u003c")
    html = """<!doctype html><meta charset="utf-8"><title>Annotation alignment review</title>
<style>body{font:16px system-ui;max-width:1200px;margin:24px auto;background:#111827;color:#eee}
img{max-width:100%}section{border-top:1px solid #888;margin-top:24px}button{margin:4px}</style>
<h1>Annotation alignment hypotheses</h1><p>Expected labels are unverified. Neither strip is
approved for training. Frames are decoded from unchanged media and matched by exact PTS.
Start/end inclusion, FPS and zero/one indexing remain open. Arrow buttons step one actual frame.
Play uses source timestamp differences. No URL timestamp offset has been applied.</p><main></main>
<script>const rows=PAYLOAD;
for(const r of rows){let k=0,timer=null;const section=document.createElement('section');
section.innerHTML=`<h2>Source row ${r.source_row}: ${r.source_class}</h2>
<p>${r.flags.join(', ')}</p><img src="row-${r.source_row}.jpg">
<p>Top: raw decoded indices. Bottom: annotation seconds at 30 FPS.</p>
<img class="frame" width="768"><p class="caption"></p>
<button class="back">Previous frame</button><button class="play">Play / stop</button>
<button class="next">Next frame</button><input type="range" min="0"
max="${r.frame_times.length-1}" value="0" style="width:80%">`;
document.querySelector('main').append(section);const slider=section.querySelector('input');
function show(){const f=r.frame_times[k];
section.querySelector('.frame').src=`frames/${f.index}.jpg`;
const active=Object.entries(r.hypotheses).filter(([n,h])=>h.in_bounds&&
f.index>=h.indices_inclusive_hypothesis[0]&&f.index<=h.indices_inclusive_hypothesis[1])
.map(([n])=>n);
section.querySelector('.caption').textContent=`Frame ${f.index}; `+
`PTS ${f.exact_pts_seconds}s (${f.seconds.toFixed(3)}s); `+
`inside: ${active.join(', ')||'context'}`;slider.value=k;}
function stop(){clearTimeout(timer);timer=null;}
function step(d){stop();k=Math.max(0,Math.min(r.frame_times.length-1,k+d));show();}
function tick(){if(k===r.frame_times.length-1){stop();return;}
const dt=r.frame_times[k+1].seconds-r.frame_times[k].seconds;
timer=setTimeout(()=>{k++;show();tick();},dt*1000);}
section.querySelector('.back').onclick=()=>step(-1);section.querySelector('.next').onclick=()=>step(1);
section.querySelector('.play').onclick=()=>{if(timer!==null)stop();else{if(k===r.frame_times.length-1)k=0;show();tick();}};
slider.oninput=()=>{stop();k=Number(slider.value);show();};show();}
</script>""".replace("PAYLOAD", data)
    (directory / "review.html").write_text(html)


def run(output):
    output = Path(output)
    if output.exists() or any(p.name == "raw" for p in output.resolve().parents):
        raise ValueError("Use a new output directory outside raw data")
    raw = Path("data/raw/boxingvi")
    evidence = Path("reports/boxingvi/development")
    registry = json.loads((evidence / "source-manifest.json").read_text())
    report = {
        "schema_version": "expansion-gate-1.0",
        "training_ready": False,
        "source_group": SOURCE_GROUP,
        "boxer_id": None,
        "session_id": None,
        "v9_accessed": False,
        "recordings": {},
    }
    for v in RECORDINGS:
        source = next(r for r in registry["recordings"] if r["id"] == v)
        video, workbook = raw / source["media_path"], raw / source["workbook"]
        for path, expected in [
            (video, source["media_sha256"]),
            (workbook, source["workbook_sha256"]),
        ]:
            if sha256(path) != expected:
                raise ValueError(f"Source hash mismatch: {path}")
        config_path = Path(f"configs/{v.lower()}_recording.yaml")
        recording = load_recording(config_path)
        annotations = read_annotations(workbook, recording)
        probe_path = evidence / f"{v}.ffprobe.json"
        probe = json.loads(probe_path.read_text())
        times = [Fraction(f["pts_time"]) for f in probe["frames"]]
        if not times or any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("Missing/non-monotonic frame inventory")
        inventory = summarize_rows(annotations["rows"], len(times), recording.class_map)
        anchors = [
            {**r, "hypotheses": timing_hypotheses(r, times)}
            for r in select_review_rows(annotations["rows"], recording.class_map)
        ]
        print(f"{v}: decoding {len(anchors)} review intervals", flush=True)
        frames = decode_review(video, times, anchors, output / v / "frames")
        write_review_html(output / v, anchors, times)
        for row in anchors:
            row.pop("review_indices")
        stream = probe["streams"][0]
        result = {
            **inventory,
            "video_id": v,
            "source_url": source["original_url"],
            "source_group": SOURCE_GROUP,
            "boxer_id": None,
            "session_id": None,
            "video": {
                k: stream.get(k)
                for k in ["width", "height", "avg_frame_rate", "r_frame_rate", "time_base"]
            },
            "source_frame_count": len(times),
            "anchors": anchors,
            "review_frame_count": frames,
            "inputs": {
                "video": {"path": str(video), "sha256": sha256(video)},
                "workbook": {"path": str(workbook), "sha256": sha256(workbook)},
                "frame_inventory_sha256": sha256(probe_path),
                "recording_config_sha256": sha256(config_path),
            },
            "mapping_status": "unverified; all rows quarantined from expansion training",
        }
        report["recordings"][v] = result
        write_json(output / v / "inventory.json", result)
        print(f"{v}: {frames} exact-PTS review frames written", flush=True)
    write_json(output / "expansion.json", report)
    links = "\n".join(
        f'<li><a href="{escape(v)}/review.html">{escape(v)}</a></li>' for v in RECORDINGS
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>V7/V8 alignment review</h1>'
        "<p>Training gate pending. Original files unchanged.</p><ul>" + links + "</ul>"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
