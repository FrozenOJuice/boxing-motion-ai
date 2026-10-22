"""Read-only V7 overlap and V6/V10 expansion evidence; no pose inference or training."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

from ..vision.annotations import read_annotations
from ..vision.pipeline import sha256, write_json
from ..vision.recording import load_recording
from .alignment import nearest_index
from .expansion_review import summarize_rows

SOURCES = ("V7", "V6", "V10")
GROUPS = {
    "V7": "precision-striking-unresolved",
    "V8": "precision-striking-unresolved",
    "V6": "natebowerfitness-unresolved",
    "V10": "utah-lee-unresolved",
}


def overlap_relations(target, rows):
    """Describe interval geometry, never infer action correctness from geometry."""
    result = []
    for other in rows:
        if other["source_row"] == target["source_row"]:
            continue
        a, b, c, d = [
            r[k]
            for r, k in [
                (target, "raw_start"),
                (target, "raw_end"),
                (other, "raw_start"),
                (other, "raw_end"),
            ]
        ]
        if b < a or d < c or max(a, c) > min(b, d):
            continue
        span = min(b, d) - max(a, c)
        if span == 0:
            kind = "shared_endpoint"
        elif (a, b) == (c, d):
            kind = "identical_interval"
        elif a <= c <= d <= b or c <= a <= b <= d:
            kind = "nested_interval"
        else:
            kind = "partial_overlap"
        result.append(
            {
                "source_row": other["source_row"],
                "source_class": other["source_class"],
                "raw_bounds": [c, d],
                "kind": kind,
                "end_minus_start_overlap": span,
                "training_restored": False,
            }
        )
    return result


def planning_scenario(ceiling, fraction=0.8):
    """Explicit capacity scenario, NOT a measured yield or a statistical interval."""
    import math

    if type(ceiling) is not int or ceiling < 0 or not 0 <= fraction <= 1:
        raise ValueError("Nonnegative integer ceiling and probability-like scenario required")
    return {
        "structural_ceiling": ceiling,
        "capacity_scenario": math.floor(ceiling * fraction),
        "assumed_fraction": fraction,
        "note": "Conditional planning assumption; sample selection is not random. "
        "Full pose/alignment/phase review can yield fewer, including zero.",
    }


def quality_comparison():
    """Read existing policy evidence, verify hashes and retain sample/full distinctions."""
    import numpy as np

    from .pose_quality import POLICIES, score_validity

    result = {
        "policy": "mask_aware70",
        "definition": POLICIES["mask_aware70"],
        "new_pose_inference": False,
        "recordings": {},
    }
    for v in ["V4", "V5", "V7", "V6", "V10"]:
        full = v in ("V4", "V5", "V7")
        dataset = Path(f"data/processed/boxingvi/{v}") / (
            "candidates-full-v1" if full else "candidates-sample-expansion-v1"
        )
        qpath = (
            Path(f"reports/boxingvi/v7-v8-full-v1/quality/{v}/quality.json")
            if v == "V7"
            else (Path(f"reports/boxingvi/expansion/quality/{v}/quality.json"))
        )
        quality = json.loads(qpath.read_text())
        prep = json.loads((dataset / "preparation.json").read_text())
        if (
            quality["policies"]["mask_aware70"] != POLICIES["mask_aware70"]
            or quality["source_mask_threshold"] != 0.5
        ):
            raise ValueError("Source quality policy differs from approved thresholds")
        for name, digest in quality["inputs"].items():
            if sha256(dataset / name) != digest:
                raise ValueError("Stale pose-quality evidence")
        if not full:
            probe = json.loads(Path(f"reports/boxingvi/development/{v}.ffprobe.json").read_text())
            with np.load(dataset / "poses.npz", allow_pickle=False) as arrays:
                if not np.array_equal(
                    arrays["valid_mask"],
                    score_validity(arrays["coordinates"], arrays["confidence"], 0.5),
                ):
                    raise ValueError("Stored masks disagree with original confidence/coordinates")
                observed = [
                    Fraction(int(p) * int(n), int(d))
                    for p, n, d in zip(
                        arrays["pts"], arrays["time_base_num"], arrays["time_base_den"], strict=True
                    )
                ]
                expected = [
                    Fraction(probe["frames"][int(i)]["pts_time"]) for i in arrays["frame_index"]
                ]
                if observed != expected:
                    raise ValueError("Sample pose timing differs from source frame inventory")
        result["recordings"][v] = {
            "scope": quality["scope"],
            "extracted_frames": prep.get("extracted_frame_count", prep.get("source_frame_count")),
            "quality_path": str(qpath),
            "quality_sha256": sha256(qpath),
            "by_class": quality["by_class"],
            "boxer_id": prep.get("boxer_id"),
            "session_id": prep.get("session_id"),
            "note": "Full annotation-eligible events"
            if full
            else "Five ordinally selected events per class; not random or full extraction",
        }
    return result


def hypotheses(row, times, video_id):
    a, b = int(row["raw_start"]), int(row["raw_end"])
    if b < a:
        return {"reversed_bounds_diagnostic_only": [min(a, b), max(a, b)]}
    h = {"raw": [a, b], "one_based": [a - 1, b - 1]}
    for rate in [Fraction(24), Fraction(25), Fraction(30), Fraction(24000, 1001)]:
        targets = [Fraction(n, 1) / rate for n in [a, b]]
        if times[0] <= targets[0] <= targets[1] <= times[-1]:
            h[f"clock_{rate}"] = [nearest_index(times, t) for t in targets]
    if video_id == "V6":
        h["helper_minus_157"] = [a - 157, b - 157]
    return {k: pair for k, pair in h.items() if 0 <= pair[0] <= pair[1] < len(times)}


def decode_frames(video, times, requested, directory):
    """Short disjoint review windows; no decoding across huge hypothesis gaps."""
    import av
    from PIL import Image

    directory.mkdir(parents=True)
    previous, decoder = None, None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 4
        for index in sorted(requested):
            target = times[index]
            if previous is None or target - previous > 1:
                container.seek(int(target / stream.time_base), stream=stream, backward=True)
                decoder = container.decode(stream)
            for frame in decoder:
                if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                    raise ValueError("Corrupt/missing source timing")
                actual = frame.pts * frame.time_base
                if actual < target:
                    continue
                if actual != target:
                    raise ValueError(f"Exact PTS missing at source index {index}")
                image = Image.fromarray(
                    frame.reformat(width=384, height=216).to_ndarray(format="rgb24")
                )
                image.save(directory / f"{index}.jpg", quality=90)
                previous = target
                break
            else:
                raise ValueError("EOF before source review index")


def render_case(case, times, directory):
    from PIL import Image, ImageDraw, ImageFont

    frames = case["indices"]
    n = min(20, len(frames))
    positions = sorted({round(i * (len(frames) - 1) / (n - 1)) for i in range(n)})
    indices = [frames[p] for p in positions]
    sheet = Image.new("RGB", (384 * 5, 252 * ((len(indices) + 4) // 5) + 60), "#101827")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=16)
    draw.text(
        (5, 3),
        f"{case['id']} expected {case['source_class']} / {case['hypothesis']} / "
        "UNVERIFIED; no labels restored",
        fill="#ffda80",
        font=font,
    )
    draw.text((5, 27), "Source intervals: " + str(case["neighbor_bounds"]), fill="white", font=font)
    for k, index in enumerate(indices):
        x, y = (k % 5) * 384, 60 + (k // 5) * 252
        with Image.open(directory / "frames" / f"{index}.jpg") as image:
            sheet.paste(image, (x, y + 33))
        active = [str(r) for r, a, b in case["neighbor_bounds"] if a <= index <= b]
        draw.text(
            (x + 2, y + 3),
            f"f{index} {float(times[index]):.3f}s rows:{','.join(active)}",
            fill="white",
            font=font,
        )
    sheet.save(directory / f"{case['id']}.jpg", quality=92)


def write_player(cases, times, directory):
    payload = [
        {
            **c,
            "frames": [
                {"i": i, "time": str(times[i]), "seconds": float(times[i])} for i in c["indices"]
            ],
        }
        for c in cases
    ]
    data = json.dumps(payload).replace("<", "\\u003c")
    html = """<!doctype html><meta charset="utf-8"><title>Source expansion review</title>
<style>body{font:16px system-ui;background:#111827;color:white;max-width:1200px;margin:24px auto}
img{max-width:100%}section{border-top:1px solid #888;margin:24px 0}button{margin:4px}</style>
<h1>Source frame review</h1><p>All classes and boundaries provisional. No exclusions restored.
Scrub exact consecutive source frames. Play follows source timestamp differences.
Hypothesis windows are independent; no helper offset or rate correction is applied to data.</p>
<main></main><script>const cases=PAYLOAD;
for(const c of cases){let k=0,timer=null;const el=document.createElement('section');
el.innerHTML=`<h2>${c.id}: expected ${c.source_class}</h2><p>${c.hypothesis}</p>
<img src="${c.id}.jpg"><img class="frame" width="768"><p class="caption"></p>
<button class="prev">Previous</button><button class="play">Play / stop</button>
<button class="next">Next</button><input type="range" min="0"
max="${c.frames.length-1}" value="0" style="width:70%">`;
document.querySelector('main').append(el);const slider=el.querySelector('input');
function stop(){clearTimeout(timer);timer=null;}
function show(){const f=c.frames[k];el.querySelector('.frame').src=`frames/${f.i}.jpg`;
el.querySelector('.caption').textContent=`Frame ${f.i}; exact PTS ${f.time}s`;slider.value=k;}
function tick(){if(k===c.frames.length-1){stop();return;}
const dt=c.frames[k+1].seconds-c.frames[k].seconds;
timer=setTimeout(()=>{k++;show();tick();},dt*1000);}
function step(d){stop();k=Math.max(0,Math.min(c.frames.length-1,k+d));show();}
el.querySelector('.prev').onclick=()=>step(-1);el.querySelector('.next').onclick=()=>step(1);
el.querySelector('.play').onclick=()=>{if(timer!==null)stop();else{
if(k===c.frames.length-1)k=0;show();tick();}};
slider.oninput=()=>{stop();k=Number(slider.value);show();};show();}
</script>""".replace("PAYLOAD", data)
    (directory / "review.html").write_text(html)


def run(output):
    output = Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh reports directory")
    raw, evidence = Path("data/raw/boxingvi"), Path("reports/boxingvi/development")
    registry = json.loads((evidence / "source-manifest.json").read_text())
    report = {
        "schema_version": "source-priorities-review-1.0",
        "training_performed": False,
        "annotations_restored": [],
        "source_groups": GROUPS,
        "recordings": {},
    }
    for v in SOURCES:
        source = next(r for r in registry["recordings"] if r["id"] == v)
        video, workbook = raw / source["media_path"], raw / source["workbook"]
        if sha256(video) != source["media_sha256"] or sha256(workbook) != source["workbook_sha256"]:
            raise ValueError("Cached source evidence does not match current inputs")
        config_path = Path(f"configs/{v.lower()}_recording.yaml")
        config = load_recording(config_path)
        annotation = read_annotations(workbook, config)
        rows = annotation["rows"]
        lookup = {r["source_row"]: r for r in rows}
        probe_path = evidence / f"{v}.ffprobe.json"
        probe = json.loads(probe_path.read_text())
        times = [Fraction(f["pts_time"]) for f in probe["frames"]]
        if any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("Non-monotonic cached source timestamps")
        inventory = summarize_rows(rows, len(times), config.class_map)
        if v == "V7":
            selected = [r for r in rows if r["source_class"] == "Rear Uppercut"]
        else:
            plan = json.loads(
                Path(f"reports/boxingvi/expansion/plans/{v}/sample-plan.json").read_text()
            )
            ids = {r["source_row"] for r in plan["selected_rows"]}
            ids.update([3, 379] if v == "V6" else [123])
            if v == "V10":
                ids.update(
                    r["source_row"]
                    for r in rows
                    if r["source_class"] in ("Rear Hook", "Lead Uppercut")
                )
            selected = [lookup[n] for n in sorted(ids)]
        cases = []
        for row in selected:
            h = hypotheses(row, times, v)
            names = ["raw"] if "raw" in h else ["reversed_bounds_diagnostic_only"]
            compare = (v == "V6" and row["source_row"] in [2, 321, 686]) or (
                v == "V10" and row["source_row"] in [2, 120, 153]
            )
            if compare:
                names += [k for k in h if k in ("clock_24", "clock_30", "helper_minus_157")]
            relations = overlap_relations(row, rows)
            for name in names:
                a, b = h[name]
                neighbors = [[row["source_row"], row["raw_start"], row["raw_end"]]]
                if name == "raw":
                    neighbors += [[r["source_row"], *r["raw_bounds"]] for r in relations]
                    a, b = min(int(n[1]) for n in neighbors), max(int(n[2]) for n in neighbors)
                lo, hi = max(0, a - 8), min(len(times), b + 9)
                cases.append(
                    {
                        "id": f"{v}-row-{row['source_row']}-{name}",
                        "source_row": row["source_row"],
                        "source_class": row["source_class"],
                        "raw_bounds": [row["raw_start"], row["raw_end"]],
                        "hypothesis": name,
                        "hypothesis_bounds": h[name],
                        "indices": list(range(lo, hi)),
                        "neighbor_bounds": neighbors,
                        "overlap_relations": relations,
                        "training_restored": False,
                    }
                )
        requested = {i for case in cases for i in case["indices"]}
        print(
            f"{v}: decoding {len(requested)} frames in {len(cases)} short review windows",
            flush=True,
        )
        directory = output / v
        decode_frames(video, times, requested, directory / "frames")
        for case in cases:
            render_case(case, times, directory)
        write_player(cases, times, directory)
        info = json.loads((raw / "RGB_videos" / f"{v}.info.json").read_text())
        report["recordings"][v] = {
            **inventory,
            "source_group_id": GROUPS[v],
            "boxer_id": None,
            "session_id": None,
            "source_url": source["original_url"],
            "channel": info.get("channel"),
            "channel_id": info.get("channel_id"),
            "source_title": info.get("title"),
            "source_frame_count": len(times),
            "video": probe["streams"][0],
            "review_cases": [{k: val for k, val in c.items() if k != "indices"} for c in cases],
            "capacity_scenario": {
                c: planning_scenario(d["structurally_eligible"])
                for c, d in inventory["by_class"].items()
            },
            "auxiliary_only_rows": annotation["auxiliary_only_rows"],
            "inputs": {
                "video_path": str(video),
                "video_sha256": source["media_sha256"],
                "workbook_path": str(workbook),
                "workbook_sha256": source["workbook_sha256"],
                "frame_inventory_sha256": sha256(probe_path),
                "recording_config_sha256": sha256(config_path),
            },
        }
        write_json(directory / "inventory.json", report["recordings"][v])
    write_json(output / "review.json", report)
    write_json(output / "pose-comparison.json", quality_comparison())
    links = "".join(f'<li><a href="{v}/review.html">{v}</a></li>' for v in SOURCES)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>V7/V6/V10 source review</h1><ul>'
        + links
        + "</ul>"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
