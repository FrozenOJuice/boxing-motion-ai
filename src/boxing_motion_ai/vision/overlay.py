"""Private review images; invalid landmarks never form confident skeleton edges."""

from .contracts import EDGES


def save_overlay(frame, record, destination, annotation_rows):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame.rgb)
    image.thumbnail((640, 360))
    width, height = image.size
    canvas = Image.new("RGB", (width, height + 74), "#101827")
    canvas.paste(image, (0, 74))
    draw = ImageDraw.Draw(canvas)
    draw.font = ImageFont.load_default(size=14)
    points, valid = record["landmarks"], record["valid_mask"]

    def xy(i):
        return points[i]["x"] * width, points[i]["y"] * height + 74

    for a, b in EDGES:
        if valid[a] and valid[b]:
            draw.line([xy(a), xy(b)], fill="#29ee9d", width=3)
    for i, point in enumerate(points):
        if point and all(point[k] is not None for k in ("x", "y")):
            if 0 <= point["x"] <= 1 and 0 <= point["y"] <= 1:
                x, y = xy(i)
                color = "#00d6ff" if valid[i] else "#ff665e"
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    candidates = [
        f"row {r['source_row']}:{r['source_class']}"
        for r in annotation_rows
        if r["candidate_start_index"] is not None
        and r["candidate_end_index"] is not None
        and r["candidate_start_index"] <= frame.index <= r["candidate_end_index"]
    ]
    draw.text(
        (8, 5),
        f"{record['video_id']} n={frame.index} PTS={record['timestamp_seconds']:.3f}s "
        f"{record['status']} body={record['valid_body_joint_count']}/22",
        fill="white",
    )
    draw.text(
        (8, 23),
        "Configured-index candidates (UNVERIFIED): "
        + (", ".join(candidates) if candidates else "none; action unknown"),
        fill="#f5cf84",
    )
    draw.text(
        (8, 42), "cyan=valid red=unreliable | green edges require both valid joints", fill="white"
    )
    canvas.save(destination, quality=92)


def contact_sheet(paths, output):
    from PIL import Image

    columns, w, h = 3, 480, 326
    sheet = Image.new("RGB", (columns * w, ((len(paths) + columns - 1) // columns) * h), "#101827")
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            image.thumbnail((w, h))
            sheet.paste(image, ((i % columns) * w, (i // columns) * h))
    sheet.save(output, quality=92)
