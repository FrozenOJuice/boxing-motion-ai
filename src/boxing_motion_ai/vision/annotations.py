"""V5 source provenance only. Deliberately not a verified canonical Event mapping."""

from openpyxl import load_workbook

from ..boxingvi import integer_cell


def read_v5_annotations(path):
    book = load_workbook(path, read_only=True, data_only=False)
    try:
        if book.sheetnames != ["Sheet1"]:
            raise ValueError("Expected inspected V5 layout: one Sheet1; review changed workbook")
        rows = []
        for n, cells in enumerate(book["Sheet1"].iter_rows(values_only=True), 1):
            if n == 1:
                if any(v is not None for v in cells):
                    raise ValueError("Expected blank V5 first row; inspect changed workbook layout")
                continue
            if not any(v is not None for v in cells):
                continue
            values = list(cells) + [None] * 5
            start, end, label, helper_start, helper_end = values[:5]
            flags = ["provisional_mapping", "class_unverified", "boundaries_unverified"]
            start_index = end_index = None
            try:
                a, b = integer_cell(start), integer_cell(end)
                numeric = a >= 0 and b >= 0
                if numeric:
                    start_index, end_index = a, b
            except ValueError:
                numeric = False
            if not numeric:
                flags.append("invalid_frame_values")
            elif end < start:
                flags.append("reversed_bounds")
            elif end == start:
                flags.append("equal_bounds_end_convention_unverified")
            if not isinstance(label, str):
                flags.append("invalid_source_class")
            elif label not in {
                "Jab",
                "Cross",
                "Lead Hook",
                "Rear Hook",
                "Lead Uppercut",
                "Rear Uppercut",
                "Jab ",
            }:
                flags.append("unrecognized_source_class")
            if isinstance(label, str) and label != label.strip():
                flags.append("source_class_whitespace")
            if helper_start != f"=A{n}-24" or helper_end != f"=B{n}-24":
                flags.append("unexpected_helper_formula")
            rows.append(
                {
                    "source_row": n,
                    "raw_start": start,
                    "raw_end": end,
                    "source_class": label,
                    "candidate_start_index": start_index,
                    "candidate_end_index": end_index,
                    "helper_D": helper_start,
                    "helper_E": helper_end,
                    "flags": flags,
                }
            )
        valid = [r for r in rows if "invalid_frame_values" not in r["flags"]]
        previous = None
        seen = set()
        for row in valid:
            start, end = row["raw_start"], row["raw_end"]
            if previous is not None and start < previous:
                row["flags"].append("out_of_row_order")
            previous = start
            key = (start, end, row["source_class"])
            if key in seen:
                row["flags"].append("duplicate")
            seen.add(key)
            if end >= start:
                # Inclusive endpoint comparison is deliberately conservative for review only.
                overlap = [
                    other["source_row"]
                    for other in valid
                    if other is not row
                    and other["raw_end"] >= other["raw_start"]
                    and max(start, other["raw_start"]) <= min(end, other["raw_end"])
                ]
                if overlap:
                    row["flags"].append("suspect_overlap_inclusive_candidate")
                    row["overlapping_source_rows"] = overlap
        return {
            "schema_version": "v5-provisional-1.0",
            "video_id": "V5",
            "sheet": "Sheet1",
            "mapping_status": "provisional_user_authorized",
            "display_candidate": "raw A/B -> zero-based decoded indices; no -24 subtraction",
            "verified_author_fps": None,
            "verified_frame_base": None,
            "verified_end_convention": None,
            "verified_trim_offset": None,
            "boxer_id": None,
            "session_id": None,
            "canonical_class_map": None,
            "unlabeled_intervals": "unknown; never automatically no_punch",
            "rows": rows,
        }
    finally:
        book.close()
