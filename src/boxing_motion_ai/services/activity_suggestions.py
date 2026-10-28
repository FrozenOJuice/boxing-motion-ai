"""Source-annotation proposals only. Gaps remain unknown until exhaustive human review."""

from fractions import Fraction
from uuid import NAMESPACE_URL, uuid5

from .review_workspace import encode, sha

VERSION = "source-activity-suggestions-1.0"


def propose(entry, rows, arrays):
    def time(index):
        return Fraction(
            int(arrays["pts"][index]) * int(arrays["time_base_num"][index]),
            int(arrays["time_base_den"][index]),
        )

    pieces, issues = [], []
    for row in rows:
        bounds = row.get("source_interval_half_open")
        if not (
            isinstance(bounds, list)
            and len(bounds) == 2
            and all(type(i) is int for i in bounds)
            and 0 <= bounds[0] < bounds[1] < len(arrays["pts"])
        ):
            issues.append(
                dict(source_row=row.get("source_row"), reason="invalid_provisional_interval")
            )
            continue
        start, stop = max(entry["start"], bounds[0]), min(entry["stop"], bounds[1])
        if start < stop:
            pieces.append(dict(start=start, stop=stop, source_annotation=row))
    groups = []
    for piece in sorted(pieces, key=lambda p: (p["start"], p["stop"])):
        if groups and time(piece["start"]) - time(groups[-1]["stop"]) < Fraction(1, 4):
            groups[-1]["stop"] = max(groups[-1]["stop"], piece["stop"])
            groups[-1]["members"].append(piece)
        else:
            groups.append(dict(start=piece["start"], stop=piece["stop"], members=[piece]))
    result = []
    for group in groups:
        origin = dict(
            schema_version=VERSION,
            interval_id=entry["id"],
            recording=entry["recording"],
            source_inputs=entry["source_inputs"],
            **group,
        )
        identifier = str(uuid5(NAMESPACE_URL, f"boxing-motion-ai:{sha(encode(origin))}"))
        result.append(
            origin
            | dict(
                event_id=identifier,
                kind="activity",
                source="automated",
                verified=False,
                timing=dict(
                    start=str(time(group["start"])), stop_exclusive=str(time(group["stop"]))
                ),
            )
        )
    return dict(
        schema_version=VERSION,
        suggestions=result,
        issues=issues,
        gaps="unknown_until_complete_human_review",
        detector_predictions_used=False,
    )
