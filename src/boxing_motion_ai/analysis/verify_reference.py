"""Read-only V5 adapter regression against the pinned checkpoint's original predictions."""

import json
from pathlib import Path

import numpy as np
import torch

from ..training.data import PoseClips
from ..training.velocity import source_gap_limit
from .common import sha, write_json
from .inference import Classifier


def verify(root):
    model = Classifier(root)
    checkpoint = root / model.pin["checkpoint"]["path"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    spec = saved["config"]["development"]
    assert spec["recording"] == "V5"
    dataset = PoseClips(
        {k: str(root / v) if k != "recording" else v for k, v in spec.items()}, "V5"
    )
    directory = root / spec["dataset"]
    with np.load(directory / "poses.npz", allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    with (root / spec["pose_run"] / "poses.jsonl").open() as stream:
        first = json.loads(next(stream))
    aspect = first["source_width"] / first["source_height"]
    gap = source_gap_limit(directory)
    predictions_file = checkpoint.parent / "development_predictions.json"
    predictions = {p["source_row"]: p for p in json.loads(predictions_file.read_text())}
    maximum = 0.0
    matched = 0
    repeated = False
    for clip in dataset.clips:
        start, stop = clip["annotation"]["array_slice_half_open"]
        result = model.classify(raw, aspect, start, stop, gap=gap)
        if result["status"] != "available":
            raise ValueError(f"Reference selection became unavailable: {clip['source_row']}")
        expected = predictions[clip["source_row"]]
        values = list(result["scores"].values())
        error = float(np.max(np.abs(np.array(values) - expected["probabilities"])))
        maximum = max(maximum, error)
        if result["prediction"] != expected["prediction"] or error > 1e-6:
            raise ValueError(f"Adapter failed reference replay: {clip['source_row']}, {error}")
        matched += 1
        if not repeated:
            repeated = (
                model.classify(raw, aspect, start, stop, gap=gap)["scores"] == result["scores"]
            )
            if not repeated:
                raise ValueError("Repeated inference was not identical")
    return dict(
        scope="V5 existing development positives; integration regression, "
        "not new generalization evidence",
        matched=matched,
        expected=len(predictions),
        max_score_difference=maximum,
        score_tolerance=1e-6,
        repeated_scores_exactly_equal=repeated,
        checkpoint_sha256=sha(checkpoint),
        predictions_sha256=sha(predictions_file),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new verification output file")
    result = verify(Path.cwd())
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
