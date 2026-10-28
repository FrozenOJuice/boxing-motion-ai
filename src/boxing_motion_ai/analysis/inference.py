"""Annotation-free adapter over the unchanged Level 1 features and temporal model."""

from fractions import Fraction
from statistics import median
from time import perf_counter

import numpy as np
import torch

from ..data.pose_quality import POLICIES, policy_metrics, retain
from ..training.data import CLASSES, exact_times, features, validate_arrays
from ..training.models import CausalTCN
from ..training.velocity import VELOCITY_VERSION, joint_velocities
from .common import pin


class Classifier:
    def __init__(self, root):
        self.pin = pin(root)
        p = self.pin
        saved = torch.load(root / p["checkpoint"]["path"], map_location="cpu", weights_only=True)
        provenance = p["provenance"]["sha256"]
        config = saved["config"]
        if not (
            saved["feature_version"] == p["feature_version"] == VELOCITY_VERSION
            and p["channels"] == 65
            and tuple(p["classes"]) == CLASSES
            and saved["seed"] == p["seed"] == 20261021
            and saved["epoch"] == p["epoch"] == 32
            and saved["run"] == "tcn"
            and saved["provenance_sha256"] == provenance
            and config["train"]["recording"] == "V4"
            and config["development"]["recording"] == "V5"
            and config["feature_representation"] == "xy_velocity"
            and config["hidden"] == 32
        ):
            raise ValueError("The pinned checkpoint has an incompatible feature/provenance schema.")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        self.model = CausalTCN(65, hidden=32, dropout=config["dropout"])
        self.model.load_state_dict(saved["model_state"], strict=True)
        self.model.eval()

    def classify(self, arrays, aspect, start, stop, *, gap=None):
        began = perf_counter()
        validate_arrays(arrays)
        if (
            type(start) is not int
            or type(stop) is not int
            or not (0 <= start < stop <= len(arrays["frame_index"]) and stop - start >= 2)
        ):
            return dict(status="unavailable", reason="Select at least two frames of one punch.")
        times = exact_times({"raw": arrays})
        gap = gap or 2 * median(b - a for a, b in zip(times, times[1:], strict=False))
        a = {key: value[start:stop].copy() for key, value in arrays.items()}
        duration = times[stop - 1] - times[start]
        if duration > 3:
            return dict(status="unavailable", reason="Select one punch spanning at most 3 seconds.")
        if any(len(np.unique(a[key])) != 1 for key in ("scene_id", "segment_id")) or any(
            b - t > gap
            for t, b in zip(times[start : stop - 1], times[start + 1 : stop], strict=True)
        ):
            return dict(status="unavailable", reason="The selection crosses a video timing gap.")
        quality = policy_metrics(a["valid_mask"], a["pose_status"] == 2)
        if not retain(a["valid_mask"], a["pose_status"] == 2, POLICIES["mask_aware70"]):
            return dict(
                status="unavailable",
                reason="Not enough reliable torso and arm poses. Try a clearer, shorter selection.",
                pose_coverage=quality,
            )
        x, _, _ = features(a["coordinates"], a["confidence"], a["valid_mask"], aspect)
        if float((x[:, 40] == 1).mean()) < 0.95:
            return dict(
                status="unavailable",
                reason="The torso pose cannot be normalized reliably.",
                pose_coverage=quality,
            )
        x = torch.from_numpy(x)
        velocity = joint_velocities(x, a, gap)
        x = torch.cat((x, velocity.reshape(len(x), 24)), dim=1)
        with torch.inference_mode():
            scores = self.model(x[None], torch.ones((1, len(x)), dtype=torch.bool)).softmax(-1)[0]
        if not torch.isfinite(scores).all():
            raise ValueError("The model returned invalid scores.")
        return dict(
            status="available",
            source="model_prediction",
            prediction=CLASSES[int(scores.argmax())],
            scores=dict(zip(CLASSES, scores.tolist(), strict=True)),
            score_semantics="uncalibrated_softmax_scores_not_verified_correctness",
            pose_coverage=quality,
            inference_seconds=perf_counter() - began,
            checkpoint_sha256=self.pin["checkpoint"]["sha256"],
            feature_version=VELOCITY_VERSION,
            start_frame=start,
            stop_frame_exclusive=stop,
            start_timestamp_exact=str(times[start]),
            last_timestamp_exact=str(times[stop - 1]),
            velocity_gap_seconds_exact=str(Fraction(gap)),
        )
