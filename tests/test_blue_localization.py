"""Selective human labels, Unknown exclusions and development isolation."""

from copy import deepcopy
from fractions import Fraction as Q

import numpy as np
import pytest

from boxing_motion_ai.data.blue_localization import select, targets
from boxing_motion_ai.services.artifacts import ServiceError


def fixture():
    entry = dict(recording="V6", role="train_candidate", source_group_guard=["V1", "V6"], id="test")
    event = dict(event_id="blue", start=10, stop=20, kind="punch", punch_type="cross")
    other = dict(event_id="excluded", start=19, stop=24, kind="punch", punch_type="jab")
    state = dict(
        complete=True,
        head="h",
        events=[event, other],
        history=[
            dict(
                source="manual",
                version=1,
                sha256="a",
                payload=dict(operation="upsert", event=event, inspected_frames=[10, 19]),
            ),
            dict(source="manual", version=2, sha256="b", payload=dict(operation="complete")),
        ],
    )
    # Type disagreement does not substitute the provisional type for the human event.
    sources = [
        dict(source_row=1, source_interval_half_open=[10, 20], punch_type_provisional="rear_hook")
    ]
    return entry, state, sources


def test_unique_exact_human_association_not_source_promotion_or_nearest_match():
    e, s, rows = fixture()
    before = deepcopy(s)
    i = select(e, s, rows)
    assert len(i["events"]) == 1 and len(i["excluded_events"]) == 1
    assert i["events"][0]["punch_type"] == "cross"
    assert not i["events"][0]["reference_eligible"]
    assert s == before
    for changed in [
        s | dict(events=[]),
        s | dict(events=[s["events"][0]] * 2),
        s | dict(complete=False),
    ]:
        with pytest.raises(ServiceError):
            select(e, changed, rows)
    shifted = deepcopy(s)
    shifted["events"][0]["start"] = 11
    with pytest.raises(ServiceError):
        select(e, shifted, rows)


def test_endpoint_evidence_and_roles_are_mandatory():
    e, s, rows = fixture()
    s["history"][0]["payload"]["inspected_frames"] = [10]
    with pytest.raises(ServiceError):
        select(e, s, rows)
    for v in ("V7", "V8", "V9", "V10"):
        with pytest.raises(ServiceError):
            select(e | dict(recording=v), s, rows)
    with pytest.raises(ServiceError):
        select(e | dict(recording="V5"), s, rows)


def test_sparse_unknown_and_conflicts_never_become_negative_labels():
    e, s, rows = fixture()
    interval = select(e, s, rows)
    video = dict(frames=[dict(timestamp_exact=str(Q(i, 25))) for i in range(30)])
    y, mask, censor = targets(interval, video, 0, Q(1, 12))
    assert not mask[:10].any() and not mask[19:].any()
    assert (y[10:19, 0] == 1).all() and mask[10:19, 0].all()
    assert not mask[17:24, 1:].any()  # Boundary support near the excluded motion is Unknown.
    assert censor[0]["onset_available"] and not censor[0]["end_available"]
    assert not censor[0]["primary_reference_eligible"]
    assert y[19, 2] == 0
    # Adding unrelated excluded motion entirely outside this cache must not mask the cache.
    interval["excluded_events"].append(dict(start=-10, stop=-5))
    a, b, _ = targets(interval, video, 0, Q(1, 12))
    np.testing.assert_array_equal(a, y)
    np.testing.assert_array_equal(b, mask)


def test_development_is_accepted_only_with_its_role_and_guard():
    e, s, rows = fixture()
    i = select(e | dict(recording="V5", role="development", source_group_guard=["V5"]), s, rows)
    assert i["provenance"]["role"] == "development" and not i["exhaustive"]


def test_v5_input_and_targets_cannot_enter_training_sequences():
    from boxing_motion_ai.detection.localization_model import sequences

    def case(train):
        return dict(
            receipt=dict(train_eligible=train),
            arrays=dict(
                pose=np.ones((4, 65), np.float32),
                visual=np.ones((4, 97), np.float32),
                observable=np.ones(4, bool),
                reset=np.array([True, False, False, False]),
                targets=np.ones((4, 3), np.float32),
                supervision_mask=np.ones((4, 3), bool),
            ),
        )

    cases = dict(training=case(True), V5=case(False))
    a = sequences(cases, "fused", training=True)
    cases["V5"]["arrays"]["pose"][:] = 1000
    cases["V5"]["arrays"]["targets"][:] = 0
    b = sequences(cases, "fused", training=True)
    assert [s["case"] for s in b] == ["training"]
    np.testing.assert_array_equal(a[0]["x"], b[0]["x"])
    np.testing.assert_array_equal(a[0]["y"], b[0]["y"])


def test_partial_matching_penalizes_broad_predictions_and_never_calls_unknown_false_positive():
    from boxing_motion_ai.detection.localization_blue_experiment import evaluate

    frames = [dict(timestamp_exact=str(Q(i, 25))) for i in range(50)]
    case = dict(
        provenance=dict(
            kind="reviewed_source",
            exhaustive=False,
            source_start=0,
            review_start=10,
            review_stop=20,
            selected_events=1,
            scorable_events=1,
        ),
        video=dict(frames=frames, stop_timestamp_exact="2"),
        references=[dict(start_frame=10, stop_frame_exclusive=20, event_id="x", type="jab")],
    )
    result = dict(
        candidates=[
            dict(start_frame=1, stop_frame_exclusive=40, peak_frame=15),
            dict(start_frame=41, stop_frame_exclusive=45, peak_frame=43),
        ]
    )
    c = evaluate(result, case)
    assert c["matches_iou_half"] == [] and c["missed_references"] == ["x"]
    assert c["false_positive_count"] is None and c["precision"] is None
    assert c["unmatched_unlabeled_candidates"] == [1]
    assert "candidate_without_reference_overlap" not in c


def test_v5_preprocessing_masks_counter_without_affecting_unmasked_motion():
    from PIL import Image, ImageDraw

    from boxing_motion_ai.detection.localization_blue_features import development_redacted

    first = Image.new("RGB", (200, 200), "black")
    second = first.copy()
    ImageDraw.Draw(second).rectangle((170, 150, 199, 199), fill="white")
    a, ma = development_redacted(first, (0, 0, 200, 200), [])
    b, mb = development_redacted(second, (0, 0, 200, 200), [])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ma, mb)
    assert not ma[130:, 145:].any()


def test_v5_identifier_masks_are_fixed_not_annotation_dependent():
    from PIL import Image, ImageDraw

    from boxing_motion_ai.detection.localization_blue_features import development_redacted

    first = Image.new("RGB", (200, 200), "black")
    second = first.copy()
    draw = ImageDraw.Draw(second)
    draw.rectangle((0, 0, 199, 30), fill="white")  # Flags/header lettering.
    draw.rectangle((2, 48, 55, 80), fill="white")  # Gym sign.
    draw.rectangle((115, 10, 195, 40), fill="white")  # Wall posters.
    a, ma = development_redacted(first, (0, 0, 200, 200), [])
    b, mb = development_redacted(second, (0, 0, 200, 200), [])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ma, mb)
