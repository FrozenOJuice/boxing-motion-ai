"""Exact-time synthetic decoder tests; no fitting or real human decisions."""

from copy import deepcopy
from fractions import Fraction as Q

import pytest

from boxing_motion_ai.detection.decoder import (
    CausalEventDecoder,
    DecoderConfig,
    Observation,
    decode_batch,
)


def frame(i, p=0.9, fps=25, **kw):
    return Observation(
        recording=kw.pop("recording", "V4"),
        interval_id=kw.pop("interval_id", "test"),
        frame_index=i,
        pts=kw.pop("pts", i),
        time_base_num=1,
        time_base_den=fps,
        probability=p,
        **kw,
    )


def feed(values, config=None, fps=25):
    d = CausalEventDecoder(config)
    for i, p in enumerate(values):
        d.push(frame(i, p, fps))
    d.finish(stop_frame=len(values), stop_time=Q(len(values), fps))
    return d


def test_activation_onset_emission_and_delayed_termination_are_distinct():
    d = feed([0.9] * 5 + [0.1] * 4)
    e = d.events[0]
    assert e["estimated_start"]["frame_index"] == 0
    assert e["first_emission"]["frame_index"] == 2
    assert e["activation_delay_seconds_exact"] == "2/25"
    assert e["estimated_end_inclusive"]["frame_index"] == 4
    assert e["stop_frame_exclusive"] == 5 and e["stop_timestamp_exact"] == "1/5"
    assert e["finalized_at"]["observed_through_frame"] == 7
    assert e["finalized_at"]["timestamp_exact"] == "7/25"
    assert e["right_censored"] is False and e["left_censored"] is True
    assert e["confidence"]["mean_probability"] == pytest.approx(0.9)
    assert e["confidence"]["calibrated"] is False


def test_brief_high_fluctuations_never_become_events_or_background_certificates():
    d = feed([0.1, 0.8, 0.1, 0.8, 0.1])
    assert d.events == [] and d.updates == [] and d.discarded_candidates == 2


def test_hysteresis_band_and_short_observed_dip_are_bridged():
    values = [0.9, 0.5, 0.4, 0.6, 0.2, 0.4, 0.5, 0.1, 0.1, 0.1]
    e = feed(values).events[0]
    assert e["first_emission"]["frame_index"] == 2
    assert e["stop_frame_exclusive"] == 7
    assert e["confidence"]["observed_samples"] == 7
    assert e["confidence"]["mean_probability"] == pytest.approx(sum(values[:7]) / 7)


def test_distinct_punches_require_a_sufficient_observed_low_valley():
    assert len(feed([0.1] + [0.9] * 4 + [0.1] + [0.9] * 4 + [0.1] * 3).events) == 1
    d = feed([0.1] + [0.9] * 4 + [0.1] * 3 + [0.9] * 4 + [0.1] * 3)
    assert len(d.events) == 2
    assert d.events[0]["stop_frame_exclusive"] == 5
    assert d.events[1]["estimated_start"]["frame_index"] == 8
    assert d.events[0]["left_censored"] is False


@pytest.mark.parametrize("fps", [24, 25, 30, 60])
def test_durations_use_elapsed_source_time_not_fixed_frame_counts(fps):
    values = [0.9 if Q(1, 10) <= Q(i, fps) < Q(2, 5) else 0.1 for i in range(fps)]
    e = feed(values, fps=fps).events[0]
    onset = Q(e["estimated_start"]["timestamp_exact"])
    emit = Q(e["first_emission"]["timestamp_exact"])
    stop = Q(e["stop_timestamp_exact"])
    final = Q(e["finalized_at"]["timestamp_exact"])
    assert Q(2, 25) <= emit - onset < Q(2, 25) + Q(1, fps)
    assert Q(2, 25) <= final - stop < Q(2, 25) + Q(1, fps)


def test_activation_prefix_is_emitted_immediately_and_never_rewritten_by_future():
    d = CausalEventDecoder()
    assert d.push(frame(0)) == [] and d.push(frame(1)) == []
    messages = d.push(frame(2))
    saved = deepcopy(messages)
    assert messages[0]["kind"] == "activated"
    assert "estimated_end_inclusive" not in messages[0]
    for i in range(3, 8):
        d.push(frame(i, 1.0))
    d.finish(stop_frame=8, stop_time=Q(8, 25))
    assert messages == saved and d.updates[0] == saved[0]
    assert d.events[0]["first_emission"] == saved[0]["first_emission"]


@pytest.mark.parametrize(
    "kind",
    [
        "pose",
        "unknown",
        "cut",
        "scene",
        "segment",
        "frame_gap",
        "time_gap",
        "recording",
        "interval",
    ],
)
def test_discontinuities_interrupt_instead_of_silently_joining(kind):
    d = CausalEventDecoder()
    for i in range(4):
        d.push(frame(i))
    if kind == "pose":
        d.push(frame(4, None, observable=False))
    elif kind == "unknown":
        d.push(frame(4, 0.99, reviewed=False))
    for i in range(4 if kind not in ("pose", "unknown") else 5, 9):
        kw = {}
        index = i
        if kind == "cut":
            kw["discontinuity_before"] = i == 4
        elif kind == "scene":
            kw["scene_id"] = 1
        elif kind == "segment":
            kw["segment_id"] = 1
        elif kind == "frame_gap":
            index = i + 6
        elif kind == "time_gap":
            kw["pts"] = i + 50
        elif kind == "recording":
            kw["recording"] = "V6"
        elif kind == "interval":
            kw["interval_id"] = "second"
        d.push(frame(index, **kw))
    d.finish()
    assert len(d.events) == 2
    first, second = d.events
    assert first["right_censored"] is True and second["left_censored"] is True
    assert first["estimated_end_inclusive"]["frame_index"] == 3
    if kind in ("frame_gap", "time_gap", "recording", "interval"):
        assert first["stop_timestamp_exact"] is None


def test_end_of_interval_and_unconfirmed_edge_cases_are_explicit():
    pending = feed([0.9, 0.9])
    assert pending.events == [] and pending.discarded_candidates == 1
    d = feed([0.1, 0.9, 0.9, 0.9])
    e = d.events[0]
    assert e["left_censored"] is False and e["right_censored"] is True
    assert e["estimated_end_inclusive"]["frame_index"] == 3
    assert e["stop_frame_exclusive"] == 4 and e["stop_timestamp_exact"] == "4/25"
    assert e["finalized_at"]["timestamp_exact"] == "4/25"
    assert e["finalized_at"]["observed_through_frame"] == 3
    assert e["finalized_at"]["trigger"] == "interval_end"


def test_pending_low_tail_is_not_misreported_as_a_verified_end_at_eof():
    e = feed([0.9, 0.9, 0.9, 0.1]).events[0]
    assert e["right_censored"] is True
    assert e["estimated_end_inclusive"]["frame_index"] == 2
    assert e["stop_timestamp_exact"] == "3/25"
    assert e["finalized_at"]["timestamp_exact"] == "4/25"


def test_zero_durations_and_configurable_thresholds():
    cfg = DecoderConfig(
        activation_threshold=0.8,
        deactivation_threshold=0.2,
        minimum_activation_seconds="0",
        termination_seconds="0",
    )
    d = feed([0.7, 0.9, 0.1, 0.9, 0.1], cfg)
    assert len(d.events) == 2
    assert d.events[0]["estimated_start"]["frame_index"] == 1
    assert d.events[0]["first_emission"]["frame_index"] == 1
    assert d.events[0]["estimated_end_inclusive"]["frame_index"] == 1


@pytest.mark.parametrize(
    "kw",
    [
        dict(activation_threshold=0.2, deactivation_threshold=0.4),
        dict(minimum_activation_seconds="-1"),
        dict(termination_seconds="1/0"),
        dict(minimum_activation_seconds="NaN"),
        dict(maximum_timestamp_gap_seconds="0"),
    ],
)
def test_invalid_policy_is_rejected(kw):
    with pytest.raises(ValueError):
        DecoderConfig(**kw)


def test_unknown_inputs_are_not_activated_and_invalid_or_repeated_frames_are_rejected():
    d = CausalEventDecoder()
    for i in range(10):
        d.push(frame(i, 1.0, reviewed=False))
    assert d.events == [] and d.updates == []
    for p in (float("nan"), 1.1, -0.1):
        with pytest.raises(ValueError):
            frame(0, p)
    with pytest.raises(ValueError):
        frame(0, None)
    with pytest.raises(ValueError):
        d.push(frame(9, reviewed=False))
    with pytest.raises(ValueError):
        d.finish(stop_frame=15, stop_time=Q(15, 25))


def test_batch_streaming_arbitrary_chunks_and_label_metadata_independence():
    rows = []
    for i, p in enumerate([0.1] + [0.9] * 4 + [0.1] * 3 + [0.9] * 4):
        f = frame(i, p)
        rows.append(
            dict(
                recording=f.recording,
                interval_id=f.interval_id,
                frame_index=i,
                pts=i,
                time_base_num=1,
                time_base_den=25,
                probability_punch=p,
                observable=True,
                reviewed=True,
                scene_id=0,
                segment_id=0,
                reviewed_target=i % 2,
            )
        )
    bounds = {("V4", "test"): (len(rows), Q(len(rows), 25))}
    batch = decode_batch(rows, bounds)
    d = CausalEventDecoder()
    for chunk in (rows[:2], rows[2:7], rows[7:]):
        for row in chunk:
            d.push(Observation.from_row(row))
    d.finish(stop_frame=len(rows), stop_time=Q(len(rows), 25))
    assert batch == dict(
        events=d.events, updates=d.updates, discarded_candidates=d.discarded_candidates
    )
    changed = [
        dict(row, reviewed_target=1 - row["reviewed_target"], punch_type="fabricated")
        for row in rows
    ]
    assert decode_batch(changed, bounds) == batch
    assert decode_batch([dict(row, reviewed=False) for row in rows], bounds)["events"] == []


def test_recovery_at_termination_deadline_is_a_new_candidate_not_a_bridge():
    d = feed([0.1] + [0.9] * 4 + [0.1] * 2 + [0.9] * 3)
    assert len(d.events) == 2
    assert d.events[0]["stop_frame_exclusive"] == 5
    assert d.events[0]["finalized_at"]["observed_through_frame"] == 7
    assert d.events[1]["estimated_start"]["frame_index"] == 7


def test_finalization_never_backdates_to_before_an_already_observed_low_tail():
    d = CausalEventDecoder()
    for i, p in enumerate([0.9, 0.9, 0.9, 0.9, 0.1, 0.1]):
        d.push(frame(i, p))
    d.finish()
    assert d.events[0]["estimated_end_inclusive"]["frame_index"] == 3
    assert d.events[0]["finalized_at"]["observed_through_frame"] == 5
    assert d.events[0]["finalized_at"]["timestamp_exact"] == "1/5"


def test_zero_low_threshold_can_end_on_exact_zero_probability():
    cfg = DecoderConfig(
        deactivation_threshold=0.0, minimum_activation_seconds="0", termination_seconds="0"
    )
    event = feed([0.9, 0.0], cfg).events[0]
    assert event["right_censored"] is False and event["stop_frame_exclusive"] == 1


def test_an_uncertain_hysteresis_prefix_does_not_certify_a_clean_left_boundary():
    event = feed([0.5, 0.5, 0.9, 0.9, 0.9]).events[0]
    assert event["estimated_start"]["frame_index"] == 2
    assert event["left_censored"] is True
