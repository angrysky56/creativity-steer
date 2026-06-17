"""Tests for the training-data validity gate and record builder."""

from __future__ import annotations

from creativity_steer.backends import MockBackend
from creativity_steer.training_store import (
    assess_validity,
    build_correction_record,
    build_record,
    compute_impact,
    detect_correction,
)


def _events_by_type(selected_index: int, floor_met: bool, originality: float | None = None,
                    synthesis_collapsed: bool | None = None) -> dict[str, list]:
    scores = {"novelty": 0.8, "quality": 0.7, "coherence": 0.8}
    if originality is not None:
        scores["originality"] = originality
    by_type: dict[str, list] = {
        "variants": [{"items": [{"text": "modal", "is_modal": True},
                                 {"text": "a fresh joke", "is_modal": False}]}],
        "scored": [
            {"index": 0, "scores": {"novelty": 0.0, "quality": 0.5}},
            {"index": 1, "scores": scores},
        ],
        "selected": [{"index": selected_index, "floor_met": floor_met}],
        "controller": [{"semantic_entropy": 1.2, "num_clusters": 4,
                         "num_candidates": 12, "weights": {"novelty": 0.5},
                         "quality_floor": 0.4}],
        "response": [{"text": "a fresh joke", "synthesized": False}],
    }
    if synthesis_collapsed is not None:
        by_type["synthesis"] = [{"collapsed_to_modal": synthesis_collapsed}]
    return by_type


def test_valid_turn_accepted() -> None:
    ok, reason = assess_validity(_events_by_type(1, True, originality=0.7))
    assert ok and reason == "ok"


def test_below_floor_rejected() -> None:
    ok, reason = assess_validity(_events_by_type(1, False))
    assert not ok and reason == "below_quality_floor"


def test_modal_winner_rejected() -> None:
    ok, reason = assess_validity(_events_by_type(0, True))
    assert not ok and reason == "winner_is_modal"


def test_cliche_winner_rejected() -> None:
    ok, reason = assess_validity(_events_by_type(1, True, originality=0.1))
    assert not ok and reason == "winner_is_cliche"


def test_collapsed_synthesis_rejected() -> None:
    ok, reason = assess_validity(
        _events_by_type(1, True, originality=0.7, synthesis_collapsed=True)
    )
    assert not ok and reason == "synthesis_collapsed"


def test_build_record_shape() -> None:
    rec = build_record("Tell me a joke.", [], _events_by_type(1, True, originality=0.7))
    assert rec["response"] == "a fresh joke"
    assert rec["winner_index"] == 1
    assert rec["winner_axes"]["originality"] == 0.7
    assert rec["semantic_entropy"] == 1.2
    assert len(rec["candidates"]) == 2
    assert rec["candidates"][1]["scores"]["quality"] == 0.7
    # Impact-weight present (plan §3) — not flat-logged.
    assert "impact" in rec and rec["impact"] >= 0.0


def test_impact_rises_with_axis_gap_and_correction() -> None:
    base = _events_by_type(1, True, originality=0.7)
    low = compute_impact(base)
    # A correction is the strongest impact signal (asymmetric negative, §2a).
    high = compute_impact(base, is_correction=True)
    assert high > low


def test_detect_correction() -> None:
    b = MockBackend()
    prev = "The capital of Australia is Sydney."
    assert detect_correction(b, prev, "No, that's wrong, it's Canberra.")
    assert not detect_correction(b, prev, "Now tell me about New Zealand.")
    assert not detect_correction(b, "", "anything")  # no prior reply


def test_build_correction_record() -> None:
    rec = build_correction_record("What's the capital?", "Sydney.", "It's Canberra.")
    assert rec["is_correction"] is True
    assert rec["kind"] == "correction"
    assert rec["failed_response"] == "Sydney."
    assert rec["response"] == "It's Canberra."
    # Correction flag makes impact high (dominant term, §3).
    assert rec["impact"] >= 1.0
