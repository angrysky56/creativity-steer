"""Tests for multi-axis scoring and selection."""

from __future__ import annotations

from creativity_steer.backends import MockBackend
from creativity_steer.chat import ChatConfig, chat_turn
from creativity_steer.entailment import EmbeddingEntailment
from creativity_steer.scoring import pareto_mask_nd, select_multi

MSG = "What's a creative way to reuse an empty glass jar?"


def test_pareto_mask_nd_3d() -> None:
    rows = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.4, 0.4, 0.4], [0.2, 0.2, 0.2]]
    mask = pareto_mask_nd(rows)
    assert mask[0] and mask[1] and mask[2]   # non-dominated
    assert not mask[3]                        # dominated by row 2


def test_select_multi_weights_and_floor() -> None:
    scores = {
        "novelty": [0.0, 1.0, 0.5],
        "quality": [1.0, 0.2, 0.9],
    }
    # Heavy novelty weight -> prefer index 1, but a quality floor blocks it.
    idx, _ = select_multi(scores, {"novelty": 0.9, "quality": 0.1},
                          floor_axis="quality", floor=0.5)
    assert idx in (0, 2)            # index 1 (quality 0.2) excluded by floor
    # No floor -> novelty wins.
    idx2, _ = select_multi(scores, {"novelty": 0.9, "quality": 0.1}, floor=0.0)
    assert idx2 == 1


def test_chat_emits_axis_scores_dict() -> None:
    b = MockBackend()
    res = chat_turn(b, b, EmbeddingEntailment(b, 0.9), [], MSG,
                    ChatConfig(k=4, coherence_paraphrases=1))
    scored = res["scores"]
    assert scored and "scores" in scored[0]
    axes = scored[0]["scores"]
    assert {"novelty", "quality", "coherence"} <= set(axes)
    for ev in scored:
        for v in ev["scores"].values():
            assert 0.0 <= v <= 1.0


def test_coherence_weight_changes_nothing_when_zero() -> None:
    # With coherence_weight 0 the axis is computed but not used in selection.
    b = MockBackend()
    cfg = ChatConfig(k=4, coherence_weight=0.0, coherence_paraphrases=1)
    res = chat_turn(b, b, EmbeddingEntailment(b, 0.9), [], MSG, cfg)
    assert 0 <= res["selected"] < len(res["scores"])
