"""Tests for the Stage 1 think-and-select pipeline (MockBackend)."""

from __future__ import annotations

from creativity_steer.backends import MockBackend
from creativity_steer.entailment import EmbeddingEntailment
from creativity_steer.reference import novelty_vs_reference
from creativity_steer.stage1 import Stage1Config, think_and_select

PROBLEM = "How do you fish keys out of a drain?"


def _ent(b: MockBackend) -> EmbeddingEntailment:
    return EmbeddingEntailment(b, threshold=0.9)


def test_reference_novelty_zero_for_self() -> None:
    b = MockBackend()
    modal = b.pool[0][0]
    other = b.pool[2][0]
    nov = novelty_vs_reference(b, modal, [modal, other])
    assert nov[0] == 0.0          # identical to reference -> zero novelty
    assert nov[1] > 0.0           # different cluster -> positive novelty


def test_think_and_select_runs() -> None:
    b = MockBackend()
    cfg = Stage1Config(k=6, wrap=True)
    res = think_and_select(b, PROBLEM, "", cfg, _ent(b))
    assert res.modal
    assert res.chosen
    assert res.response
    assert len(res.candidates) >= 6


def test_modal_candidate_has_zero_novelty() -> None:
    b = MockBackend()
    cfg = Stage1Config(k=6, include_modal_in_pool=True)
    res = think_and_select(b, PROBLEM, "", cfg, _ent(b))
    modal_cands = [c for c in res.candidates if c.is_modal]
    assert modal_cands and modal_cands[0].novelty == 0.0


def test_chosen_respects_floor_when_possible() -> None:
    b = MockBackend()
    cfg = Stage1Config(k=6, convergent_floor=0.5, wrap=False)
    res = think_and_select(b, PROBLEM, "", cfg, _ent(b))
    chosen = next(c for c in res.candidates if c.text == res.chosen)
    eligible = [c for c in res.candidates if c.on_frontier and c.convergent >= 0.5]
    if eligible:
        assert chosen.convergent >= 0.5
