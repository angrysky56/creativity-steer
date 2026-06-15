"""Tests for breadth -> funnel -> branch -> synthesize stages."""

from __future__ import annotations

from creativity_steer.backends import MockBackend, _DEFAULT_POOL
from creativity_steer.chat import ChatConfig, chat_turn, chat_turn_stream
from creativity_steer.entailment import EmbeddingEntailment
from creativity_steer.scoring import funnel_representatives

MSG = "What is the most valuable thing in life?"
POOL = [t for t, _, _, _ in _DEFAULT_POOL]


def _ent(b: MockBackend) -> EmbeddingEntailment:
    return EmbeddingEntailment(b, 0.9)


def test_funnel_returns_prime_n_distinct() -> None:
    b = MockBackend()
    idx = funnel_representatives(b, POOL[0], POOL[1:], prime_n=3)
    assert len(idx) == 3
    assert len(set(idx)) == 3


def test_breadth_then_funnel_bounds_candidates() -> None:
    b = MockBackend()
    cfg = ChatConfig(k=3, breadth_k=9, prime_n=4, max_rounds=1)
    res = chat_turn(b, b, _ent(b), [], MSG, cfg)
    # modal + 4 primes = 5 scored candidates regardless of breadth
    assert len(res["scores"]) == 5


def test_branch_runs() -> None:
    b = MockBackend()
    res = chat_turn(b, b, _ent(b), [], MSG, ChatConfig(k=3, branch=True, max_rounds=1))
    assert res["response"]


def test_synthesize_emits_event_and_marks_response() -> None:
    b = MockBackend()
    evs = list(chat_turn_stream(b, b, _ent(b), [], MSG,
                                ChatConfig(k=4, synthesize=True, max_rounds=1)))
    assert any(e["type"] == "synthesis" for e in evs)
    resp = [e for e in evs if e["type"] == "response"][0]
    assert resp.get("synthesized") is True
    assert resp["text"]
