"""Tests for chat-mode think-and-select (MockBackend)."""

from __future__ import annotations

from creativity_steer.backends import MockBackend
from creativity_steer.chat import ChatConfig, chat_turn, chat_turn_stream
from creativity_steer.entailment import EmbeddingEntailment

MSG = "What's a creative way to reuse an empty glass jar?"


def _ent(b: MockBackend) -> EmbeddingEntailment:
    return EmbeddingEntailment(b, threshold=0.9)


def test_chat_stream_event_order() -> None:
    b = MockBackend()
    cfg = ChatConfig(k=4)
    types = [e["type"] for e in chat_turn_stream(b, b, _ent(b), [], MSG, cfg)]
    assert types[0] == "modal"
    assert types[1] == "variants"
    assert types.count("scored") == 5          # modal + 4 variants
    assert types[-2] == "selected"
    assert types[-1] == "response"


def test_chat_turn_assembles_result() -> None:
    b = MockBackend()
    res = chat_turn(b, b, _ent(b), [], MSG, ChatConfig(k=4))
    assert res["modal"]
    assert res["response"]
    assert len(res["variants"]) == 5
    assert 0 <= res["selected"] < 5


def test_chat_separate_judge_backend() -> None:
    gen, judge = MockBackend(), MockBackend()
    res = chat_turn(gen, judge, _ent(gen), [], MSG, ChatConfig(k=3))
    assert res["response"]
    assert len(res["scores"]) == 4  # modal + 3 variants


def test_chat_history_is_accepted() -> None:
    b = MockBackend()
    history = [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": "hello"}]
    res = chat_turn(b, b, _ent(b), history, MSG, ChatConfig(k=3))
    assert res["response"]
