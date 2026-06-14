"""Tests for the env-driven backend factory and the API backend wiring."""

from __future__ import annotations

from creativity_steer.backends import MockBackend, OllamaBackend, OpenAIBackend
from creativity_steer.config import backend_summary, build_backend
from creativity_steer.chat import ChatConfig, chat_turn
from creativity_steer.entailment import EmbeddingEntailment


def test_global_default_backend(monkeypatch) -> None:
    monkeypatch.setenv("CS_BACKEND", "ollama")
    monkeypatch.delenv("CS_GEN_BACKEND", raising=False)
    monkeypatch.setenv("CS_GEN_MODEL", "granite4.1:3b")
    b = build_backend("gen")
    assert isinstance(b, OllamaBackend)
    assert b.gen_model == "granite4.1:3b"


def test_per_role_override(monkeypatch) -> None:
    monkeypatch.setenv("CS_BACKEND", "ollama")
    monkeypatch.setenv("CS_JUDGE_BACKEND", "mock")
    assert isinstance(build_backend("judge"), MockBackend)


def test_shared_mock_instance(monkeypatch) -> None:
    monkeypatch.setenv("CS_BACKEND", "mock")
    shared = MockBackend()
    assert build_backend("gen", shared) is shared
    assert build_backend("embed", shared) is shared


def test_api_backend_constructs(monkeypatch) -> None:
    monkeypatch.setenv("CS_GEN_BACKEND", "api")
    monkeypatch.setenv("CS_GEN_MODEL", "unsloth/test")
    monkeypatch.setenv("CS_API_BASE_URL", "http://localhost:9/v1")
    monkeypatch.setenv("CS_API_KEY", "EMPTY")
    b = build_backend("gen")
    assert isinstance(b, OpenAIBackend)
    assert b.model == "unsloth/test"


def test_per_role_api_base_url(monkeypatch) -> None:
    monkeypatch.setenv("CS_GEN_BACKEND", "api")
    monkeypatch.setenv("CS_GEN_MODEL", "gemma-4-E4B-it-qat-GGUF")
    monkeypatch.setenv("CS_GEN_API_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("CS_GEN_API_KEY", "sk-unsloth-abc")
    b = build_backend("gen")
    assert isinstance(b, OpenAIBackend)
    assert "localhost:8001" in str(b._client.base_url)


def test_summary_reflects_env(monkeypatch) -> None:
    monkeypatch.setenv("CS_BACKEND", "ollama")
    monkeypatch.setenv("CS_GEN_MODEL", "granite4.1:3b")
    assert "gen=ollama:granite4.1:3b" in backend_summary()


def test_chat_with_separate_embed_backend() -> None:
    gen, judge, embed = MockBackend(), MockBackend(), MockBackend()
    res = chat_turn(gen, judge, EmbeddingEntailment(embed, 0.9), [], "hi",
                    ChatConfig(k=3), embed_backend=embed)
    assert res["response"]
