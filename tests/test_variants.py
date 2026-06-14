"""Tests for the two variant sources using the MockBackend."""

from __future__ import annotations

from creativity_steer.backends import MockBackend
from creativity_steer.variants import (
    brainstorm_variants,
    get_variants,
    independent_variants,
    parse_numbered_list,
)

PROBLEM = "How do you fish keys out of a drain?"


def test_parse_numbered_list_formats() -> None:
    raw = "1) first\n2. second\n3 - third\nnot a list line\n4: fourth"
    assert parse_numbered_list(raw, 4) == ["first", "second", "third", "fourth"]


def test_parse_numbered_list_respects_k() -> None:
    raw = "1) a\n2) b\n3) c"
    assert parse_numbered_list(raw, 2) == ["a", "b"]


def test_independent_variants_count_and_logprobs() -> None:
    b = MockBackend()
    vs = independent_variants(b, PROBLEM, "", k=5)
    assert len(vs) == 5
    assert all(v.logprob is not None for v in vs)


def test_brainstorm_variants_parses_population() -> None:
    b = MockBackend()
    vs = brainstorm_variants(b, PROBLEM, "", k=4)
    assert len(vs) == 4
    assert all(v.text for v in vs)
    assert all(v.logprob is None for v in vs)  # single call -> no per-variant logprob


def test_get_variants_dispatch() -> None:
    b = MockBackend()
    assert len(get_variants("independent", b, PROBLEM, "", 3)) == 3
    assert len(get_variants("brainstorm", b, PROBLEM, "", 3)) == 3
