"""Tests for the metacognitive turn controller."""

from __future__ import annotations

from creativity_steer.control import TurnController


def test_diversity() -> None:
    c = TurnController()
    assert c.diversity([0.0, 0.0, 0.0]) == 0.0        # all collapsed onto modal
    assert c.diversity([0.0, 1.0, 1.0]) == 1.0        # all escaped


def test_should_continue_when_collapsed() -> None:
    c = TurnController(explore_diversity_threshold=0.5)
    # collapsed set + rounds remaining -> explore more
    assert c.should_continue(0, [0.0, 0.0, 0.0, 1.0], max_rounds=2) is True
    # already diverse -> commit
    assert c.should_continue(0, [0.0, 1.0, 1.0, 1.0], max_rounds=2) is False
    # round cap reached -> commit regardless of collapse
    assert c.should_continue(1, [0.0, 0.0, 0.0], max_rounds=2) is False


def test_next_temperature_caps() -> None:
    c = TurnController(temperature_step=0.3, max_temperature=1.5)
    assert c.next_temperature(1.0) == 1.3
    assert c.next_temperature(1.4) == 1.5
