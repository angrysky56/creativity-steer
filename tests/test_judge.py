"""Tests for the multi-agent retrieval judge and its fragment store."""

from __future__ import annotations

from creativity_steer.backends import MockBackend, _DEFAULT_POOL
from creativity_steer.criteria import MACGYVER_CRITERIA
from creativity_steer.judge import FragmentStore, multi_agent_judge

PROBLEM = "How do you fish keys out of a drain?"
HIGH_Q = _DEFAULT_POOL[3][0]   # quality 0.85
LOW_Q = _DEFAULT_POOL[2][0]    # quality 0.30


def test_fragment_store_retrieval() -> None:
    store = FragmentStore(MockBackend())
    store.add(HIGH_Q, "feasibility", temporary=False)
    store.add(LOW_Q, "feasibility", temporary=True)
    hits = store.query(HIGH_Q, k=1)
    assert hits and hits[0] == HIGH_Q


def test_clear_temporary_keeps_permanent() -> None:
    store = FragmentStore(MockBackend())
    store.add(HIGH_Q, "all", temporary=False)
    store.add(LOW_Q, "feasibility", temporary=True)
    store.clear_temporary()
    assert store.query("anything", k=5) == [HIGH_Q]


def test_judge_returns_all_criteria() -> None:
    res = multi_agent_judge(
        MockBackend(), PROBLEM, HIGH_Q, MACGYVER_CRITERIA, num_rounds=1
    )
    assert set(res.verdicts) == set(MACGYVER_CRITERIA)
    assert 0.0 <= res.score <= 1.0


def test_judge_separates_quality() -> None:
    # The mock votes YES when solution quality >= cutoff (0.5) and NO below.
    high = multi_agent_judge(
        MockBackend(), PROBLEM, HIGH_Q, MACGYVER_CRITERIA, num_rounds=1
    )
    low = multi_agent_judge(
        MockBackend(), PROBLEM, LOW_Q, MACGYVER_CRITERIA, num_rounds=1
    )
    assert high.score > low.score
