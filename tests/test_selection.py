"""End-to-end selection tests using the deterministic MockBackend."""

from __future__ import annotations

from creativity_steer.backends import MockBackend
from creativity_steer.entailment import EmbeddingEntailment
from creativity_steer.selection import (
    SelectionConfig,
    run_trajectory,
    select_candidate,
)

PROBLEM = "How do you fish keys out of a drain with a magnet, string, and cup?"


def _ent(b: MockBackend) -> EmbeddingEntailment:
    return EmbeddingEntailment(b, threshold=0.9)


def test_mock_samples_are_deterministic() -> None:
    b = MockBackend()
    a = b.generate_samples(PROBLEM, n=6, temperature=1.0)
    c = b.generate_samples(PROBLEM, n=6, temperature=1.0)
    assert [s.text for s in a] == [s.text for s in c]
    assert all(s.logprob is not None for s in a)


def test_temperature_changes_diversity() -> None:
    b = MockBackend()
    cold = {s.text for s in b.generate_samples(PROBLEM, 20, 0.1)}
    hot = {s.text for s in b.generate_samples(PROBLEM, 20, 2.0)}
    assert len(hot) >= len(cold)


def test_all_strategies_run() -> None:
    b = MockBackend()
    cfg = SelectionConfig(n_candidates=6, max_steps=2, run_final_judge=False)
    for strat in ("greedy", "convergent", "pareto"):
        res = run_trajectory(b, PROBLEM, cfg, strat, _ent(b))
        assert res.solution
        assert len(res.steps) == 2


def test_pareto_not_less_novel_than_greedy() -> None:
    b = MockBackend()
    cfg = SelectionConfig(
        n_candidates=8, max_steps=3, convergent_floor=0.0, run_final_judge=False
    )
    greedy = run_trajectory(b, PROBLEM, cfg, "greedy", _ent(b))
    pareto = run_trajectory(b, PROBLEM, cfg, "pareto", _ent(b))
    assert pareto.mean_novelty >= greedy.mean_novelty


def test_convergent_floor_blocks_low_quality() -> None:
    b = MockBackend()
    cfg = SelectionConfig(n_candidates=8, convergent_floor=1.0, run_final_judge=False)
    rec = select_candidate(b, PROBLEM, "", cfg, "pareto", _ent(b))
    chosen = rec.candidates[rec.chosen_index]
    eligible = [c for c in rec.candidates if c.on_frontier and c.convergent >= 1.0]
    if eligible:
        assert chosen.convergent >= 1.0


def test_final_judge_runs_on_mock() -> None:
    b = MockBackend()
    cfg = SelectionConfig(
        n_candidates=4, max_steps=2, run_final_judge=True, judge_rounds=1
    )
    res = run_trajectory(b, PROBLEM, cfg, "pareto", _ent(b))
    assert res.final_judge is not None
    assert set(res.final_judge.verdicts) == {"feasibility", "safety", "effectiveness"}
    assert 0.0 <= res.final_score <= 1.0
