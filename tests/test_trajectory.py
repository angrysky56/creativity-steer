"""Tests for multi-step Stage 1 trajectories (MockBackend)."""

from __future__ import annotations

from creativity_steer.backends import MockBackend
from creativity_steer.entailment import EmbeddingEntailment
from creativity_steer.stage1 import (
    Stage1Config,
    run_greedy_trajectory,
    run_stage1_trajectory,
)

PROBLEM = "How do you fish keys out of a drain?"


def test_stage1_trajectory_builds_solution() -> None:
    b = MockBackend()
    cfg = Stage1Config(k=4, wrap=False)
    traj = run_stage1_trajectory(b, PROBLEM, cfg, EmbeddingEntailment(b, 0.9),
                                 max_steps=3)
    assert len(traj.steps) == 3
    assert len(traj.results) == 3
    assert traj.solution
    assert 0.0 <= traj.mean_novelty <= 1.0


def test_separate_judge_backend_runs() -> None:
    gen = MockBackend()
    judge = MockBackend()
    cfg = Stage1Config(k=4, wrap=False)
    traj = run_stage1_trajectory(gen, PROBLEM, cfg, EmbeddingEntailment(gen, 0.9),
                                 judge_backend=judge, max_steps=2)
    assert len(traj.steps) == 2


def test_greedy_trajectory_runs() -> None:
    b = MockBackend()
    traj = run_greedy_trajectory(b, PROBLEM, max_steps=3)
    assert len(traj.steps) == 3
    assert traj.results == []  # greedy records no per-step selection
