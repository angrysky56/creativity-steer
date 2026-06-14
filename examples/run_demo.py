"""Minimal demo: compare greedy vs pareto selection on one sample problem.

Model-free (no Ollama needed):

    uv run python examples/run_demo.py
"""

from __future__ import annotations

from creativity_steer.backends import MockBackend
from creativity_steer.data import criteria_for, get_problem
from creativity_steer.entailment import EmbeddingEntailment
from creativity_steer.selection import SelectionConfig, run_trajectory


def main() -> None:
    backend = MockBackend()
    entailment = EmbeddingEntailment(backend, threshold=0.9)
    prob = get_problem("keys-in-drain")
    config = SelectionConfig(
        n_candidates=8, max_steps=3, convergent_floor=0.34,
        run_final_judge=True, judge_rounds=1,
        judge_criteria=dict(criteria_for(prob)),
    )

    print(f"PROBLEM: {prob['problem']}\n")
    for strategy in ("greedy", "pareto"):
        res = run_trajectory(backend, prob["problem"], config, strategy, entailment)
        print(f"--- {strategy} ---")
        print(f"  mean novelty      : {res.mean_novelty:.3f}")
        print(f"  mean step-conv    : {res.mean_step_convergent:.3f}")
        print(f"  mean sem-entropy  : {res.mean_semantic_entropy:.3f}")
        print(f"  FINAL judge score : {res.final_score:.3f}")
        print(f"  verdicts          : {res.final_judge.verdicts}")
        print("  solution:\n    " + res.solution.replace("\n", "\n    ") + "\n")


if __name__ == "__main__":
    main()
