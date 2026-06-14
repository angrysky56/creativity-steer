"""Multi-step Stage 1 trajectory vs greedy baseline (experiment b).

Generation/brainstorm/wrap run on a fast small model (granite4.1:3b); quality
judging runs on a stronger model (gemma4:12b). Novelty uses DeBERTa-anchored
distance from the modal answer. Builds a full multi-step solution two ways and
compares how novelty/quality look per step.

    uv run python examples/stage1_trajectory_demo.py
"""

from __future__ import annotations

from creativity_steer.backends import OllamaBackend
from creativity_steer.data import get_problem
from creativity_steer.entailment import make_entailment
from creativity_steer.stage1 import (
    Stage1Config,
    run_greedy_trajectory,
    run_stage1_trajectory,
)

GEN_MODEL = "granite4.1:3b"
JUDGE_MODEL = "gemma4:12b"
STEPS = 3
K = 4


def main() -> None:
    gen = OllamaBackend(gen_model=GEN_MODEL)
    judge = OllamaBackend(gen_model=JUDGE_MODEL)
    entailment = make_entailment("deberta", gen)
    prob = get_problem("keys-in-drain")
    cfg = Stage1Config(k=K, convergent_floor=0.34, novelty_weight=0.5, wrap=False)

    print(f"gen={GEN_MODEL}  judge={JUDGE_MODEL}  steps={STEPS}  K={K}")
    print(f"PROBLEM: {prob['problem']}\n")

    s1 = run_stage1_trajectory(gen, prob["problem"], cfg, entailment, judge, STEPS)
    print("--- STAGE 1 (think-and-select) ---")
    for i, r in enumerate(s1.results):
        print(f"  step {i}: nov={r.chosen_novelty:.2f} conv={r.chosen_convergent:.2f}"
              f"\n    modal : {r.modal[:80]}"
              f"\n    chosen: {r.chosen[:80]}")
    print(f"  mean novelty={s1.mean_novelty:.3f}  mean conv={s1.mean_convergent:.3f}")

    g = run_greedy_trajectory(gen, prob["problem"], STEPS)
    print("\n--- GREEDY (modal only) ---")
    for i, step in enumerate(g.steps):
        print(f"  step {i}: {step[:80]}")

    print("\n=== FINAL SOLUTIONS ===")
    print("STAGE 1:\n  " + s1.solution.replace("\n", "\n  "))
    print("GREEDY:\n  " + g.solution.replace("\n", "\n  "))


if __name__ == "__main__":
    main()
