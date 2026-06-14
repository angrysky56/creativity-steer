"""Matched benchmark: greedy (modal) vs Stage 1 (chosen) selection.

Each think-and-select run yields BOTH arms from the same candidate pool:
the modal candidate is the greedy baseline, the chosen candidate is Stage 1.
We repeat over problems x repeats and report mean +/- std of absolute novelty
(distance from modal) and quality. Generation on a fast model, judging on a
stronger one.

    uv run python examples/benchmark.py
"""

from __future__ import annotations

import json
from statistics import mean, pstdev

from creativity_steer.backends import OllamaBackend
from creativity_steer.data import SAMPLE_PROBLEMS
from creativity_steer.entailment import make_entailment
from creativity_steer.stage1 import Stage1Config, think_and_select

GEN_MODEL = "granite4.1:3b"
JUDGE_MODEL = "gemma4:12b"
REPEATS = 3
K = 4


def main() -> None:
    gen = OllamaBackend(gen_model=GEN_MODEL)
    judge = OllamaBackend(gen_model=JUDGE_MODEL)
    ent = make_entailment("deberta", gen)
    cfg = Stage1Config(k=K, convergent_floor=0.34, novelty_weight=0.5, wrap=False)

    arms = {"greedy": {"nov": [], "q": []}, "stage1": {"nov": [], "q": []}}
    rows = []
    print(f"gen={GEN_MODEL} judge={JUDGE_MODEL} repeats={REPEATS} K={K}\n")

    for prob in SAMPLE_PROBLEMS:
        for r in range(REPEATS):
            res = think_and_select(gen, prob["problem"], "", cfg, ent, judge)
            modal = next(c for c in res.candidates if c.is_modal)
            arms["greedy"]["nov"].append(modal.distance)   # 0 by definition
            arms["greedy"]["q"].append(modal.convergent)
            arms["stage1"]["nov"].append(res.chosen_distance)
            arms["stage1"]["q"].append(res.chosen_convergent)
            print(f"  {prob['id'][:14]:<14} r{r}: "
                  f"greedy q={modal.convergent:.2f} | "
                  f"stage1 nov={res.chosen_distance:.3f} q={res.chosen_convergent:.2f}")
            rows.append({"problem": prob["id"], "repeat": r,
                         "greedy_q": modal.convergent,
                         "stage1_nov": res.chosen_distance,
                         "stage1_q": res.chosen_convergent})

    print("\n================ AGGREGATE (mean +/- std) ================")
    print(f"{'arm':<10}{'abs-novelty':>16}{'quality':>16}")
    for name, d in arms.items():
        print(f"{name:<10}{_ms(d['nov']):>16}{_ms(d['q']):>16}")

    with open("/tmp/cs_benchmark.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print("\nwrote /tmp/cs_benchmark.json")


def _ms(xs: list[float]) -> str:
    return f"{mean(xs):.3f}+/-{pstdev(xs):.3f}" if xs else "n/a"


if __name__ == "__main__":
    main()
