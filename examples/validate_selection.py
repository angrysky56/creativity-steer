"""Isolated validation of the selection effect.

For each problem we generate ONE shared candidate set, score every candidate on
novelty (divergent) and the rubric judge (convergent), then report what each
selection rule would pick from that *same* pool. This isolates the selection
effect with no trajectory-divergence confound and only one generation pass per
problem, so it is fast.

    uv run python examples/validate_selection.py
"""

from __future__ import annotations

import os
from statistics import mean

from creativity_steer.backends import OllamaBackend
from creativity_steer.convergent import judge_candidate
from creativity_steer.data import SAMPLE_PROBLEMS
from creativity_steer.divergent import analyze_divergent
from creativity_steer.entailment import make_entailment
from creativity_steer.selection import CandidateEval, _choose, SelectionConfig, pareto_mask

N = 6
THRESHOLD = 0.88
# Entailment instrument: "deberta" (default), "llm", or "embedding".
KIND = os.getenv("CS_ENTAILMENT", "deberta")


def main() -> None:
    backend = OllamaBackend()
    cfg = SelectionConfig(n_candidates=N, convergent_floor=0.34)
    ent_kwargs = {"threshold": THRESHOLD} if KIND == "embedding" else {}
    ent = make_entailment(KIND, backend, **ent_kwargs)
    print(f"entailment={KIND}")
    rows: dict[str, list[tuple[float, float]]] = {
        "greedy": [], "convergent": [], "pareto": []
    }

    for prob in SAMPLE_PROBLEMS:
        prompt = (
            f"PROBLEM:\n{prob['problem']}\n\nPropose ONE creative, useful next "
            "step. Reply with the step only, in one sentence."
        )
        samples = backend.generate_samples(prompt, N, 1.0, 100)
        div = analyze_divergent(prob["problem"], samples, ent)
        conv = [
            judge_candidate(backend, prob["problem"], "", s.text).score
            for s in samples
        ]
        frontier = pareto_mask(list(zip(div.novelty, conv)))
        evals = [
            CandidateEval(samples[i].text, div.cluster_ids[i], div.novelty[i],
                          conv[i], samples[i].logprob, frontier[i])
            for i in range(N)
        ]
        print(f"\n=== {prob['id']} ===  clusters={div.num_clusters}  "
              f"SE={div.semantic_entropy:.3f}")
        for strat in rows:
            i = _choose(evals, strat, cfg)
            e = evals[i]
            rows[strat].append((e.novelty, e.convergent))
            print(f"  {strat:<11} -> novelty={e.novelty:.2f} conv={e.convergent:.2f} "
                  f"| {e.text[:70]}")

    print("\n================ AGGREGATE (mean over problems) ================")
    print(f"{'strategy':<12}{'novelty':>9}{'convergent':>12}")
    for strat, vals in rows.items():
        nov = mean(v[0] for v in vals)
        cv = mean(v[1] for v in vals)
        print(f"{strat:<12}{nov:>9.3f}{cv:>12.3f}")


if __name__ == "__main__":
    main()
