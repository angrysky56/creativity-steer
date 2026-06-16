"""Command-line entry point.

Runs selection strategies over the sample problems and prints, per strategy,
mean novelty and step-convergent (from the inner loop) plus the final
multi-agent judge score (the authoritative convergent number).
Use ``--backend mock`` to validate the whole pipeline with no model running.
"""

from __future__ import annotations

import argparse
import json
import math

from creativity_steer.backends import LLMBackend, MockBackend, OllamaBackend
from creativity_steer.data import SAMPLE_PROBLEMS, criteria_for, get_problem
from creativity_steer.entailment import make_entailment
from creativity_steer.selection import SelectionConfig, run_trajectory

STRATEGIES = ("greedy", "convergent", "pareto")


def _build_backend(args: argparse.Namespace) -> LLMBackend:
    if args.backend == "mock":
        return MockBackend()
    return OllamaBackend(gen_model=args.model, embed_model=args.embed_model)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pareto creativity steering prototype.")
    p.add_argument("--backend", choices=("mock", "ollama"), default="mock")
    p.add_argument("--model", default=None, help="Ollama generation model tag.")
    p.add_argument("--embed-model", default=None, help="Ollama embedding model tag.")
    p.add_argument(
        "--entailment",
        choices=("llm", "embedding", "deberta"),
        default="llm",
        help="Semantic-clustering entailment model (deberta needs the extra).",
    )
    p.add_argument(
        "--strategy",
        choices=(*STRATEGIES, "all"),
        default="all",
    )
    p.add_argument("--problem", default="all", help="Problem id, or 'all'.")
    p.add_argument("-n", "--n-candidates", type=int, default=6)
    p.add_argument("-t", "--temperature", type=float, default=1.0)
    p.add_argument("-s", "--max-steps", type=int, default=3)
    p.add_argument("--novelty-weight", type=float, default=0.5)
    p.add_argument("--convergent-floor", type=float, default=0.34)
    p.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.88,
        help="Cosine cut-off for embedding entailment clustering "
        "(tuned for embeddinggemma).",
    )
    p.add_argument("--judge-rounds", type=int, default=2)
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument(
        "--no-final-judge",
        action="store_true",
        help="Skip the multi-agent judge (faster).",
    )
    p.add_argument("--output", default=None, help="Write full results to JSON.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI main: run strategies and print a comparison table."""
    from creativity_steer.config import load_env

    load_env()
    args = _parse_args(argv)
    backend = _build_backend(args)
    ent_kwargs = (
        {"threshold": args.cluster_threshold} if args.entailment == "embedding" else {}
    )
    entailment = make_entailment(args.entailment, backend, **ent_kwargs)
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    problems = SAMPLE_PROBLEMS if args.problem == "all" else [get_problem(args.problem)]

    dump: list[dict] = []
    print(
        f"\nbackend={args.backend}  entailment={args.entailment}  "
        f"n={args.n_candidates}  T={args.temperature}  steps={args.max_steps}\n"
    )
    for prob in problems:
        config = SelectionConfig(
            n_candidates=args.n_candidates,
            temperature=args.temperature,
            novelty_weight=args.novelty_weight,
            convergent_floor=args.convergent_floor,
            max_steps=args.max_steps,
            run_final_judge=not args.no_final_judge,
            judge_rounds=args.judge_rounds,
            judge_confidence_threshold=args.confidence_threshold,
            judge_criteria=dict(criteria_for(prob)),
        )
        print(f"=== {prob['id']} ===")
        print(
            f"{'strategy':<12}{'novelty':>9}{'step-conv':>11}"
            f"{'sem-ent':>9}{'FINAL-judge':>13}"
        )
        for strat in strategies:
            res = run_trajectory(backend, prob["problem"], config, strat, entailment)
            final = res.final_score
            final_s = "  n/a" if math.isnan(final) else f"{final:.3f}"
            print(
                f"{strat:<12}{res.mean_novelty:>9.3f}"
                f"{res.mean_step_convergent:>11.3f}"
                f"{res.mean_semantic_entropy:>9.3f}{final_s:>13}"
            )
            dump.append(_serialise(prob["id"], res))
        print()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(dump, fh, indent=2)
        print(f"wrote {args.output}")


def _serialise(problem_id: str, res) -> dict:
    """Flatten a SelectionResult to JSON-friendly form."""
    return {
        "problem_id": problem_id,
        "strategy": res.strategy,
        "mean_novelty": res.mean_novelty,
        "mean_step_convergent": res.mean_step_convergent,
        "mean_semantic_entropy": res.mean_semantic_entropy,
        "final_judge": (
            None
            if res.final_judge is None
            else {
                "score": res.final_judge.score,
                "verdicts": res.final_judge.verdicts,
                "confidence": res.final_judge.confidence,
            }
        ),
        "solution": res.solution,
        "steps": [
            {
                "step": s.step,
                "chosen": s.candidates[s.chosen_index].text,
                "semantic_entropy": s.semantic_entropy,
                "candidates": [
                    {
                        "text": c.text,
                        "novelty": c.novelty,
                        "convergent": c.convergent,
                        "cluster_id": c.cluster_id,
                        "logprob": c.logprob,
                        "on_frontier": c.on_frontier,
                    }
                    for c in s.candidates
                ],
            }
            for s in res.steps
        ],
    }


if __name__ == "__main__":
    main()
