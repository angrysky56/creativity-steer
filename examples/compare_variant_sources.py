"""Does single-call brainstorming collapse diversity vs independent sampling?

For each problem we draw K variants both ways and measure semantic diversity
with DeBERTa-NLI clustering (clusters formed + semantic entropy). More clusters
/ higher SE = more diverse. This decides which source feeds the Stage 1 pipeline.

    uv run python examples/compare_variant_sources.py
"""

from __future__ import annotations

from statistics import mean

from creativity_steer.backends import OllamaBackend
from creativity_steer.data import SAMPLE_PROBLEMS
from creativity_steer.divergent import analyze_divergent
from creativity_steer.entailment import make_entailment
from creativity_steer.variants import brainstorm_variants, independent_variants

K = 6


def main() -> None:
    backend = OllamaBackend()
    ent = make_entailment("deberta", backend)
    agg: dict[str, list[tuple[int, float]]] = {"independent": [], "brainstorm": []}

    for prob in SAMPLE_PROBLEMS:
        print(f"\n=== {prob['id']} ===")
        sources = {
            "independent": independent_variants(backend, prob["problem"], "", K),
            "brainstorm": brainstorm_variants(backend, prob["problem"], "", K),
        }
        for name, samples in sources.items():
            div = analyze_divergent(prob["problem"], samples, ent)
            agg[name].append((div.num_clusters, div.semantic_entropy))
            print(f"  {name:<12} n={len(samples)} clusters={div.num_clusters} "
                  f"SE={div.semantic_entropy:.3f}")

    print("\n========= AGGREGATE (mean over problems) =========")
    print(f"{'source':<12}{'clusters':>10}{'SE':>8}")
    for name, vals in agg.items():
        print(f"{name:<12}{mean(c for c, _ in vals):>10.2f}"
              f"{mean(s for _, s in vals):>8.3f}")


if __name__ == "__main__":
    main()
