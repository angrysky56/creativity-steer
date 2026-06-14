"""Compare embedding models for semantic-clustering discrimination.

Generates one candidate set per problem (once, with the gen model), then embeds
the SAME texts with each embedding model and reports the pairwise-cosine spread
and how many clusters form across thresholds. A good discriminator has a wide
cosine spread and produces sensible cluster counts over a usable threshold band.

    uv run python examples/compare_embedders.py
"""

from __future__ import annotations

import numpy as np

from creativity_steer.backends import OllamaBackend
from creativity_steer.data import SAMPLE_PROBLEMS
from creativity_steer.divergent import cluster_by_entailment
from creativity_steer.entailment import EmbeddingEntailment

EMBED_MODELS = (
    "nomic-embed-text",
    "embeddinggemma",
    "qwen3-embedding:0.6b",
    "qwen3-embedding:4b",
)
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9)


def _spread(vecs: list[np.ndarray]) -> tuple[float, float, float]:
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            a, b = vecs[i], vecs[j]
            sims.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return min(sims), float(np.mean(sims)), max(sims)


def main() -> None:
    gen = OllamaBackend()
    for prob in SAMPLE_PROBLEMS:
        prompt = (
            f"PROBLEM:\n{prob['problem']}\n\nPropose ONE creative, useful next "
            "step. Reply with the step only, in one sentence."
        )
        texts = [s.text for s in gen.generate_samples(prompt, 6, 1.0, 100)]
        print(f"\n################ {prob['id']} ################")
        for model in EMBED_MODELS:
            try:
                eb = OllamaBackend(embed_model=model)
                vecs = [np.asarray(v, dtype=float) for v in eb.embed(texts)]
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  [{model}] ERROR: {exc}")
                continue
            lo, mean, hi = _spread(vecs)
            counts = []
            for th in THRESHOLDS:
                ids = cluster_by_entailment(
                    prob["problem"], texts, EmbeddingEntailment(eb, threshold=th)
                )
                counts.append(f"{th}:{len(set(ids))}")
            print(f"  [{model}] cos lo/mean/hi={lo:.2f}/{mean:.2f}/{hi:.2f}  "
                  f"clusters {' '.join(counts)}")


if __name__ == "__main__":
    main()
