"""Calibrate the embedding-clustering threshold on real candidates.

Generates candidates for each sample problem, embeds them, and reports how many
semantic clusters form at a range of cosine thresholds -- so we can pick a
threshold that neither collapses everything into one cluster nor splits every
candidate apart.

    uv run python examples/calibrate_threshold.py
"""

from __future__ import annotations

import numpy as np

from creativity_steer.backends import OllamaBackend
from creativity_steer.data import SAMPLE_PROBLEMS
from creativity_steer.divergent import cluster_by_entailment
from creativity_steer.entailment import EmbeddingEntailment

THRESHOLDS = (0.6, 0.7, 0.75, 0.8, 0.85, 0.9)


def main() -> None:
    backend = OllamaBackend()
    for prob in SAMPLE_PROBLEMS:
        prompt = (
            f"PROBLEM:\n{prob['problem']}\n\nPropose ONE creative, useful next "
            "step. Reply with the step only, in one sentence."
        )
        samples = backend.generate_samples(prompt, n=6, temperature=1.0, max_tokens=100)
        texts = [s.text for s in samples]
        vecs = [np.asarray(v, dtype=float) for v in backend.embed(texts)]

        # Pairwise cosine summary.
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                a, b = vecs[i], vecs[j]
                sims.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        print(f"\n=== {prob['id']} ===")
        print(f"pairwise cosine: min={min(sims):.3f} mean={np.mean(sims):.3f} "
              f"max={max(sims):.3f}")
        for th in THRESHOLDS:
            ids = cluster_by_entailment(
                prob["problem"], texts, EmbeddingEntailment(backend, threshold=th)
            )
            print(f"  threshold {th:.2f} -> {len(set(ids))} clusters  {ids}")


if __name__ == "__main__":
    main()
