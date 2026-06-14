"""Novelty measured against a reference (the model's modal answer).

When variants are all distinct (e.g. from brainstorming), within-set cluster
surprisal saturates and gives no gradient to select on. Instead we anchor on the
model's MODAL answer -- what greedy decoding would have produced -- and score
each variant's novelty as its semantic distance from that anchor.

``reference_distances`` returns the raw, absolute distance (1 - cosine), which
is graded and comparable across problems -- use this for reporting "how much
more novel". ``novelty_vs_reference`` max-normalises those distances into [0, 1]
for use as a selection objective alongside the [0, 1] convergent score.
"""

from __future__ import annotations

import numpy as np

from creativity_steer.backends import LLMBackend
from creativity_steer.entailment import EntailmentModel, bidirectional_equivalent


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def reference_distances(
    backend: LLMBackend,
    reference: str,
    candidates: list[str],
    question: str = "",
    entailment: EntailmentModel | None = None,
) -> list[float]:
    """Raw absolute distance (1 - cosine) of each candidate from ``reference``.

    If an entailment model is given, any candidate bidirectionally equivalent to
    the reference is forced to 0.0 (it is the modal idea restated).
    """
    if not candidates:
        return []
    vecs = [np.asarray(v, dtype=float) for v in backend.embed([reference, *candidates])]
    ref, cand_vecs = vecs[0], vecs[1:]
    dists = [1.0 - _cosine(ref, cv) for cv in cand_vecs]
    if entailment is not None:
        for i, cand in enumerate(candidates):
            if bidirectional_equivalent(entailment, question, reference, cand):
                dists[i] = 0.0
    return dists


def novelty_vs_reference(
    backend: LLMBackend,
    reference: str,
    candidates: list[str],
    question: str = "",
    entailment: EntailmentModel | None = None,
) -> list[float]:
    """Max-normalised distances in [0, 1] (selection objective)."""
    dists = reference_distances(backend, reference, candidates, question, entailment)
    max_d = max(dists) if dists else 0.0
    return [d / max_d if max_d > 0 else 0.0 for d in dists]


def normalize_max(dists: list[float]) -> list[float]:
    """Max-normalise a list of distances into [0, 1]."""
    max_d = max(dists) if dists else 0.0
    return [d / max_d if max_d > 0 else 0.0 for d in dists]
