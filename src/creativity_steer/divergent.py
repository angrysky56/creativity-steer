"""Divergent-creativity signal (semantic entropy + per-candidate novelty).

Faithful to the paper: candidates are clustered by *bidirectional entailment*
(paper §3.2), class probabilities are formed from sequence probabilities when
available (paper Eq. 2), and semantic entropy uses the Rao-Blackwellised
estimator that renormalises the class distribution before taking entropy
(paper Appendix C.1, Eq. 4). Each candidate also receives a novelty score equal
to the normalised surprisal of its semantic class, for per-candidate selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from creativity_steer.backends import GenSample
from creativity_steer.entailment import EntailmentModel, bidirectional_equivalent


@dataclass
class DivergentResult:
    """Per-step output of the divergent analysis."""

    cluster_ids: list[int]      # semantic class index per candidate
    semantic_entropy: float     # Rao-Blackwellised entropy (nats)
    novelty: list[float]        # per-candidate novelty in [0, 1]
    num_clusters: int
    prob_weighted: bool         # True if sequence logprobs were used


def cluster_by_entailment(
    question: str, candidates: list[str], model: EntailmentModel
) -> list[int]:
    """Greedy bidirectional-entailment clustering (paper Appendix C.3.3).

    Each candidate is compared against the *first member* of every existing
    class; it joins the first class it is bidirectionally equivalent to, else
    it starts a new class.
    """
    reps: list[str] = []          # first member (representative) of each class
    cluster_ids: list[int] = []
    for cand in candidates:
        assigned = -1
        for idx, rep in enumerate(reps):
            if bidirectional_equivalent(model, question, rep, cand):
                assigned = idx
                break
        if assigned < 0:
            reps.append(cand)
            assigned = len(reps) - 1
        cluster_ids.append(assigned)
    return cluster_ids


def _class_mass(cluster_ids: list[int], samples: list[GenSample]) -> dict[int, float]:
    """Unnormalised probability mass per class.

    Uses summed sequence probabilities when every sample has a logprob
    (paper Eq. 2), otherwise falls back to membership counts.
    """
    have_probs = all(s.logprob is not None for s in samples)
    mass: dict[int, float] = {}
    for cid, s in zip(cluster_ids, samples):
        weight = math.exp(s.logprob) if have_probs else 1.0
        mass[cid] = mass.get(cid, 0.0) + weight
    return mass


def _rao_blackwell_entropy(mass: dict[int, float]) -> float:
    """Entropy of the renormalised class distribution (paper Eq. 4)."""
    total = sum(mass.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for m in mass.values():
        p = m / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def analyze_divergent(
    question: str, samples: list[GenSample], model: EntailmentModel
) -> DivergentResult:
    """Cluster candidates, then compute semantic entropy and novelty."""
    if not samples:
        return DivergentResult([], 0.0, [], 0, False)

    candidates = [s.text for s in samples]
    cluster_ids = cluster_by_entailment(question, candidates, model)
    mass = _class_mass(cluster_ids, samples)
    total = sum(mass.values())
    have_probs = all(s.logprob is not None for s in samples)

    entropy = _rao_blackwell_entropy(mass)

    # Novelty = normalised surprisal of a candidate's class. The rarest class
    # scores 1.0; the modal class scores lowest. Zero when only one class.
    surprisal = {cid: -math.log(m / total) for cid, m in mass.items()}
    max_surprisal = max(surprisal.values()) if surprisal else 0.0
    if max_surprisal <= 0:
        novelty = [0.0] * len(candidates)
    else:
        novelty = [surprisal[cid] / max_surprisal for cid in cluster_ids]

    return DivergentResult(cluster_ids, entropy, novelty, len(mass), have_probs)
