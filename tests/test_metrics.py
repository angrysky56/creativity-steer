"""Tests for entailment clustering, semantic entropy, and the Pareto frontier."""

from __future__ import annotations

import math

from creativity_steer.backends import GenSample, MockBackend, _DEFAULT_POOL
from creativity_steer.divergent import analyze_divergent, cluster_by_entailment
from creativity_steer.entailment import EmbeddingEntailment
from creativity_steer.selection import pareto_mask

Q = "How do you fish keys out of a drain?"
POOL = [t for t, _, _, _ in _DEFAULT_POOL]


def _ent() -> EmbeddingEntailment:
    # Mock embeddings are cluster-aware, so cosine entailment recovers clusters.
    return EmbeddingEntailment(MockBackend(), threshold=0.9)


def test_clustering_groups_equivalent_ideas() -> None:
    # POOL[0] and POOL[1] share cluster 0; POOL[2] is a different cluster.
    ids = cluster_by_entailment(Q, [POOL[0], POOL[1], POOL[2]], _ent())
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]


def test_entropy_zero_when_all_same() -> None:
    samples = [GenSample(POOL[0], math.log(0.2)) for _ in range(5)]
    res = analyze_divergent(Q, samples, _ent())
    assert res.num_clusters == 1
    assert res.semantic_entropy == 0.0
    assert all(n == 0.0 for n in res.novelty)


def test_entropy_positive_when_distinct() -> None:
    samples = [GenSample(POOL[i], math.log(0.25)) for i in (0, 2, 3, 4)]
    res = analyze_divergent(Q, samples, _ent())
    assert res.num_clusters == 4
    assert res.semantic_entropy > 1.0
    assert res.prob_weighted is True


def test_novelty_higher_for_rarer_cluster() -> None:
    # Three copies of cluster 0, one of cluster 1 -> the singleton is novel.
    samples = [GenSample(POOL[0], math.log(0.25)) for _ in range(3)]
    samples.append(GenSample(POOL[2], math.log(0.25)))
    res = analyze_divergent(Q, samples, _ent())
    assert res.novelty[3] > res.novelty[0]
    assert math.isclose(max(res.novelty), 1.0, rel_tol=1e-9)


def test_count_fallback_when_no_logprobs() -> None:
    samples = [GenSample(POOL[i], None) for i in (0, 2, 3)]
    res = analyze_divergent(Q, samples, _ent())
    assert res.prob_weighted is False
    assert res.num_clusters == 3


def test_pareto_mask_basic() -> None:
    pts = [(0.0, 1.0), (1.0, 0.0), (0.5, 0.5), (0.2, 0.2)]
    mask = pareto_mask(pts)
    assert mask[0] and mask[1] and mask[2]
    assert not mask[3]
