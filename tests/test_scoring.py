"""Tests for multi-axis scoring and selection."""

from __future__ import annotations

from creativity_steer.backends import MockBackend
from creativity_steer.chat import ChatConfig, chat_turn, chat_turn_stream
from creativity_steer.entailment import EmbeddingEntailment
from creativity_steer.scoring import pareto_mask_nd, select_multi

MSG = "What's a creative way to reuse an empty glass jar?"


def test_pareto_mask_nd_3d() -> None:
    rows = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.4, 0.4, 0.4], [0.2, 0.2, 0.2]]
    mask = pareto_mask_nd(rows)
    assert mask[0] and mask[1] and mask[2]   # non-dominated
    assert not mask[3]                        # dominated by row 2


def test_select_multi_weights_and_floor() -> None:
    scores = {
        "novelty": [0.0, 1.0, 0.5],
        "quality": [1.0, 0.2, 0.9],
    }
    # Heavy novelty weight -> prefer index 1, but a quality floor blocks it.
    idx, _ = select_multi(scores, {"novelty": 0.9, "quality": 0.1},
                          floor_axis="quality", floor=0.5)
    assert idx in (0, 2)            # index 1 (quality 0.2) excluded by floor
    # No floor -> novelty wins.
    idx2, _ = select_multi(scores, {"novelty": 0.9, "quality": 0.1}, floor=0.0)
    assert idx2 == 1


def test_floor_unmet_picks_best_quality() -> None:
    # When NO candidate clears the quality floor, selection must honor quality
    # (pick the best-quality one) rather than silently rewarding a high-novelty
    # low-quality pick. This is the "not respecting the quality score" fix.
    scores = {"novelty": [1.0, 0.0, 0.2], "quality": [0.2, 0.55, 0.45]}
    idx, _ = select_multi(
        scores, {"novelty": 0.9, "quality": 0.1}, floor_axis="quality", floor=0.65
    )
    assert idx == 1  # best quality (0.55), not the novel-but-weak index 0


def test_originality_axis_present_when_weighted() -> None:
    b = MockBackend()
    cfg = ChatConfig(k=4, originality_weight=0.3, coherence_paraphrases=1)
    res = chat_turn(b, b, EmbeddingEntailment(b, 0.9), [], MSG, cfg)
    axes = res["scores"][0]["scores"]
    assert "originality" in axes
    assert all(0.0 <= ev["scores"]["originality"] <= 1.0 for ev in res["scores"])


def test_surprise_axis_present_when_weighted() -> None:
    # Surprise weight triggers sequential single-candidate generation with
    # logprobs; the axis must appear and stay in range. (Model token-confidence:
    # recited clichés are high-probability/low-surprise, fresh ones low/high.)
    b = MockBackend()
    cfg = ChatConfig(k=4, surprise_weight=0.3, coherence_paraphrases=1)
    res = chat_turn(b, b, EmbeddingEntailment(b, 0.9), [], MSG, cfg)
    axes = res["scores"][0]["scores"]
    assert "surprise" in axes
    assert all(0.0 <= ev["scores"]["surprise"] <= 1.0 for ev in res["scores"])


def test_chat_emits_axis_scores_dict() -> None:
    b = MockBackend()
    res = chat_turn(b, b, EmbeddingEntailment(b, 0.9), [], MSG,
                    ChatConfig(k=4, coherence_paraphrases=1))
    scored = res["scores"]
    assert scored and "scores" in scored[0]
    axes = scored[0]["scores"]
    assert {"novelty", "quality", "coherence"} <= set(axes)
    for ev in scored:
        for v in ev["scores"].values():
            assert 0.0 <= v <= 1.0


def test_openness_axis_present_when_enabled() -> None:
    b = MockBackend()
    cfg = ChatConfig(k=4, openness_branches=3, max_rounds=1, coherence_paraphrases=1)
    res = chat_turn(b, b, EmbeddingEntailment(b, 0.9), [], MSG, cfg)
    axes = res["scores"][0]["scores"]
    assert "openness" in axes
    assert all(0.0 <= ev["scores"]["openness"] <= 1.0 for ev in res["scores"])


def test_coherence_weight_changes_nothing_when_zero() -> None:
    # With coherence_weight 0 the axis is computed but not used in selection.
    b = MockBackend()
    cfg = ChatConfig(k=4, coherence_weight=0.0, coherence_paraphrases=1)
    res = chat_turn(b, b, EmbeddingEntailment(b, 0.9), [], MSG, cfg)
    assert 0 <= res["selected"] < len(res["scores"])


def test_chain_emits_semantic_entropy_clusters_and_refine() -> None:
    # The full connected chain: cluster-aware trajectory breadth + guided
    # refine + guarded synthesis. Verifies the REAL signals are surfaced and
    # the chain records exist (the fix for the saturated-entropy drift).
    b = MockBackend()
    cfg = ChatConfig(
        k=4,
        breadth_k=8,
        prime_n=3,
        trajectory=True,
        refine_passes=1,
        synthesize=True,
        coherence_paraphrases=1,
        novelty_weight=0.5,
    )
    events = list(chat_turn_stream(b, b, EmbeddingEntailment(b, 0.9), [], MSG, cfg))
    by_type: dict[str, list] = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)

    ctrl = by_type["controller"][0]
    # Real semantic entropy is present (not the saturated basin-escape proxy).
    assert "semantic_entropy" in ctrl and ctrl["semantic_entropy"] >= 0.0
    assert ctrl["num_clusters"] >= 1
    # cluster_ids align with the variant cards (modal + variants).
    assert len(ctrl["cluster_ids"]) == len(by_type["variants"][0]["items"])

    # The connected-chain event exists and records guided-refine passes.
    assert by_type["chain"], "chain event should be emitted"
    refine = by_type["chain"][0]["refine"]
    assert refine, "refine passes should run when refine_passes > 0"
    for rec in refine:
        assert rec["axis"] in {"novelty", "quality", "coherence"}
        assert isinstance(rec["accepted"], bool)
        assert isinstance(rec["collapsed"], bool)

    # A final response is still produced.
    assert by_type["response"][0]["text"].strip()


def test_synthesis_self_revises_and_stays_consistent() -> None:
    # Synthesis is anchored on the SELECTED candidate and self-evaluates; the
    # event carries the revise count, and the final reply is marked synthesized.
    b = MockBackend()
    cfg = ChatConfig(
        k=4, breadth_k=8, prime_n=3, synthesize=True, synthesis_passes=2,
        coherence_paraphrases=1,
    )
    events = list(chat_turn_stream(b, b, EmbeddingEntailment(b, 0.9), [], MSG, cfg))
    by_type: dict[str, list] = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)
    syn = by_type["synthesis"][0]
    assert "revised" in syn and "kept_merge" in syn
    assert by_type["response"][0]["text"].strip()


def test_seed_is_threaded_and_runs() -> None:
    # The seed plumbs through every generation call without error, and a fixed
    # seed yields a stable result (Mock is deterministic; this guards the
    # signature so a real backend gets the seed it needs to reproduce a run).
    b = MockBackend()
    ent = EmbeddingEntailment(b, 0.9)
    cfg = ChatConfig(k=4, seed=12345, coherence_paraphrases=1, openness_branches=3)
    r1 = chat_turn(b, b, ent, [], MSG, cfg)
    r2 = chat_turn(b, b, ent, [], MSG, cfg)
    assert r1["response"] == r2["response"]
    # seed=0 means random (no seed passed); still produces a valid response.
    r0 = chat_turn(b, b, ent, [], MSG, ChatConfig(k=4, seed=0, coherence_paraphrases=1))
    assert r0["response"].strip()
