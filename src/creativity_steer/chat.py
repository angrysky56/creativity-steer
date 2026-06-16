"""Chat mode: think-and-select for open conversation.

Each turn: produce a modal (greedy) reply, brainstorm K diverse replies, score
every candidate on multiple reference-free axes (novelty, quality, coherence,
…), and Pareto-select. ``chat_turn_stream`` yields process-trace events for a
UI; ``chat_turn`` consumes the stream and returns the assembled result.

Axes live in :mod:`creativity_steer.scoring` (a pluggable registry); selection
generalises to any number of them. See docs/CONCEPT.md and docs/PLAN.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from creativity_steer.backends import GenSample, LLMBackend
from creativity_steer.control import TurnController
from creativity_steer.divergent import analyze_divergent, cluster_by_entailment
from creativity_steer.entailment import EntailmentModel, bidirectional_equivalent
from creativity_steer.grounding import GroundingProvider
from creativity_steer.reference import normalize_max, reference_distances
from creativity_steer.scoring import (
    CoherenceScorer,
    OpennessScorer,
    OriginalityScorer,
    QualityScorer,
    ScoringContext,
    SurpriseScorer,
    _history_block,
    funnel_representatives,
    judge_comparative,
    score_extra,
    select_multi,
)
from creativity_steer.variants import parse_numbered_list


@dataclass
class ChatConfig:
    """Knobs for a chat turn. ``novelty_weight`` dials novelty vs quality (the
    2-axis default); ``coherence_weight`` adds the basin-depth axis on top."""

    k: int = 5
    temperature: float = 0.9
    seed: int = 0  # 0 = random; any other value makes the run reproducible
    # generation length caps (tokens) — separate from the model context window.
    # A value <= 0 means *no ceiling*: the model's own EOS decides when the
    # reply is complete (bounded only by the context window). oMCD governs
    # deliberation effort (rounds/breadth), not the length of a single reply.
    modal_tokens: int = 256  # greedy baseline restatement (kept short)
    brainstorm_tokens: int = 700  # the numbered seed list (shared budget)
    branch_tokens: int = 0  # deepened prime (real content) -> unbounded
    response_tokens: int = 0  # the final answer (synthesis/chosen) -> unbounded
    novelty_weight: float = 0.5
    coherence_weight: float = 0.0
    openness_weight: float = 0.0
    originality_weight: float = 0.0  # freshness vs. recognised clichés (judge)
    surprise_weight: float = 0.0  # model's own token-confidence (recitation vs.
    #                               composition); generates breadth one-at-a-time
    convergent_floor: float = 0.34
    zero_out_modal_restatements: bool = True
    coherence_paraphrases: int = 2
    openness_branches: int = 0  # 0 = skip the (expensive) openness axis
    max_rounds: int = 2  # controller may explore extra rounds when collapsed
    # breadth -> funnel -> branch/refine -> synthesize (all off by default)
    breadth_k: int = 0  # 0 = single batch of k; else generate ~this many candidates
    prime_n: int = 0  # 0 = no funnel; else keep this many diverse primes
    branch: bool = False  # deepen each prime candidate before scoring
    synthesize: bool = False  # merge the frontier set into one final reply
    # Connected-chain controls (each step is driven by, and checked against, the
    # real signals — semantic clusters + axis scores — so it cannot silently
    # collapse back to the obvious/common answer).
    trajectory: bool = False  # breadth in cluster-aware waves that diverge from
    #                           the meanings already covered (uses entailment SE)
    refine_passes: int = 0  # >0 replaces one-shot branch with critique->revise->
    #                         re-score loops; revisions that collapse onto the
    #                         modal class or fall below the floor are rejected
    trajectory_stall: int = 2  # stop a trajectory early after this many waves
    #                            with no new semantic cluster (it is resampling)

    def weights(self) -> dict[str, float]:
        return {
            "novelty": self.novelty_weight,
            "quality": 1.0 - self.novelty_weight,
            "coherence": self.coherence_weight,
            "openness": self.openness_weight,
            "originality": self.originality_weight,
            "surprise": self.surprise_weight,
        }


def _seed_at(base: int, offset: int = 0) -> int | None:
    """Per-call seed from a base (0 = random -> ``None``). Distinct offsets keep
    sibling samples from duplicating while staying reproducible for a fixed base."""
    return (base + offset) if base else None


def _gen_record(
    gen: LLMBackend,
    prompt: str,
    *,
    temperature: float,
    num_predict: int | None,
    seed: int | None,
    lp_map: dict[str, tuple[float | None, int | None]] | None,
) -> str:
    """Generate text; when ``lp_map`` is given, also capture the generated-token
    logprobs (for the surprise axis) keyed by the produced text."""
    if lp_map is not None:
        s = gen.chat_logprob(
            prompt, temperature=temperature, num_predict=num_predict, seed=seed
        )
        text = s.text.strip()
        if text:
            lp_map[text] = (s.logprob, s.n_tokens)
        return text
    return gen.chat(
        prompt, temperature=temperature, num_predict=num_predict, seed=seed
    ).strip()


def _one_creative_prompt(history: list[dict[str, str]], user_msg: str) -> str:
    return (
        f"{_history_block(history)}User: {user_msg}\n\n"
        "Write ONE creative, original reply — avoid the obvious or common "
        "answer. Reply only."
    )


def _modal_prompt(history: list[dict[str, str]], user_msg: str) -> str:
    return (
        f"{_history_block(history)}User: {user_msg}\n\n"
        "Reply directly and helpfully in 1-3 sentences. Reply only."
    )


def _brainstorm_prompt(history: list[dict[str, str]], user_msg: str, k: int) -> str:
    return (
        f"{_history_block(history)}User: {user_msg}\n\n"
        f"BRAINSTORM TASK: Write exactly {k} genuinely DIFFERENT replies to the "
        "user, each taking a distinct angle, tone, or idea (not reworded "
        "versions of one reply). Each reply 1-3 sentences.\n"
        "Return them as a numbered list and nothing else:\n"
        "1) <reply one>\n2) <reply two>\n..."
    )


def _branch(
    gen: LLMBackend,
    history,
    user_msg: str,
    seed: str,
    max_tokens: int = 512,
    rng_seed: int | None = None,
    lp_map: dict[str, tuple[float | None, int | None]] | None = None,
) -> str:
    """Develop a short candidate into a richer, deeper reply (depth creativity)."""
    prompt = (
        f"{_history_block(history)}User: {user_msg}\n\n"
        "Develop this idea into a richer, deeper reply (2-4 sentences) that keeps "
        f"its distinct angle and adds real substance:\n{seed}\nDeeper reply:"
    )
    return _gen_record(
        gen,
        prompt,
        temperature=0.7,
        num_predict=max_tokens,
        seed=rng_seed,
        lp_map=lp_map,
    )


def _cluster_reps(
    question: str, items: list[str], entailment: EntailmentModel
) -> list[str]:
    """First member (representative) of each distinct semantic class in ``items``."""
    if not items:
        return []
    ids = cluster_by_entailment(question, items, entailment)
    seen: dict[int, str] = {}
    for cid, text in zip(ids, items, strict=False):
        seen.setdefault(cid, text)
    return list(seen.values())


def _diverge_prompt(history, user_msg: str, k: int, existing_reps: list[str]) -> str:
    """Brainstorm prompt that names the meanings already covered and asks for
    genuinely new semantic regions (the cluster-aware trajectory step)."""
    avoid = "\n".join(f"- {r}" for r in existing_reps)
    return (
        f"{_history_block(history)}User: {user_msg}\n\n"
        f"DISTINCT angles already proposed (do NOT repeat or reword these):\n{avoid}\n\n"
        f"Propose {k} MORE replies, each occupying a genuinely NEW conceptual "
        "region — a different frame, mechanism, or stance — not equivalent to "
        "any above. No rephrasings, no safe restatements. Numbered list only:\n"
        "1) <reply one>\n2) <reply two>\n..."
    )


_REFINE_GUIDANCE = {
    "novelty": (
        "Its weakest MEASURED dimension is NOVELTY: it sits too close to the "
        "obvious baseline. Rewrite it to occupy a genuinely different conceptual "
        "region — a distinct frame, mechanism, or angle — staying on-topic. Do "
        "NOT make it safer, milder, or more generic."
    ),
    "coherence": (
        "Its weakest MEASURED dimension is COHERENCE: the idea is unstable or "
        "muddled. Rewrite it so ONE clear, self-consistent idea lands harder — "
        "without flattening into a platitude."
    ),
    "quality": (
        "Its weakest MEASURED dimension is SUBSTANCE: it is thin or generic. "
        "Rewrite it with sharper specificity, insight, and depth. Do NOT retreat "
        "to a safe, common answer."
    ),
}


def _revise_for_axis(
    gen: LLMBackend,
    history,
    user_msg: str,
    modal: str,
    cand: str,
    axis: str,
    max_tokens: int,
    rng_seed: int | None = None,
    lp_map: dict[str, tuple[float | None, int | None]] | None = None,
) -> str:
    """One critique-and-revise step targeting the candidate's weakest axis.

    The instruction pushes *further out* on the measured axis and explicitly
    forbids retreating to the safe/common answer — exploration as master,
    approval never the goal.
    """
    guidance = _REFINE_GUIDANCE.get(axis, _REFINE_GUIDANCE["quality"])
    prompt = (
        f"{_history_block(history)}User: {user_msg}\n\n"
        f"Obvious baseline answer to diverge FROM:\n{modal}\n\n"
        f"Current reply:\n{cand}\n\n{guidance}\nRewrite (reply only):"
    )
    return _gen_record(
        gen,
        prompt,
        temperature=0.8,
        num_predict=max_tokens,
        seed=rng_seed,
        lp_map=lp_map,
    )


def _novelty_vs_modal(
    embed: LLMBackend, modal: str, cands: list[str], ent: EntailmentModel | None
) -> list[float]:
    """Per-candidate novelty in [0,1] relative to the modal (modal excluded)."""
    if not cands:
        return []
    dist = reference_distances(embed, modal, [modal, *cands], "", ent)
    return normalize_max(dist)[1:]


def _refine_primes(
    gen: LLMBackend,
    judge: LLMBackend,
    embed: LLMBackend,
    entailment: EntailmentModel,
    history,
    user_msg: str,
    modal: str,
    primes: list[str],
    passes: int,
    weights: dict[str, float],
    floor: float,
    ent: EntailmentModel | None,
    max_tokens: int,
    base_seed: int = 0,
    lp_map: dict[str, tuple[float | None, int | None]] | None = None,
) -> tuple[list[str], list[dict]]:
    """Critique -> revise -> re-score each prime, ``passes`` times.

    A revision is ACCEPTED only if it (a) does not bidirectionally entail the
    modal (i.e. did not collapse onto the obvious answer), (b) stays above the
    quality floor, and (c) improves the weighted axis score. Otherwise the prior
    version is kept. This is the chain that builds depth *using* the signals.
    """
    cur = list(primes)
    events: list[dict] = []
    w_n = weights.get("novelty", 0.5)
    w_q = weights.get("quality", 0.5)
    for p in range(max(0, passes)):
        nov = _novelty_vs_modal(embed, modal, cur, ent)
        qual = judge_comparative(judge, history, user_msg, [modal, *cur])[1:]
        weakest = [
            "novelty" if n * w_n <= q * w_q else "quality"
            for n, q in zip(nov, qual, strict=False)
        ]
        revised = [
            _revise_for_axis(
                gen,
                history,
                user_msg,
                modal,
                c,
                w,
                max_tokens,
                rng_seed=_seed_at(base_seed, 1000 + p * 100 + i),
                lp_map=lp_map,
            )
            for i, (c, w) in enumerate(zip(cur, weakest, strict=False))
        ]
        collapsed = [
            bidirectional_equivalent(entailment, user_msg, modal, r) for r in revised
        ]
        nov2 = _novelty_vs_modal(embed, modal, revised, ent)
        qual2 = judge_comparative(judge, history, user_msg, [modal, *revised])[1:]
        accepted = 0
        for i, _ in enumerate(cur):
            old_s = nov[i] * w_n + qual[i] * w_q
            new_s = nov2[i] * w_n + qual2[i] * w_q
            keep = (not collapsed[i]) and qual2[i] >= floor and new_s >= old_s
            if keep:
                cur[i] = revised[i]
                accepted += 1
            events.append(
                {
                    "pass": p,
                    "prime": i,
                    "axis": weakest[i],
                    "accepted": keep,
                    "collapsed": collapsed[i],
                    "old": round(old_s, 3),
                    "new": round(new_s, 3),
                }
            )
        if accepted == 0:  # nothing improved this pass; further passes won't help
            break
    return cur, events


def _synthesize(
    gen: LLMBackend,
    history,
    user_msg: str,
    sources: list[str],
    max_tokens: int = 1024,
    rng_seed: int | None = None,
) -> str:
    """Integrate the strongest distinct angles into ONE argument (not a stitch).

    The prompt forces integration into a single line of reasoning rather than
    listing or concatenating the angles.
    """
    listing = "\n".join(f"- {s}" for s in sources)
    prompt = (
        f"{_history_block(history)}User: {user_msg}\n\n"
        f"Several DISTINCT angles on a reply:\n{listing}\n\n"
        "Write ONE reply that INTEGRATES the strongest insight from these into a "
        "single, coherent line of reasoning — not a list, not a tour of each "
        "angle, but one unified original answer in which the ideas genuinely "
        "build on each other. Do not hedge or fall back to the generic. Reply only."
    )
    return gen.chat(
        prompt, temperature=0.6, num_predict=max_tokens, seed=rng_seed
    ).strip()


def chat_turn_stream(
    gen_backend: LLMBackend,
    judge_backend: LLMBackend,
    entailment: EntailmentModel,
    history: list[dict[str, str]],
    user_msg: str,
    config: ChatConfig | None = None,
    embed_backend: LLMBackend | None = None,
    controller: TurnController | None = None,
    grounding: GroundingProvider | None = None,
):
    """Yield process-trace events for one chat turn, ending with the response.

    The controller may run extra, hotter brainstorm rounds when the set
    collapses onto the modal basin (self-tuning exploration). Event types:
    ``modal``, ``variants``, ``controller`` (additive metadata), ``scored``
    (per candidate, carrying a ``scores`` axis dict plus back-compat
    ``novelty``/``distance``/``quality``), ``selected``, ``response``.
    """
    config = config or ChatConfig()
    controller = controller or TurnController()
    embed = embed_backend or gen_backend
    ent = entailment if config.zero_out_modal_restatements else None

    ctx_block = ""
    if grounding:
        grounding_ctx = grounding.gather(user_msg, history)
        ctx_block = grounding_ctx.block()
        yield {
            "type": "grounding",
            "memory": len(grounding_ctx.memory),
            "tools": len(grounding_ctx.tool_results),
            "snippets": [m.content for m in grounding_ctx.memory]
            + [t.get("text", str(t)) for t in grounding_ctx.tool_results],
        }

    modal_prompt_text = ctx_block + _modal_prompt(history, user_msg)
    brainstorm_prompt_text = ctx_block + _brainstorm_prompt(history, user_msg, config.k)

    # Surprise axis (model token-confidence) needs clean per-candidate logprobs,
    # so it generates breadth one-at-a-time (still conditioned to diverge). The
    # ``lp_map`` collects logprobs for every text we generate this way.
    want_lp = config.surprise_weight > 0
    lp_map: dict[str, tuple[float | None, int | None]] = {}

    modal = _gen_record(
        gen_backend,
        modal_prompt_text,
        temperature=0.0,
        num_predict=config.modal_tokens,
        seed=_seed_at(config.seed),
        lp_map=lp_map if want_lp else None,
    )
    yield {"type": "modal", "text": modal}

    # ---- generate variants (breadth mode, or controller-adaptive rounds) ----
    variants: list[str] = []
    temp = config.temperature
    rounds_used = 0
    traj_waves: list[dict] = []
    breadth_mode = bool(config.breadth_k and config.breadth_k > config.k)
    if want_lp:
        # Sequential, conditioned, single-candidate generation: each call sees
        # the distinct meanings already produced and is told to diverge, AND
        # returns its own clean token logprobs. Conditioning supplies diversity;
        # the single call supplies the measurement.
        target = (
            config.breadth_k
            if (config.breadth_k and config.breadth_k > config.k)
            else config.k
        )
        stall = 0
        best_clusters = 0
        guard = 0
        while len(variants) < target and guard < target * 3 + 3:
            guard += 1
            rounds_used += 1
            reps = _cluster_reps(user_msg, variants, entailment)
            prompt = ctx_block + (
                _diverge_prompt(history, user_msg, 1, reps)
                if reps
                else _one_creative_prompt(history, user_msg)
            )
            s = gen_backend.chat_logprob(
                prompt,
                temperature=temp,
                num_predict=config.brainstorm_tokens,
                seed=_seed_at(config.seed, rounds_used),
            )
            cands = parse_numbered_list(s.text, 1) or (
                [s.text.strip()] if s.text.strip() else []
            )
            for t in cands:
                lp_map[t] = (s.logprob, s.n_tokens)
            variants += cands
            n_clusters = len(_cluster_reps(user_msg, variants, entailment))
            traj_waves.append(
                {"wave": rounds_used, "pool": len(variants), "clusters": n_clusters}
            )
            if n_clusters > best_clusters:
                best_clusters = n_clusters
                stall = 0
            else:
                stall += 1
            if len(variants) >= config.k and stall >= max(1, config.trajectory_stall):
                break
            temp = min(1.3, round(temp + 0.1, 3))
        variants = variants[:target]
    elif breadth_mode and config.trajectory:
        # Cluster-aware trajectory: each wave is told the meanings already
        # covered and asked to occupy NEW semantic regions. Stop early if the
        # cluster count stalls (the model is just resampling the common).
        stall = 0
        best_clusters = 0  # most distinct meanings seen so far
        while len(variants) < config.breadth_k:
            rounds_used += 1
            reps = _cluster_reps(user_msg, variants, entailment)
            prompt = (
                ctx_block + _diverge_prompt(history, user_msg, config.k, reps)
                if reps
                else brainstorm_prompt_text
            )
            raw = gen_backend.chat(
                prompt,
                temperature=temp,
                num_predict=config.brainstorm_tokens,
                seed=_seed_at(config.seed, rounds_used),
            )
            variants += parse_numbered_list(raw, config.k)
            n_clusters = len(_cluster_reps(user_msg, variants, entailment))
            traj_waves.append(
                {"wave": rounds_used, "pool": len(variants), "clusters": n_clusters}
            )
            # Stall = no NEW distinct meaning vs the best so far (robust to the
            # cluster count oscillating up and down between waves).
            if n_clusters > best_clusters:
                best_clusters = n_clusters
                stall = 0
            else:
                stall += 1
            if stall >= max(1, config.trajectory_stall):
                break
            temp = min(1.3, round(temp + 0.1, 3))
        variants = variants[: config.breadth_k]
    elif breadth_mode:
        while len(variants) < config.breadth_k:
            rounds_used += 1
            raw = gen_backend.chat(
                brainstorm_prompt_text,
                temperature=temp,
                num_predict=config.brainstorm_tokens,
                seed=_seed_at(config.seed, rounds_used),
            )
            variants += parse_numbered_list(raw, config.k)
            temp = min(1.3, round(temp + 0.1, 3))
        variants = variants[: config.breadth_k]
    else:
        for round_idx in range(max(1, config.max_rounds)):
            rounds_used += 1
            raw = gen_backend.chat(
                brainstorm_prompt_text,
                temperature=temp,
                num_predict=config.brainstorm_tokens,
                seed=_seed_at(config.seed, rounds_used),
            )
            variants += parse_numbered_list(raw, config.k)
            nov = normalize_max(
                reference_distances(embed, modal, [modal, *variants], "", ent)
            )
            if not controller.should_continue(round_idx, nov, config.max_rounds):
                break
            temp = controller.next_temperature(temp)
    breadth_n = len(variants)
    breadth_pool = list(variants)  # full pool BEFORE the diversity funnel

    # ---- funnel to a diverse prime set (keeps expensive scoring bounded) ----
    if config.prime_n and len(variants) > config.prime_n:
        keep = funnel_representatives(embed, modal, variants, config.prime_n)
        variants = [variants[i] for i in keep]

    # ---- deepen primes: guided refine chain (preferred) or one-shot branch ----
    refine_events: list[dict] = []
    if config.refine_passes > 0 and variants:
        variants, refine_events = _refine_primes(
            gen_backend,
            judge_backend,
            embed,
            entailment,
            history,
            user_msg,
            modal,
            variants,
            config.refine_passes,
            config.weights(),
            config.convergent_floor,
            ent,
            config.branch_tokens,
            base_seed=config.seed,
            lp_map=lp_map if want_lp else None,
        )
    elif config.branch:
        variants = [
            _branch(
                gen_backend,
                history,
                user_msg,
                v,
                config.branch_tokens,
                rng_seed=_seed_at(config.seed, 500 + i),
                lp_map=lp_map if want_lp else None,
            )
            for i, v in enumerate(variants)
        ]

    texts = [modal, *variants]
    modal_flags = [True, *([False] * len(variants))]

    # Novelty over the FINAL candidate set (embedding distance from the modal).
    distances = reference_distances(embed, modal, texts, "", ent)
    novelty = normalize_max(distances)
    basin_escape = controller.diversity(novelty)  # fraction off the modal basin

    # The paper's actual divergent signal: cluster the candidate pool by
    # bidirectional entailment and take the Rao-Blackwellised semantic entropy.
    # This SEES mode collapse (many phrasings of ONE idea -> few clusters -> low
    # entropy) which the basin-escape proxy cannot. Computed over the FULL
    # breadth pool (before the funnel deliberately spread them out) so it
    # reflects how many genuinely distinct ideas the model actually produced.
    div = analyze_divergent(user_msg, [GenSample(t) for t in breadth_pool], entailment)
    n_cand = max(1, len(breadth_pool))
    max_entropy = math.log(n_cand) if n_cand > 1 else 0.0
    norm_entropy = (div.semantic_entropy / max_entropy) if max_entropy > 0 else 0.0
    # Separate clustering over the DISPLAYED candidates for the card tags
    # (prefix the modal's own class with -1 so indices line up with `texts`).
    display_ids = cluster_by_entailment(user_msg, variants, entailment)
    cluster_ids = [-1, *display_ids]

    yield {
        "type": "variants",
        "items": [
            {"text": t, "is_modal": f} for t, f in zip(texts, modal_flags, strict=False)
        ],
    }
    yield {
        "type": "controller",
        "mode": "breadth" if breadth_mode else "adaptive",
        "rounds": rounds_used,
        # Real semantic entropy (paper Eq. 4), plus a [0,1] normalised version.
        "semantic_entropy": round(div.semantic_entropy, 3),
        "norm_entropy": round(norm_entropy, 3),
        "num_clusters": div.num_clusters,
        "num_candidates": len(breadth_pool),
        "prob_weighted": div.prob_weighted,
        "cluster_ids": cluster_ids,
        "basin_escape": round(basin_escape, 3),
        # Back-compat: the "diversity" the panel historically showed is now the
        # normalised semantic entropy (a true, non-saturating divergence score).
        "diversity": round(norm_entropy, 3),
        "final_temperature": round(temp, 3),
        "breadth": breadth_n,
        "primes": len(variants),
        "branched": config.branch,
        "trajectory": config.trajectory,
        "trajectory_waves": len(traj_waves),
        "refine_passes": config.refine_passes,
        "refine_accepted": sum(1 for e in refine_events if e["accepted"]),
        "refine_collapsed": sum(1 for e in refine_events if e["collapsed"]),
        "refine_total": len(refine_events),
        "weights": config.weights(),
        "quality_floor": config.convergent_floor,
    }

    if traj_waves or refine_events:
        yield {
            "type": "chain",
            "trajectory_waves": traj_waves,
            "refine": refine_events,
        }

    ctx = ScoringContext(
        gen=gen_backend,
        judge=judge_backend,
        embed=embed,
        entailment=entailment,
        history=history,
        user_msg=user_msg,
        modal=modal,
        texts=texts,
        samples=[GenSample(t, *lp_map.get(t, (None, None))) for t in texts],
        coherence_paraphrases=config.coherence_paraphrases,
        seed=(config.seed or None),
    )
    scorers = [QualityScorer(), CoherenceScorer()]
    if config.openness_branches >= 2:
        scorers.append(OpennessScorer(config.openness_branches))
    if config.originality_weight > 0:
        scorers.append(OriginalityScorer())
    if config.surprise_weight > 0:
        scorers.append(SurpriseScorer())
    scores = {"novelty": novelty, **score_extra(ctx, scorers)}

    for i in range(len(texts)):
        yield {
            "type": "scored",
            "index": i,
            "novelty": novelty[i],
            "distance": distances[i],
            "quality": scores["quality"][i],
            "scores": {a: scores[a][i] for a in scores},
        }

    chosen, frontier = select_multi(
        scores, config.weights(), floor_axis="quality", floor=config.convergent_floor
    )
    floor_met = scores["quality"][chosen] >= config.convergent_floor
    yield {
        "type": "selected",
        "index": chosen,
        "frontier": frontier,
        "floor_met": floor_met,
        "chosen_quality": round(scores["quality"][chosen], 3),
    }

    if config.synthesize:
        prime = [texts[i] for i in range(len(texts)) if frontier[i]] or [texts[chosen]]
        final = _synthesize(
            gen_backend,
            history,
            user_msg,
            prime,
            config.response_tokens,
            rng_seed=_seed_at(config.seed, 9000),
        )
        # Anti-collapse guard: if the merge fell back to the obvious answer
        # (bidirectionally entails the modal), keep the selected frontier
        # candidate instead — synthesis must not launder the common.
        collapsed = bidirectional_equivalent(entailment, user_msg, modal, final)
        if collapsed:
            final = texts[chosen]
        yield {
            "type": "synthesis",
            "sources": len(prime),
            "collapsed_to_modal": collapsed,
        }
        yield {
            "type": "response",
            "text": final,
            "index": chosen,
            "synthesized": not collapsed,
        }
    else:
        yield {"type": "response", "text": texts[chosen], "index": chosen}


def chat_turn(
    gen_backend: LLMBackend,
    judge_backend: LLMBackend,
    entailment: EntailmentModel,
    history: list[dict[str, str]],
    user_msg: str,
    config: ChatConfig | None = None,
    embed_backend: LLMBackend | None = None,
    controller: TurnController | None = None,
    grounding: GroundingProvider | None = None,
) -> dict:
    """Run a chat turn and return the assembled result (consumes the stream)."""
    events = list(
        chat_turn_stream(
            gen_backend,
            judge_backend,
            entailment,
            history,
            user_msg,
            config,
            embed_backend,
            controller,
            grounding,
        )
    )
    by_type: dict[str, list] = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)
    response = by_type["response"][0]
    return {
        "modal": by_type["modal"][0]["text"],
        "variants": by_type["variants"][0]["items"],
        "scores": by_type.get("scored", []),
        "selected": by_type["selected"][0]["index"],
        "response": response["text"],
    }
