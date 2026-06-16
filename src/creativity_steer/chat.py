"""Chat mode: think-and-select for open conversation.

Each turn: produce a modal (greedy) reply, brainstorm K diverse replies, score
every candidate on multiple reference-free axes (novelty, quality, coherence,
…), and Pareto-select. ``chat_turn_stream`` yields process-trace events for a
UI; ``chat_turn`` consumes the stream and returns the assembled result.

Axes live in :mod:`creativity_steer.scoring` (a pluggable registry); selection
generalises to any number of them. See docs/CONCEPT.md and docs/PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from creativity_steer.backends import GenSample, LLMBackend
from creativity_steer.control import TurnController
from creativity_steer.entailment import EntailmentModel
from creativity_steer.grounding import GroundingProvider
from creativity_steer.reference import normalize_max, reference_distances
from creativity_steer.scoring import (
    CoherenceScorer,
    OpennessScorer,
    QualityScorer,
    ScoringContext,
    _history_block,
    funnel_representatives,
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
    novelty_weight: float = 0.5
    coherence_weight: float = 0.0
    openness_weight: float = 0.0
    convergent_floor: float = 0.34
    zero_out_modal_restatements: bool = True
    coherence_paraphrases: int = 2
    openness_branches: int = 0  # 0 = skip the (expensive) openness axis
    max_rounds: int = 2  # controller may explore extra rounds when collapsed
    # breadth -> funnel -> branch -> synthesize (all off by default)
    breadth_k: int = 0  # 0 = single batch of k; else generate ~this many candidates
    prime_n: int = 0  # 0 = no funnel; else keep this many diverse primes
    branch: bool = False  # deepen each prime candidate before scoring
    synthesize: bool = False  # merge the frontier set into one final reply

    def weights(self) -> dict[str, float]:
        return {
            "novelty": self.novelty_weight,
            "quality": 1.0 - self.novelty_weight,
            "coherence": self.coherence_weight,
            "openness": self.openness_weight,
        }


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


def _branch(gen: LLMBackend, history, user_msg: str, seed: str) -> str:
    """Develop a short candidate into a richer, deeper reply (depth creativity)."""
    prompt = (
        f"{_history_block(history)}User: {user_msg}\n\n"
        "Develop this idea into a richer, deeper reply (2-4 sentences) that keeps "
        f"its distinct angle and adds real substance:\n{seed}\nDeeper reply:"
    )
    return gen.chat(prompt, temperature=0.7, num_predict=240).strip()


def _synthesize(gen: LLMBackend, history, user_msg: str, sources: list[str]) -> str:
    """Merge the strongest, most distinct angles into one final reply."""
    listing = "\n".join(f"- {s}" for s in sources)
    prompt = (
        f"{_history_block(history)}User: {user_msg}\n\n"
        f"Here are several distinct angles on a reply:\n{listing}\n\n"
        "Write ONE excellent reply that weaves together the strongest, most "
        "distinct insights from these into a coherent, original answer. Reply only."
    )
    return gen.chat(prompt, temperature=0.6, num_predict=320).strip()


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

    modal = gen_backend.chat(
        modal_prompt_text, temperature=0.0, num_predict=160
    ).strip()
    yield {"type": "modal", "text": modal}

    # ---- generate variants (breadth mode, or controller-adaptive rounds) ----
    variants: list[str] = []
    temp = config.temperature
    rounds_used = 0
    diversity = 0.0
    if config.breadth_k and config.breadth_k > config.k:
        while len(variants) < config.breadth_k:
            rounds_used += 1
            raw = gen_backend.chat(
                brainstorm_prompt_text,
                temperature=temp,
                num_predict=500,
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
                num_predict=500,
            )
            variants += parse_numbered_list(raw, config.k)
            nov = normalize_max(
                reference_distances(embed, modal, [modal, *variants], "", ent)
            )
            diversity = controller.diversity(nov)
            if not controller.should_continue(round_idx, nov, config.max_rounds):
                break
            temp = controller.next_temperature(temp)
    breadth_n = len(variants)

    # ---- funnel to a diverse prime set (keeps expensive scoring bounded) ----
    if config.prime_n and len(variants) > config.prime_n:
        keep = funnel_representatives(embed, modal, variants, config.prime_n)
        variants = [variants[i] for i in keep]

    # ---- branch primes into deeper replies ----
    if config.branch:
        variants = [_branch(gen_backend, history, user_msg, v) for v in variants]

    texts = [modal, *variants]
    modal_flags = [True, *([False] * len(variants))]
    yield {
        "type": "variants",
        "items": [
            {"text": t, "is_modal": f} for t, f in zip(texts, modal_flags, strict=False)
        ],
    }
    yield {
        "type": "controller",
        "rounds": rounds_used,
        "diversity": round(diversity, 3),
        "final_temperature": round(temp, 3),
        "breadth": breadth_n,
        "primes": len(variants),
        "branched": config.branch,
        "weights": config.weights(),
        "quality_floor": config.convergent_floor,
    }

    distances = reference_distances(embed, modal, texts, "", ent)
    novelty = normalize_max(distances)

    ctx = ScoringContext(
        gen=gen_backend,
        judge=judge_backend,
        embed=embed,
        entailment=entailment,
        history=history,
        user_msg=user_msg,
        modal=modal,
        texts=texts,
        samples=[GenSample(t) for t in texts],
        coherence_paraphrases=config.coherence_paraphrases,
    )
    scorers = [QualityScorer(), CoherenceScorer()]
    if config.openness_branches >= 2:
        scorers.append(OpennessScorer(config.openness_branches))
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
    yield {"type": "selected", "index": chosen, "frontier": frontier}

    if config.synthesize:
        prime = [texts[i] for i in range(len(texts)) if frontier[i]] or [texts[chosen]]
        final = _synthesize(gen_backend, history, user_msg, prime)
        yield {"type": "synthesis", "sources": len(prime)}
        yield {"type": "response", "text": final, "index": chosen, "synthesized": True}
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
