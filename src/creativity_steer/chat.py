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
from creativity_steer.reference import normalize_max, reference_distances
from creativity_steer.scoring import (
    ScoringContext,
    _history_block,
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
    convergent_floor: float = 0.34
    zero_out_modal_restatements: bool = True
    coherence_paraphrases: int = 2
    max_rounds: int = 2  # controller may explore extra rounds when collapsed

    def weights(self) -> dict[str, float]:
        return {
            "novelty": self.novelty_weight,
            "quality": 1.0 - self.novelty_weight,
            "coherence": self.coherence_weight,
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


def chat_turn_stream(
    gen_backend: LLMBackend,
    judge_backend: LLMBackend,
    entailment: EntailmentModel,
    history: list[dict[str, str]],
    user_msg: str,
    config: ChatConfig | None = None,
    embed_backend: LLMBackend | None = None,
    controller: TurnController | None = None,
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

    modal = gen_backend.chat(
        _modal_prompt(history, user_msg), temperature=0.0, num_predict=160
    ).strip()
    yield {"type": "modal", "text": modal}

    # Brainstorm rounds; the controller decides whether to explore another,
    # hotter round when the accumulated set is collapsed.
    texts = [modal]
    modal_flags = [True]
    temp = config.temperature
    distances: list[float] = [0.0]
    novelty: list[float] = [0.0]
    rounds_used = 0
    for round_idx in range(max(1, config.max_rounds)):
        rounds_used += 1
        raw = gen_backend.chat(
            _brainstorm_prompt(history, user_msg, config.k),
            temperature=temp,
            num_predict=500,
        )
        new = parse_numbered_list(raw, config.k)
        texts += new
        modal_flags += [False] * len(new)
        distances = reference_distances(embed, modal, texts, "", ent)
        novelty = normalize_max(distances)
        if not controller.should_continue(round_idx, novelty, config.max_rounds):
            break
        temp = controller.next_temperature(temp)

    yield {
        "type": "variants",
        "items": [
            {"text": t, "is_modal": f} for t, f in zip(texts, modal_flags, strict=False)
        ],
    }
    yield {
        "type": "controller",
        "rounds": rounds_used,
        "diversity": round(controller.diversity(novelty), 3),
        "final_temperature": round(temp, 3),
        "weights": config.weights(),
        "quality_floor": config.convergent_floor,
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
        samples=[GenSample(t) for t in texts],
        coherence_paraphrases=config.coherence_paraphrases,
    )
    scores = {"novelty": novelty, **score_extra(ctx)}

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
