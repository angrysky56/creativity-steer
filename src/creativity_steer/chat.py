"""Chat mode: think-and-select for open conversation.

Generalises the Stage 1 pipeline from "propose a next step" to "respond to a
user message". Each turn:

1. produce the MODAL reply (greedy) -- the reference;
2. brainstorm K diverse replies in one call;
3. score novelty as distance from the modal reply, quality with a chat rubric
   (relevance / helpfulness / coherence);
4. Pareto-select (novelty, quality) above the floor;
5. emit the chosen reply.

``chat_turn_stream`` yields process-trace events so a UI can show the model
"thinking"; ``chat_turn`` consumes the stream and returns the final result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from creativity_steer.backends import LLMBackend
from creativity_steer.entailment import EntailmentModel
from creativity_steer.reference import normalize_max, reference_distances
from creativity_steer.selection import pareto_mask
from creativity_steer.variants import parse_numbered_list

CHAT_CRITERIA: tuple[str, ...] = ("relevance", "helpfulness", "coherence")
_VERDICT_RE = re.compile(r"\[\[\s*(YES|NO)\s*\]\]", re.IGNORECASE)


@dataclass
class ChatConfig:
    """Knobs for a chat turn."""

    k: int = 5
    temperature: float = 0.9
    criteria: tuple[str, ...] = CHAT_CRITERIA
    novelty_weight: float = 0.5
    convergent_floor: float = 0.34
    zero_out_modal_restatements: bool = True


def _history_block(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in history]
    return "CONVERSATION SO FAR:\n" + "\n".join(lines) + "\n\n"


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


def _judge_prompt(
    history: list[dict[str, str]],
    user_msg: str,
    reply: str,
    criteria: tuple[str, ...],
) -> str:
    qs = {
        "relevance": "Relevance (does it address what the user said?)",
        "helpfulness": "Helpfulness (is it useful or substantive?)",
        "coherence": "Coherence (is it clear and well-formed?)",
    }
    lines = "\n".join(f"{c.capitalize()}: {qs.get(c, c)}" for c in criteria)
    fmt = ", ".join(f"{c.capitalize()}: [[YES/NO]]" for c in criteria)
    return (
        "Judge a candidate reply to a user message. RATE THE REPLY on each "
        f"criterion.\n{_history_block(history)}User: {user_msg}\n\n"
        f"CANDIDATE REPLY:\n{reply}\n\nCriteria:\n{lines}\n\n"
        f"After a brief note, end with STRICTLY: {fmt}"
    )


def _judge_one(
    judge: LLMBackend,
    history: list[dict[str, str]],
    user_msg: str,
    reply: str,
    criteria: tuple[str, ...],
) -> float:
    """Fraction of chat criteria judged YES for one candidate reply."""
    raw = judge.chat(
        _judge_prompt(history, user_msg, reply, criteria),
        temperature=0.0,
        num_predict=200,
    )
    verdicts = _VERDICT_RE.findall(raw)
    yes = sum(
        1
        for i in range(len(criteria))
        if i < len(verdicts) and verdicts[i].upper() == "YES"
    )
    return yes / len(criteria) if criteria else 0.0


def _select(
    novelty: list[float],
    convergent: list[float],
    frontier: list[bool],
    config: ChatConfig,
) -> int:
    """Pareto pick on (novelty, quality) above the floor; quality fallback."""
    eligible = [
        i
        for i in range(len(novelty))
        if frontier[i] and convergent[i] >= config.convergent_floor
    ]
    if not eligible:
        return max(range(len(convergent)), key=lambda i: convergent[i])
    w = config.novelty_weight
    return max(eligible, key=lambda i: w * novelty[i] + (1.0 - w) * convergent[i])


def chat_turn_stream(
    gen_backend: LLMBackend,
    judge_backend: LLMBackend,
    entailment: EntailmentModel,
    history: list[dict[str, str]],
    user_msg: str,
    config: ChatConfig | None = None,
    embed_backend: LLMBackend | None = None,
):
    """Yield process-trace events for one chat turn, ending with the response.

    ``embed_backend`` (default: ``gen_backend``) supplies embeddings for the
    novelty signal, so embeddings can stay local even when generation runs on
    a remote/trained model.

    Event types: ``modal``, ``variants``, ``scored`` (per candidate),
    ``selected``, ``response``.
    """
    config = config or ChatConfig()
    embed = embed_backend or gen_backend

    modal = gen_backend.chat(
        _modal_prompt(history, user_msg), temperature=0.0, num_predict=160
    ).strip()
    yield {"type": "modal", "text": modal}

    raw = gen_backend.chat(
        _brainstorm_prompt(history, user_msg, config.k),
        temperature=config.temperature,
        num_predict=500,
    )
    variants = parse_numbered_list(raw, config.k)
    texts = [modal, *variants]
    modal_flags = [True, *([False] * len(variants))]
    yield {
        "type": "variants",
        "items": [
            {"text": t, "is_modal": f} for t, f in zip(texts, modal_flags, strict=False)
        ],
    }

    ent = entailment if config.zero_out_modal_restatements else None
    distances = reference_distances(embed, modal, texts, "", ent)
    novelty = normalize_max(distances)

    convergent: list[float] = []
    for i, t in enumerate(texts):
        q = _judge_one(judge_backend, history, user_msg, t, config.criteria)
        convergent.append(q)
        yield {
            "type": "scored",
            "index": i,
            "novelty": novelty[i],
            "distance": distances[i],
            "quality": q,
        }

    frontier = pareto_mask(list(zip(novelty, convergent, strict=False)))
    chosen = _select(novelty, convergent, frontier, config)
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
