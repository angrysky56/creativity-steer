"""Selection axes (scorers) and multi-axis Pareto selection.

Each axis is a reference-free ``Scorer`` mapping the turn's candidates to a
value in [0, 1]. Adding a creative dimension = adding a scorer; the selection
generalises to any number of axes. See docs/CONCEPT.md for the framing
(attractor basins, counterfactual surfacing) and docs/PLAN.md for the roadmap.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from creativity_steer.backends import GenSample, LLMBackend
from creativity_steer.entailment import EntailmentModel


@dataclass
class ScoringContext:
    """Everything a scorer might need for one turn."""

    gen: LLMBackend
    judge: LLMBackend
    embed: LLMBackend
    entailment: EntailmentModel
    history: list[dict[str, str]]
    user_msg: str
    modal: str
    texts: list[str]              # modal + variants
    samples: list[GenSample]      # parallel to texts; logprob may be None
    coherence_paraphrases: int = 2


class Scorer(Protocol):
    """Maps candidates to one [0, 1] value each (higher = more of the axis)."""

    name: str

    def score(self, ctx: ScoringContext) -> list[float]: ...


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _history_block(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in history]
    return "CONVERSATION SO FAR:\n" + "\n".join(lines) + "\n\n"


# --------------------------------------------------------------------------- #
# Quality (convergent) — comparative, graded, critical                        #
# --------------------------------------------------------------------------- #

_SCORE_NUM_RE = re.compile(r'"score"\s*:\s*([01](?:\.\d+)?)')

_COMPARATIVE_INSTR = (
    "You are a demanding, skeptical critic. Score each candidate reply from 0.0 "
    "to 1.0 on SUBSTANTIVE quality for THIS question — insight, specificity, "
    "depth, and how non-generic it is. Be discriminating and USE THE FULL "
    "RANGE: reserve 0.85+ for genuinely exceptional replies; a competent but "
    "generic reply scores ~0.4-0.6; vague, shallow, or flawed replies lower. "
    "Do not give everything the same score. Name each reply's single biggest "
    "weakness in <=6 words, then score it.\n"
    "Return ONLY a JSON array, one object per reply IN ORDER:\n"
    '[{"i":1,"weakness":"...","score":0.5}, ...]'
)


def _parse_scores(raw: str, n: int) -> list[float]:
    """Extract n scores in [0, 1] from a JSON array (best-effort)."""
    scores: list[float] = []
    start, end = raw.find("["), raw.rfind("]")
    if 0 <= start < end:
        try:
            arr = json.loads(raw[start : end + 1])
            scores = [float(o.get("score", 0.5)) for o in arr if isinstance(o, dict)]
        except (json.JSONDecodeError, TypeError, ValueError):
            scores = []
    if not scores:
        scores = [float(m) for m in _SCORE_NUM_RE.findall(raw)]
    scores = [min(1.0, max(0.0, s)) for s in scores]
    if len(scores) < n:
        scores += [0.5] * (n - len(scores))
    return scores[:n]


def judge_comparative(
    judge: LLMBackend,
    history: list[dict[str, str]],
    user_msg: str,
    candidates: list[str],
) -> list[float]:
    """Critically score all candidates in ONE comparative pass (defeats the
    saturation of an absolute yes/no rubric and self-judge leniency)."""
    listing = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    prompt = (
        f"{_history_block(history)}User question:\n{user_msg}\n\n"
        f"Candidate replies:\n{listing}\n\n{_COMPARATIVE_INSTR}"
    )
    raw = judge.chat(prompt, temperature=0.0, num_predict=40 * len(candidates) + 160)
    return _parse_scores(raw, len(candidates))


class QualityScorer:
    name = "quality"

    def score(self, ctx: ScoringContext) -> list[float]:
        return judge_comparative(ctx.judge, ctx.history, ctx.user_msg, ctx.texts)


# --------------------------------------------------------------------------- #
# Coherence (basin depth) — self-consistency under paraphrase                 #
# --------------------------------------------------------------------------- #

_PARAPHRASE_PROMPT = (
    "Restate the following in different words, same meaning, one sentence:\n"
    "{text}\nRestatement:"
)


class CoherenceScorer:
    """A stable idea (deep attractor) re-paraphrases back to itself; noise
    (shallow fluctuation) scatters. Score = mean pairwise cosine of the
    candidate and its paraphrases. The anti-Goodhart axis."""

    name = "coherence"

    def score(self, ctx: ScoringContext) -> list[float]:
        p = max(0, ctx.coherence_paraphrases)
        if p == 0:
            return [1.0] * len(ctx.texts)
        out: list[float] = []
        for t in ctx.texts:
            paras = [
                ctx.gen.chat(
                    _PARAPHRASE_PROMPT.format(text=t), temperature=0.7, num_predict=80
                ).strip()
                for _ in range(p)
            ]
            vecs = [np.asarray(v, dtype=float) for v in ctx.embed.embed([t, *paras])]
            sims = [
                _cosine(vecs[i], vecs[j])
                for i in range(len(vecs))
                for j in range(i + 1, len(vecs))
            ]
            out.append(min(1.0, max(0.0, sum(sims) / len(sims))) if sims else 1.0)
        return out


# --------------------------------------------------------------------------- #
# Surprise (information gain) — sequence improbability                        #
# --------------------------------------------------------------------------- #


class SurpriseScorer:
    """Less probable = more surprising. Uses per-candidate sequence logprobs
    when available (e.g. independent-sample generation); returns neutral 0.5
    when they are not (e.g. a single brainstorm call has no per-variant logprob)."""

    name = "surprise"

    def score(self, ctx: ScoringContext) -> list[float]:
        lps = [s.logprob for s in ctx.samples]
        if any(lp is None for lp in lps) or len(lps) < 2:
            return [0.5] * len(ctx.texts)
        vals = [-lp for lp in lps]  # higher = more surprising
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return [0.5] * len(ctx.texts)
        return [(v - lo) / (hi - lo) for v in vals]


# --------------------------------------------------------------------------- #
# Multi-axis selection                                                        #
# --------------------------------------------------------------------------- #


def pareto_mask_nd(rows: list[list[float]]) -> list[bool]:
    """Non-dominated mask for maximising every axis."""
    mask = [True] * len(rows)
    for i, ri in enumerate(rows):
        for j, rj in enumerate(rows):
            if i == j:
                continue
            if all(rj[k] >= ri[k] for k in range(len(ri))) and any(
                rj[k] > ri[k] for k in range(len(ri))
            ):
                mask[i] = False
                break
    return mask


def select_multi(
    scores: dict[str, list[float]],
    weights: dict[str, float],
    floor_axis: str = "quality",
    floor: float = 0.0,
) -> tuple[int, list[bool]]:
    """Pareto-select over weighted axes, above a floor on ``floor_axis``."""
    axes = [a for a in weights if a in scores and weights[a] > 0]
    n = len(next(iter(scores.values()))) if scores else 0
    if not axes or n == 0:
        return 0, [True] * n
    rows = [[scores[a][i] for a in axes] for i in range(n)]
    frontier = pareto_mask_nd(rows)
    floor_vals = scores.get(floor_axis, [1.0] * n)
    eligible = [i for i in range(n) if frontier[i] and floor_vals[i] >= floor]
    if not eligible:
        eligible = list(range(n))
    wsum = sum(weights[a] for a in axes) or 1.0

    def scalar(i: int) -> float:
        return sum(weights[a] * scores[a][i] for a in axes) / wsum

    return max(eligible, key=scalar), frontier


# Extra axes computed via the registry (novelty is computed inline in chat.py
# because it also yields the raw distance used in the trace). Order is stable.
EXTRA_SCORERS: list[Scorer] = [QualityScorer(), CoherenceScorer()]


def score_extra(ctx: ScoringContext, scorers: list[Scorer] | None = None) -> dict[str, list[float]]:
    """Run each scorer and return {axis_name: per-candidate scores}."""
    return {s.name: s.score(ctx) for s in (scorers if scorers is not None else EXTRA_SCORERS)}
