"""Selection axes (scorers) and multi-axis Pareto selection.

Each axis is a reference-free ``Scorer`` mapping the turn's candidates to a
value in [0, 1]. Adding a creative dimension = adding a scorer; the selection
generalises to any number of axes. See docs/CONCEPT.md for the framing
(attractor basins, counterfactual surfacing) and docs/PLAN.md for the roadmap.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
    texts: list[str]  # modal + variants
    samples: list[GenSample]  # parallel to texts; logprob may be None
    coherence_paraphrases: int = 2
    seed: int | None = None  # base seed; scorers offset it per sample


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
        for ti, t in enumerate(ctx.texts):
            paras = [
                ctx.gen.chat(
                    _PARAPHRASE_PROMPT.format(text=t),
                    temperature=0.7,
                    num_predict=80,
                    # distinct offset per (candidate, paraphrase) keeps the
                    # samples diverse while remaining reproducible under a seed
                    seed=(ctx.seed + ti * 97 + j + 1) if ctx.seed else None,
                ).strip()
                for j in range(p)
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
        # Length-normalise: mean per-token logprob, so a long reply is not judged
        # "more surprising" merely for being long. Higher mean surprisal (the
        # model was less certain token-to-token) = more likely freshly composed
        # than recited. Memorised clichés sit at very high probability (~0 logprob).
        vals = []
        for s in ctx.samples:
            n = s.n_tokens if (s.n_tokens and s.n_tokens > 0) else 1
            vals.append(-(s.logprob / n))  # higher = more surprising
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return [0.5] * len(ctx.texts)
        return [(v - lo) / (hi - lo) for v in vals]


# --------------------------------------------------------------------------- #
# Counterfactual openness — does the reply open the option space?             #
# --------------------------------------------------------------------------- #

_CONTINUE_PROMPT = "Continue this thought in one short sentence:\n{text}\nNext:"


class OpennessScorer:
    """An option-opening reply fans out into many directions; a closed
    pronouncement collapses. Score = normalised spread of a few continuations
    conditioned on the candidate. The most direct measure of "surfaces
    counterfactuals" (power-as-narrative-compression)."""

    name = "openness"

    def __init__(self, branches: int = 3) -> None:
        self.branches = branches

    def score(self, ctx: ScoringContext) -> list[float]:
        if self.branches < 2:
            return [0.5] * len(ctx.texts)
        raw: list[float] = []
        for ti, t in enumerate(ctx.texts):
            conts = [
                ctx.gen.chat(
                    _CONTINUE_PROMPT.format(text=t),
                    temperature=0.9,
                    num_predict=60,
                    seed=(ctx.seed + ti * 89 + j + 1) if ctx.seed else None,
                ).strip()
                for j in range(self.branches)
            ]
            vecs = [np.asarray(v, dtype=float) for v in ctx.embed.embed(conts)]
            sims = [
                _cosine(vecs[i], vecs[j])
                for i in range(len(vecs))
                for j in range(i + 1, len(vecs))
            ]
            raw.append(1.0 - (sum(sims) / len(sims)) if sims else 0.5)
        lo, hi = min(raw), max(raw)
        if hi - lo < 1e-9:
            return [0.5] * len(raw)
        return [(r - lo) / (hi - lo) for r in raw]


# --------------------------------------------------------------------------- #
# Originality — is it fresh, or a remembered cliché?                          #
# --------------------------------------------------------------------------- #

_ORIGINALITY_INSTR = (
    "You have read very widely and remember common jokes, puns, sayings, memes, "
    "and stock phrases. For EACH candidate reply, judge ORIGINALITY for this "
    "request: 1.0 = genuinely fresh, you have not seen it before; 0.0 = a "
    "well-known existing joke / pun / meme / stock phrase you clearly recognise. "
    "A clever reply that is still a recognised classic scores LOW — recognition, "
    "not cleverness, is what lowers the score. Name what it resembles in <=6 "
    "words, then score.\n"
    "Return ONLY a JSON array, one object per reply IN ORDER:\n"
    '[{"i":1,"resembles":"...","score":0.5}, ...]'
)


def judge_originality(
    judge: LLMBackend,
    history: list[dict[str, str]],
    user_msg: str,
    candidates: list[str],
) -> list[float]:
    """Score each candidate's freshness vs. recognised existing material.

    Novelty-from-modal cannot tell an original idea from a memorised cliché that
    merely differs from the greedy answer; this axis closes that gap.
    """
    listing = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    prompt = (
        f"{_history_block(history)}User request:\n{user_msg}\n\n"
        f"Candidate replies:\n{listing}\n\n{_ORIGINALITY_INSTR}"
    )
    raw = judge.chat(prompt, temperature=0.0, num_predict=40 * len(candidates) + 160)
    return _parse_scores(raw, len(candidates))


class OriginalityScorer:
    name = "originality"

    def score(self, ctx: ScoringContext) -> list[float]:
        return judge_originality(ctx.judge, ctx.history, ctx.user_msg, ctx.texts)


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
        # No frontier candidate clears the floor. Do NOT silently fall back to a
        # high-novelty / low-quality pick (that defeats the floor, the
        # anti-Goodhart gate). Honor the floor axis: choose the best-quality
        # candidate available. Caller can detect "floor not met" via the score.
        return max(range(n), key=lambda i: floor_vals[i]), frontier
    wsum = sum(weights[a] for a in axes) or 1.0

    def scalar(i: int) -> float:
        return sum(weights[a] * scores[a][i] for a in axes) / wsum

    return max(eligible, key=scalar), frontier


# Extra axes computed via the registry (novelty is computed inline in chat.py
# because it also yields the raw distance used in the trace). Order is stable.
EXTRA_SCORERS: list[Scorer] = [QualityScorer(), CoherenceScorer()]


def score_extra(
    ctx: ScoringContext, scorers: list[Scorer] | None = None
) -> dict[str, list[float]]:
    """Run each scorer and return {axis_name: per-candidate scores}."""
    return {
        s.name: s.score(ctx)
        for s in (scorers if scorers is not None else EXTRA_SCORERS)
    }


def funnel_representatives(
    embed: LLMBackend, modal: str, variants: list[str], prime_n: int
) -> list[int]:
    """Pick ``prime_n`` maximally-spread variants (farthest-first / k-center).

    Cheap embedding-only diverse subset selection — the funnel that makes large
    breadth tractable, and the submodular spanning selector in one. Returns
    indices into ``variants``.
    """
    if prime_n <= 0 or len(variants) <= prime_n:
        return list(range(len(variants)))
    vecs = [np.asarray(v, dtype=float) for v in embed.embed([modal, *variants])]
    mvec, cand = vecs[0], vecs[1:]
    # seed with the variant farthest from the modal
    selected = [max(range(len(cand)), key=lambda i: 1.0 - _cosine(mvec, cand[i]))]
    while len(selected) < prime_n:
        rest = [i for i in range(len(cand)) if i not in selected]
        nxt = max(
            rest,
            key=lambda i: min(1.0 - _cosine(cand[i], cand[j]) for j in selected),
        )
        selected.append(nxt)
    return sorted(selected)
