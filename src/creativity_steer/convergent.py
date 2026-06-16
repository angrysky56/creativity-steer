"""Per-step convergent signal: fast single-agent rubric judge.

This is the cheap convergent scorer used *inside* the selection loop, where
every candidate at every step must be scored. It mirrors the paper's
single-agent binary judge (Table 2 baseline). The authoritative multi-agent
retrieval judge (:mod:`creativity_steer.judge`) is reserved for scoring final
solutions, matching the paper's two-stage design.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from creativity_steer.backends import LLMBackend

DEFAULT_CRITERIA: tuple[str, ...] = ("feasibility", "safety", "effectiveness")

_VERDICT_RE = re.compile(r"\[\[\s*(YES|NO)\s*\]\]", re.IGNORECASE)


@dataclass
class ConvergentResult:
    """Rubric-judge output for a single candidate."""

    score: float  # fraction of criteria judged YES, in [0, 1]
    breakdown: dict[str, bool]  # per-criterion YES/NO


def build_judge_prompt(
    problem: str,
    partial_solution: str,
    candidate: str,
    criteria: tuple[str, ...] = DEFAULT_CRITERIA,
) -> str:
    """Single-agent rubric prompt (paper's binary judge style)."""
    so_far = partial_solution.strip() or "(none yet)"
    crit_q = {
        "feasibility": "Feasibility (can this step realistically be executed?)",
        "safety": "Safety (can it be done without severe risk to the person?)",
        "effectiveness": "Effectiveness (does it meaningfully advance the solution?)",
    }
    lines = "\n".join(f"{c.capitalize()}: {crit_q.get(c, c)}" for c in criteria)
    fmt = ", ".join(f"{c.capitalize()}: [[YES/NO]]" for c in criteria)
    return (
        "Act as an impartial, critical judge of ONE next step of a solution.\n"
        f"PROBLEM:\n{problem}\n\nSOLUTION SO FAR:\n{so_far}\n\n"
        f"PROPOSED NEXT STEP:\n{candidate}\n\n"
        f"Judge the proposed step on each criterion:\n{lines}\n\n"
        "After a brief justification, end with STRICTLY this format:\n"
        f"{fmt}"
    )


def judge_candidate(
    backend: LLMBackend,
    problem: str,
    partial_solution: str,
    candidate: str,
    criteria: tuple[str, ...] = DEFAULT_CRITERIA,
) -> ConvergentResult:
    """Score one candidate; verdicts are read positionally from the response."""
    prompt = build_judge_prompt(problem, partial_solution, candidate, criteria)
    raw = backend.chat(prompt, temperature=0.0, num_predict=256)
    verdicts = _VERDICT_RE.findall(raw)

    breakdown: dict[str, bool] = {}
    for i, crit in enumerate(criteria):
        breakdown[crit] = bool(i < len(verdicts) and verdicts[i].upper() == "YES")
    score = sum(breakdown.values()) / len(breakdown) if breakdown else 0.0
    return ConvergentResult(score, breakdown)
