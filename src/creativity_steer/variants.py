"""Candidate-variant sources for Stage 1 (efficient think-and-select).

Two ways to obtain the population of next-step candidates whose novelty we then
measure:

* ``independent_variants`` -- K independent samples (proven-diverse, K calls).
* ``brainstorm_variants``  -- ONE call asking the model to emit K contrasting
  variants in its reasoning, parsed from a numbered list (1 call, but risks
  in-context mode collapse).

Both return ``list[GenSample]`` so the divergent/selection code is unchanged.
Novelty is always measured externally over the returned population; we never
trust the model to self-report its own novelty.
"""

from __future__ import annotations

import re

from creativity_steer.backends import GenSample, LLMBackend

_BRAINSTORM_TMPL = (
    "PROBLEM:\n{problem}{tail}\n\n"
    "BRAINSTORM TASK: Propose exactly {k} genuinely DIFFERENT next steps, each "
    "using a distinct principle or mechanism (not reworded versions of one "
    "idea). Each step must be one sentence.\n"
    "Return them as a numbered list and nothing else:\n"
    "1) <step one>\n2) <step two>\n...\n{k}) <step {k}>"
)

_NUMBERED_RE = re.compile(r"^\s*\d+\s*[).:\-]\s*(.+?)\s*$")


def _partial_tail(partial_solution: str) -> str:
    so_far = partial_solution.strip()
    return f"\n\nSOLUTION SO FAR:\n{so_far}" if so_far else ""


def independent_variants(
    backend: LLMBackend,
    problem: str,
    partial_solution: str,
    k: int,
    temperature: float = 1.0,
    max_tokens: int = 100,
) -> list[GenSample]:
    """K independent samples (each its own generation call, with logprobs)."""
    prompt = (
        f"PROBLEM:\n{problem}{_partial_tail(partial_solution)}\n\n"
        "Propose ONE creative, useful next step. Reply with the step only, in "
        "one sentence."
    )
    return backend.generate_samples(prompt, k, temperature, max_tokens)


def parse_numbered_list(raw: str, k: int) -> list[str]:
    """Extract up to ``k`` items from a numbered list in model output."""
    items: list[str] = []
    for line in raw.splitlines():
        m = _NUMBERED_RE.match(line)
        if m:
            text = m.group(1).strip()
            if text:
                items.append(text)
    return items[:k]


def brainstorm_variants(
    backend: LLMBackend,
    problem: str,
    partial_solution: str,
    k: int,
    temperature: float = 0.9,
    max_tokens: int = 400,
) -> list[GenSample]:
    """K contrasting variants from a SINGLE call (no per-variant logprob)."""
    prompt = _BRAINSTORM_TMPL.format(
        problem=problem, tail=_partial_tail(partial_solution), k=k
    )
    raw = backend.chat(prompt, temperature=temperature, num_predict=max_tokens)
    return [GenSample(t, None) for t in parse_numbered_list(raw, k)]


def get_variants(
    source: str,
    backend: LLMBackend,
    problem: str,
    partial_solution: str,
    k: int,
    temperature: float = 1.0,
) -> list[GenSample]:
    """Dispatch to a variant source: 'independent' or 'brainstorm'."""
    if source == "independent":
        return independent_variants(backend, problem, partial_solution, k, temperature)
    if source == "brainstorm":
        return brainstorm_variants(backend, problem, partial_solution, k, temperature)
    raise ValueError(f"unknown variant source: {source!r}")
