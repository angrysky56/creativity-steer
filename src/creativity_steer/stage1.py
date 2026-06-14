"""Stage 1: efficient think-and-select.

One pass per step:

1. get the model's MODAL answer (greedy / temperature 0) -- the reference;
2. brainstorm K contrasting variants in a single call;
3. score each variant's novelty as distance from the modal answer, and its
   quality with the fast rubric judge;
4. Pareto-select (novelty, quality) above the convergent floor;
5. optionally wrap the chosen variant as a natural response.

Cost is ~K+3 calls per step (modal + brainstorm + K judges + wrap), versus the
explicit pipeline's larger per-candidate cost. Novelty stays a measured signal
(distance from the modal answer), never self-reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from creativity_steer.backends import LLMBackend
from creativity_steer.convergent import DEFAULT_CRITERIA, judge_candidate
from creativity_steer.entailment import EntailmentModel
from creativity_steer.reference import normalize_max, reference_distances
from creativity_steer.selection import pareto_mask
from creativity_steer.variants import brainstorm_variants


@dataclass
class Stage1Config:
    """Knobs for the think-and-select step."""

    k: int = 6
    temperature: float = 0.9
    step_criteria: tuple[str, ...] = DEFAULT_CRITERIA
    novelty_weight: float = 0.5
    convergent_floor: float = 0.34
    include_modal_in_pool: bool = True
    zero_out_modal_restatements: bool = True
    wrap: bool = True


@dataclass
class Stage1Candidate:
    """A scored variant in the think-and-select pool."""

    text: str
    novelty: float          # max-normalised distance from modal (selection)
    convergent: float
    distance: float = 0.0   # raw absolute distance from modal (reporting)
    is_modal: bool = False
    on_frontier: bool = False


@dataclass
class Stage1Result:
    """Outcome of one think-and-select step."""

    modal: str
    chosen: str
    response: str
    chosen_novelty: float = 0.0
    chosen_distance: float = 0.0
    chosen_convergent: float = 0.0
    candidates: list[Stage1Candidate] = field(default_factory=list)


def _modal_prompt(problem: str, partial_solution: str) -> str:
    so_far = partial_solution.strip()
    tail = f"\n\nSOLUTION SO FAR:\n{so_far}" if so_far else ""
    return (
        f"PROBLEM:\n{problem}{tail}\n\n"
        "Propose ONE creative, useful next step. Reply with the step only, in "
        "one sentence."
    )


def _wrap(backend: LLMBackend, problem: str, chosen: str) -> str:
    prompt = (
        f"PROBLEM:\n{problem}\n\nCHOSEN NEXT STEP:\n{chosen}\n\n"
        "Present this step as a clear, helpful one-sentence response to the "
        "problem. Reply with the response only."
    )
    return backend.chat(prompt, temperature=0.3, num_predict=120)


def _select(cands: list[Stage1Candidate], config: Stage1Config) -> int:
    """Pareto pick on (novelty, convergent) above the floor; quality fallback."""
    points = [(c.novelty, c.convergent) for c in cands]
    frontier = pareto_mask(points)
    for i, c in enumerate(cands):
        c.on_frontier = frontier[i]
    eligible = [
        i for i, c in enumerate(cands)
        if c.on_frontier and c.convergent >= config.convergent_floor
    ]
    if not eligible:
        return max(range(len(cands)), key=lambda i: cands[i].convergent)
    w = config.novelty_weight
    return max(
        eligible, key=lambda i: w * cands[i].novelty + (1.0 - w) * cands[i].convergent
    )


def think_and_select(
    backend: LLMBackend,
    problem: str,
    partial_solution: str,
    config: Stage1Config,
    entailment: EntailmentModel,
    judge_backend: LLMBackend | None = None,
) -> Stage1Result:
    """Run one efficient think-and-select step.

    ``backend`` drives generation (modal, brainstorm, wrap) and embeddings;
    ``judge_backend`` (default: ``backend``) scores quality, so a fast small
    model can generate while a stronger model judges.
    """
    judge = judge_backend or backend
    modal = backend.chat(_modal_prompt(problem, partial_solution),
                         temperature=0.0, num_predict=100).strip()
    variants = [v.text for v in brainstorm_variants(
        backend, problem, partial_solution, config.k, config.temperature
    )]

    texts = ([modal] + variants) if config.include_modal_in_pool else variants
    modal_flags = ([True] + [False] * len(variants)
                   if config.include_modal_in_pool else [False] * len(variants))

    ent = entailment if config.zero_out_modal_restatements else None
    distances = reference_distances(backend, modal, texts, problem, ent)
    novelty = normalize_max(distances)
    convergent = [
        judge_candidate(judge, problem, partial_solution, t,
                        config.step_criteria).score
        for t in texts
    ]

    cands = [
        Stage1Candidate(texts[i], novelty[i], convergent[i], distances[i],
                        modal_flags[i])
        for i in range(len(texts))
    ]
    chosen_idx = _select(cands, config)
    c = cands[chosen_idx]
    response = _wrap(backend, problem, c.text) if config.wrap else c.text
    return Stage1Result(
        modal=modal, chosen=c.text, response=response,
        chosen_novelty=c.novelty, chosen_distance=c.distance,
        chosen_convergent=c.convergent, candidates=cands,
    )


@dataclass
class Stage1Trajectory:
    """A multi-step solution built under one strategy."""

    strategy: str
    steps: list[str] = field(default_factory=list)
    results: list[Stage1Result] = field(default_factory=list)  # empty for greedy

    @property
    def solution(self) -> str:
        return "\n".join(self.steps)

    @property
    def mean_novelty(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.chosen_novelty for r in self.results) / len(self.results)

    @property
    def mean_convergent(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.chosen_convergent for r in self.results) / len(self.results)


def run_stage1_trajectory(
    backend: LLMBackend,
    problem: str,
    config: Stage1Config,
    entailment: EntailmentModel,
    judge_backend: LLMBackend | None = None,
    max_steps: int = 3,
) -> Stage1Trajectory:
    """Build a full solution by think-and-select at each step."""
    steps: list[str] = []
    results: list[Stage1Result] = []
    for _ in range(max_steps):
        res = think_and_select(
            backend, problem, "\n".join(steps), config, entailment, judge_backend
        )
        steps.append(res.chosen)
        results.append(res)
    return Stage1Trajectory("stage1", steps, results)


def run_greedy_trajectory(
    backend: LLMBackend, problem: str, max_steps: int = 3
) -> Stage1Trajectory:
    """Greedy baseline: take the modal (temperature 0) answer at each step."""
    steps: list[str] = []
    for _ in range(max_steps):
        modal = backend.chat(
            _modal_prompt(problem, "\n".join(steps)), temperature=0.0, num_predict=100
        ).strip()
        steps.append(modal)
    return Stage1Trajectory("greedy", steps, [])
