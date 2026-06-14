"""Pareto selection loop, greedy baseline, and final multi-agent scoring.

Two-stage design matching the paper:

* **Per step** -- sample N candidates with logprobs, cluster by entailment,
  score each on novelty (divergent) and a fast rubric judge (convergent), then
  pick by the chosen selection rule.
* **Per trajectory** -- once the full solution is built, score it with the
  authoritative multi-agent retrieval judge.

Selection rules:

* ``greedy``     -- pick the modal idea (largest semantic cluster). Baseline.
* ``convergent`` -- maximise the step rubric score only. Ablation.
* ``pareto``     -- Pareto-best on (novelty, convergent) above a convergent
  floor. The floor is the anti-Goodhart guard: novelty is never maximised
  alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from creativity_steer.backends import LLMBackend
from creativity_steer.convergent import DEFAULT_CRITERIA, judge_candidate
from creativity_steer.criteria import MACGYVER_CRITERIA
from creativity_steer.divergent import analyze_divergent
from creativity_steer.entailment import EntailmentModel
from creativity_steer.judge import JudgeResult, multi_agent_judge


@dataclass
class SelectionConfig:
    """Knobs for the selection loop and final judge."""

    n_candidates: int = 6
    temperature: float = 1.0
    gen_max_tokens: int = 100
    step_criteria: tuple[str, ...] = DEFAULT_CRITERIA
    novelty_weight: float = 0.5         # scalarisation weight on the frontier
    convergent_floor: float = 0.34      # anti-Goodhart guard (>=1 of 3 criteria)
    max_steps: int = 3
    run_final_judge: bool = True
    judge_rounds: int = 2
    judge_confidence_threshold: float = 0.5
    judge_criteria: dict[str, str] = field(
        default_factory=lambda: dict(MACGYVER_CRITERIA)
    )


@dataclass
class CandidateEval:
    """A single scored candidate."""

    text: str
    cluster_id: int
    novelty: float
    convergent: float
    logprob: float | None = None
    on_frontier: bool = False


@dataclass
class StepRecord:
    """Everything that happened at one step."""

    step: int
    strategy: str
    candidates: list[CandidateEval]
    chosen_index: int
    semantic_entropy: float


@dataclass
class SelectionResult:
    """A full trajectory under one strategy."""

    strategy: str
    solution: str
    steps: list[StepRecord] = field(default_factory=list)
    final_judge: JudgeResult | None = None

    @property
    def mean_novelty(self) -> float:
        return _mean(s.candidates[s.chosen_index].novelty for s in self.steps)

    @property
    def mean_step_convergent(self) -> float:
        return _mean(s.candidates[s.chosen_index].convergent for s in self.steps)

    @property
    def mean_semantic_entropy(self) -> float:
        return _mean(s.semantic_entropy for s in self.steps)

    @property
    def final_score(self) -> float:
        return self.final_judge.score if self.final_judge else float("nan")


def _mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def pareto_mask(points: list[tuple[float, float]]) -> list[bool]:
    """Mark points on the Pareto frontier (both objectives maximised)."""
    mask = [True] * len(points)
    for i, (ai, bi) in enumerate(points):
        for j, (aj, bj) in enumerate(points):
            if i == j:
                continue
            if aj >= ai and bj >= bi and (aj > ai or bj > bi):
                mask[i] = False
                break
    return mask


def _gen_prompt(problem: str, partial_solution: str) -> str:
    """Prompt asking the model for a single next step."""
    so_far = partial_solution.strip()
    tail = f"\n\nSOLUTION SO FAR:\n{so_far}" if so_far else ""
    return (
        f"PROBLEM:\n{problem}{tail}\n\n"
        "Propose ONE creative, useful next step. Reply with the step only, "
        "in one sentence."
    )


def evaluate_step(
    backend: LLMBackend,
    problem: str,
    partial_solution: str,
    config: SelectionConfig,
    entailment: EntailmentModel,
) -> tuple[list[CandidateEval], float]:
    """Sample and score candidates for one step; returns evals + semantic entropy."""
    samples = backend.generate_samples(
        _gen_prompt(problem, partial_solution),
        config.n_candidates,
        config.temperature,
        config.gen_max_tokens,
    )
    div = analyze_divergent(problem, samples, entailment)
    convergent = [
        judge_candidate(
            backend, problem, partial_solution, s.text, config.step_criteria
        ).score
        for s in samples
    ]
    points = list(zip(div.novelty, convergent))
    frontier = pareto_mask(points)
    evals = [
        CandidateEval(
            text=samples[i].text,
            cluster_id=div.cluster_ids[i],
            novelty=div.novelty[i],
            convergent=convergent[i],
            logprob=samples[i].logprob,
            on_frontier=frontier[i],
        )
        for i in range(len(samples))
    ]
    return evals, div.semantic_entropy


def _choose(evals: list[CandidateEval], strategy: str, config: SelectionConfig) -> int:
    """Apply a selection rule and return the chosen candidate index."""
    if strategy == "greedy":
        sizes: dict[int, int] = {}
        for e in evals:
            sizes[e.cluster_id] = sizes.get(e.cluster_id, 0) + 1
        return max(
            range(len(evals)),
            key=lambda i: (sizes[evals[i].cluster_id], evals[i].convergent),
        )
    if strategy == "convergent":
        return max(range(len(evals)), key=lambda i: evals[i].convergent)
    if strategy == "pareto":
        eligible = [
            i for i, e in enumerate(evals)
            if e.on_frontier and e.convergent >= config.convergent_floor
        ]
        if not eligible:
            return max(range(len(evals)), key=lambda i: evals[i].convergent)
        w = config.novelty_weight
        return max(
            eligible,
            key=lambda i: w * evals[i].novelty + (1.0 - w) * evals[i].convergent,
        )
    raise ValueError(f"unknown strategy: {strategy!r}")


def select_candidate(
    backend: LLMBackend,
    problem: str,
    partial_solution: str,
    config: SelectionConfig,
    strategy: str,
    entailment: EntailmentModel,
    step: int = 0,
) -> StepRecord:
    """Evaluate one step and pick a candidate under ``strategy``."""
    evals, entropy = evaluate_step(backend, problem, partial_solution, config, entailment)
    chosen = _choose(evals, strategy, config)
    return StepRecord(step, strategy, evals, chosen, entropy)


def run_trajectory(
    backend: LLMBackend,
    problem: str,
    config: SelectionConfig,
    strategy: str,
    entailment: EntailmentModel,
) -> SelectionResult:
    """Build a full multi-step solution, then score it with the final judge."""
    solution_steps: list[str] = []
    result = SelectionResult(strategy=strategy, solution="")
    for step in range(config.max_steps):
        partial = "\n".join(solution_steps)
        record = select_candidate(
            backend, problem, partial, config, strategy, entailment, step
        )
        solution_steps.append(record.candidates[record.chosen_index].text)
        result.steps.append(record)
    result.solution = "\n".join(solution_steps)

    if config.run_final_judge:
        result.final_judge = multi_agent_judge(
            backend, problem, result.solution, config.judge_criteria,
            num_rounds=config.judge_rounds,
            confidence_threshold=config.judge_confidence_threshold,
        )
    return result
