"""Retrieval-based multi-agent judge (paper §4, Fig. 2, Appendix D.2).

Faithful port of ``modified_chateval_combined``: a Problem Analyst, Solution
Analyst and Criterion Analyst hold up to ``num_rounds`` rounds of structured
discussion. Each turn, an agent retrieves the top-``k`` most relevant prior
fragments from a vector store (rather than replaying the whole history), posts
new points and queries, and emits a confidence score. When mean confidence
crosses the threshold the loop exits early and the highest-confidence agent
gives a binary verdict per criterion.

The paper uses ChromaDB + a SentenceTransformer; this port uses a light
in-memory cosine store over the backend's own embeddings, which is functionally
equivalent for retrieval and keeps the project dependency-light.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from creativity_steer.backends import LLMBackend

_SCORE_RE = re.compile(r"\[\[\s*([01](?:\.\d+)?)\s*\]\]")
_YES_VOTE_RE = re.compile(r"\(\s*\[?\s*YES\s*\]?\s*\)", re.IGNORECASE)


@dataclass
class JudgeResult:
    """Output of the multi-agent judge for one solution."""

    score: float                       # mean binary verdict over criteria, [0, 1]
    verdicts: dict[str, int]           # criterion -> 1 / 0 / -1 (parse fail)
    confidence: dict[str, float]       # criterion -> final mean confidence


@dataclass
class _Fragment:
    text: str
    vec: np.ndarray
    criterion: str
    temporary: bool


class FragmentStore:
    """Minimal in-memory cosine-similarity fragment store."""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend
        self._frags: list[_Fragment] = []

    def add(self, text: str, criterion: str, temporary: bool) -> None:
        """Embed and store one discussion fragment."""
        text = text.strip()
        if not text:
            return
        vec = np.asarray(self.backend.embed([text])[0], dtype=float)
        self._frags.append(_Fragment(text, vec, criterion, temporary))

    def clear_temporary(self) -> None:
        """Drop per-criterion fragments before starting a new criterion."""
        self._frags = [f for f in self._frags if not f.temporary]

    def query(self, text: str, k: int) -> list[str]:
        """Return the texts of the top-``k`` fragments by cosine similarity."""
        if not self._frags:
            return []
        q = np.asarray(self.backend.embed([text])[0], dtype=float)
        qn = np.linalg.norm(q) or 1.0
        scored = []
        for f in self._frags:
            fn = np.linalg.norm(f.vec) or 1.0
            scored.append((float(np.dot(q, f.vec) / (qn * fn)), f.text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]


_INIT_PROMPT = """You are an impartial but critical '{role}', examining the problem, \
solution and criteria below.

Problem:
{problem}

Proposed solution:
{solution}

Criteria and definitions:
{criteria_list}

Your task: {task}
Be concise, critical and analytical, raising only the most pertinent points. \
Limit to 250 words. Present each new idea as a new point beginning with the \
header [[POINT]]. For example: [[POINT]] Explicit constraints: <...>"""

_DISCUSSION_PROMPT = """You are an impartial but critical '{role}' in a discussion \
to determine whether the solution fulfils the criterion. {focus}

Problem:
{problem}

Proposed solution:
{solution}

Criterion: {criterion}
Definition: {definition}

Be concise, critical and analytical. Fill missing details with reasonable \
assumptions. Limit to 150 words. Respond in three parts, each beginning with a \
[[label]]:
[[Answering]] Answer any questions from other agents (or 'none').
[[Opinion]] Your view on whether the solution fulfils the {criterion} criterion.
[[Queries]] Queries for other agents, formatted 'To <analyst>: <query>' (or 'none').

Relevant discussion:
{relevant_discussion}"""

_CONFIDENCE_PROMPT = """You are the impartial but critical {role} in the discussion \
below, {focus}.

Problem:
{problem}

Solution:
{solution}

Criterion: {criterion}
Definition: {definition}

Discussion points:
{discussion}

How certain are you that you can reach a correct conclusion about whether the \
solution fulfils the {criterion} criterion? Give a <=20 word explanation, then a \
certainty score between 0 and 1 STRICTLY as [[Score]] to one decimal place, then \
your current stance as ([YES]) or ([NO]). Example: <explanation> Thus, [[0.6]]. ([NO])"""

_VERDICT_PROMPT = """You are the {role}, acting as an impartial but critical judge. \
Based on the problem, solution, criterion definition and discussion below, give a \
final binary verdict on whether the solution fulfils the criterion. Heavily weight \
the criterion definition's phrasing.

Problem:
{problem}

Solution:
{solution}

Criterion: {criterion}
Definition: {definition}

Discussion:
{discussion}

Provide your verdict STRICTLY as [[YES]] or [[NO]] with a one-sentence justification."""

_ROLES = ("problem analyst", "solution analyst", "criterion analyst")
_FOCUS = (
    "focusing on a comprehensive understanding of the problem",
    "focusing on articulating the solution's details and nuances",
    "focusing on how the criterion should be defined in this problem's context",
)


def _split_points(text: str) -> list[str]:
    """Extract [[POINT]]-delimited fragments."""
    return [p.strip() for p in text.split("[[POINT]]")[1:] if p.strip()]


def _extract_queries(text: str) -> list[str]:
    """Pull 'To <analyst>: <query>' lines from a discussion response."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("to ") and ":" in line and "none" not in line.lower():
            out.append(line)
    return out


def _opinion(text: str) -> str:
    """Return the [[Opinion]] segment of a discussion response, if present."""
    m = re.search(r"\[\[\s*Opinion\s*\]\](.*?)(\[\[|$)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def _agent_confidence(
    backend: LLMBackend, store: FragmentStore, idx: int,
    problem: str, solution: str, criterion: str, definition: str,
) -> tuple[float, int]:
    """Run one agent's confidence turn; return (confidence, yes_vote)."""
    discussion = "\n".join(store.query(
        f"Whether the solution fulfils {criterion}; {_FOCUS[idx]}", k=5
    ))
    prompt = _CONFIDENCE_PROMPT.format(
        role=_ROLES[idx], focus=_FOCUS[idx], problem=problem, solution=solution,
        criterion=criterion, definition=definition, discussion=discussion,
    )
    resp = backend.chat(prompt, temperature=1.0, num_predict=220)
    m = _SCORE_RE.search(resp)
    conf = float(m.group(1)) if m else 0.3
    vote = 1 if _YES_VOTE_RE.search(resp) else 0
    return conf, vote


def _discussion_turn(
    backend: LLMBackend, store: FragmentStore, idx: int, queries: list[str],
    problem: str, solution: str, criterion: str, definition: str,
) -> None:
    """Run one analyst's discussion turn and update the store + query queue."""
    mine = [q for q in queries if _ROLES[idx].split()[0] in q.lower()]
    for q in mine:
        queries.remove(q)
    context = store.query(
        f"Discussion about {criterion}; {_FOCUS[idx]}", k=4
    ) + mine
    prompt = _DISCUSSION_PROMPT.format(
        role=_ROLES[idx], focus=_FOCUS[idx], problem=problem, solution=solution,
        criterion=criterion, definition=definition,
        relevant_discussion="\n".join(context),
    )
    resp = backend.chat(prompt, temperature=0.7, num_predict=320)
    store.add(_opinion(resp), criterion, temporary=True)
    queries.extend(_extract_queries(resp))


def multi_agent_judge(
    backend: LLMBackend,
    problem: str,
    solution: str,
    criteria_definitions: dict[str, str],
    num_rounds: int = 2,
    confidence_threshold: float = 0.5,
) -> JudgeResult:
    """Score a complete solution with the retrieval-based multi-agent judge."""
    store = FragmentStore(backend)
    criteria_list = "".join(f"{c}: {d}\n" for c, d in criteria_definitions.items())

    # Shared initial analyses over all criteria.
    for idx, task in (
        (0, "List explicit/implicit constraints, desired outcomes and key difficulties."),
        (1, "Describe the solution's steps, object properties, and logical coherence."),
    ):
        resp = backend.chat(_INIT_PROMPT.format(
            role=_ROLES[idx], problem=problem, solution=solution,
            criteria_list=criteria_list, task=task,
        ), temperature=0.7, num_predict=400)
        for pt in _split_points(resp):
            store.add(pt, "all", temporary=False)

    verdicts: dict[str, int] = {}
    confidences: dict[str, float] = {}

    for criterion, definition in criteria_definitions.items():
        store.clear_temporary()
        resp = backend.chat(_INIT_PROMPT.format(
            role=_ROLES[2], problem=problem, solution=solution,
            criteria_list=f"{criterion}: {definition}",
            task=f"Outline what a solution must do to fulfil '{criterion}'.",
        ), temperature=0.7, num_predict=400)
        for pt in _split_points(resp):
            store.add(pt, criterion, temporary=True)

        queries: list[str] = []
        mean_conf = 0.0
        for _ in range(num_rounds):
            for idx in range(3):
                _discussion_turn(backend, store, idx, queries,
                                 problem, solution, criterion, definition)
            confs, votes = [], []
            for idx in range(3):
                c, v = _agent_confidence(backend, store, idx,
                                         problem, solution, criterion, definition)
                confs.append(c)
                votes.append(v)
            mean_conf = sum(confs) / len(confs)
            best_idx = confs.index(max(confs))
            if mean_conf >= confidence_threshold:
                break

        verdict_text = backend.chat(_VERDICT_PROMPT.format(
            role=_ROLES[best_idx], problem=problem, solution=solution,
            criterion=criterion, definition=definition,
            discussion="\n".join(store.query(
                f"Whether the solution fulfils {criterion}; {_FOCUS[best_idx]}", k=8
            )),
        ), temperature=0.0, num_predict=160)
        if "[[YES]]" in verdict_text.upper():
            verdicts[criterion] = 1
        elif "[[NO]]" in verdict_text.upper():
            verdicts[criterion] = 0
        else:
            verdicts[criterion] = -1
        confidences[criterion] = mean_conf if verdicts[criterion] == 1 else 1 - mean_conf

    valid = [v for v in verdicts.values() if v >= 0]
    score = sum(valid) / len(valid) if valid else 0.0
    return JudgeResult(score, verdicts, confidences)
