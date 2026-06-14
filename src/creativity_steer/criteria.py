"""Criterion definitions per domain (paper Appendix D.1).

The MacGyver definitions are verbatim from the reference implementation; the
HypoGen / BookMIA sets are provided for swapping in other domains.
"""

from __future__ import annotations

SAFETY_DEF = (
    "A solution is Safe if the solution has minimal risk of causing severe "
    "bodily harm to the person if they execute the actions in the solution "
    "cautiously. Do NOT consider the outcomes/results of the solution, or any "
    "of its effects on other people/animals/objects when determining safety - "
    "we are only evaluating the safety of the solution only for the person "
    "WHILE it is being executed. Assume that the person is aware of any "
    "potential risks of the solution, is well-prepared and takes precautionary "
    "measures."
)

FEASIBILITY_DEF = (
    "A solution is Feasible if a knowledgeable person is able to realistically "
    "execute the solution given the available materials and their specific "
    "properties, REGARDLESS of whether the actions or solution themselves are "
    "effective. A solution being ineffective DOES NOT MEAN the solution is "
    "infeasible."
)

EFFECTIVENESS_DEF = (
    "A solution is Effective if it will FULLY resolve the problem given and "
    "achieve ALL pertinent desired outcomes of the task."
)

MACGYVER_CRITERIA: dict[str, str] = {
    "feasibility": FEASIBILITY_DEF,
    "safety": SAFETY_DEF,
    "effectiveness": EFFECTIVENESS_DEF,
}

HYPOGEN_CRITERIA: dict[str, str] = {
    "feasibility": "A hypothesis is Feasible if it can realistically be tested.",
    "relevance": "A hypothesis is Relevant if it addresses the stated research gap.",
    "scientific accuracy": (
        "A hypothesis is Scientifically Accurate if it is consistent with "
        "established scientific knowledge."
    ),
}

BOOKMIA_CRITERIA: dict[str, str] = {
    "coherence": "The narrative is Coherent if it flows logically and consistently.",
    "realism": "The narrative is Realistic if characters and events are believable.",
    "plot completion": (
        "The narrative achieves Plot Completion if it connects the given start "
        "and end in a satisfying way."
    ),
}

DOMAIN_CRITERIA: dict[str, dict[str, str]] = {
    "macgyver": MACGYVER_CRITERIA,
    "hypogen": HYPOGEN_CRITERIA,
    "bookmia": BOOKMIA_CRITERIA,
}
