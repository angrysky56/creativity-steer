"""Sample open-ended problems (MacGyver-style) for quick demos.

These mirror the problem-solving domain from the source paper. Real
benchmarks (MacGyver / HypoGen / BookMIA) can be swapped in behind the same
``{"id", "domain", "problem"}`` shape.
"""

from __future__ import annotations

SAMPLE_PROBLEMS: list[dict[str, str]] = [
    {
        "id": "keys-in-drain",
        "domain": "macgyver",
        "problem": (
            "How can you retrieve a set of keys dropped into a deep drain "
            "using only a magnet, a string, and a plastic cup?"
        ),
    },
    {
        "id": "wine-no-opener",
        "domain": "macgyver",
        "problem": (
            "You need to open a bottle of wine but have no corkscrew. You have "
            "a tea towel, a wooden spoon, a screw, and a pair of scissors. How?"
        ),
    },
    {
        "id": "phone-stand",
        "domain": "macgyver",
        "problem": (
            "Build a hands-free phone stand for watching video using only a "
            "binder clip, a rubber band, and an old credit card."
        ),
    },
]


def get_problem(problem_id: str) -> dict[str, str]:
    """Look up a sample problem by id."""
    for p in SAMPLE_PROBLEMS:
        if p["id"] == problem_id:
            return p
    raise KeyError(f"no sample problem with id {problem_id!r}")


def criteria_for(problem: dict[str, str]) -> dict[str, str]:
    """Return the judge criteria definitions for a problem's domain."""
    from creativity_steer.criteria import DOMAIN_CRITERIA, MACGYVER_CRITERIA

    return DOMAIN_CRITERIA.get(problem.get("domain", ""), MACGYVER_CRITERIA)
