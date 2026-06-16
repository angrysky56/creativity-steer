"""Metacognitive control — the system pulls its own levers.

A transparent first version of CONCEPT.md §6: the controller reads the diversity
of the brainstormed set (a cheap entropy proxy) and decides whether to explore
another, hotter round (oMCD stop/continue) before committing. Collapsed onto the
modal basin → explore harder; already diverse → commit. Future versions can also
set k, the axis weights, and the floors per turn, and learn the mapping from
logged conversations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnController:
    """Decides, after each brainstorm round, whether to keep exploring."""

    explore_diversity_threshold: float = 0.34  # min fraction of non-collapsed variants
    temperature_step: float = 0.3
    max_temperature: float = 1.5
    novelty_eps: float = 0.05

    def diversity(self, novelty: list[float]) -> float:
        """Fraction of variants that escaped the modal basin (novelty > eps)."""
        variants = novelty[1:]  # exclude the modal itself
        if not variants:
            return 0.0
        return sum(1 for v in variants if v > self.novelty_eps) / len(variants)

    def should_continue(
        self, round_idx: int, novelty: list[float], max_rounds: int
    ) -> bool:
        """Continue while under the round cap AND the set is collapsed."""
        if round_idx + 1 >= max(1, max_rounds):
            return False
        return self.diversity(novelty) < self.explore_diversity_threshold

    def next_temperature(self, temp: float) -> float:
        return min(self.max_temperature, round(temp + self.temperature_step, 3))
