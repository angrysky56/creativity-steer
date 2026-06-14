"""Live Stage 1 demo: think-and-select on gemma4 with DeBERTa novelty anchoring.

Shows the modal (greedy) answer, the brainstormed variants with their
novelty-vs-modal and quality, the Pareto choice, and the wrapped response.

    uv run python examples/stage1_demo.py
"""

from __future__ import annotations

from creativity_steer.backends import OllamaBackend
from creativity_steer.data import get_problem
from creativity_steer.entailment import make_entailment
from creativity_steer.stage1 import Stage1Config, think_and_select


def main() -> None:
    backend = OllamaBackend()
    entailment = make_entailment("deberta", backend)
    prob = get_problem("keys-in-drain")
    cfg = Stage1Config(k=6, convergent_floor=0.34, novelty_weight=0.5, wrap=True)

    res = think_and_select(backend, prob["problem"], "", cfg, entailment)

    print(f"PROBLEM: {prob['problem']}\n")
    print(f"MODAL (greedy) answer:\n  {res.modal}\n")
    print("CANDIDATES (novelty-vs-modal / quality):")
    for c in res.candidates:
        tag = " [modal]" if c.is_modal else ""
        star = " *FRONTIER" if c.on_frontier else ""
        print(f"  nov={c.novelty:.2f} conv={c.convergent:.2f}{star}{tag}"
              f"\n      {c.text[:90]}")
    print(f"\nCHOSEN:\n  {res.chosen}\n")
    print(f"WRAPPED RESPONSE:\n  {res.response}")


if __name__ == "__main__":
    main()
