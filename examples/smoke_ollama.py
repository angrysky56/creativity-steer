"""Smoke test against a live Ollama server (gemma4:12b + nomic-embed-text).

Checks: sampling with logprobs, embeddings, LLM entailment clustering, and the
per-step rubric judge -- WITHOUT the heavy multi-agent judge.

    uv run python examples/smoke_ollama.py
"""

from __future__ import annotations

from creativity_steer.backends import OllamaBackend
from creativity_steer.convergent import judge_candidate
from creativity_steer.divergent import analyze_divergent
from creativity_steer.entailment import make_entailment

PROBLEM = (
    "How can you retrieve a set of keys dropped into a deep drain using only a "
    "magnet, a string, and a plastic cup?"
)


def main() -> None:
    backend = OllamaBackend()
    print(f"gen={backend.gen_model}  embed={backend.embed_model}")

    prompt = (
        f"PROBLEM:\n{PROBLEM}\n\nPropose ONE creative, useful next step. "
        "Reply with the step only, in one sentence."
    )
    samples = backend.generate_samples(prompt, n=4, temperature=1.0)
    print("\nsamples (logprob shown if server supports it):")
    for s in samples:
        lp = "None" if s.logprob is None else f"{s.logprob:.2f}"
        print(f"  [{lp}] {s.text}")

    ent = make_entailment("llm", backend)
    div = analyze_divergent(PROBLEM, samples, ent)
    print(f"\nclusters={div.num_clusters}  SE={div.semantic_entropy:.3f}  "
          f"prob_weighted={div.prob_weighted}")
    print(f"novelty={[round(n, 2) for n in div.novelty]}")

    conv = judge_candidate(backend, PROBLEM, "", samples[0].text)
    print(f"\nstep judge of sample 0: score={conv.score:.2f} {conv.breakdown}")
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
