"""Entailment models for semantic clustering (paper §3.2, Appendix C.3).

Two generations are placed in the same semantic class iff they are
*bidirectionally* entailing. Three interchangeable entailment backends:

* ``LLMEntailment``       -- asks the generation model (paper's GPT-4o-entailment
                             variant; here a local Ollama model). No extra deps.
* ``EmbeddingEntailment`` -- cosine-similarity proxy; fastest, least faithful.
* ``DebertaEntailment``   -- the paper's primary ``tasksource/deberta-base-long-nli``
                             NLI model. Requires the ``deberta`` extra (torch).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from creativity_steer.backends import LLMBackend


class EntailmentModel(Protocol):
    """Directional entailment check."""

    def entails(self, question: str, premise: str, hypothesis: str) -> bool:
        """True if ``premise`` semantically entails ``hypothesis``."""
        ...


def bidirectional_equivalent(
    model: EntailmentModel, question: str, a: str, b: str
) -> bool:
    """True if a and b entail each other (same semantic class)."""
    if a == b:
        return True
    return model.entails(question, a, b) and model.entails(question, b, a)


_ENTAIL_SYS = (
    "You are a skilled linguist studying semantic entailment. Determine whether "
    "one sentence semantically entails another."
)


class LLMEntailment:
    """Entailment via a chat model, following the paper's prompt format."""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def entails(self, question: str, premise: str, hypothesis: str) -> bool:
        prompt = (
            f"{_ENTAIL_SYS}\n\n"
            f"We are evaluating answers to the question: {question}.\n"
            f"Here are 2 possible answers:\nSentence A: {premise}\n"
            f"Sentence B: {hypothesis}\n"
            "Does Sentence A semantically entail Sentence B? Respond with STRICTLY "
            "one word: entailment, or neutral."
        )
        out = self.backend.chat(prompt, temperature=0.0, num_predict=8)
        return "entailment" in out.lower()


class EmbeddingEntailment:
    """Cosine-similarity proxy for entailment (symmetric, fast)."""

    def __init__(self, backend: LLMBackend, threshold: float = 0.8) -> None:
        self.backend = backend
        self.threshold = threshold
        self._cache: dict[str, np.ndarray] = {}

    def _vec(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = np.asarray(self.backend.embed([text])[0], dtype=float)
        return self._cache[text]

    def entails(self, question: str, premise: str, hypothesis: str) -> bool:
        a, b = self._vec(premise), self._vec(hypothesis)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return False
        return float(np.dot(a, b) / (na * nb)) >= self.threshold


class DebertaEntailment:
    """Paper's primary NLI entailment via ``tasksource/deberta-base-long-nli``."""

    def __init__(self, model_name: str = "tasksource/deberta-base-long-nli") -> None:
        from transformers import pipeline  # requires the `deberta` extra

        device = _pick_device()
        self._pipe = pipeline("text-classification", model=model_name, device=device)

    def entails(self, question: str, premise: str, hypothesis: str) -> bool:
        label = self._pipe({"text": premise, "text_pair": hypothesis})["label"]
        return label.lower() == "entailment"


def _pick_device() -> int:
    """Return CUDA device index if available, else -1 (CPU)."""
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


def make_entailment(kind: str, backend: LLMBackend, **kwargs) -> EntailmentModel:
    """Factory: 'llm' (default, no extra deps), 'embedding', or 'deberta'."""
    if kind == "llm":
        return LLMEntailment(backend)
    if kind == "embedding":
        return EmbeddingEntailment(backend, **kwargs)
    if kind == "deberta":
        import os

        kwargs.setdefault(
            "model_name",
            os.getenv("CS_DEBERTA_MODEL", "tasksource/deberta-base-long-nli"),
        )
        return DebertaEntailment(**kwargs)
    raise ValueError(f"unknown entailment kind: {kind!r}")
