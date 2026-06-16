"""LLM backends.

A backend can (a) sample N candidate continuations *with sequence
log-probabilities*, (b) act as a chat agent for the judge, and (c) embed text.
This keeps the selection logic and the judge independent of the model provider.

Logprobs enable probability-weighted semantic entropy (paper Eq. 4). When a
backend cannot supply them, ``GenSample.logprob`` is ``None`` and the divergent
metric falls back to count-based class probabilities.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class GenSample:
    """One sampled continuation and its sequence log-probability."""

    text: str
    logprob: float | None = None  # sum of token logprobs, or None if unavailable


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal interface the selection loop and judge depend on."""

    def generate_samples(
        self, prompt: str, n: int, temperature: float, max_tokens: int = 128
    ) -> list[GenSample]:
        """Sample ``n`` candidate continuations with sequence logprobs."""

    def chat(
        self, prompt: str, temperature: float = 0.0, num_predict: int | None = None
    ) -> str:
        """Single completion (used by the judge agents)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string."""


class OllamaBackend:
    """Backend backed by a local Ollama server (sync client).

    Generation requests token logprobs and sums them into a sequence
    log-probability. Embeddings use a dedicated embedding model.
    """

    def __init__(
        self,
        gen_model: str | None = None,
        embed_model: str | None = None,
        host: str | None = None,
        think: bool = False,
    ) -> None:
        import ollama  # imported lazily so the mock backend needs no server

        # Many recent models (gemma4, deepseek-r1, ...) emit a separate
        # "thinking" channel. Left on, it consumes the token budget and leaves
        # `content` empty, so default to off for this inference-time pipeline.
        self.think = think

        self.gen_model = gen_model or os.getenv(
            "CREATIVITY_STEER_GEN_MODEL", "gemma4:12b"
        )
        self.embed_model = embed_model or os.getenv(
            "CREATIVITY_STEER_EMBED_MODEL", "embeddinggemma"
        )
        self._client = ollama.Client(host=host or os.getenv("OLLAMA_HOST"))

    def _seq_logprob(self, resp) -> float | None:
        """Sum per-token logprobs from a chat response, if the server gave them."""
        lps = getattr(resp, "logprobs", None)
        if not lps:
            return None
        try:
            return float(sum(tok.logprob for tok in lps))
        except (TypeError, AttributeError):
            return None

    def generate_samples(
        self, prompt: str, n: int, temperature: float, max_tokens: int = 128
    ) -> list[GenSample]:
        """Draw ``n`` independent samples, each with its sequence logprob."""
        out: list[GenSample] = []
        for _ in range(n):
            resp = self._client.chat(
                model=self.gen_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature, "num_predict": max_tokens},
                logprobs=True,
                think=self.think,
            )
            out.append(
                GenSample(resp["message"]["content"].strip(), self._seq_logprob(resp))
            )
        return out

    def chat(
        self, prompt: str, temperature: float = 0.0, num_predict: int | None = None
    ) -> str:
        """One completion for a judge agent."""
        options: dict = {"temperature": temperature}
        if num_predict is not None:
            options["num_predict"] = num_predict
        resp = self._client.chat(
            model=self.gen_model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
            think=self.think,
        )
        return resp["message"]["content"].strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each string with the embedding model."""
        resp = self._client.embed(model=self.embed_model, input=texts)
        return [list(v) for v in resp["embeddings"]]


# (text, base_logit, quality, cluster_id). base_logit drives sampling
# probability and the mock sequence logprob; quality is the latent
# task-fulfilment the judge recovers; cluster_id groups equivalent ideas.
_DEFAULT_POOL: list[tuple[str, float, float, int]] = [
    ("Tie the string around the magnet and lower it to grab the keys.", 3.0, 0.55, 0),
    ("Fasten the magnet to the end of the string and dip it in.", 2.6, 0.55, 0),
    ("Drop the magnet straight into the drain to catch the keys.", 2.2, 0.30, 1),
    (
        "Punch a hole in the plastic cup, thread the string, lower the cup.",
        0.8,
        0.85,
        2,
    ),
    ("Use two strings to tilt the cup and scoop the keys out.", 0.4, 0.80, 3),
    ("Swing the magnet like a pendulum to snag the keys sideways.", 0.2, 0.70, 4),
]


class MockBackend:
    """Deterministic, model-free backend for tests and offline demos.

    Drives the *full* pipeline -- generation, entailment (via embeddings),
    and the multi-agent judge -- without any server. ``chat`` inspects the
    prompt type and returns deterministically parseable agent output that
    encodes the candidate's latent quality, so the real judge code path runs
    end-to-end offline.
    """

    def __init__(
        self,
        pool: list[tuple[str, float, float, int]] | None = None,
        embed_dim: int = 16,
        quality_cutoff: float = 0.5,
    ) -> None:
        self.pool = pool or _DEFAULT_POOL
        self.embed_dim = embed_dim
        self.quality_cutoff = quality_cutoff
        self._quality = {t: q for t, _, q, _ in self.pool}
        self._cluster = {t: c for t, _, _, c in self.pool}

    def generate_samples(
        self, prompt: str, n: int, temperature: float, _max_tokens: int = 128
    ) -> list[GenSample]:
        """Temperature-scaled softmax sampling; logprob = scaled base logit."""
        # trunk-ignore(bandit/B311)
        rng = random.Random(int(hashlib.sha256(prompt.encode()).hexdigest(), 16))
        temp = max(temperature, 1e-3)
        logits = [logit / temp for _, logit, _, _ in self.pool]
        m = max(logits)
        weights = [math.exp(x - m) for x in logits]
        total = sum(weights)
        idxs = rng.choices(range(len(self.pool)), weights=weights, k=n)
        return [GenSample(self.pool[i][0], math.log(weights[i] / total)) for i in idxs]

    def _quality_of(self, prompt: str) -> float:
        """Latent quality of whichever pool idea appears in the prompt."""
        for text, quality in self._quality.items():
            if text in prompt:
                return quality
        return 0.5

    def chat(
        self, prompt: str, _temperature: float = 0.0, _num_predict: int | None = None
    ) -> str:
        """Return deterministic, judge-parseable text by prompt type."""
        q = self._quality_of(prompt)
        verdict = "YES" if q >= self.quality_cutoff else "NO"
        if "BRAINSTORM TASK" in prompt:  # single-call variant brainstorm
            return "\n".join(
                f"{i + 1}) {text}" for i, (text, _, _, _) in enumerate(self.pool)
            )
        if "Develop this idea" in prompt:  # branch / deepen
            seed = prompt.rsplit("\nDeeper reply:", 1)[0].rsplit(":\n", 1)[-1]
            return seed.strip() or "mock deeper reply"
        if "weaves together" in prompt:  # synthesis
            found = next((t for t in self._quality if t in prompt), None)
            return found or "mock synthesis"
        if "Restate the following" in prompt:  # coherence paraphrase
            body = prompt.split(":\n", 1)[-1].rsplit("\nRestatement:", 1)[0]
            return body.strip() or "mock restatement"
        if "Continue this thought" in prompt:  # openness continuation
            seed = prompt.rsplit("\nNext:", 1)[0].rsplit(":\n", 1)[-1]
            return seed.strip() or "mock continuation"
        if "skeptical critic" in prompt:  # chat-mode comparative judge
            scores: list[float] = []
            for line in prompt.splitlines():
                s = line.strip()
                if s and s[0].isdigit() and ". " in s[:6]:
                    scores.append(
                        next((q for t, q in self._quality.items() if t in s), 0.5)
                    )
            scores = scores or [0.5]
            return json.dumps([{"i": i + 1, "score": q} for i, q in enumerate(scores)])
        if "RATE THE REPLY" in prompt:  # chat-mode rubric judge
            return (
                f"Relevance: [[{verdict}]], Helpfulness: [[{verdict}]], "
                f"Coherence: [[{verdict}]]"
            )
        if "Reply directly and helpfully" in prompt:  # chat-mode modal reply
            return self.pool[0][0]
        if "Present this step" in prompt:  # Stage 1 wrap step
            found = next((t for t in self._quality if t in prompt), "the chosen step")
            return f"To solve it: {found}"
        if "Propose ONE creative" in prompt:  # Stage 1 modal answer (greedy)
            return self.pool[0][0]  # highest-logit (modal) idea
        if "PROPOSED NEXT STEP" in prompt:  # per-step rubric judge
            return (
                f"Feasibility: [[{verdict}]], Safety: [[{verdict}]], "
                f"Effectiveness: [[{verdict}]]"
            )
        if "final binary verdict" in prompt:
            return f"[[{verdict}]] mock verdict (q={q:.2f})."
        if "certainty score" in prompt or "[[Score]]" in prompt:
            return f"Mock certainty. Thus, [[0.9]]. ([{verdict}])"
        if "Queries for other agents" in prompt:
            return (
                "[[Answering questions]] none. "
                "[[Opinion]] mock opinion. "
                "[[Queries]] none."
            )
        if "[[POINT]]" in prompt:
            return "[[POINT]] mock analysis A. [[POINT]] mock analysis B."
        if "Extract a concise, factual lesson" in prompt:
            return "Apples fall down."
        return "mock response"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Cluster-aware deterministic embeddings (same cluster -> similar)."""
        vectors: list[list[float]] = []
        for text in texts:
            cluster = self._cluster.get(text)
            vec = [0.0] * self.embed_dim
            if cluster is not None:
                vec[cluster % self.embed_dim] = 1.0
            else:
                h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
                vec[h % self.embed_dim] = 1.0
            vectors.append(vec)
        return vectors


def parse_json_object(raw: str) -> dict | None:
    """Best-effort extraction of the first JSON object from model text."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


class OpenAIBackend:
    """Backend for any OpenAI-compatible endpoint.

    Covers a trained unsloth model served via vLLM (OpenAI-compatible API),
    a Colab-tunnelled endpoint, or other compatible servers. Configure via
    ``base_url`` / ``api_key`` (or CS_API_BASE_URL / CS_API_KEY). Many local
    servers accept any key, so it defaults to ``"EMPTY"``.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        from openai import OpenAI  # requires the `api` extra

        self.model = model
        self.embed_model = embed_model or model
        self._resolved: str | None = None
        self._client = OpenAI(
            base_url=base_url or os.getenv("CS_API_BASE_URL"),
            api_key=api_key or os.getenv("CS_API_KEY") or "EMPTY",
            timeout=float(os.getenv("CS_API_TIMEOUT", "120")),
            max_retries=1,
        )

    def _model_id(self) -> str:
        """Resolve the configured name to an id the server actually serves.

        Some servers (e.g. the Unsloth llama-server router) ignore ``--alias``
        and expose models under ``repo:quant`` ids. We query ``/v1/models`` once
        and pick an exact match, else one containing the configured name, else
        the only/first model. Falls back to the configured name on any error.
        """
        if self._resolved is not None:
            return self._resolved
        rid = self.model
        try:
            ids = [m.id for m in self._client.models.list().data]
            if self.model in ids:
                rid = self.model
            else:
                matches = [i for i in ids if self.model in i]
                rid = matches[0] if matches else (ids[0] if ids else self.model)
        except Exception:
            rid = self.model
        self._resolved = rid
        return rid

    def generate_samples(
        self, prompt: str, n: int, temperature: float, max_tokens: int = 128
    ) -> list[GenSample]:
        """Draw ``n`` samples, summing token logprobs when the server gives them."""
        model = self._model_id()
        out: list[GenSample] = []
        for _ in range(n):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=True,
                )
            except Exception:  # server may reject logprobs -> retry without
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            choice = resp.choices[0]
            lp: float | None = None
            content = getattr(choice, "logprobs", None)
            if content and getattr(content, "content", None):
                lp = float(sum(t.logprob for t in content.content))
            out.append(GenSample((choice.message.content or "").strip(), lp))
        return out

    def chat(
        self, prompt: str, temperature: float = 0.0, num_predict: int | None = None
    ) -> str:
        """One completion for a judge / pipeline call."""
        resp = self._client.chat.completions.create(
            model=self._model_id(),
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=num_predict,
        )
        if not hasattr(resp, "choices"):
            raise RuntimeError(f"unexpected API response from {self.model}: {resp!r}")
        return (resp.choices[0].message.content or "").strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed via an OpenAI-compatible embeddings endpoint."""
        resp = self._client.embeddings.create(model=self.embed_model, input=texts)
        return [list(d.embedding) for d in resp.data]
