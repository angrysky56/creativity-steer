"""Environment-driven configuration: load ``.env`` and build backends per role.

Each role (gen / judge / embed) is selected independently, so you can run a
trained model over an API for generation while keeping local embeddings:

    CS_BACKEND=ollama                 # global default for all roles
    CS_GEN_BACKEND=api                # override generation to the API model
    CS_GEN_MODEL=unsloth/my-gemma
    CS_API_BASE_URL=https://<tunnel>/v1
    CS_API_KEY=EMPTY
    CS_JUDGE_BACKEND=ollama
    CS_JUDGE_MODEL=gemma4:12b
    CS_EMBED_BACKEND=ollama
    CS_EMBED_MODEL=embeddinggemma
"""

from __future__ import annotations

import os

from creativity_steer.backends import LLMBackend, MockBackend, OllamaBackend

_DEFAULT_MODELS = {
    "GEN": "granite4.1:3b",
    "JUDGE": "gemma4:12b",
    "EMBED": "embeddinggemma",
}


def load_env() -> None:
    """Load a local ``.env`` if python-dotenv is available (no-op otherwise)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _kind(role: str) -> str:
    """Resolve the backend kind for a role, honouring CS_WEB_BACKEND alias."""
    default = os.getenv("CS_BACKEND") or os.getenv("CS_WEB_BACKEND") or "ollama"
    return os.getenv(f"CS_{role}_BACKEND", default).lower()


def build_backend(role: str, shared_mock: MockBackend | None = None) -> LLMBackend:
    """Construct the backend for ``role`` (gen / judge / embed) from the env."""
    role = role.upper()
    kind = _kind(role)
    model = os.getenv(f"CS_{role}_MODEL", _DEFAULT_MODELS.get(role, ""))
    embed_model = os.getenv("CS_EMBED_MODEL", _DEFAULT_MODELS["EMBED"])

    if kind == "mock":
        return shared_mock or MockBackend()
    if kind == "api":
        from creativity_steer.backends import OpenAIBackend

        base = os.getenv(f"CS_{role}_API_BASE_URL") or os.getenv("CS_API_BASE_URL")
        key = os.getenv(f"CS_{role}_API_KEY") or os.getenv("CS_API_KEY")
        return OpenAIBackend(
            model=model, base_url=base, api_key=key, embed_model=embed_model
        )
    return OllamaBackend(gen_model=model, embed_model=embed_model)


def backend_summary() -> str:
    """Human-readable summary of the configured backends."""
    parts = []
    for role in ("GEN", "JUDGE", "EMBED"):
        kind = _kind(role)
        model = os.getenv(f"CS_{role}_MODEL", _DEFAULT_MODELS.get(role, ""))
        parts.append(f"{role.lower()}={kind}:{model}")
    return "  ".join(parts)
