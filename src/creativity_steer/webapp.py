"""FastAPI app: streams the chat-mode process trace over SSE.

Backends are chosen by environment:
  CS_WEB_BACKEND = mock | ollama   (default ollama)
  CS_GEN_MODEL    (default granite4.1:3b)
  CS_JUDGE_MODEL  (default gemma4:12b)
  CS_ENTAILMENT   = deberta | embedding | llm   (default deberta)

Each turn is appended to results/conversations.jsonl as future training data.
Run with:  uv run creativity-steer-serve   (or uvicorn creativity_steer.webapp:app)
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from creativity_steer.backends import MockBackend
from creativity_steer.chat import ChatConfig, chat_turn_stream
from creativity_steer.config import backend_summary, build_backend, load_env
from creativity_steer.consolidation import consolidate
from creativity_steer.entailment import EmbeddingEntailment, make_entailment
from creativity_steer.grounding import DefaultGrounding
from creativity_steer.mcp_client import McpClient
from creativity_steer.memory import build_memory

LOG_PATH = Path("results/conversations.jsonl")
_state: dict | None = None
_lock = threading.Lock()


def _env_int(name: str, default: int) -> int:
    """Read an integer env var, falling back to ``default`` if unset/invalid."""
    try:
        return int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []
    k: int = 5
    seed: int = 0  # 0 = random; any other value reproduces the run
    novelty_weight: float = 0.35
    coherence_weight: float = 0.2
    openness_weight: float = 0.0
    originality_weight: float = 0.0
    surprise_weight: float = 0.0
    convergent_floor: float = 0.4
    temperature: float = 0.7
    openness_branches: int = 0
    breadth_k: int = 10
    prime_n: int = 4
    branch: bool = False
    synthesize: bool = False
    trajectory: bool = False
    refine_passes: int = 0
    # generation length budgets (tokens); <= 0 means unlimited (model EOS).
    # Defaults come from .env so the whole system can be tuned in one place.
    modal_tokens: int = Field(default_factory=lambda: _env_int("CS_MODAL_TOKENS", 256))
    brainstorm_tokens: int = Field(
        default_factory=lambda: _env_int("CS_BRAINSTORM_TOKENS", 700)
    )
    branch_tokens: int = Field(default_factory=lambda: _env_int("CS_BRANCH_TOKENS", 0))
    response_tokens: int = Field(
        default_factory=lambda: _env_int("CS_RESPONSE_TOKENS", 0)
    )


def _build_state() -> dict:
    """Instantiate backends + entailment once (DeBERTa load is slow)."""
    load_env()
    is_mock = (
        os.getenv("CS_BACKEND") or os.getenv("CS_WEB_BACKEND") or "ollama"
    ).lower() == "mock"
    shared = MockBackend() if is_mock else None
    gen = build_backend("gen", shared)
    judge = build_backend("judge", shared)
    embed = build_backend("embed", shared)
    ent_kind = os.getenv("CS_ENTAILMENT", "embedding" if is_mock else "deberta")
    ent = (
        EmbeddingEntailment(embed, 0.9)
        if (is_mock and ent_kind == "embedding")
        else make_entailment(ent_kind, embed)
    )

    memory = build_memory(embed)
    grounding = None
    if os.getenv("CS_GROUNDING", "off").lower() == "on" and memory is not None:
        mcp_client = McpClient().connect()
        whitelist = os.getenv("CS_MCP_TOOLS")
        tools = [t.strip() for t in whitelist.split(",")] if whitelist else []
        grounding = DefaultGrounding(memory, mcp_client, tools)

    return {
        "gen": gen,
        "judge": judge,
        "embed": embed,
        "ent": ent,
        "memory": memory,
        "grounding": grounding,
        "backend": backend_summary(),
    }


def get_state() -> dict:
    global _state
    with _lock:
        if _state is None:
            _state = _build_state()
    return _state


def _log_turn(req: ChatRequest, events: list[dict], state: dict) -> None:
    """Append one completed turn to the JSONL training log."""
    by_type: dict[str, list] = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)
    if "response" not in by_type:
        return
    record = {
        "ts": time.time(),
        "message": req.message,
        "history": req.history,
        "modal": by_type["modal"][0]["text"],
        "variants": by_type["variants"][0]["items"],
        "scores": by_type.get("scored", []),
        "selected": by_type["selected"][0]["index"],
        "response": by_type["response"][0]["text"],
    }

    # Extract controller variables if available for impact_score
    if "controller" in by_type and by_type["controller"]:
        ctrl = by_type["controller"][0]
        record["rounds"] = ctrl.get("rounds", 1)
        record["initial_diversity"] = ctrl.get("diversity", 0.0)

    # See if there's any is_correction logic we can infer, or let the user pass it
    # Currently just passing defaults

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


app = FastAPI(title="creativity-steer chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "backend": get_state()["backend"]}


class ConsolidateRequest(BaseModel):
    threshold: float = 0.4


@app.post("/api/consolidate")
def api_consolidate(req: ConsolidateRequest) -> dict:
    state = get_state()
    memory = state.get("memory")
    if not memory:
        return {"error": "Memory is disabled or unconfigured."}

    if not LOG_PATH.exists():
        return {"consolidated": 0, "message": "No logs to consolidate"}

    turns = []
    with LOG_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                turns.append(json.loads(line))

    new_memories = consolidate(turns, memory, state["gen"], threshold=req.threshold)
    return {"consolidated": len(new_memories)}


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    state = get_state()
    cfg = ChatConfig(
        k=req.k,
        temperature=req.temperature,
        seed=req.seed,
        novelty_weight=req.novelty_weight,
        coherence_weight=req.coherence_weight,
        openness_weight=req.openness_weight,
        originality_weight=req.originality_weight,
        surprise_weight=req.surprise_weight,
        openness_branches=req.openness_branches,
        convergent_floor=req.convergent_floor,
        breadth_k=req.breadth_k,
        prime_n=req.prime_n,
        branch=req.branch,
        synthesize=req.synthesize,
        trajectory=req.trajectory,
        refine_passes=req.refine_passes,
        modal_tokens=req.modal_tokens,
        brainstorm_tokens=req.brainstorm_tokens,
        branch_tokens=req.branch_tokens,
        response_tokens=req.response_tokens,
    )

    def event_gen():
        collected: list[dict] = []
        try:
            for ev in chat_turn_stream(
                state["gen"],
                state["judge"],
                state["ent"],
                req.history,
                req.message,
                cfg,
                state["embed"],
                grounding=state.get("grounding"),
            ):
                collected.append(ev)
                yield f"data: {json.dumps(ev)}\n\n"
            _log_turn(req, collected, state)
        except Exception as exc:  # surface to the UI instead of hanging
            msg = f"{type(exc).__name__}: {exc}"
            yield f"data: {json.dumps({'type': 'error', 'message': msg})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def run() -> None:
    """Console-script entry point."""
    import uvicorn

    load_env()
    uvicorn.run(
        app,
        host=os.getenv("CS_HOST", "127.0.0.1"),
        port=int(os.getenv("CS_PORT", "8000")),
    )


if __name__ == "__main__":
    run()
