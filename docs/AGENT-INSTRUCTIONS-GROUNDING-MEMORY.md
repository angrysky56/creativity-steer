# Agent Instructions — Grounding (MCP) + Memory

Agent-executable build plan for the next phases: give the explorer **substance**
(tools via MCP) and **memory** (consolidated, impact-curated, option-space-
preserving). Read the companion docs first — they are binding:

- [CONCEPT.md](CONCEPT.md) — the axes, controller, attractor framing.
- [GROUNDING-MEMORY-AND-TRAINING.md](GROUNDING-MEMORY-AND-TRAINING.md) — §2a/§2b
  (MOP occupancy; reward the explorer not the answer; the KL trap), §3
  (McGaugh impact/consolidation).
- [PLAN.md](PLAN.md) — the selector phases already shipped.

## Non-negotiables (every phase)

1. **MOP / occupancy first.** Grounding and memory must *expand* the option
   space, never collapse it toward one answer. Memory stores the **space of good
   approaches**, not "the verdict." This is the anti-narrative-compression
   property at the storage layer.
2. **No thought-leak.** Persist only final, self-contained lessons/answers —
   never the model's intermediate reasoning. Re-injecting prior "thoughts" is a
   known failure (multi-turn reasoning loop).
3. **Corrections, not approval.** Impact/consolidation learn from *objective
   failures* (errors, user fixes, surfaced consequences), weighted delayed over
   immediate. Never add an approval/thumbs-up optimizer (§2a).
4. **Memory ≠ a new axis.** Grounding/memory change *what candidates get
   generated* (richer, grounded). The four selection axes still measure the
   candidates; do not entangle them.
5. **Atomic + emergent + dormant** (knowledge-architecture): one idea per item;
   structure emerges from usage (no upfront taxonomy); decay to **dormant**,
   never hard-delete.
6. **Additive event protocol.** New SSE events (`grounding`, `tool`) only; never
   remove/rename existing fields. Pluggable behind protocols, default-off.
7. **Reference-free, fast, local-first.** Reuse the embed backend; no heavy deps
   beyond the MCP SDK. `uv run pytest` green after each phase; `MockBackend` +
   a mock MCP/memory must keep tests deterministic and offline.

---

## Phase G1 — MCP client (`src/creativity_steer/mcp_client.py`)

A thin, sync-friendly client over the official MCP Python SDK.

- Dep: add an extra `mcp = ["mcp>=1.0"]` in `pyproject.toml`. Lazy-import.
- Config: read a standard `mcp.json` (servers map: name → {transport: stdio|sse,
  command/args/env or url}). Path from `CS_MCP_CONFIG` (default `./mcp.json`).
  Optional `CS_MCP_TOOLS` whitelist (comma list of `server.tool`) for focus/safety.
- `class McpClient`:
  - `connect()` / context manager — start configured servers.
  - `list_tools() -> list[ToolSpec]` (name, server, description, input schema).
  - `call_tool(server, name, args: dict) -> ToolResult` (text/structured).
  - Sync wrappers over the async SDK (run a private event loop in a thread); the
    pipeline is synchronous.
  - Robust: timeouts, per-call try/except returning an error ToolResult; never
    crash the turn.
- `class MockMcpClient` (for tests): in-memory tools (e.g. `echo`, `search`
  returning canned snippets) so grounding/tests run offline & deterministically.

Acceptance: connect to a stdio server from `mcp.json`, list + call a tool live;
`MockMcpClient` drives the grounding tests.

---

## Phase M1 — Memory store (`src/creativity_steer/memory.py`)

Atomic, embedding-retrieved, dormancy-managed. **Stores option spaces.**

```python
@dataclass
class MemoryItem:
    id: str
    created: float
    last_used: float
    uses: int
    kind: str               # "lesson" | "correction" | "options" | "fact"
    content: str            # atomic, final form — NO reasoning
    context: str            # when/where it applies
    tags: list[str]
    impact: float
    alternatives: list[str] = field(default_factory=list)  # MOP: the option space
    embedding: list[float] | None = None
    status: str = "active"  # "active" | "dormant"
    source: str = ""        # turn ref
```

```python
class MemoryStore(Protocol):
    def write(self, item: MemoryItem) -> None: ...
    def retrieve(self, query: str, k: int, include_dormant: bool = False) -> list[MemoryItem]: ...
    def touch(self, ids: list[str]) -> None: ...           # bump last_used/uses
    def decay(self, max_idle_seconds: float) -> int: ...   # active -> dormant, never delete
    def all(self) -> list[MemoryItem]: ...
```

- `LocalMemoryStore(embed_backend, path="results/memory.jsonl")`: JSONL persistence;
  embed `content` with embeddinggemma; cosine retrieval over **active** items
  (optionally include dormant); dedupe on write by high cosine similarity (merge,
  bump impact, don't duplicate).
- `McpMemoryStore(mcp_client, server)`: adapter that maps `write/retrieve` onto an
  MCP memory server's tools (the user provides the server). Same protocol.
- Factory `build_memory()` reads `CS_MEMORY_BACKEND` (local|mcp|none).

Acceptance: write/retrieve round-trips by similarity; `decay()` flips stale items
to dormant (kept, not deleted); dedupe merges near-duplicates.

---

## Phase G2 — Grounding layer (`src/creativity_steer/grounding.py`) + pipeline

Gather context *before* generation; inject into modal + brainstorm; expand, never
prescribe.

```python
@dataclass
class GroundingContext:
    memory: list[MemoryItem]
    tool_results: list[dict]
    def block(self) -> str: ...   # formatted "KNOWN CONTEXT" text for prompts

class GroundingProvider(Protocol):
    def gather(self, query: str, history: list[dict]) -> GroundingContext: ...
```

- `DefaultGrounding(memory, mcp_client, retrieval_tools)`:
  `gather` = `memory.retrieve(query, k)` + (optional) call whitelisted retrieval
  tools (e.g. a `search`/`docs` MCP tool); return a context block framed as
  *"relevant things known / sources — use these to inform a RANGE of replies,"*
  never *"the answer is."* (MOP guard, enforced in the prompt wording.)
- Pipeline (`chat.py`): add optional `grounding: GroundingProvider | None`. If set:
  - `ctx = grounding.gather(user_msg, history)`; prepend `ctx.block()` to
    `_modal_prompt` and `_brainstorm_prompt`.
  - emit additive `{"type":"grounding","memory":N,"tools":M,"snippets":[...]}`.
  - `memory.touch()` the retrieved ids.
- (Optional, advanced — Phase G3) tool execution on **primes** only: a config
  flag; prime candidates that propose an action call a tool; result enriches the
  candidate before scoring/branch. Bounded by `prime_n`. Emit `tool` events.

Acceptance: with `MockMcpClient` + `LocalMemoryStore` seeded, grounding events
carry retrieved snippets and generations visibly reflect injected context;
default-off path unchanged.

---

## Phase M2 — Impact + consolidation (`src/creativity_steer/consolidation.py`)

Post-session, background. Turns the flat log into curated, atomic, option-space
memory. **Never approval-driven.**

- `impact_score(turn) -> float` from already-logged signals:
  ```
  impact = a*(controller_rounds - 1)            # had to work / explore
         + b*mean(|chosen_axis - modal_axis|)   # the selection mattered
         + c*(1 - initial_diversity)            # had to escape the modal basin
         + d*correction_flag                    # objective failure/fix (delayed)
  ```
  No approval term. Corrections weighted strongest and may arrive later.
- `consolidate(turns, memory, gen) -> list[MemoryItem]`:
  1. keep only turns with `impact >= threshold` (drop neutral chatter);
  2. for each, ask `gen` for an **atomic, transferable lesson in final form, no
     reasoning** (strip thoughts);
  3. **preserve the option space**: store the frontier/diverse primes as
     `alternatives` and set `kind="options"` for creative turns; `kind="correction"`
     when a failure/fix was detected;
  4. dedupe against existing memory (embedding); `memory.write()`.
- Triggers: a `POST /api/consolidate` endpoint (consume the session log) and/or a
  CLI `examples/consolidate.py`. Reads `results/conversations.jsonl`.
- Correction detection: heuristic over consecutive turns (user contradicts/fixes,
  or an error surfaced) → `correction_flag`, `kind="correction"`.

Acceptance: a synthetic session yields memory only for high-impact turns; lessons
contain no reasoning text; creative turns carry `alternatives`; low-impact turns
produce nothing.

---

## Phase M3 — Wire into the server (`webapp.py`)

- Build `memory` + `mcp_client` + `grounding` once in `_build_state()` from env;
  pass `grounding` into `chat_turn_stream`.
- `_log_turn` already records modal/candidates/scores/controller — ensure it also
  records enough for `impact_score` (controller `rounds`, `diversity`, per-axis
  chosen-vs-modal). Extend additively if missing.
- Add `POST /api/consolidate` (runs Phase M2 over the current log). Optionally a
  scheduled background consolidation on session idle.
- `.env` keys (document in `.env.example`):
  ```
  CS_MCP_CONFIG=./mcp.json
  CS_MCP_TOOLS=                     # optional whitelist
  CS_MEMORY_BACKEND=local           # local|mcp|none
  CS_MEMORY_PATH=results/memory.jsonl
  CS_GROUNDING=on                   # on|off
  CS_IMPACT_THRESHOLD=0.4
  ```

Acceptance: live turn shows a `grounding` event sourced from seeded memory;
`/api/consolidate` writes curated memory; next turn retrieves it.

---

## Tests & cross-cutting

- New tests: `test_memory.py` (write/retrieve/decay/dedupe), `test_grounding.py`
  (MockMcpClient + seeded memory → grounding events + prompt injection),
  `test_consolidation.py` (impact ordering; high-impact-only; no-thought-leak;
  alternatives preserved).
- `MockBackend` branches: an atomic-lesson extraction prompt → returns a terse
  lesson; the grounding "KNOWN CONTEXT" wording must not match existing branches.
- Keep all 51 existing tests green; behavior with grounding/memory **off** must be
  byte-identical to today.

## Order & checkpoints

G1 → M1 → G2 → M2 → M3 → M3-wire. `uv run pytest` + a live mock smoke after each.
Phase-sized commits. Do NOT touch the selection axes or the event fields the UI
already consumes.

## Out of scope (separate track)

Training (GRPO with the axes as occupancy reward — see
GROUNDING-MEMORY-AND-TRAINING.md §2b) consumes the curated memory + impact-weighted
traces; not part of this plan.
