# Creativity Steer: Grounding and Memory Module

I've implemented the **MCP Client**, **Memory Store**, **Grounding Module**, and **Consolidation** processes as outlined in your instructions (`AGENT-INSTRUCTIONS-GROUNDING-MEMORY.md`).

## Implementation Summary

### Phase G1 — MCP Client
Created `src/creativity_steer/mcp_client.py` containing `McpClient`, which offers a synchronous wrapper around the asynchronous official MCP Python SDK (`mcp>=1.0`).
- Implemented `connect()`, `list_tools()`, and `call_tool()` over `stdio` transport.
- Added `MockMcpClient` for offline testing.
- Added an example configuration file `mcp.json.example`.

### Phase M1 — Memory Store
Created `src/creativity_steer/memory.py` utilizing `chromadb`.
- Stores option spaces (lessons, facts, creative alternatives) atomically, without thought leaks.
- Supports retrieval via cosine similarity against the `embed_backend`.
- Added support for dormancy management through `decay()` and activity tracking through `touch()`.
- Implemented `LocalMemoryStore` using JSONL as a fallback for offline/deterministic testing.

### Phase G2 — Grounding Layer
Created `src/creativity_steer/grounding.py` for retrieving context before generation.
- Queries `MemoryStore` and `McpClient` tools (when enabled via config).
- Injects a `[KNOWN CONTEXT]` block into `chat.py` prompts, explicitly guiding the explorer to *inform a RANGE of replies*, preventing premature collapse.
- Emits additive `grounding` events in `chat_turn_stream`.

### Phase M2 — Impact & Consolidation
Created `src/creativity_steer/consolidation.py` to curate high-impact turns and store them.
- `impact_score()` rates turns based on exploration depth, axis difference, and explicit corrections.
- Extracts factual, thought-free lessons and preserves creative alternatives using the generator backend.
- Added `POST /api/consolidate` in `webapp.py` to trigger this phase over the session log.

### Phase M3 — Wiring into the Server
Updated `src/creativity_steer/webapp.py` and `src/creativity_steer/chat.py`.
- Instantiates `memory` and `grounding` in `_build_state()` configured via `.env` keys (`CS_MEMORY_BACKEND`, `CS_GROUNDING`, etc.).
- `chat_turn_stream` now gathers grounding contexts, prefixes prompts, and emits the `grounding` event.
- Ensured process trace is extended additively without breaking the existing front-end.

## Verification
- Added test coverage: `tests/test_memory.py`, `tests/test_grounding.py`, `tests/test_consolidation.py`.
- Run all 58 tests successfully (`uv run pytest tests/`).
- Verified zero-regression against `smoke_ollama.py`.
- Checked backend configurations.

> [!NOTE] 
> The system requires the latest `.env` keys. See `CS_MEMORY_BACKEND`, `CS_GROUNDING`, etc., in the updated `webapp.py` config variables. When turning grounding "on", ensure the configured `CS_MCP_CONFIG` path exists.

Let me know if you want any tweaks to the consolidation triggers or grounding prompt block phrasing.
