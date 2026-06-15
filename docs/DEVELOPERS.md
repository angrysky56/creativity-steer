# Developer guide

Technical reference for `creativity-steer`. For the end-user quickstart see the
top-level [README](../README.md).

## The idea

The source paper ([creativity-eval](https://github.com/tanminsen/creativity-eval))
measures LLM creativity with two reference-free signals — **semantic entropy**
(divergent / novelty) and a **retrieval-based multi-agent judge** (convergent /
task fulfilment) — and finds the two are *separable*: more novelty does not
imply worse task fulfilment.

Because both signals are reference-free, they can score the model's own
candidates at generation time. So instead of only measuring creativity, we
**select** for it: sample candidates, score novelty and quality, and pick the
Pareto-best — novel *and* good — subject to a quality floor (the anti-Goodhart
guard; novelty is never maximised alone).

## Pipeline

- **Semantic entropy** — greedy bidirectional-entailment clustering with the
  Rao-Blackwellised, probability-weighted estimator. Sequence probabilities come
  from logprobs when the backend supplies them, else count-based.
- **Entailment** — `deberta` (the paper's `tasksource/deberta-base-long-nli`,
  default), `embedding` (cosine proxy), or `llm`.
- **Convergent judge** — fast single-agent rubric per candidate (selection
  loop); the full retrieval-based multi-agent judge for final solution scoring
  (`judge.py`). Two-stage, matching the paper.
- **Stage 1 (chat / efficient)** — one modal (greedy) answer as a reference, one
  brainstorm call for K diverse variants, novelty measured as distance from the
  modal answer, quality via the rubric judge, Pareto select, optional wrap.

## Backends (per role)

Three roles — `gen`, `judge`, `embed` — are selected independently via env, so a
fast local generator can pair with a stronger judge and local embeddings.

| kind     | class            | notes |
|----------|------------------|-------|
| `ollama` | `OllamaBackend`  | local Ollama; logprobs + `think=False` for reasoning models |
| `api`    | `OpenAIBackend`  | any OpenAI-compatible endpoint (Unsloth/vLLM, Colab tunnel) |
| `mock`   | `MockBackend`    | deterministic, model-free; for tests and offline checks |

### Environment variables

```
CS_BACKEND                global default kind (ollama|api|mock)
CS_{GEN,JUDGE,EMBED}_BACKEND     per-role override
CS_{GEN,JUDGE,EMBED}_MODEL       per-role model id
CS_{GEN,JUDGE}_API_BASE_URL      per-role API endpoint (kind=api)
CS_{GEN,JUDGE}_API_KEY           per-role API key (start.sh captures these)
CS_API_BASE_URL / CS_API_KEY     shared API fallback
CS_ENTAILMENT             deberta|embedding|llm
CS_DEBERTA_MODEL          NLI model id
OLLAMA_HOST               Ollama base URL
CS_HOST / CS_PORT         web server bind
```

### All-local (no Unsloth)

```bash
CS_BACKEND=ollama CS_GEN_MODEL=granite4.1:3b CS_JUDGE_MODEL=gemma4:12b \
  uv run creativity-steer-serve
```

### Model-free (logic check)

```bash
CS_BACKEND=mock uv run creativity-steer-serve
```

## Running Unsloth GGUFs (the default)

`start.sh` serves API-backend models by running **llama.cpp's `llama-server`
directly** (the binary Unsloth installs at `~/.unsloth/llama.cpp/llama-server`),
NOT the `unsloth run` Studio wrapper:

```bash
LD_LIBRARY_PATH=~/.unsloth/llama.cpp/build/bin ~/.unsloth/llama.cpp/llama-server \
  -hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL \
  --host 127.0.0.1 --port 8001 -c 8192 -ngl 99 --jinja --reasoning off \
  --temp 1.0 --top-p 0.95 --top-k 64 --alias gemma-4-E4B-it-qat-GGUF
```

Why not `unsloth run`: the Studio wrapper forces `enable_thinking: true` and
ignores CLI / per-request overrides, so gemma-4 emits chain-of-thought into
`content` instead of answers. Running `llama-server` directly with
`--reasoning off` produces clean output. It needs no API key on localhost (the
backend sends `EMPTY`). `start.sh` launches one server per unique API port
(gen + judge share a port → one model) and waits on `/health`.

Per-role env: `CS_{GEN,JUDGE}_HF` (the `-hf` download spec, `repo:quant` or a
local path), `CS_{GEN,JUDGE}_MODEL` (the API alias). `CS_LLAMA_SERVER` /
`CS_LLAMA_LIBS` override the binary/lib paths; `CS_CONTEXT` the window.

Memory: a 12 GB GPU fits one E4B (~5 GB) + embeddinggemma + DeBERTa. A 12B + an
E4B together do not — use one model for both roles, or put the bigger judge on
a second GPU / a Colab-tunnelled `llama-server`.

## Web app

- `webapp.py` — FastAPI; `POST /api/chat` streams SSE trace events
  (`modal`, `variants`, `scored`, `selected`, `response`, `done`); `GET
  /api/health`. Each turn is appended to `results/conversations.jsonl`
  (modal-rejected vs chosen — ready as DPO/ORPO pairs).
- `web/` — Vite + React + TypeScript SPA; dev server proxies `/api` to the
  backend.

## Testing

```bash
uv run pytest            # mock backend, no models needed
cd web && npm run typecheck
```

## Experiments (examples/)

| script | what |
|--------|------|
| `validate_selection.py`    | greedy vs pareto on one shared candidate pool |
| `benchmark.py`             | matched greedy-vs-stage1 across problems x repeats |
| `compare_variant_sources.py` | brainstorm vs independent diversity |
| `compare_embedders.py`     | embedding-model clustering discrimination |
| `stage1_demo.py`           | single-turn think-and-select trace |
| `stage1_trajectory_demo.py`| multi-step Stage 1 vs greedy |
| `smoke_ollama.py`          | live generation + logprobs + entailment + judge |

## Layout

```
src/creativity_steer/
  backends.py    GenSample, LLMBackend, Ollama/OpenAI/Mock backends
  config.py      .env loading + per-role backend factory
  entailment.py  bidirectional entailment: LLM / embedding / DeBERTa-NLI
  divergent.py   clustering, semantic entropy, novelty
  reference.py   novelty-vs-modal distance
  convergent.py  per-step rubric judge
  judge.py       multi-agent retrieval judge
  criteria.py    paper criterion definitions
  selection.py   Pareto frontier, selection rules, trajectories
  stage1.py      think-and-select + trajectories
  chat.py        chat-mode streaming pipeline
  webapp.py      FastAPI SSE server
  cli.py         comparison CLI
web/             React/TS SPA
tests/           pytest (mock backend)
examples/        experiments + demos
```

## Install extras

```bash
uv sync --extra dev --extra deberta --extra web --extra api
```

## License

MIT.
