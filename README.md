# creativity-steer

Turn the **creativity-eval** metrics into a *generation-time steering signal*.

The source paper ([creativity-eval](https://github.com/tanminsen/creativity-eval))
measures LLM creativity with two reference-free signals: **semantic entropy**
(divergent / novelty) and a **retrieval-based multi-agent judge** (convergent /
task fulfilment). Its key empirical finding is that the two are *separable* —
higher novelty does not imply worse task fulfilment.

This project tests the consequence: because both signals are reference-free,
they can score the model's **own candidates at generation time**, so we can
*select* for creativity instead of only measuring it.

## The idea

At each step the standard pipeline greedy-decodes the most probable (modal)
continuation. Instead, we:

1. sample N candidate next-steps (with sequence logprobs),
2. cluster them by **bidirectional entailment** and score each candidate's
   **novelty** (surprisal of its semantic cluster),
3. score each candidate's **convergent** quality with a fast rubric judge,
4. pick the **Pareto-best** candidate (novel *and* task-fulfilling), subject to
   a convergent floor.

The convergent floor is the **anti-Goodhart guard**: semantic entropy alone is
trivially maximised by diverse garbage, so novelty is only ever optimised
*subject to* a quality threshold. We never optimise SE alone.

## Faithfulness to the paper

This is a faithful port of the measurement apparatus, not an approximation:

- **Semantic entropy** — greedy bidirectional-entailment clustering (paper
  §3.2 / App. C.3.3) with the Rao-Blackwellised, probability-weighted estimator
  (App. C.1, Eq. 4). Sequence probabilities come from Ollama logprobs; if a
  server can't supply them it falls back to count-based class probabilities.
- **Entailment model** — `llm` (default, via the generation model — the paper's
  GPT-4o-entailment variant), `embedding` (fast cosine proxy), or `deberta`
  (the paper's primary `tasksource/deberta-base-long-nli`; opt-in extra).
- **Convergent judge** — the retrieval-based multi-agent judge (paper §4,
  Fig. 2, App. D.2): Problem / Solution / Criterion analysts exchange fragments
  through a vector store, retrieve top-k per turn, score confidence, early-exit
  at threshold T, and deliver a binary verdict per criterion. Criterion
  definitions are verbatim from the reference implementation.

### Two-stage design (matches the paper)

The multi-agent judge evaluates a *complete solution* (~dozens of model calls),
so it is **not** run per candidate. Mirroring the paper:

- **per step** — fast single-agent rubric judge scores every candidate (the
  paper's Table 2 single-agent baseline);
- **per trajectory** — the full multi-agent judge scores the finished solution
  (the authoritative convergent number).

Two deliberate, documented substitutions keep the project light without changing
the algorithm: the fragment store is an in-memory cosine store rather than
ChromaDB, and embeddings come from the Ollama embedding model rather than a
bundled SentenceTransformer.

## Selection strategies

| strategy     | rule                                                        |
|--------------|-------------------------------------------------------------|
| `greedy`     | pick the modal idea (largest semantic cluster) — baseline   |
| `convergent` | maximise the step rubric score only (ablation)              |
| `pareto`     | Pareto frontier of (novelty, convergent) above the floor    |

## Install

Uses [uv](https://docs.astral.sh/uv/). Python ≥ 3.12.

```bash
cd creativity-steer
uv sync --extra dev
# optional: the paper's exact DeBERTa-NLI entailment (pulls in torch)
uv sync --extra deberta
```

## Run

Model-free (validates the whole pipeline instantly, no server needed):

```bash
uv run creativity-steer --backend mock --strategy all
uv run python examples/run_demo.py
```

Live, on a local model via [Ollama](https://ollama.com):

```bash
# smoke test: generation + logprobs + entailment + step judge
uv run python examples/smoke_ollama.py

# full comparison (note: the multi-agent judge is slow on local models)
uv run creativity-steer --backend ollama --model gemma4:12b \
    --strategy all --problem keys-in-drain -n 3 -s 2 \
    --judge-rounds 1 --output results.json
```

Reasoning models (gemma4, deepseek-r1, …) emit a separate "thinking" channel;
the Ollama backend disables it by default so `content` is populated and the
token budget isn't wasted. Pass `think=True` to `OllamaBackend` to re-enable.

Key flags: `--entailment {llm,embedding,deberta}`, `-n/--n-candidates`,
`-t/--temperature`, `-s/--max-steps`, `--novelty-weight`, `--convergent-floor`,
`--judge-rounds`, `--confidence-threshold`, `--no-final-judge`.

Tip: `--entailment embedding` is much faster than `llm` (no O(k²) entailment
calls) and is a good default for larger sweeps.

## Test

```bash
uv run pytest
```

## Layout

```
src/creativity_steer/
  backends.py    # GenSample, LLMBackend protocol, OllamaBackend, MockBackend
  entailment.py  # bidirectional entailment: LLM / embedding / DeBERTa-NLI
  divergent.py   # entailment clustering, Rao-Blackwell semantic entropy, novelty
  convergent.py  # fast per-step single-agent rubric judge
  judge.py       # retrieval-based multi-agent judge + fragment store
  criteria.py    # paper criterion definitions (MacGyver / HypoGen / BookMIA)
  selection.py   # Pareto frontier, selection rules, two-stage trajectory runner
  data.py        # sample problems + per-domain criteria lookup
  cli.py         # comparison CLI
tests/           # metric, judge, and selection tests (mock backend)
examples/        # mock demo + live Ollama smoke test
```

## Status & next steps

The inference-time selection prototype is complete and validated end-to-end on
`gemma4:12b`. Natural extensions:

- tree search over steps (expand toward sparse clusters, prune with the judge);
- use the per-step novelty + judge signals as a reward for rejection-sampling /
  RLAIF fine-tuning — an `UnslothBackend` slots into the same protocol for the
  training experiments;
- batch entailment / parallelise judge calls to speed up large sweeps.

Self-contained; does not import from the original `creativity-eval` repo.

## License

MIT.
