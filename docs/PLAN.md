# Implementation Plan — Multi-Axis, Self-Tuning Creative Selection

Agent-executable plan for the ideas in [CONCEPT.md](CONCEPT.md). Read CONCEPT.md
first for the *why*; this is the *how*, sequenced so each phase ships
independently and is visible in the existing trace UI.

## Ground rules (read before touching code)

- **Reference-free only.** Every axis must be computable from the candidates +
  the model, no gold answer. This is what keeps it scalable.
- **The SSE event protocol is a contract.** A redesigned React/TS frontend
  consumes it. Extend events **additively** — never remove or rename existing
  fields (`type`, `index`, `novelty`, `distance`, `quality`, `frontier`,
  `is_modal`, `text`). Add new data alongside.
- **Keep it fast and local.** New axes cost extra short generations; that's
  fine on the small local model, but cap counts and run in the existing async
  flow. No new heavyweight deps.
- **Tests stay green.** `MockBackend` must deterministically support every new
  scorer/prompt (see existing branches in `backends.py`). `uv run pytest` must
  pass after each phase.
- **Don't break Ollama/llama-server/mock backends or the `.env` factory.**

## Architecture foundation (Phase 0 — do first)

### 0a. Scorer registry
Add `src/creativity_steer/scoring.py`:
```python
class Scorer(Protocol):
    name: str
    def score(self, ctx: ScoringContext) -> list[float]: ...   # one [0,1] per candidate
```
`ScoringContext` carries: `gen`, `judge`, `embed` backends, `history`,
`user_msg`, `modal`, `texts` (modal + variants), `samples` (with logprobs),
`entailment`, `config`. Refactor today's two signals into scorers:
- `NoveltyScorer` — wraps `reference_distances` + `normalize_max`.
- `QualityScorer` — wraps `judge_comparative`.
Register scorers in an ordered dict so axes have stable identity.

### 0b. N-axis selection
Generalize `pareto_mask` and `_select` to operate on an `M×A` matrix (M
candidates, A axes) with a weight vector `w` (sum-normalised). 2-axis behaviour
(today's `novelty_weight`) must be the exact special case. Add `axis_weights:
dict[str,float]` to `ChatConfig` (default `{novelty:0.5, quality:0.5}`).

### 0c. Event protocol extension
`scored` event gains `scores: {axisName: value}` while **keeping** the existing
`novelty`/`distance`/`quality` keys (populate them from the registry for
back-compat). Coordinate the exact shape with the maintainer before the UI
relies on the dict.

Acceptance: pytest green; live chat identical output to pre-refactor (2 axes);
new `scores` dict present in events.

## Phase 1 — cheap axes (surprise + coherence)

- **SurpriseScorer** (free): use `GenSample.logprob` already captured per
  candidate. Length-normalise (avg token logprob), invert and min-max to [0,1]
  so *less probable = more surprising*. Modal/brainstorm gens must carry
  logprobs (Ollama + llama-server do; if `None`, return neutral 0.5).
- **CoherenceScorer** (basin depth, anti-Goodhart): for each candidate, generate
  `p` (≈3) terse paraphrases at low temp, embed all, score = mean pairwise
  cosine (tight cluster → deep basin → high coherence). Cache embeddings.
- Wire both into the registry and default `axis_weights`. Emit in `scores`.
- Tests: MockBackend branches for the paraphrase prompt; assert surprise tracks
  logprob ordering and coherence separates a stable pool sentence from a
  hashed-noise string.

Acceptance: Pareto is multi-axis; "novelty" effectively becomes "surprising +
coherent departure"; novel-but-incoherent candidates score low coherence.

## Phase 2 — counterfactual openness

- **OpennessScorer**: for each candidate, generate `b` (≈3) short continuations
  conditioned on it; embed; score = spread (1 − mean pairwise cosine). High
  spread = the answer opens many paths. Costs `M×b` short gens — cap `b`, run
  once per turn, reuse where possible.
- Register; add to weights; emit. Mock support + test (spread vs collapse).

Acceptance: openness axis present and visibly different from novelty on
option-opening vs closed-pronouncement answers.

## Phase 3 — portfolio selector

- Add `SelectorMode`: `pareto` (current) and `portfolio`.
- Portfolio: greedy **submodular** set selection over the candidates maximising
  axis coverage with diminishing returns for redundancy; size `s` (≈1–3).
- Optional `synthesize=True`: one gen call braiding the selected set into a
  single answer, then re-score it for coherence as a guard.
- Emit a `portfolio` event (additive) listing chosen indices. Tests on mock.

Acceptance: selecting `s>1` returns a spanning set, not near-duplicates;
synthesis (if on) passes a coherence floor.

## Phase 4 — metacognitive controller (the lever)

- Add `src/creativity_steer/control.py` with `TurnController`:
  - `observe(state) -> ChatConfig` — maps a `TurnState` (set entropy/SE, quality
    spread, coherence of high-novelty candidates, a cheap "needs-creativity"
    read of the prompt) to params: `k`, `axis_weights` (esp. the exploration
    dial β = novelty/surprise vs quality), `convergent_floor`, `temperature`.
    Use the entropy-confidence rule from CONCEPT §6.1 (collapsed→explore harder;
    diverse→select harder).
  - `should_continue(rounds, frontier) -> bool` — oMCD stop/continue: keep
    exploring while expected marginal gain > cost, commit when confident.
- Make `chat_turn_stream` **iterable**: a loop over rounds, controller sets
  params per round, accumulates candidates, stops on `should_continue` or a
  round cap. Emit `controller` events (additive) recording each decision so the
  UI/logs show the system pulling its own levers.
- Start with a **transparent heuristic** controller (documented rules); leave a
  seam to learn it later (Phase 5).

Acceptance: with the controller on, params vary per turn and are logged; easy
prompts commit in one round, open prompts explore more; manual sliders still
override when set.

## Dataset track (parallel; for the training agent)

- Enrich `results/conversations.jsonl` per turn to include: all candidates with
  every axis score, the chosen index, the modal (rejected) baseline, the
  controller's params, and the final selector mode. (Schema additive to current
  logging in `webapp.py::_log_turn`.)
- Add `examples/build_preference_dataset.py`: reads the JSONL, emits DPO/ORPO
  pairs `(prompt, chosen, rejected=modal)` plus an optional scored reward column
  from the axis weights — a clean handoff artifact for the unsloth agent.
- **Sequencing:** only harvest for training *after* Phases 1–2 land, so the
  preferences encode the sharper multi-axis selection, not today's 2-axis one.

## Out of scope here (Stage 2 — different substrate)

Training the model to internalise this (LoRA/DPO via unsloth) and any
latent-space steering are a separate track that consumes the dataset above.
Don't attempt inside this plan.

## Suggested order & checkpoints

0 → 1 → (collect data) → 2 → 3 → 4. Run `uv run pytest` and a live mock + live
llama-server smoke after each phase. Keep PRs phase-sized.
