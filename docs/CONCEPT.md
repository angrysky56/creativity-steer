# Self-Tuning, Multi-Axis Creative Selection — Concept Note

Status: design / research direction (not yet implemented).
Builds on the shipped pipeline (brainstorm → score → select) and on three
internal frameworks: **attractor dynamics**, **portfolio-of-policies
metacognition** (oMCD), and **power-as-narrative-compression**.

The premise that makes all of this affordable: inference is now fast (small
model, thinking off, ~one judge call). When generation and scoring are cheap,
we can produce and evaluate *far* more freely — and, crucially, let the system
**tune its own selection parameters** instead of relying on human-set sliders.

---

## 1. What we have

Per turn we produce a *modal* (greedy) answer plus K brainstormed variants,
score each on two axes — **novelty** (distance from the modal answer) and
**quality** (a comparative rubric judge) — and Pareto-select the best trade-off
above a quality floor. Humans set the levers: `k`, `novelty_weight`,
`quality_floor`, `temperature`.

Two limitations motivate the extension:

1. **The axes are too few.** Novelty rewards *distance from the default*, but
   nothing checks whether the distant point is a real idea or noise.
2. **The levers are static and human-set.** The right amount of exploration
   depends on the question and on what the brainstorm actually produced; a fixed
   `novelty_weight` can't know that.

---

## 2. Reframe I — purpose: counterfactual surfacing

(*power-as-narrative-compression*)

`power = (control over narrative) / (exposure to counterfactuals)`.

The modal reply is the **dominant narrative** — the single output distribution
a post-RLHF model is engineered toward. The brainstormed variants are a
**counterfactual codebook**. Selecting away from the modal *lowers the user's
narrative-compression ratio*: it surfaces options the default suppresses.

So the tool is not merely "more creative chat." It is a personal
**anti-monoculture-of-thought instrument** — it keeps the user's option space
open against a model biased toward one answer. This is the load-bearing purpose,
and it implies a measurable axis (§4: *counterfactual openness*).

## 3. Reframe II — mechanism: attractor navigation

(*attractor dynamics*)

Generation settles into **attractor basins**. The modal answer is the **deepest
basin** (most probable, the default). "Distance from modal" is how far a
candidate has climbed out of that basin.

This exposes the geometric form of the Goodhart failure: novelty rewards being
*far from the basin floor*, but a far-away point may be **another real basin or
just noise off the edge of the map**. The missing measurement is **basin depth /
coherence** — is the candidate a stable idea or a fluctuation? Operating *at
criticality* (deep enough to be coherent, shallow enough to explore) is the
target regime for creativity.

---

## 4. The selection space (axes)

Each axis must be **reference-free** — computable from the candidates and the
model, no gold answer — the property that made the source metrics scalable.

| Axis | Frame | Reference-free measurement | Cost |
|------|-------|----------------------------|------|
| **Distance from modal** | narrative / attractor | embedding distance of candidate from the modal answer (shipped) | cheap |
| **Quality** | convergent | comparative critical rubric judge (shipped) | 1 call |
| **Coherence / basin depth** | attractor; anti-Goodhart | self-consistency under perturbation: re-paraphrase the candidate a few times, embed, measure cluster tightness. Deep basin → returns to itself; noise → scatters | a few short gens |
| **Surprise / information gain** | entropy-confidence duality | sequence log-probability (already captured): low prob = high surprise. "Insight" = surprising **and** coherent | free |
| **Counterfactual openness** | narrative-compression | branching factor: generate a few short continuations conditioned on the candidate; measure their spread. Option-opening answers fan out, closed pronouncements collapse | a few short gens |

The current 2-D Pareto becomes a multi-objective frontier. Novelty sharpens
from "far" to "**surprising, coherent** departure from the default" — which is
the anti-Goodhart guard expressed as geometry rather than a bolt-on floor.

---

## 5. From point-selection to portfolio

(*portfolio-of-policies metacognition*)

Stop collapsing to a single winner. The oMCD/portfolio view: allocate a budget
across *threads* by marginal benefit/cost. Here:

- **Submodular spanning** — pick a small *set* of candidates that best covers
  the basins, scoring each by its *marginal* contribution to the set's
  coverage, not in isolation. This avoids choosing three near-duplicates of the
  same good idea.
- **Synthesis** — optionally compose the spanning set into one answer that
  braids distinct basins (a deliberate counterfactual *combination*).
- `novelty_weight` is then the human-visible special case of the principled
  **entropy-confidence dial β**: the Lagrange multiplier trading log-SNR
  (confidence/quality) against entropy (diversity/novelty).

---

## 6. The metacognitive control layer — *its own levers*

(*oMCD: portfolio allocation + stop/continue*)

This is the centerpiece. The parameters currently exposed as human sliders
become **state-dependent decisions the system makes per turn**, reading its own
entropy/confidence signals. Two oMCD-flavored controls:

### 6.1 Allocation — set the creativity parameters

The controller observes a **state** summarizing the turn:

- semantic entropy / spread of the brainstormed set (are we stuck in the modal
  basin, or is there real diversity?),
- the spread of quality scores (is anything actually good, or all mediocre?),
- coherence of the high-novelty candidates (is the far region real or noise?),
- a read of the question itself (does it even call for creativity?).

…and sets `k`, β (exploration weight), the quality floor, and temperature
accordingly. The **entropy-confidence duality** gives the adaptive rule:

- **Low-entropy set** (variants collapse onto the modal — a deep basin we can't
  escape) → *increase* exploration: raise temperature, raise `k`, raise β, push
  the perturbation budget. Climb harder out of the basin.
- **High-entropy set** (many diverse, coherent options) → *select harder*: raise
  the quality/coherence floor, lower β; the diversity is already there, now be
  discerning.

Cost itself becomes entropy-dependent (from the note: `ν(H) = ν₀ + γ·H`) — in
chaotic regimes, spend more conservatively.

### 6.2 Stopping — explore more vs commit

oMCD's single-action stop/continue MDP maps **exactly** onto: *brainstorm
another round of candidates, or commit to the best so far?* After each round the
controller estimates confidence that the current frontier is good enough; it
continues exploring while the expected marginal benefit of another round exceeds
its cost, and commits when confidence crosses threshold. Easy questions stop
after one round; rich, open questions earn more exploration. The "creativity
budget" `Z_max` is itself allocated by the demand of the prompt.

### 6.3 Why now

This is only practical because inference is fast. Multi-round exploration,
perturbation-based coherence, and branching all cost extra generations — cheap
on a small local model with thinking off. Speed is what lets the system "produce
and select more freely, work through its own parameter selection."

---

## 7. Phased plan (when we build)

1. **Cheap axes first** — add *surprise* (free; logprobs already captured) and
   *coherence* (perturbation self-consistency). Make the Pareto multi-axis; the
   UI gains two bars. Kills the novel-garbage failure mode.
2. **Counterfactual openness** — the branching-factor axis; the most direct
   measure of the §2 thesis. Costs short gens.
3. **Portfolio selector** — swap argmax for submodular spanning; optional
   synthesis.
4. **Metacognitive controller** — start with a transparent heuristic mapping
   (state → parameters) per §6, log its decisions, then optionally learn the
   mapping from the accumulating `conversations.jsonl` (which already records
   modal-vs-chosen with scores). The stop/continue rule comes last.

Each phase is independently shippable and observable in the existing trace UI.

---

## 8. Open questions

- **Learning the controller.** §6 can be a hand-tuned heuristic or learned
  (policy gradient / Bayesian opt over the parameter simplex, per the portfolio
  note's weight-learning question). The logged conversations are the dataset.
- **Coherence estimator bias.** Perturbation self-consistency approximates basin
  depth; what are its failure modes (e.g., bland answers are trivially
  self-consistent)? Pair it with surprise so blandness isn't rewarded.
- **Counterfactual openness vs. coherence tension.** Maximally open answers
  (many branches) may be less coherent; these may be genuinely opposed axes,
  which is the point of a multi-objective frontier rather than a scalar.
- **Synthesis honesty.** Braiding basins risks incoherent mash-ups; synthesis
  needs its own coherence check.
- **Does self-tuning drift?** A controller that sets its own exploration could
  spiral (always explore) or collapse (always commit). The entropy-confidence
  equilibrium and a budget cap are the damping mechanisms; needs verification.

---

## References

- Source paper (measurement apparatus we build on): *Automated Creativity
  Evaluation of Language Models Across Open-Ended Tasks*, Tan Min Sen et al.,
  ACL 2026 — semantic entropy (divergent) + multi-agent judge (convergent),
  shown empirically separable.
- Internal: `attractor-dynamics`, `portfolio-policies-metacognition` (oMCD,
  entropy-confidence duality), `power-as-narrative-compression`.
