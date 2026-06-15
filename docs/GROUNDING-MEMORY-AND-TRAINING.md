# Grounding, Impact-Memory, and What Training Actually Teaches

Status: concept / research direction. Captures the "weak traces" problem and a
McGaugh-inspired impact-memory + consolidation layer. Companion to
[CONCEPT.md](CONCEPT.md) and [PLAN.md](PLAN.md).

## 1. The weak-traces problem

Today the system is creative only over **raw text** — no skills, tools, memory,
retrieval, or consequences. So the (chosen vs modal) pairs we log capture
*stylistic* divergence, not grounded problem-solving creativity. As training
signal they are thin.

Two independent upgrades make traces strong:

1. **Grounding** — attach creativity to substance: a creative *use of a tool*, a
   surprising *plan*, a non-obvious *retrieval*. Then the trace carries outcome,
   not just style.
2. **Impact-curation** — even ungrounded, not all turns are equal; keep the
   high-impact ones (see §3).

## 2. What training teaches: basins vs the stepping-through skill

A central question: does fine-tuning on our traces give a model with new
*defaults*, or one that has learned to *search*?

- **SFT on chosen outputs → relocates attractor basins.** New defaults; learns
  *what* to say, not *how to explore*.
- **DPO on (chosen ≻ modal) → learns a direction** (away from the default toward
  the selected). A gradient — slightly more than SFT, still distributional.
- **Process-trace training → can teach the stepping-through.** Train on the
  *search* (modal → alternatives → selection rationale), the way reasoning
  models learn to reason from process, not answer keys.
- **RL with the multi-axis creativity score as reward → trains a policy** of
  exploration rather than a distribution. This is the most direct route to
  "learn to step through basins toward novelty." The four axes (novelty,
  quality, coherence, openness) are ready-made reward components.

Takeaway: **endpoints → new basins; process-traces / RL-reward → the meta-skill.**
Absent that, the harness *is* the search and the model stays the generator —
itself a valid architecture, just not an internalised one.

## 2a. Primary constraint — reward the explorer, not the answer

**Preference optimization is narrative compression.** Optimizing toward what a
rater (human or AI) approves narrows the model's codebook toward the rater's —
the exact counterfactual-suppression this project exists to resist. Making
approval the primary objective makes the tool defeat its own purpose: regression
to the common, suppression of the contrarian-but-right, sycophancy, mode
collapse (RLHF's documented failure modes).

Therefore, non-negotiably:

1. **Primary objective = process/structural axes** (novelty, coherence,
   openness). These measure the *search*, not anyone's approval; they are
   reference-free and cannot be sycophantically gamed.
2. **Evaluation is a floor, never the objective.** The quality judge and any
   feedback prune garbage (novelty *subject to* a quality floor). Promoting
   quality or preference to the maximization target re-introduces Goodhart and
   destroys the creativity.
3. **Corrections, not compliments.** Learn from *objective, demonstrated
   failure* (errors, wrong facts, surfaced consequences) — asymmetric negative
   signal. *Approval* ("thumbs up") is corruptible and regression-inducing; at
   most a weak impact marker, never a quality verdict to optimize toward.
4. **Delayed > immediate.** Per McGaugh, consolidation happens after the event:
   the "looks great now / catastrophic later" case means the late, objective
   outcome must dominate the in-the-moment reaction. Weight outcomes that
   surface over time; discount immediate approval.
5. **Anti-monoculture is the safeguard against confidently-wrong outputs.** A
   wrong answer is dangerous only as a monopoly. Because the system never
   collapses to one answer, a flawed "good" answer always sits beside its
   alternatives. The defense against bad evaluation is not better evaluation —
   it is never letting evaluation foreclose the option space. Human = exploration
   partner, not grader.

## 2b. The formal backbone — Maximum Occupancy Principle (MOP)

§2a is not just a design preference; it is the **Maximum Occupancy Principle**.

- **MOP maximizes path entropy** — occupy *all* the distinct high-reward states,
  weighted by rareness; always stochastic; **no reference model**. That is our
  novelty/openness axes, and the brainstorm-and-select harness (BoN as
  occupancy exploitation).
- **Absolute vs relative entropy** (different measurement primitives): MOP =
  absolute (variety of trajectories actually taken). KL/RLHF = relative
  (divergence from a reference) → converges to the *mode*. Our axes are the
  absolute/occupancy kind by construction.
- **The KL theorem (MOP paper, Suppl. F):** under KL-regularization the immediate
  return is `H(A|s) − ln|A(s)|`. The `−ln|A(s)|` term **penalizes states with
  many available actions** — RLHF structurally suppresses the rich-option states
  creativity needs. This is the formal proof of §2a: preference/KL optimization
  destroys diversity by construction, not by mis-tuning.

### Training prescription (steer away from the KL trap)

- **Avoid** standard DPO/PPO with **KL-against-a-reference** — it collapses
  toward one new deterministic basin (loses the creativity). DPO ≈ this trap.
- **Prefer MOP-compatible objectives:**
  - **GRPO** — no reference model, group-relative advantage; the most compatible
    existing algorithm (preserves stochasticity).
  - **Occupancy / entropy reward** — RL where the reward is the multi-axis
    *exploration* score (novelty, coherence, openness), i.e. reward visiting
    diverse coherent states. Trains *trajectory diversity* = "stepping through
    basins," not mode-convergence.
  - **Occupancy-relative regularization** — replace `KL(π‖π_ref)` with
    `KL(π(·|s)‖π_group(·|s))`.
- **Absorbing states = the floor.** MOP replaces the reference tether with
  designed absorbing states (deontological boundaries); inside them, pure
  occupancy maximization. Our quality/safety floor *is* that boundary — and this
  matches the project's ethics: deontology as the hard boundary, utilitarian
  reward as servant (a floor), never master.

Caveat — open research: entropy bonuses are unstable; GRPO-for-this and
axes-as-occupancy-reward are untested at scale. Direction, not recipe.

## 3. Impact-memory (McGaugh) — the curator we half-have

Subordinate to §2a: impact selects *which turns are worth keeping* (friction,
surprise, corrections, surfaced consequences) — it does **not** define the
optimization target. The target stays the exploration axes.

Memories consolidate in proportion to emotional/adrenergic impact, *after* the
event. The agent analogue: weight by **processing friction and resolution**, and
consolidate post-session. Flat logging (every turn → vector DB) is the "bad
student": bloated, noisy, weak.

creativity-steer already produces the impact signal:

| McGaugh trigger | Our existing signal |
|---|---|
| Surprise / prediction error | novelty + surprise axes (divergence from the modal = the model's expected path) |
| Friction / correction loops | controller's explore-when-collapsed (extra rounds = "worked hard") |
| Stakes | axis weights, quality floor |
| Correction (objective failure) | **missing** — capture user fixes / errors / surfaced consequences (NOT approval; see §2a) |

### Impact Score (per turn)

A scalar from signals we already compute, e.g.:

```
impact = w1 * controller_rounds_beyond_1
       + w2 * (mean modal-vs-chosen gap across axes)
       + w3 * low_initial_diversity        # had to escape the modal basin
       + w4 * user_feedback                 # +1 validation, -1 correction
```

### Consolidation (post-session, background)

Not real-time. After a conversation ends, a consolidation pass:
1. drops neutral, low-impact turns (the leaky default track);
2. keeps high-impact turns as **strong training traces**;
3. extracts the "lesson" (the resolved correct path / the winning creative move)
   for **cross-session memory**.

One signal, two layers: it curates the **training set** *and* defines the
**memory** policy (remember what cost friction or earned reward).

## 4. Where this plugs in

- `results/conversations.jsonl` already logs modal, candidates, all axis scores,
  the controller's decisions, and the chosen reply — most of the impact inputs.
- The specced `build_preference_dataset.py` (PLAN.md) should become
  **impact-weighted**: keep only high-impact turns, and (per §2a) build pairs
  from the **axes and the exploration process**, not from approval. RL reward, if
  used, is the multi-axis exploration score — never a preference scalar.
- Capture **corrections** (objective failures the user fixed, errors, surfaced
  consequences) as asymmetric negative signal. Do **not** add an approval
  ("thumbs up") optimizer — it compresses toward the rater (§2a).
- Grounding (tools/memory/skills) is the parallel track that raises trace
  strength at the source; it is out of scope for the selector but is the
  precondition for traces worth training on.

## 5. Open questions

- Calibrating the impact weights; is controller-effort a good friction proxy?
- Does process-trace SFT actually transfer the search, or only imitate its
  surface? (Compare against RL-reward on the same axes.)
- Memory retrieval: how does consolidated memory feed back into a future turn's
  modal/brainstorm without re-injecting stale "thoughts" (the multi-turn
  thinking-leak failure)?
- Grounded impact: with tools, friction becomes tool-error/retry — a much
  stronger, more objective signal than text divergence.
