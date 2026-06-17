"""Persist process-valid turns as a creativity training dataset.

This is SEPARATE from the grounding memory (different Chroma collection) so it
never pollutes retrieval. A turn is stored only when it passed the *process*
bar — quality floor cleared, the steering produced a non-modal winner, and the
synthesis (if any) did not collapse onto the obvious answer. Validity is
process- and axis-based, never user approval: we keep what the exploration
earned, not what someone liked. The records (prompt, winning reply, full axis
scores, candidate pool, divergence stats) are exactly what a later GRPO /
occupancy-reward training run needs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Protocol

try:
    import chromadb
except ImportError:  # pragma: no cover - exercised only without chromadb
    chromadb = None

logger = logging.getLogger(__name__)


class TrainingStore(Protocol):
    def add(self, record: dict) -> bool: ...
    def count(self) -> int: ...


def _turn_id(message: str, response: str, ts: float) -> str:
    h = hashlib.sha256(f"{message}\x00{response}\x00{ts}".encode()).hexdigest()
    return h[:24]


def assess_validity(by_type: dict[str, list]) -> tuple[bool, str]:
    """Decide whether a completed turn is worth keeping, from its trace events.

    Process-based: the quality floor must be met, the chosen reply must not be
    the greedy modal (the steering actually did something), and a synthesis must
    not have collapsed onto the modal. When the originality/surprise axes are
    present, a winner that is a recognised cliché is rejected.
    """
    if "response" not in by_type or "selected" not in by_type:
        return False, "incomplete"
    sel = by_type["selected"][0]
    if not sel.get("floor_met", True):
        return False, "below_quality_floor"
    if sel.get("index", 0) == 0:
        return False, "winner_is_modal"
    syn = by_type.get("synthesis")
    if syn and syn[0].get("collapsed_to_modal"):
        return False, "synthesis_collapsed"
    # Cliché guard when the relevant axes were actually computed.
    winner = next(
        (s for s in by_type.get("scored", []) if s.get("index") == sel.get("index")),
        None,
    )
    if winner:
        axes = winner.get("scores", {})
        if "originality" in axes and axes["originality"] < 0.35:
            return False, "winner_is_cliche"
    return True, "ok"


def _winner_axes(by_type: dict[str, list]) -> dict[str, float]:
    sel = by_type["selected"][0]
    winner = next(
        (s for s in by_type.get("scored", []) if s.get("index") == sel.get("index")),
        None,
    )
    return dict(winner.get("scores", {})) if winner else {}


def _mean_axis_gap(by_type: dict[str, list]) -> float:
    """Mean positive axis gap between the chosen reply and the modal (index 0).

    A wide gap = the selection moved far off the model's default path (McGaugh
    'prediction error' / friction). Modal is the reference, so this is how hard
    the steering pulled away from the obvious answer.
    """
    scored = {s["index"]: s.get("scores", {}) for s in by_type.get("scored", [])}
    sel = by_type["selected"][0].get("index", 0)
    modal, chosen = scored.get(0, {}), scored.get(sel, {})
    axes = set(modal) & set(chosen)
    if not axes:
        return 0.0
    return sum(max(0.0, chosen[a] - modal[a]) for a in axes) / len(axes)


def compute_impact(by_type: dict[str, list], is_correction: bool = False) -> float:
    """Impact score (plan §3 / McGaugh) from signals already in the trace.

    Reuses ``consolidation.impact_score``: controller effort (rounds beyond 1),
    modal-vs-chosen axis gap, and low initial diversity (had to escape the modal
    basin). NO approval term — corrections only, per §2a.
    """
    from creativity_steer.consolidation import impact_score

    ctrl = (by_type.get("controller") or [{}])[0]
    init_div = ctrl.get("norm_entropy", ctrl.get("diversity", 0.5))
    return impact_score(
        {
            "rounds": ctrl.get("rounds", 1),
            "mean_axis_diff": _mean_axis_gap(by_type),
            "initial_diversity": init_div,
            "is_correction": is_correction,
        }
    )


def build_record(message: str, history: list, by_type: dict[str, list]) -> dict:
    """Assemble the stored training record from a turn's trace events."""
    resp = by_type["response"][0]
    ctrl = (by_type.get("controller") or [{}])[0]
    return {
        "ts": time.time(),
        "message": message,
        "history": history,
        "response": resp["text"],
        "synthesized": bool(resp.get("synthesized")),
        "winner_index": by_type["selected"][0].get("index", 0),
        "winner_axes": _winner_axes(by_type),
        # Impact-weight (plan §3) so the dataset is curated, not flat-logged.
        "impact": round(compute_impact(by_type), 4),
        "is_correction": False,  # hook: correction capture is still TODO (§4)
        "candidates": [
            {
                "text": v["text"],
                "is_modal": v.get("is_modal", False),
                "scores": next(
                    (
                        s.get("scores", {})
                        for s in by_type.get("scored", [])
                        if s.get("index") == i
                    ),
                    {},
                ),
            }
            for i, v in enumerate(by_type.get("variants", [{}])[0].get("items", []))
        ],
        "semantic_entropy": ctrl.get("semantic_entropy"),
        "num_clusters": ctrl.get("num_clusters"),
        "num_candidates": ctrl.get("num_candidates"),
        "weights": ctrl.get("weights", {}),
        "quality_floor": ctrl.get("quality_floor"),
    }


_CORRECTION_JUDGE = (
    "Decide whether the user's NEW message corrects, rejects, or fixes the "
    "PREVIOUS reply (e.g. 'no, that's wrong', 'actually it's X', pointing out an "
    "error or bad answer). A new unrelated request is NOT a correction. Answer "
    "strictly YES or NO.\n\n"
    "Previous reply:\n{prev}\n\nUser's new message:\n{msg}\n\nAnswer (YES/NO):"
)


def detect_correction(judge, prev_reply: str, user_msg: str) -> bool:
    """Classify whether ``user_msg`` corrects ``prev_reply`` (one cheap call).

    Corrections are the asymmetric NEGATIVE signal the plan prizes (§2a #3) —
    objective, demonstrated failure, never approval.
    """
    if not prev_reply or not user_msg:
        return False
    out = judge.chat(
        _CORRECTION_JUDGE.format(prev=prev_reply[:1500], msg=user_msg[:800]),
        temperature=0.0,
        num_predict=4,
    )
    return out.strip().upper().startswith("Y")


def build_correction_record(
    prior_prompt: str, failed_reply: str, correction_msg: str
) -> dict:
    """A negative training example: for this prompt, this reply was wrong, and
    here is the correction. Impact is high (correction flag dominates §3)."""
    from creativity_steer.consolidation import impact_score

    return {
        "ts": time.time(),
        "kind": "correction",
        "is_correction": True,
        "message": prior_prompt,
        "failed_response": failed_reply,
        "response": correction_msg,  # the fix (asymmetric negative signal)
        "winner_axes": {},
        "candidates": [],
        "impact": round(impact_score({"is_correction": True}), 4),
    }


class ChromaTrainingStore:
    """Append-only store of valid exemplars in a dedicated Chroma collection."""

    def __init__(self, embed_backend, path: str = "results/training_db") -> None:
        if chromadb is None:
            raise ImportError("chromadb is required for ChromaTrainingStore.")
        self.embed_backend = embed_backend
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name="creativity_steer_training", metadata={"hnsw:space": "cosine"}
        )

    def add(self, record: dict) -> bool:
        try:
            doc = record["response"]
            emb = self.embed_backend.embed([doc])[0]
            axes = record.get("winner_axes", {})
            meta = {
                "ts": record["ts"],
                "kind": record.get("kind", "exemplar"),
                "is_correction": bool(record.get("is_correction", False)),
                "message": record["message"][:2000],
                "synthesized": record.get("synthesized", False),
                "impact": float(record.get("impact") or 0.0),
                "semantic_entropy": float(record.get("semantic_entropy") or 0.0),
                "num_clusters": int(record.get("num_clusters") or 0),
                # full record kept as JSON so a training run gets everything
                "record": json.dumps(record)[:60000],
            }
            for axis in (
                "novelty",
                "quality",
                "coherence",
                "openness",
                "originality",
                "surprise",
            ):
                if axis in axes:
                    meta[axis] = float(axes[axis])
            self.collection.add(
                ids=[_turn_id(record["message"], doc, record["ts"])],
                embeddings=[emb],
                documents=[doc],
                metadatas=[meta],
            )
            return True
        except Exception as e:  # never break a chat turn over logging
            logger.error(f"TrainingStore.add failed: {e}")
            return False

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    def all(self) -> list[dict]:
        """Return every stored record (parsed from the JSON metadata blob)."""
        try:
            res = self.collection.get(include=["metadatas"])
        except Exception as e:
            logger.error(f"TrainingStore.all failed: {e}")
            return []
        out: list[dict] = []
        for meta in res.get("metadatas") or []:
            blob = meta.get("record") if meta else None
            if blob:
                try:
                    out.append(json.loads(blob))
                except json.JSONDecodeError:
                    continue
        return out


def export_training_set(
    store: ChromaTrainingStore, out_path: str, min_impact: float = 0.4
) -> int:
    """Write an impact-weighted JSONL training set (plan §3/§4).

    Drops low-impact turns (the leaky default track) and keeps the curated,
    high-impact exemplars. Each line is the full record (prompt, winning reply,
    candidate pool with axis scores, divergence stats, impact) — a GRPO /
    occupancy-reward run reads the axes as the reward components (never a
    preference/approval scalar, per §2a). Returns the number written.
    """
    records = [r for r in store.all() if (r.get("impact") or 0.0) >= min_impact]
    records.sort(key=lambda r: r.get("impact", 0.0), reverse=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return len(records)


def build_training_store(embed_backend) -> ChromaTrainingStore | None:
    """Build the training store if enabled (CS_TRAINING_STORE=chroma)."""
    kind = os.getenv("CS_TRAINING_STORE", "none").lower()
    if kind != "chroma":
        return None
    path = os.getenv("CS_TRAINING_PATH", "results/training_db")
    try:
        return ChromaTrainingStore(embed_backend, path)
    except Exception as e:
        logger.error(f"Could not build training store: {e}")
        return None


def _cli() -> None:
    """`python -m creativity_steer.training_store --export [out] [min_impact]`."""
    import sys

    from creativity_steer.config import build_backend, load_env

    load_env()
    embed = build_backend("embed", None)
    store = ChromaTrainingStore(
        embed, os.getenv("CS_TRAINING_PATH", "results/training_db")
    )
    if "--export" in sys.argv:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        out = args[0] if args else "results/training_set.jsonl"
        min_impact = float(args[1]) if len(args) > 1 else 0.4
        n = export_training_set(store, out, min_impact)
        print(f"exported {n} turns (impact >= {min_impact}) -> {out}")
    else:
        print(f"training store: {store.count()} stored turns")


if __name__ == "__main__":
    _cli()
