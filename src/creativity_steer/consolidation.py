"""Memory impact and consolidation logic."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List

from creativity_steer.backends import LLMBackend
from creativity_steer.memory import MemoryItem, MemoryStore

logger = logging.getLogger(__name__)


def impact_score(
    turn_data: Dict[str, Any],
    a: float = 0.5,
    b: float = 0.3,
    c: float = 0.5,
    d: float = 1.0,
) -> float:
    """Calculate the impact score of a turn."""
    # This logic assumes the log structure format created by the controller
    # and the frontend metrics
    rounds = turn_data.get("rounds", 1)
    
    # difference between modal and chosen axes (we'll assume turn_data has this precomputed
    # or the raw scores to compute it)
    mean_axis_diff = turn_data.get("mean_axis_diff", 0.0)
    
    # initial diversity (semantic entropy) of the very first batch
    initial_diversity = turn_data.get("initial_diversity", 0.5)
    
    # Explicit user/system correction
    correction_flag = 1.0 if turn_data.get("is_correction", False) else 0.0

    impact = (
        a * max(0, rounds - 1)
        + b * mean_axis_diff
        + c * (1.0 - initial_diversity)
        + d * correction_flag
    )
    
    return impact


def _extract_lesson(gen: LLMBackend, prompt_text: str, is_correction: bool) -> str:
    """Ask the model to extract a lesson from the turn."""
    kind = "a correction to avoid" if is_correction else "an insight or useful discovery"
    prompt = f"""
Extract a concise, factual lesson from the following interaction.
You must return ONLY the final, self-contained lesson.
DO NOT include any reasoning, thoughts, or preamble.
The lesson should represent {kind}.

Interaction:
{prompt_text}

Lesson:"""
    
    # Use max_tokens to prevent it rambling
    response = gen.chat(prompt, temperature=0.1, num_predict=150)
    return response.strip()


def consolidate(
    turns: List[Dict[str, Any]], 
    memory: MemoryStore, 
    gen: LLMBackend,
    threshold: float = 0.4
) -> List[MemoryItem]:
    """Process a session log and store high-impact turns to memory."""
    
    new_memories = []
    
    for idx, turn in enumerate(turns):
        impact = impact_score(turn)
        
        if impact < threshold:
            continue
            
        is_correction = turn.get("is_correction", False)
        
        # Build a text representation of the turn to feed to the extractor
        user_msg = turn.get("prompt", "Unknown prompt")
        chosen_reply = turn.get("chosen_text", turn.get("response", ""))
        
        turn_text = f"User: {user_msg}\nResponse: {chosen_reply}"
        
        # Extract the lesson
        lesson = _extract_lesson(gen, turn_text, is_correction)
        
        # Preserve the option space (alternatives)
        alternatives = turn.get("alternatives", [])
        
        kind = "correction" if is_correction else ("options" if alternatives else "lesson")
        
        item_id = hashlib.sha256(f"{time.time()}_{idx}_{lesson}".encode()).hexdigest()[:16]
        
        item = MemoryItem(
            id=item_id,
            created=time.time(),
            last_used=time.time(),
            uses=0,
            kind=kind,
            content=lesson,
            context=f"Session processing turn {idx}",
            tags=[],
            impact=impact,
            alternatives=alternatives,
            source=turn.get("id", f"turn_{idx}")
        )
        
        memory.write(item)
        new_memories.append(item)
        
    return new_memories
