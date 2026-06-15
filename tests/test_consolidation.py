"""Tests for the consolidation module."""

from creativity_steer.consolidation import consolidate, impact_score
from creativity_steer.backends import MockBackend
from creativity_steer.memory import LocalMemoryStore

def test_impact_score():
    # Low impact
    turn_low = {
        "rounds": 1,
        "mean_axis_diff": 0.0,
        "initial_diversity": 0.9,
        "is_correction": False
    }
    score_low = impact_score(turn_low)
    assert score_low < 0.2

    # High impact
    turn_high = {
        "rounds": 3,
        "mean_axis_diff": 0.5,
        "initial_diversity": 0.2,
        "is_correction": True
    }
    score_high = impact_score(turn_high)
    assert score_high > 1.0


def test_consolidate(tmp_path):
    backend = MockBackend()
    memory = LocalMemoryStore(backend, str(tmp_path / "memory.jsonl"))
    
    turns = [
        # Should be skipped (impact < 0.4)
        {
            "id": "1",
            "prompt": "Hello",
            "chosen_text": "Hi there",
            "rounds": 1,
            "mean_axis_diff": 0.0,
            "initial_diversity": 0.9,
            "is_correction": False,
            "alternatives": []
        },
        # Should be stored
        {
            "id": "2",
            "prompt": "Solve this complex thing",
            "chosen_text": "Here is the solution",
            "rounds": 4,
            "mean_axis_diff": 0.6,
            "initial_diversity": 0.3,
            "is_correction": False,
            "alternatives": ["Alt 1", "Alt 2"]
        },
        # Should be stored as correction
        {
            "id": "3",
            "prompt": "Fix this error",
            "chosen_text": "Fixed",
            "rounds": 2,
            "mean_axis_diff": 0.1,
            "initial_diversity": 0.8,
            "is_correction": True,
            "alternatives": []
        }
    ]
    
    # We use MockBackend which returns standard mock text for things it doesn't recognize
    new_memories = consolidate(turns, memory, backend, threshold=0.4)
    
    # Only turn 2 and 3 should be consolidated
    assert len(new_memories) == 2
    
    # Turn 2: kind options (since it has alternatives)
    assert new_memories[0].kind == "options"
    assert new_memories[0].source == "2"
    assert len(new_memories[0].alternatives) == 2
    
    # Turn 3: kind correction
    assert new_memories[1].kind == "correction"
    assert new_memories[1].source == "3"
