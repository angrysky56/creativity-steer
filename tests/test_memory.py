"""Tests for the memory module."""

import os
import time

import pytest
from creativity_steer.backends import MockBackend
from creativity_steer.memory import LocalMemoryStore, MemoryItem


@pytest.fixture
def mock_backend():
    return MockBackend()


@pytest.fixture
def memory_store(mock_backend, tmp_path):
    db_path = tmp_path / "memory.jsonl"
    return LocalMemoryStore(embed_backend=mock_backend, path=str(db_path))


def test_write_and_retrieve(memory_store):
    item1 = MemoryItem(
        id="1",
        created=time.time(),
        last_used=time.time(),
        uses=1,
        kind="lesson",
        content="This is a test lesson.",
        context="Testing",
        tags=["test"],
        impact=0.8,
    )
    
    item2 = MemoryItem(
        id="2",
        created=time.time(),
        last_used=time.time(),
        uses=1,
        kind="fact",
        content="Apples are red.",
        context="General",
        tags=["fruit"],
        impact=0.5,
    )
    
    memory_store.write(item1)
    memory_store.write(item2)
    
    # MockBackend embedding is just a deterministic vector. Since we rely on similarity,
    # the mock embeddings will all be [1.0] unless we mock it more carefully,
    # but let's test if both items are returned.
    results = memory_store.retrieve("lesson", k=2)
    assert len(results) == 2
    
    # We can fetch all and check the items are there
    all_items = memory_store.all()
    assert len(all_items) == 2
    ids = [item.id for item in all_items]
    assert "1" in ids
    assert "2" in ids


def test_decay(memory_store):
    now = time.time()
    
    # Old item
    item1 = MemoryItem(
        id="1",
        created=now - 100,
        last_used=now - 100,
        uses=1,
        kind="lesson",
        content="Old lesson.",
        context="Testing",
        tags=[],
        impact=0.5,
    )
    
    # New item
    item2 = MemoryItem(
        id="2",
        created=now,
        last_used=now,
        uses=1,
        kind="lesson",
        content="New lesson.",
        context="Testing",
        tags=[],
        impact=0.5,
    )
    
    memory_store.write(item1)
    memory_store.write(item2)
    
    decayed = memory_store.decay(max_idle_seconds=50)
    assert decayed == 1
    
    # Retrieve without dormant should return 1 item
    active_results = memory_store.retrieve("test", k=10, include_dormant=False)
    assert len(active_results) == 1
    assert active_results[0].id == "2"
    
    # Retrieve with dormant should return 2 items
    all_results = memory_store.retrieve("test", k=10, include_dormant=True)
    assert len(all_results) == 2


def test_touch(memory_store):
    now = time.time()
    
    item = MemoryItem(
        id="1",
        created=now - 100,
        last_used=now - 100,
        uses=1,
        kind="lesson",
        content="A lesson to touch.",
        context="Testing",
        tags=[],
        impact=0.5,
    )
    
    memory_store.write(item)
    
    # Touch it
    memory_store.touch(["1"])
    
    # Fetch it and check
    all_items = memory_store.all()
    updated_item = all_items[0]
    
    assert updated_item.uses == 2
    assert updated_item.last_used > now - 50
