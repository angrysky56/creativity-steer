"""Tests for the grounding module."""

from creativity_steer.grounding import DefaultGrounding, GroundingContext
from creativity_steer.mcp_client import MockMcpClient
from creativity_steer.memory import LocalMemoryStore, MemoryItem
from creativity_steer.backends import MockBackend

def test_grounding_context_block():
    mem1 = MemoryItem(
        id="1",
        created=0,
        last_used=0,
        uses=0,
        kind="lesson",
        content="Apples fall down.",
        context="",
        tags=[],
        impact=1.0,
        alternatives=["Apples float in space if you're lucky"],
    )
    
    ctx = GroundingContext(
        memory=[mem1],
        tool_results=[{"tool": "mock.echo", "text": "echo result"}]
    )
    
    block = ctx.block()
    
    assert "[KNOWN CONTEXT]" in block
    assert "--- Past Discoveries ---" in block
    assert "Apples fall down." in block
    assert "Apples float in space if you're lucky" in block
    assert "--- Tool Data ---" in block
    assert "echo result" in block

def test_default_grounding(tmp_path):
    # Setup
    backend = MockBackend()
    memory = LocalMemoryStore(backend, str(tmp_path / "memory.jsonl"))
    mcp_client = MockMcpClient()
    
    # MockBackend embeddings are of size `embed_dim`, default is 16.
    # We must match the dimension for cosine similarity retrieval.
    mock_emb = backend.embed(["apples"])[0]

    # Add a memory
    memory.write(MemoryItem(
        id="1",
        created=0,
        last_used=0,
        uses=0,
        kind="lesson",
        content="Apples fall down.",
        context="",
        tags=[],
        impact=1.0,
        embedding=mock_emb # Use the correctly sized embedding
    ))
    
    grounding = DefaultGrounding(
        memory=memory,
        mcp_client=mcp_client,
        retrieval_tools=["mock.search"]
    )
    
    ctx = grounding.gather(query="apples", history=[])
    
    # Check memory retrieved
    assert len(ctx.memory) == 1
    assert ctx.memory[0].content == "Apples fall down."
    
    # Check tool retrieved
    assert len(ctx.tool_results) == 1
    assert "Results for apples" in ctx.tool_results[0]["text"]
