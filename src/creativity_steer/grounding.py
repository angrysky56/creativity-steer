"""Grounding layer for context injection before generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from creativity_steer.mcp_client import McpClient
from creativity_steer.memory import MemoryItem, MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class GroundingContext:
    memory: list[MemoryItem]
    tool_results: list[dict]

    def block(self) -> str:
        """Format the retrieved context as a prompt block."""
        if not self.memory and not self.tool_results:
            return ""

        lines = [
            "\n[KNOWN CONTEXT] Use these facts/lessons to inform a RANGE of replies (do not collapse to a single answer):"
        ]

        if self.memory:
            lines.append("--- Past Discoveries ---")
            for item in self.memory:
                lines.append(f"- {item.content}")
                if item.alternatives:
                    lines.append(
                        f"  Alternatives to consider: {', '.join(item.alternatives)}"
                    )

        if self.tool_results:
            lines.append("--- Tool Data ---")
            for result in self.tool_results:
                if isinstance(result, dict) and "text" in result:
                    lines.append(f"- {result['text']}")
                else:
                    lines.append(f"- {str(result)}")

        lines.append("[END CONTEXT]\n")
        return "\n".join(lines)


class GroundingProvider(Protocol):
    def gather(self, query: str, history: list[dict]) -> GroundingContext: ...


class DefaultGrounding:
    """Default grounding implementation using memory and (optional) MCP tools."""

    def __init__(
        self,
        memory: MemoryStore,
        mcp_client: McpClient | None = None,
        retrieval_tools: list[str] | None = None,
    ):
        self.memory = memory
        self.mcp_client = mcp_client
        self.retrieval_tools = retrieval_tools or []

    def gather(self, query: str, history: list[dict]) -> GroundingContext:
        """Gather context from memory and tools."""
        # 1. Memory retrieval
        try:
            memories = self.memory.retrieve(query, k=3, include_dormant=False)
            if memories:
                self.memory.touch([m.id for m in memories])
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            memories = []

        # 2. Tool retrieval (if configured)
        tool_results = []
        if self.mcp_client and self.retrieval_tools:
            for tool_fqn in self.retrieval_tools:
                try:
                    server, tool = tool_fqn.split(".", 1)
                    # We might want to pass more nuanced arguments to the tool, but simple query for now
                    result = self.mcp_client.call_tool(server, tool, {"query": query})
                    if not result.is_error:
                        tool_results.append({"tool": tool_fqn, "text": result.text})
                except Exception as e:
                    logger.error(f"Tool retrieval failed for {tool_fqn}: {e}")

        return GroundingContext(memory=memories, tool_results=tool_results)
