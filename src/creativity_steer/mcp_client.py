"""Synchronous client over the official MCP Python SDK for Creativity Steer."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import mcp.client.session
import mcp.client.stdio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    name: str
    server: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class ToolResult:
    is_error: bool
    text: str
    content: List[Dict[str, Any]]


class McpClient:
    """A sync-friendly wrapper around MCP's async Python SDK."""

    def __init__(self, config_path: str | None = None, tools_whitelist: str | None = None):
        self.config_path = config_path or os.getenv("CS_MCP_CONFIG", "./mcp.json")
        self.whitelist = (
            set(t.strip() for t in tools_whitelist.split(",") if t.strip())
            if tools_whitelist
            else None
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self.servers: Dict[str, dict] = self._load_config()
        # server_name -> (read_stream, write_stream, session)
        self._active_sessions: Dict[str, Any] = {}
        # Used to hold the ExitStack handling the stdio contexts
        self._exit_stack = contextlib.AsyncExitStack()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_sync(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=15.0)

    def _load_config(self) -> Dict[str, dict]:
        if not os.path.exists(self.config_path):
            logger.warning(f"MCP config not found at {self.config_path}")
            return {}
        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)
                return data.get("mcpServers", {})
        except Exception as e:
            logger.error(f"Failed to load MCP config {self.config_path}: {e}")
            return {}

    def connect(self) -> McpClient:
        """Synchronous connect."""
        self._run_sync(self._connect_all())
        return self

    def disconnect(self):
        """Synchronous disconnect."""
        self._run_sync(self._exit_stack.aclose())

    async def _connect_all(self):
        for name, config in self.servers.items():
            if config.get("transport", "stdio") != "stdio":
                logger.warning(f"Unsupported transport for server {name}: {config.get('transport')}")
                continue

            command = config.get("command")
            args = config.get("args", [])
            env = config.get("env", {})
            full_env = os.environ.copy()
            full_env.update(env)
            
            try:
                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=full_env,
                )
                
                # We need to maintain the context managers for the lifetime of the client
                stdio_ctx = mcp.client.stdio.stdio_client(server_params)
                read_stream, write_stream = await self._exit_stack.enter_async_context(stdio_ctx)
                
                session_ctx = ClientSession(read_stream, write_stream)
                session = await self._exit_stack.enter_async_context(session_ctx)
                
                await session.initialize()
                self._active_sessions[name] = session
                logger.info(f"Connected to MCP server: {name}")
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {name}: {e}")

    def list_tools(self) -> List[ToolSpec]:
        """List available tools across all connected servers."""
        return self._run_sync(self._list_tools_async())

    async def _list_tools_async(self) -> List[ToolSpec]:
        tools = []
        for server_name, session in self._active_sessions.items():
            try:
                result = await session.list_tools()
                for tool in result.tools:
                    fqn = f"{server_name}.{tool.name}"
                    if self.whitelist and fqn not in self.whitelist:
                        continue
                    
                    tools.append(
                        ToolSpec(
                            name=tool.name,
                            server=server_name,
                            description=tool.description or "",
                            input_schema=tool.inputSchema,
                        )
                    )
            except Exception as e:
                logger.error(f"Failed to list tools for {server_name}: {e}")
        return tools

    def call_tool(self, server: str, name: str, args: Dict[str, Any]) -> ToolResult:
        """Call a tool synchronously."""
        return self._run_sync(self._call_tool_async(server, name, args))

    async def _call_tool_async(self, server: str, name: str, args: Dict[str, Any]) -> ToolResult:
        session = self._active_sessions.get(server)
        if not session:
            return ToolResult(is_error=True, text=f"Server {server} not connected", content=[])
            
        fqn = f"{server}.{name}"
        if self.whitelist and fqn not in self.whitelist:
             return ToolResult(is_error=True, text=f"Tool {fqn} not in whitelist", content=[])

        try:
            result = await session.call_tool(name, arguments=args)
            
            # Basic text extraction from the result content
            text_parts = []
            content_list = []
            
            if hasattr(result, "content"):
                for item in result.content:
                    if hasattr(item, "text"):
                        text_parts.append(item.text)
                    if hasattr(item, "model_dump"):
                        content_list.append(item.model_dump())
                    elif isinstance(item, dict):
                        content_list.append(item)
            
            return ToolResult(
                is_error=getattr(result, "isError", False),
                text="\n".join(text_parts),
                content=content_list
            )
        except Exception as e:
            return ToolResult(is_error=True, text=str(e), content=[])


class MockMcpClient:
    """Mock client for offline testing."""
    
    def __init__(self, *args, **kwargs):
        self._tools = {
            "echo": lambda args: ToolResult(is_error=False, text=json.dumps(args), content=[{"type": "text", "text": json.dumps(args)}]),
            "search": lambda args: ToolResult(is_error=False, text=f"Results for {args.get('query')}", content=[{"type": "text", "text": f"Results for {args.get('query')}"}])
        }

    def connect(self) -> MockMcpClient:
        return self

    def disconnect(self):
        pass

    def list_tools(self) -> List[ToolSpec]:
        return [
            ToolSpec("echo", "mock", "Echo args", {"type": "object", "properties": {"msg": {"type": "string"}}}),
            ToolSpec("search", "mock", "Mock search", {"type": "object", "properties": {"query": {"type": "string"}}})
        ]

    def call_tool(self, server: str, name: str, args: Dict[str, Any]) -> ToolResult:
        if server != "mock":
            return ToolResult(is_error=True, text="Unknown server", content=[])
        if name not in self._tools:
            return ToolResult(is_error=True, text="Unknown tool", content=[])
        return self._tools[name](args)
