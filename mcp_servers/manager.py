"""MCP server lifecycle management and tool execution."""

import asyncio
import os
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent


class MCPManager:
    """Manage multiple MCP server connections and tool execution."""

    def __init__(self) -> None:
        self.sessions: dict[str, tuple[ClientSession, Any, Any]] = {}
        self._active = False

    async def start(self, server_configs: list[dict[str, Any]]) -> dict[str, list[str]]:
        """Start selected MCP servers and discover their tools.

        Args:
            server_configs: Selected server configs from config.yaml.

        Returns:
            Dict mapping server_name -> [tool_name, ...].
        """
        tools_catalog: dict[str, list[str]] = {}

        for config in server_configs:
            name = config["name"]
            params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env={**os.environ, **(config.get("env") or {})},
            )

            read, write = await stdio_client(params).__aenter__()
            session = ClientSession(read, write)
            await session.initialize()

            tools_result = await session.list_tools()
            all_tools = [t.name for t in tools_result.tools]

            # Apply tool filtering if specified
            enabled = config.get("enabled_tools", [])
            if enabled:
                all_tools = [t for t in all_tools if t in enabled]

            self.sessions[name] = (session, read, write)
            tools_catalog[name] = all_tools

        self._active = True
        return tools_catalog

    async def call_tool(self, server_name: str, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool on a connected MCP server."""
        if server_name not in self.sessions:
            return f"Error: MCP server '{server_name}' is not connected."

        session, _, _ = self.sessions[server_name]
        result = await session.call_tool(tool_name, args)

        parts = []
        for content in result.content:
            if isinstance(content, TextContent):
                parts.append(content.text)
            elif hasattr(content, "structuredContent") and content.structuredContent:
                import json
                parts.append(json.dumps(content.structuredContent))

        if result.isError and not parts:
            return f"Error: tool '{tool_name}' returned isError flag"

        return "\n".join(parts)

    async def shutdown(self) -> None:
        """Close all MCP server connections."""
        for _, (session, read, write) in self.sessions.items():
            try:
                await read.aclose()
                await write.aclose()
            except Exception:
                pass
        self.sessions.clear()
        self._active = False

    @property
    def available_servers(self) -> list[str]:
        """Names of connected MCP servers."""
        return list(self.sessions.keys())
