"""Tests for MCPManager."""

import pytest
from src.mcp_servers.manager import MCPManager


@pytest.fixture
def manager():
    return MCPManager()


def test_available_servers_empty(manager):
    """Before starting, no servers should be available."""
    assert manager.available_servers == []


def test_call_tool_not_connected(manager):
    """Calling a tool on a non-connected server should return an error string."""
    import asyncio
    result = asyncio.run(manager.call_tool("nonexistent", "some_tool", {}))
    assert "not connected" in result


def test_start_no_servers(manager):
    """Starting with no servers should be a no-op."""
    import asyncio
    tools = asyncio.run(manager.start([]))
    assert tools == {}
