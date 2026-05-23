"""Integration test: verify the graph can be built and invoked with minimal data."""

import pytest

from agents.graph import build_graph
from agents.state import AgentState
from notepad import Notepad
from mcp_servers.manager import MCPManager


@pytest.fixture
def tmp_notepad(tmp_path):
    """Create a temporary notepad file."""
    p = tmp_path / "test_notepad.md"
    return Notepad(str(p))


@pytest.fixture
def tmp_mcp_manager():
    """Create an MCP manager with no active connections."""
    return MCPManager()


def test_graph_builds(tmp_notepad, tmp_mcp_manager):
    """Verify the graph can be built with mocked MCP manager."""
    graph = build_graph(tmp_mcp_manager, tmp_notepad)
    assert graph is not None


def test_initial_state_has_all_keys(tmp_notepad, tmp_mcp_manager):
    """Verify the initial state has all required keys."""
    graph = build_graph(tmp_mcp_manager, tmp_notepad)
    # The compiled graph should accept a valid initial state
    state: AgentState = {
        "messages": [],
        "project_idea": "Test project",
        "spec": "Test spec",
        "research_queue": [],
        "research_findings": "",
        "pending_research": [],
        "open_questions": [],
        "planning_complete": False,
        "final_report": "",
        "progress_messages": [],
        "mcp_tools": {},
        "selected_mcp_servers": [],
        "report_signed_off": False,
    }
    # The input schema wraps AgentState in a 'root' field.
    # Check that all AgentState keys are present in the state dict.
    input_schema = graph.get_input_schema()
    root_field = input_schema.model_fields.get("root")
    if root_field:
        expected_keys = set(root_field.annotation.__annotations__.keys())
    else:
        expected_keys = set(input_schema.model_fields.keys())
    assert expected_keys.issubset(state.keys()), f"Missing keys: {expected_keys - set(state.keys())}"
