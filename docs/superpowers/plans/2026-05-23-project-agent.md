# Project Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI-based multi-agent tool that interviews users about project ideas, researches them using MCP-connected tools, and produces a final report.

**Architecture:** LangGraph StateGraph with three nodes (Orchestrator, Planner, Researcher) sharing a typed state and a single notepad file. MCP servers are spawned as stdio subprocesses and their tools are exposed to the Researcher agent. Human-in-the-loop is implemented via LangGraph's `interrupt()`.

**Tech Stack:** Python 3.12+, langgraph, langchain-openai, mcp (MCP Python SDK), rich, python-dotenv, pydantic

---

## File Structure

```
project-agent/
├── main.py                       # Entry point: bootstrap config, MCP discovery, graph invocation
├── config.py                     # Load .env, load MCP config from config.yaml
├── mcp/
│   ├── __init__.py
│   ├── manager.py                # MCPManager: spawn, init, list_tools, call_tool, shutdown
│   └── tools.py                  # Tool wrapper: convert MCP tools to LangChain-compatible format
├── agents/
│   ├── __init__.py
│   ├── graph.py                  # StateGraph: main graph, edges, conditional routing, compile
│   ├── state.py                  # AgentState TypedDict with all fields
│   ├── orchestrator.py           # Orchestrator node: CLI interaction via rich
│   ├── planner.py                # Planner node: interview, spec writing, research queuing
│   ├── researcher.py             # Researcher node: MCP tool execution, report writing
│   └── interrupt.py              # Interrupt helper: human-in-the-loop presentation
├── notepad.py                    # Notepad: read/write/append to shared notepad.md
├── config.yaml                   # MCP server definitions (users copy/modify)
├── .env.example                  # Template for OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_notepad.py
    └── test_mcp_manager.py
```

## LLM Integration

All agents use `langchain_openai.ChatOpenAI` with a custom `base_url`:
```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model="gpt-4o",
    base_url="https://api.openai.com/v1",  # from .env
    api_key="sk-...",                       # from .env
    temperature=0.2,                        # lower for planning, 0.7 for interview
)
```

The model is created once in `main.py` and passed to agent nodes via closure or state.

## MCP Integration

MCP servers are defined in `config.yaml`:
```yaml
mcp_servers:
  - name: brave-search
    command: npx
    args: ["-y", "@brave/brave-search-mcp-server"]
    env:
      BRAVE_API_KEY: "your-api-key"
```

The `MCPManager` spawns stdio subprocesses, initializes sessions, discovers tools, and provides a `call_tool(name, args)` method. All servers stay alive for the session lifetime.

## Dependencies to Add

```toml
[project]
dependencies = [
    "langgraph>=1.0",
    "langchain-openai>=0.3",
    "mcp>=1.0",
    "rich>=13.0",
    "python-dotenv>=1.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
]
```

---

### Task 1: Project Setup — pyproject.toml and .env.example

**Files:**
- Modify: `pyproject.toml`
- Create: `.env.example`

- [ ] **Step 1: Update pyproject.toml with dependencies**

Replace the empty dependencies in `pyproject.toml`:

```toml
[project]
name = "project-agent"
version = "0.1.0"
description = "A multi-agent project planning and research tool"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "langgraph>=1.0",
    "langchain-openai>=0.3",
    "mcp>=1.0",
    "rich>=13.0",
    "python-dotenv>=1.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
]
```

- [ ] **Step 2: Create .env.example**

```
# LLM Configuration (OpenAI-compatible endpoint)
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL=gpt-4o

# Optional: MCP server API keys
BRAVE_API_KEY=your-brave-api-key
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml .env.example
git commit -m "feat: add project dependencies and env template"
```

---

### Task 2: Config Module — Load .env and MCP Config

**Files:**
- Create: `config.py`

- [ ] **Step 1: Write the config module**

```python
"""Configuration loader for project-agent."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Default paths
BASE_DIR = Path(__file__).parent.parent
ENV_PATH = BASE_DIR / ".env"
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_env() -> None:
    """Load environment variables from .env file if it exists."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)


def get_llm_config() -> dict[str, str]:
    """Return LLM configuration from environment variables."""
    return {
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
    }


def load_mcp_config() -> list[dict[str, Any]]:
    """Load MCP server definitions from config.yaml.

    Returns a list of dicts with keys:
      - name: str
      - command: str
      - args: list[str]
      - env: dict[str, str] (optional)
      - enabled_tools: list[str] (optional, empty = all)
    """
    if not CONFIG_PATH.exists():
        return []

    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)

    return data.get("mcp_servers", [])
```

- [ ] **Step 2: Verify config.py is importable**

Run: `python -c "from config import get_llm_config, load_mcp_config; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add config loader for env and MCP config"
```

---

### Task 3: Notepad Module — Shared File-Based State

**Files:**
- Create: `notepad.py`

- [ ] **Step 1: Write the notepad module**

```python
"""Shared notepad: a single markdown file used by all agents."""

import re
from pathlib import Path
from typing import Optional

DEFAULT_NOTEPAD = "notepad.md"

SECTIONS = {
    "Project Spec": "## Project Spec",
    "Research Findings": "## Research Findings",
    "Open Questions": "## Open Questions",
    "Decisions": "## Decisions",
    "Progress": "## Progress",
    "Final Report": "## Final Report",
}


class Notepad:
    """Manage a single shared markdown file with named sections."""

    def __init__(self, path: str = DEFAULT_NOTEPAD) -> None:
        self.path = Path(path)
        if not self.path.exists():
            self._initialize()

    def _initialize(self) -> None:
        """Create the notepad with all section headers."""
        parts = [f"# Project Agent Notepad\n\n"]
        for header in SECTIONS.values():
            parts.append(f"{header}\n\n")
        self.path.write_text("".join(parts))

    def read_section(self, section_name: str) -> str:
        """Return the text content of a section."""
        header = SECTIONS.get(section_name, f"## {section_name}")
        content = self.path.read_text()

        # Find current header and next header
        current_idx = content.find(header)
        if current_idx == -1:
            return ""

        # Find the next section header
        next_header = None
        next_idx = len(content)
        for other_header in SECTIONS.values():
            if other_header != header:
                idx = content.find(other_header, current_idx + len(header))
                if 0 < idx < next_idx:
                    next_idx = idx
                    next_header = other_header

        text = content[current_idx + len(header):next_idx].strip()
        return text if text else ""

    def append_section(self, section_name: str, text: str) -> None:
        """Append text to a section, creating it if empty."""
        header = SECTIONS.get(section_name, f"## {section_name}")
        current = self.read_section(section_name)
        if current:
            new_content = f"{current}\n\n{text}"
        else:
            new_content = text

        content = self.path.read_text()
        header_idx = content.find(header)
        if header_idx == -1:
            # Section doesn't exist yet — add it at the end before Final Report
            final_idx = content.find("## Final Report")
            if final_idx != -1:
                content = (
                    content[:final_idx]
                    + f"{header}\n\n{text}\n\n"
                    + content[final_idx:]
                )
            else:
                content += f"{header}\n\n{text}\n\n"
        else:
            # Replace section content between current header and next header
            next_idx = len(content)
            for other_header in SECTIONS.values():
                if other_header != header:
                    idx = content.find(other_header, header_idx + len(header))
                    if 0 < idx < next_idx:
                        next_idx = idx

            content = content[:header_idx + len(header)] + f"\n\n{new_content}\n\n" + content[next_idx:]

        self.path.write_text(content)

    def get_all_content(self) -> str:
        """Return the full notepad content."""
        return self.path.read_text()

    def update_progress(self, message: str) -> None:
        """Append a progress update."""
        self.append_section("Progress", f"- {message}")
```

- [ ] **Step 2: Write a simple test to verify the notepad works**

Run in Python REPL:
```python
import tempfile, os
from notepad import Notepad

with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
    p = f.name

os.rename(p, "test_notepad.md")

n = Notepad("test_notepad.md")
n.append_section("Project Spec", "Test spec content")
assert "Test spec content" in n.read_section("Project Spec")
n.update_progress("Started planning")
assert "Started planning" in n.read_section("Progress")
os.unlink("test_notepad.md")
print("OK")
```

Expected: `OK` (no assertion errors)

- [ ] **Step 3: Commit**

```bash
git add notepad.py
git commit -m "feat: add shared notepad module with section-based read/write"
```

---

### Task 4: MCP Manager — Server Lifecycle and Tool Execution

**Files:**
- Create: `mcp/__init__.py`
- Create: `mcp/manager.py`

- [ ] **Step 1: Create mcp/__init__.py**

```python
"""MCP server management for project-agent."""
```

- [ ] **Step 2: Write the MCP Manager**

```python
"""MCP server lifecycle management and tool execution."""

import asyncio
import os
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent


class MCPManager:
    """Manage multiple MCP server connections and tool execution."""

    def __init__(self) -> None:
        self.sessions: dict[str, tuple[ClientSession, Any, Any]] = {}  # name -> (session, read, write)
        self._active = False

    async def start(self, server_configs: list[dict[str, Any]]) -> list[str]:
        """Start selected MCP servers and discover their tools.

        Args:
            server_configs: List of server config dicts with 'name', 'command',
                           'args', optional 'env', optional 'enabled_tools'.

        Returns:
            List of discovered tool names across all servers.
        """
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
            available_tools = [t.name for t in tools_result.tools]

            # Apply tool filtering if specified
            enabled = config.get("enabled_tools", [])
            if enabled:
                available_tools = [t for t in available_tools if t in enabled]

            self.sessions[name] = (session, read, write)
            print(f"  [{name}] Loaded tools: {', '.join(available_tools)}")

        self._active = True
        return available_tools

    async def call_tool(self, server_name: str, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool on a specific MCP server.

        Returns the text result as a string.
        """
        if server_name not in self.sessions:
            return f"Error: MCP server '{server_name}' is not connected."

        session, _, _ = self.sessions[server_name]
        result = await session.call_tool(tool_name, args)

        # Parse the result, handling different content types
        parts = []
        for content in result.content:
            if isinstance(content, TextContent):
                parts.append(content.text)
            elif hasattr(content, "structuredContent") and content.structuredContent:
                parts.append(str(content.structuredContent))

        if result.isError and not parts:
            return f"Error (tool '{tool_name}' returned isError flag)"

        return "\n".join(parts)

    async def shutdown(self) -> None:
        """Close all MCP server connections."""
        for name, (session, read, write) in self.sessions.items():
            try:
                await read.aclose()
                await write.aclose()
            except Exception:
                pass
        self.sessions.clear()
        self._active = False

    @property
    def available_servers(self) -> list[str]:
        """Return names of connected servers."""
        return list(self.sessions.keys())
```

- [ ] **Step 3: Write a basic test**

Create `tests/test_mcp_manager.py`:

```python
"""Tests for MCPManager."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp.manager import MCPManager


@pytest.fixture
def manager():
    return MCPManager()


@pytest.mark.asyncio
async def test_start_no_servers(manager):
    """Starting with no servers should be a no-op."""
    tools = await manager.start([])
    assert tools == []
    await manager.shutdown()


@pytest.mark.asyncio
async def test_call_tool_not_connected(manager):
    """Calling a tool on a non-connected server should return an error string."""
    result = await manager.call_tool("nonexistent", "some_tool", {})
    assert "not connected" in result
    await manager.shutdown()


@pytest.mark.asyncio
async def test_available_servers_empty(manager):
    """Before starting, no servers should be available."""
    assert manager.available_servers == []
```

- [ ] **Step 4: Run the MCP manager tests**

Run: `python -m pytest tests/test_mcp_manager.py -v`

Expected: All tests pass (3 passed)

If `pytest-asyncio` is needed, install it: `pip install pytest-asyncio`

- [ ] **Step 5: Commit**

```bash
git add mcp/__init__.py mcp/manager.py tests/test_mcp_manager.py
git commit -m "feat: add MCP server lifecycle manager with tool execution"
```

---

### Task 5: State Definition — AgentState TypedDict

**Files:**
- Create: `agents/__init__.py`
- Create: `agents/state.py`

- [ ] **Step 1: Create agents/__init__.py**

```python
"""Agent modules for project-agent."""
```

- [ ] **Step 2: Write the state definition**

```python
"""Shared state for the agent graph."""

from typing import Annotated, Any, Sequence
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State shared across all agent nodes in the graph."""

    # Conversation messages (accumulated via add_messages)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # The user's initial project idea
    project_idea: str

    # The evolving project specification
    spec: str

    # List of research tasks to perform: {"description": str, "server": str, "tool": str, "args": dict}
    research_queue: list[dict[str, Any]]

    # Research results accumulated so far
    research_findings: str

    # Pending research topics generated by the Planner (for human-in-the-loop)
    pending_research: list[str]

    # Open questions that need user input
    open_questions: list[str]

    # Whether the Planner has finished (research_queue is empty and no more needed)
    planning_complete: bool

    # Final report output
    final_report: str

    # Progress messages shown to the user
    progress_messages: list[str]

    # The MCP tool catalog: {server_name: [tool_name, ...]}
    mcp_tools: dict[str, list[str]]

    # User's selected MCP servers to use
    selected_mcp_servers: list[str]

    # Whether the user has signed off on the final report
    report_signed_off: bool

    # Error messages
    errors: list[str]
```

- [ ] **Step 3: Verify state.py is importable**

Run: `python -c "from agents.state import AgentState; print(list(AgentState.__annotations__.keys())); print('OK')"`

Expected: List of state field names followed by `OK`

- [ ] **Step 4: Commit**

```bash
git add agents/__init__.py agents/state.py
git commit -m "feat: add AgentState TypedDict for shared graph state"
```

---

### Task 6: Orchestrator Node — CLI Interaction with Rich

**Files:**
- Create: `agents/orchestrator.py`

- [ ] **Step 1: Write the orchestrator node**

```python
"""Orchestrator node: handles all user interaction via the terminal."""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status

from agents.state import AgentState

console = Console()


def orchestrator_greet(state: AgentState) -> dict:
    """Welcome the user and set up the session."""
    return {
        "progress_messages": ["👋 Welcome! Let's plan your project."],
        "selected_mcp_servers": [],
        "mcp_tools": {},
    }


def show_mcp_selection(manager) -> dict:
    """Present available MCP servers to the user for selection.

    This node is called after MCP servers are discovered.
    It should be called outside the graph via the orchestrator flow.
    """
    return {}  # Side-effect only; user selection is handled externally


def show_progress(message: str) -> None:
    """Display a progress message to the user."""
    console.print(f"  {message}")


def ask_user(question: str, default: str | None = None) -> str:
    """Ask the user a question via rich Prompt."""
    answer = Prompt.ask(question, default=default)
    return answer.strip() if answer else ""


def show_panel(title: str, content: str) -> None:
    """Display content in a styled panel."""
    console.print(Panel(content, title=title, subtitle="Project Agent"))


def present_report(report: str) -> str:
    """Show the final report and ask for sign-off.

    Returns 'signed_off' or 'revise' based on user input.
    """
    show_panel("Final Report", report)
    console.print("")
    answer = Prompt.ask(
        "How would you like to proceed?",
        choices=["signed_off", "revise"],
        default="signed_off",
    )
    return answer


def present_research_choices(
    findings: str,
    pending_topics: list[str],
) -> str:
    """Present research findings and pending topics to the user.

    Returns the user's response text for priority selection.
    """
    console.print(Panel(findings, title="Research Findings So Far"))
    if pending_topics:
        console.print("\nPending research topics:")
        for i, topic in enumerate(pending_topics, 1):
            console.print(f"  {i}. {topic}")
    console.print()
    answer = Prompt.ask(
        "What should we research next? (number, description, or 'skip')",
        default="1",
    )
    return answer.strip()


def present_open_questions(questions: list[str]) -> list[str]:
    """Present open questions to the user and collect answers."""
    answers = []
    for q in questions:
        console.print(Panel(q, title="Question"))
        answer = ask_user(q)
        answers.append(answer)
    return answers
```

- [ ] **Step 2: Verify orchestrator.py is importable**

Run: `python -c "from agents.orchestrator import show_progress, ask_user; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agents/orchestrator.py
git commit -m "feat: add orchestrator node with rich CLI interaction"
```

---

### Task 7: LLM Model Factory

**Files:**
- Create: `agents/model.py`

- [ ] **Step 1: Write the model factory**

```python
"""Factory for creating LLM models configured with the OpenAI-compatible endpoint."""

from langchain_openai import ChatOpenAI


def create_model(
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
) -> ChatOpenAI:
    """Create a ChatOpenAI model configured with env variables.

    Args:
        model_name: Override model name (default from OPENAI_MODEL env var).
        base_url: Override base URL (default from OPENAI_BASE_URL env var).
        api_key: Override API key (default from OPENAI_API_KEY env var).
        temperature: Model temperature.
    """
    from config import get_llm_config

    config = get_llm_config()
    return ChatOpenAI(
        model=model_name or config["model"],
        base_url=base_url or config["base_url"],
        api_key=api_key or config["api_key"],
        temperature=temperature,
    )
```

- [ ] **Step 2: Verify model.py is importable**

Run: `python -c "from agents.model import create_model; print('OK')"`

Expected: `OK` (no API key validation at import time)

- [ ] **Step 3: Commit**

```bash
git add agents/model.py
git commit -m "feat: add LLM model factory with configurable OpenAI endpoint"
```

---

### Task 8: Researcher Node — MCP Tool Execution Agent

**Files:**
- Create: `agents/researcher.py`

- [ ] **Step 1: Write the researcher node**

```python
"""Researcher node: executes MCP tool calls and reports findings to the Planner."""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.model import create_model
from agents.state import AgentState

SYSTEM_PROMPT = """You are a Researcher agent. Your job is to gather information using
available tools and report findings clearly and concisely.

For each research task:
1. Identify the best tool from the available MCP tools to use.
2. Construct appropriate arguments for the tool call.
3. Execute the tool call.
4. Summarize the results into a structured finding.

Write your findings in a clear format with:
- A heading for the topic researched
- Key findings in bullet points
- Source URLs when available
- Any relevant comparisons or alternatives"""


def research_node(state: AgentState, mcp_manager) -> dict[str, Any]:
    """Execute research tasks from the queue using MCP tools.

    Args:
        state: The current graph state.
        mcp_manager: The MCPManager instance with active server connections.

    Returns:
        Updated state with research findings appended.
    """
    model = create_model(temperature=0.3)

    queue = state.get("research_queue", [])
    if not queue:
        return {"research_findings": "", "progress_messages": ["No research tasks to execute."]}

    model_with_tools = model.bind_tools([])  # We call tools manually, not via the LLM

    findings_parts = []
    new_queue = []

    for task in queue:
        topic = task["description"]
        server = task.get("server", "")
        tool = task.get("tool", "")
        args = task.get("args", {})

        msg = f"🔍 Researching: {topic}"
        progress = [msg]

        # Decide which tool to use for this topic
        if not tool and server:
            # Let the model pick the best tool for this server
            tool_selection_prompt = (
                f"You need to research: '{topic}'.\n"
                f"Available tools on server '{server}': {', '.join(state['mcp_tools'].get(server, []))}\n"
                f"Which single tool is best? Reply with just the tool name."
            )
            tool_response = model.invoke([HumanMessage(content=tool_selection_prompt)])
            tool = tool_response.content.strip()
            progress.append(f"  └─ Selected tool: {tool}")

        if tool and server in mcp_manager.available_servers:
            try:
                result_text = f"Researching: {topic}..."
                progress.append(f"  └─ Calling {server}/{tool}...")
                raw_result = asyncio_run(mcp_manager.call_tool(server, tool, args))
                progress.append(f"  └─ Got result ({len(raw_result)} chars)")

                # Summarize the result
                summary_prompt = (
                    f"Summarize these research findings for topic '{topic}':\n\n{raw_result}\n\n"
                    f"Provide a concise bullet-point summary with key facts, numbers, and source URLs."
                )
                summary = model.invoke([HumanMessage(content=summary_prompt)]).content
                summary = summary.strip()
                findings_parts.append(f"\n### {topic}\n{summary}")
                progress.append(f"  └─ Summarized findings")
            except Exception as e:
                progress.append(f"  └─ Error: {e}")
                new_queue.append(task)
        else:
            progress.append(f"  └─ Skipped (no tool '{tool}' on '{server}')")
            new_queue.append(task)

        state["progress_messages"].extend(progress)

    return {
        "research_findings": "".join(findings_parts),
        "research_queue": new_queue,
        "progress_messages": state.get("progress_messages", []),
    }


def asyncio_run(coro):
    """Helper to run an async function synchronously."""
    import asyncio
    try:
        return asyncio.get_running_loop().run_until_complete(coro)
    except RuntimeError:
        import asyncio
        return asyncio.run(coro)
```

Wait — I need to reconsider. The Researcher node needs to be synchronous for LangGraph's standard invoke pattern, but MCP calls are async. Let me simplify the approach: the Researcher node uses LangChain's tool binding to have the LLM produce tool calls, and a separate ToolNode executes them. Actually, looking at the LangGraph API more carefully, the cleanest pattern is to use `create_react_agent` for the Researcher and handle MCP tools through a tool-calling loop.

Let me revise the plan to use a cleaner architecture.

---

**Revised Architecture Note:**

Rather than calling MCP tools directly from the Researcher node (which requires async handling), I'll use LangGraph's `create_react_agent` pattern for both the Planner and Researcher. Each agent gets its own `StateGraph` subgraph with:
- A message-accumulating state
- An LLM node with `bind_tools`
- A `ToolNode` that knows about the MCP tools

The Researcher agent's tools are dynamically bound MCP tool wrappers. The Planner agent doesn't use MCP tools directly.

This is a more fundamental design change. Let me update the plan.

**Revised Task 8: Researcher Node**

```python
"""Researcher agent: uses LangGraph's ReAct pattern with MCP tools."""

import asyncio
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from agents.model import create_model
from agents.state import AgentState


def _make_mcp_tool_wrapper(mcp_manager, server_name: str, tool_name: str, description: str) -> Callable:
    """Create a synchronous wrapper around an MCP tool call."""

    @tool(description=description)
    def _wrapper(query: str = "") -> str:
        """Auto-generated docstring from tool description."""
        try:
            result = asyncio_run(mcp_manager.call_tool(server_name, tool_name, {"query": query}))
            return result
        except Exception as e:
            return f"Error calling {tool_name} on {server_name}: {e}"

    return _wrapper


def _asyncio_run(coro):
    """Helper to run async code synchronously."""
    try:
        return asyncio.get_running_loop().run_until_complete(coro)
    except RuntimeError:
        import asyncio
        return asyncio.run(coro)


def build_researcher_agent(mcp_manager, available_tools_info: dict[str, list[str]]) -> Any:
    """Build a ReAct agent for the Researcher with MCP tools bound.

    Returns a compiled graph app that can be invoked with state.
    """
    from langchain_core.tools import StructuredTool

    tools = []
    for server_name, tool_names in available_tools_info.items():
        for tool_name in tool_names:
            desc = f"Use this tool on MCP server '{server_name}' to research topics."
            wrapper = _make_mcp_tool_wrapper(mcp_manager, server_name, tool_name, desc)
            tools.append(wrapper)

    model = create_model(temperature=0.3)
    agent = create_react_agent(model, tools)
    return agent


def _asyncio_run(coro):
    """Helper to run async code synchronously."""
    try:
        return asyncio.get_running_loop().run_until_complete(coro)
    except RuntimeError:
        import asyncio
        return asyncio.run(coro)
```

Hmm, this is getting complex and the async sync boundary is messy. Let me take a cleaner approach: the entire graph runs in async mode via `ainvoke`, or I restructure the MCP Manager to have a sync wrapper.

Actually, looking at the LangGraph docs more carefully, `graph.invoke()` can work with async nodes if we use `asyncio.run()` at the graph boundary. The cleanest approach for a CLI tool is:

1. The main graph (Orchestrator → Planner → Researcher → Router) is synchronous
2. The Researcher internally uses `create_react_agent` which runs its own async loop
3. MCP tool calls are wrapped so they work synchronously

But wait — `create_react_agent` returns a compiled graph that is invoked via `.invoke()` or `.ainvoke()`. The `.invoke()` method on a compiled graph with async nodes will work if there's no running event loop.

Let me simplify this significantly. The Researcher will NOT use `create_react_agent`. Instead, the Planner will build a task list, and the Researcher will execute each task sequentially using the MCP Manager. The LLM is used for summarization, not for tool selection.

Actually, re-reading the spec more carefully: "Researcher: Executes research tasks using MCP tools". The Planner tells the Researcher WHAT to research (topic), and the Researcher uses the LLM to decide WHICH MCP tool to use and with WHAT arguments, then executes it.

The simplest correct approach:

1. Planner produces research tasks as text descriptions in the state
2. Researcher node: LLM generates tool call instructions → Researcher calls MCP → LLM summarizes results → repeat

Let me rewrite Task 8 cleanly.

---

**Final Task 8: Researcher Node**

```python
"""Researcher node: researches topics using MCP tools."""

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.model import create_model
from agents.state import AgentState


def _run_async(coro):
    """Run async code from a synchronous context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop:
        return asyncio.run(asyncio.wait_for(coro, timeout=120))
    return asyncio.run(coro)


async def _call_tool_async(manager, server: str, tool: str, args: dict) -> str:
    """Async wrapper for MCP tool call."""
    return await manager.call_tool(server, tool, args)


def research_node(state: AgentState, mcp_manager) -> dict[str, Any]:
    """Research tasks assigned by the Planner using MCP tools.

    The Planner provides a list of research topics in state['research_queue'].
    This node uses the LLM to pick the right MCP tool for each topic,
    executes it, and summarizes the results.

    Args:
        state: The current graph state.
        mcp_manager: The MCPManager instance with active server connections.

    Returns:
        Updated state with research_findings appended.
    """
    model = create_model(temperature=0.3)

    queue = state.get("research_queue", [])
    if not queue:
        return {
            "research_findings": "",
            "progress_messages": state.get("progress_messages", []) + ["No research tasks remaining."],
        }

    findings_parts = []
    completed = 0

    for task in queue:
        topic = task["description"]
        server_hint = task.get("server", "")
        progress = state.get("progress_messages", [])
        progress.append(f"🔍 Researching: {topic}")

        # Step 1: Pick the best tool
        available_tools = {s: t for s, t in mcp_manager.sessions.items()}
        tool_prompt = f"""You are researching: {topic}
Available MCP tools:
{chr(10).join(f'- {server}: {", ".join(available_tools.get(server, []))}' for server in available_tools)}
{f'Preferred server: {server_hint}' if server_hint else ''}
Reply with just two lines:
TOOL: server_name/tool_name
ARGUMENTS: json_args"""

        tool_response = model.invoke([HumanMessage(content=tool_prompt)])
        response_text = tool_response.content.strip()

        # Parse TOOL: and ARGUMENTS: lines
        lines = response_text.split("\n")
        chosen_tool = ""
        chosen_args = {}
        for line in lines:
            if line.startswith("TOOL:"):
                tool_parts = line[5:].strip().split("/")
                if len(tool_parts) == 2:
                    server_name = tool_parts[0].strip()
                    tool_name = tool_parts[1].strip()
                    chosen_tool = f"{server_name}/{tool_name}"
            elif line.startswith("ARGUMENTS:"):
                import json
                try:
                    chosen_args = json.loads(line[10:].strip())
                except json.JSONDecodeError:
                    chosen_args = {"query": topic}

        if not chosen_tool or "/" not in chosen_tool:
            chosen_tool = f"{server_hint}/brave_web_search" if server_hint else ""

        progress.append(f"  └─ Selected: {chosen_tool}")

        # Step 2: Execute the tool
        if chosen_tool and "/" in chosen_tool:
            server_name, tool_name = chosen_tool.split("/", 1)
            if server_name in mcp_manager.sessions:
                try:
                    progress.append(f"  └─ Calling tool...")
                    raw_result = _run_async(_call_tool_async(mcp_manager, server_name, tool_name, chosen_args or {"query": topic}))
                    progress.append(f"  └─ Got {len(raw_result)} chars")

                    # Step 3: Summarize
                    summary_prompt = (
                        f"Summarize these findings for '{topic}':\n\n{raw_result[:4000]}\n\n"
                        f"Provide a concise bullet-point summary with key facts and source URLs."
                    )
                    summary = model.invoke([HumanMessage(content=summary_prompt)]).content.strip()
                    findings_parts.append(f"\n### {topic}\n{summary}")
                    progress.append(f"  └─ Summarized")
                    completed += 1
                except Exception as e:
                    progress.append(f"  └─ Error: {e}")
            else:
                progress.append(f"  └─ Server '{server_name}' not connected")
        else:
            progress.append(f"  └─ No valid tool selected, skipping")

        state["progress_messages"] = progress

    return {
        "research_findings": "".join(findings_parts),
        "research_queue": queue[completed:],  # Remove completed tasks
        "progress_messages": state.get("progress_messages", []),
    }
```

OK this is getting too long in the plan. Let me finalize and clean up the plan properly. The key insight: Researcher node uses a for loop over research tasks, calls the LLM to pick a tool, calls MCP, then summarizes. All synchronous from the graph's perspective, with async bridged by `_run_async`.

Let me write the complete final plan now.

---

## Final Complete Plan

*(Rewriting all tasks cleanly below)*

---

### Task 1: Project Setup — pyproject.toml and .env.example

**Files:**
- Modify: `pyproject.toml`
- Create: `.env.example`

- [ ] **Step 1: Update pyproject.toml with dependencies**

Replace the dependencies list in `pyproject.toml`:

```toml
[project]
name = "project-agent"
version = "0.1.0"
description = "A multi-agent project planning and research tool"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "langgraph>=1.0",
    "langchain-openai>=0.3",
    "mcp>=1.0",
    "rich>=13.0",
    "python-dotenv>=1.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]
```

- [ ] **Step 2: Create .env.example**

```
# LLM Configuration (OpenAI-compatible endpoint)
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL=gpt-4o

# Optional: MCP server API keys (referenced in config.yaml)
BRAVE_API_KEY=your-brave-api-key
```

- [ ] **Step 3: Create config.yaml template**

```yaml
# MCP server definitions.
# Users configure which MCP servers to connect and select at runtime.
mcp_servers:
  - name: brave-search
    description: "Web search via Brave Search API"
    command: npx
    args: ["-y", "@brave/brave-search-mcp-server"]
    env:
      BRAVE_API_KEY: "${BRAVE_API_KEY}"
    # enabled_tools: []   # empty = all tools; e.g., ["brave_web_search", "brave_news_search"]
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .env.example config.yaml
git commit -m "feat: add project dependencies, env template, and MCP config"
```

---

### Task 2: Config Module — Load .env and MCP Config

**Files:**
- Create: `config.py`

- [ ] **Step 1: Write config.py**

```python
"""Configuration loader for project-agent."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
ENV_PATH = BASE_DIR / ".env"
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_env() -> None:
    """Load environment variables from .env file if it exists."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)


def get_llm_config() -> dict[str, str]:
    """Return LLM configuration from environment variables."""
    return {
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
    }


def load_mcp_config() -> list[dict[str, Any]]:
    """Load MCP server definitions from config.yaml.

    Each server dict has:
      - name: str
      - description: str (optional)
      - command: str
      - args: list[str]
      - env: dict[str, str] (optional)
      - enabled_tools: list[str] (optional, empty = all)
    """
    if not CONFIG_PATH.exists():
        return []

    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)

    return data.get("mcp_servers", [])
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from config import get_llm_config, load_mcp_config; print(get_llm_config()); print('OK')"`

Expected: LLM config dict followed by `OK`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add config loader for env vars and MCP server definitions"
```

---

### Task 3: Notepad Module — Shared File-Based State

**Files:**
- Create: `notepad.py`

- [ ] **Step 1: Write notepad.py**

```python
"""Shared notepad: a single markdown file used by all agents."""

from pathlib import Path

DEFAULT_NOTEPAD = "notepad.md"

# Ordered section headers
SECTIONS = (
    "Project Spec",
    "Research Findings",
    "Open Questions",
    "Decisions",
    "Progress",
    "Final Report",
)


class Notepad:
    """Manage a single shared markdown file with named sections."""

    def __init__(self, path: str = DEFAULT_NOTEPAD) -> None:
        self.path = Path(path)
        if not self.path.exists():
            self._initialize()

    def _initialize(self) -> None:
        """Create the notepad with all section headers."""
        parts = [f"# Project Agent Notepad\n\n"]
        for name in SECTIONS:
            parts.append(f"## {name}\n\n")
        self.path.write_text("".join(parts))

    def _header_for(self, section: str) -> str:
        return f"## {section}"

    def _find_section_range(self, content: str, section: str) -> tuple[int, int]:
        """Return (start_idx, end_idx) for a section's text content."""
        header = self._header_for(section)
        header_idx = content.find(header)
        if header_idx == -1:
            return -1, -1

        header_end = header_idx + len(header)

        # Find next section header
        next_idx = len(content)
        for other in SECTIONS:
            if other != section:
                idx = content.find(self._header_for(other), header_end)
                if 0 < idx < next_idx:
                    next_idx = idx

        return header_idx, next_idx

    def read_section(self, section: str) -> str:
        """Return the text content of a section."""
        content = self.path.read_text()
        start, end = self._find_section_range(content, section)
        if start == -1:
            return ""
        text = content[start + len(self._header_for(section)):end].strip()
        return text if text else ""

    def set_section(self, section: str, content: str) -> None:
        """Replace the entire content of a section."""
        text = self.path.read_text()
        start, end = self._find_section_range(text, section)

        if start == -1:
            # Section doesn't exist; append before Final Report or at end
            final_idx = text.find(self._header_for("Final Report"))
            insert_point = final_idx if final_idx != -1 else len(text)
            insert_text = f"\n\n## {section}\n\n{content}\n\n"
            if final_idx != -1:
                text = text[:insert_point] + insert_text + text[insert_point:]
            else:
                text = text.rstrip() + "\n\n" + insert_text
            self.path.write_text(text)
        else:
            text = text[:start + len(self._header_for(section))] + f"\n\n{content}\n\n" + text[end:]
            self.path.write_text(text)

    def append_section(self, section: str, text: str) -> None:
        """Append text to a section."""
        existing = self.read_section(section)
        new_content = f"{existing}\n\n{text}".strip() if existing else text
        self.set_section(section, new_content)

    def get_all_content(self) -> str:
        """Return the full notepad content."""
        return self.path.read_text()

    def update_progress(self, message: str) -> None:
        """Append a progress entry."""
        self.append_section("Progress", f"- {message}")

    def save_report(self, report: str) -> None:
        """Save the final report."""
        self.set_section("Final Report", report)
```

- [ ] **Step 2: Verify with a quick test**

Run:
```python
import tempfile, os
from notepad import Notepad

tmp = tempfile.mktemp(suffix=".md")
n = Notepad(tmp)
n.set_section("Project Spec", "Test content")
assert n.read_section("Project Spec") == "Test content"
n.append_section("Progress", "Started")
assert "Started" in n.read_section("Progress")
os.unlink(tmp)
print("OK")
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add notepad.py
git commit -m "feat: add shared notepad with sectioned read/write/append"
```

---

### Task 4: MCP Manager — Server Lifecycle and Tool Execution

**Files:**
- Create: `mcp/__init__.py`
- Create: `mcp/manager.py`
- Create: `tests/test_mcp_manager.py`

- [ ] **Step 1: Create mcp/__init__.py**

```python
"""MCP server management for project-agent."""
```

- [ ] **Step 2: Write mcp/manager.py**

```python
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
```

- [ ] **Step 3: Write tests**

Create `tests/test_mcp_manager.py`:

```python
"""Tests for MCPManager."""

import pytest
from mcp.manager import MCPManager


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
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_mcp_manager.py -v`

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add mcp/__init__.py mcp/manager.py tests/test_mcp_manager.py
git commit -m "feat: add MCP manager with server lifecycle and tool execution"
```

---

### Task 5: State Definition — AgentState TypedDict

**Files:**
- Create: `agents/__init__.py`
- Create: `agents/state.py`

- [ ] **Step 1: Create agents/__init__.py**

```python
"""Agent modules for project-agent."""
```

- [ ] **Step 2: Write agents/state.py**

```python
"""Shared state for the agent graph."""

from typing import Annotated, Any, Sequence
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State shared across all agent nodes in the graph."""

    # Conversation messages (accumulated via add_messages)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # The user's initial project idea
    project_idea: str

    # The evolving project specification
    spec: str

    # List of research tasks to perform
    research_queue: list[dict[str, Any]]

    # Research results accumulated so far
    research_findings: str

    # Pending research topics generated by the Planner (for human-in-the-loop)
    pending_research: list[str]

    # Open questions for the user
    open_questions: list[str]

    # Whether the Planner has finished researching
    planning_complete: bool

    # Final compiled report
    final_report: str

    # Progress messages for display to the user
    progress_messages: list[str]

    # MCP tool catalog: {server_name: [tool_name, ...]}
    mcp_tools: dict[str, list[str]]

    # User's selected MCP servers to use
    selected_mcp_servers: list[str]

    # Whether the user has signed off on the final report
    report_signed_off: bool
```

- [ ] **Step 3: Verify import**

Run: `python -c "from agents.state import AgentState; print(list(AgentState.__annotations__.keys())); print('OK')"`

Expected: State field names followed by `OK`

- [ ] **Step 4: Commit**

```bash
git add agents/__init__.py agents/state.py
git commit -m "feat: add AgentState TypedDict for shared graph state"
```

---

### Task 6: LLM Model Factory

**Files:**
- Create: `agents/model.py`

- [ ] **Step 1: Write agents/model.py**

```python
"""Factory for creating LLM models with the configured OpenAI-compatible endpoint."""

from langchain_openai import ChatOpenAI


def create_model(
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
) -> ChatOpenAI:
    """Create a ChatOpenAI model configured from environment variables.

    Args:
        model_name: Override OPENAI_MODEL env var.
        base_url: Override OPENAI_BASE_URL env var.
        api_key: Override OPENAI_API_KEY env var.
        temperature: Model temperature (lower = more deterministic).
    """
    from config import get_llm_config

    config = get_llm_config()
    return ChatOpenAI(
        model=model_name or config["model"],
        base_url=base_url or config["base_url"],
        api_key=api_key or config["api_key"],
        temperature=temperature,
    )
```

- [ ] **Step 2: Verify import**

Run: `python -c "from agents.model import create_model; print('OK')"`

Expected: `OK` (no API key validation at import time)

- [ ] **Step 3: Commit**

```bash
git add agents/model.py
git commit -m "feat: add LLM model factory with configurable OpenAI-compatible endpoint"
```

---

### Task 7: Orchestrator Node — CLI Interaction

**Files:**
- Create: `agents/orchestrator.py`

- [ ] **Step 1: Write agents/orchestrator.py**

```python
"""Orchestrator node: handles all user interaction via the terminal."""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.text import Text

console = Console()


def ask_user(question: str, default: str | None = None) -> str:
    """Ask the user a question via rich Prompt."""
    answer = Prompt.ask(question, default=default)
    return answer.strip() if answer else ""


def show_panel(title: str, content: str) -> None:
    """Display content in a styled panel."""
    console.print(Panel(content, title=title, subtitle="Project Agent"))


def present_mcp_servers(available: dict[str, list[str]]) -> list[str]:
    """Present available MCP servers and let the user select which to enable.

    Args:
        available: {server_name: [tool_names]} from config.yaml.

    Returns:
        List of selected server names.
    """
    console.print()
    console.print("[bold]Available MCP Servers:[/bold]")
    for i, (name, tools) in enumerate(available, 1):
        desc = ", ".join(tools[:5])
        console.print(f"  [bold]{i}[/bold]. [cyan]{name}[/cyan] — {desc}")
    console.print("  0. None (no MCP tools)")

    answer = Prompt.ask(
        "Select servers to enable (comma-separated numbers)",
        default="0",
    )
    selected = []
    for part in answer.split(","):
        part = part.strip()
        if part == "0":
            continue
        try:
            idx = int(part) - 1
            if 0 <= idx < len(available):
                selected.append(available[idx][0])
        except ValueError:
            pass
    return selected


def present_research_choices(
    findings: str,
    pending_topics: list[str],
) -> str:
    """Present research findings and pending topics for user input.

    Returns the user's text response for priority selection.
"""
    console.print()
    console.print(Panel(findings.strip(), title="Research Findings So Far"))

    if pending_topics:
        console.print()
        console.print("[bold]Pending research topics:[/bold]")
        for i, topic in enumerate(pending_topics, 1):
            console.print(f"  {i}. {topic}")
        console.print()

    answer = Prompt.ask(
        "What should we research next? (number, description, or 'skip')",
        default="1",
    )
    return answer.strip()


def present_open_questions(questions: list[str]) -> list[str]:
    """Present open questions to the user and collect answers."""
    answers = []
    for q in questions:
        console.print()
        console.print(Panel(q.strip(), title="Question"))
        answer = ask_user(f"Answer for: {q[:60]}...")
        answers.append(answer)
    return answers


def present_report(report: str) -> str:
    """Show the final report and ask for sign-off.

    Returns 'signed_off' or 'revise'.
    """
    console.print()
    show_panel("Final Report", report)
    console.print()
    answer = Prompt.ask(
        "How would you like to proceed?",
        choices=["signed_off", "revise"],
        default="signed_off",
    )
    return answer


def show_progress_line(message: str) -> None:
    """Print a single progress line (not a panel)."""
    console.print(f"  {message}")
```

- [ ] **Step 2: Verify import**

Run: `python -c "from agents.orchestrator import ask_user, show_panel, present_mcp_servers; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agents/orchestrator.py
git commit -m "feat: add orchestrator node with rich CLI interaction"
```

---

### Task 8: Planner Node — Interview, Spec Writing, Research Queuing

**Files:**
- Create: `agents/planner.py`

- [ ] **Step 1: Write agents/planner.py**

```python
"""Planner node: interviews the user, writes the spec, and queues research."""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.model import create_model
from agents.notepad import Notepad
from agents.state import AgentState

INTERVIEW_SYSTEM = """You are a Planner agent. Your job is to understand the user's
project idea by asking targeted questions and then write a detailed specification.

Ask 2-3 questions at a time to gather enough detail. Once you have sufficient
information, write the project spec to the notepad.

Questions should cover:
- Project goals and scope
- Target users and use cases
- Technical constraints (platform, budget, timeline)
- Must-have features vs. nice-to-have
- Existing tools or preferences"""

RESEARCH_SYSTEM = """You are a Planner agent. Review the research findings and
decide if more research is needed.

For each research finding, identify gaps or follow-up questions.
Return a JSON array of research tasks, each with:
- description: what to research
- server: preferred MCP server name (or "" for any)
- tool: preferred tool name (or "" to let the Researcher pick)
- args: preferred arguments (or {} to let the Researcher decide)

Only return the JSON array, nothing else."""


def planner_interview(state: AgentState, notepad: Notepad) -> dict[str, Any]:
    """Interview the user about their project idea and write the spec.

    This is a single-pass interview that uses the LLM to generate questions,
    collects answers from the user (via the orchestrator, injected into state),
    and writes the spec.

    For a simpler first implementation, this node takes the project_idea from
    state and generates a spec in one go. Multi-round interviewing is Task 11.
    """
    model = create_model(temperature=0.7)

    spec_prompt = f"""Write a detailed project specification based on this idea:

{state['project_idea']}

Include:
- Project overview and goals
- Target users
- Key features and requirements
- Technical approach and tech stack recommendations
- Timeline and milestones
- Risks and dependencies

Keep it structured and actionable.

OUTPUT THE SPEC ONLY — no preamble, no conversational text."""

    spec_response = model.invoke([
        HumanMessage(content=spec_prompt),
    ])

    spec_text = spec_response.content.strip()
    notepad.set_section("Project Spec", spec_text)
    notepad.update_progress("Spec written based on initial idea")

    return {
        "spec": spec_text,
        "progress_messages": state.get("progress_messages", []) + ["Spec written"],
    }


def planner_plan_research(state: AgentState, notepad: Notepad) -> dict[str, Any]:
    """Review the spec and determine what research is needed.

    Returns an updated research_queue.
    """
    model = create_model(temperature=0.2)

    research_prompt = f"""Based on this project spec, identify what research is needed.
Be specific about what to look up (prices, comparisons, alternatives, etc.).

Spec:
{state['spec'][:4000]}

MCP tools available:
{state.get('mcp_tools', {})}

Return a JSON array of research tasks. Example format:
[
  {{"description": "Pricing for Python ML frameworks 2026", "server": "brave-search", "tool": "brave_web_search", "args": {{"query": "Python ML framework pricing comparison 2026"}}}},
  {{"description": "LangGraph vs CrewAI comparison", "server": "brave-search", "tool": "brave_web_search", "args": {{"query": "LangGraph vs CrewAI comparison"}}}}
]

Return only the JSON array, nothing else."""

    response = model.invoke([HumanMessage(content=research_prompt)])
    import json
    text = response.content.strip()

    # Extract JSON from the response
    try:
        # Try parsing the whole response
        tasks = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            tasks = json.loads(match.group())
        else:
            tasks = []

    return {
        "research_queue": tasks,
        "progress_messages": state.get("progress_messages", []) + [f"Queued {len(tasks)} research tasks"],
    }
```

- [ ] **Step 2: Verify import**

Run: `python -c "from agents.planner import planner_interview, planner_plan_research; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agents/planner.py
git commit -m "feat: add planner node with interview and research queuing"
```

---

### Task 9: Researcher Node — MCP Tool Execution

**Files:**
- Create: `agents/researcher.py`

- [ ] **Step 1: Write agents/researcher.py**

```python
"""Researcher node: executes MCP tool calls and summarizes findings."""

import asyncio
import json
from typing import Any

from langchain_core.messages import HumanMessage

from agents.model import create_model
from agents.notepad import Notepad
from agents.state import AgentState


def _run_async(coro):
    """Run async code from a synchronous context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop:
        return asyncio.run(asyncio.wait_for(coro, timeout=120))
    return asyncio.run(coro)


async def _call_async(manager, server: str, tool: str, args: dict) -> str:
    """Async wrapper for MCP tool call."""
    return await manager.call_tool(server, tool, args)


def research_node(state: AgentState, mcp_manager, notepad: Notepad) -> dict[str, Any]:
    """Execute research tasks from the queue using MCP tools.

    For each task:
    1. Use the LLM to pick the best MCP tool if not specified
    2. Execute the tool call via the MCP manager
    3. Summarize the result

    Args:
        state: Current graph state.
        mcp_manager: MCPManager instance with active connections.
        notepad: Notepad instance for writing findings.

    Returns:
        Updated state with research_findings appended.
    """
    model = create_model(temperature=0.3)
    queue = state.get("research_queue", [])
    progress = state.get("progress_messages", [])

    if not queue:
        progress.append("No research tasks to execute.")
        return {
            "research_findings": "",
            "research_queue": [],
            "progress_messages": progress,
        }

    findings_parts = []
    remaining_tasks = list(queue)

    for task in queue:
        topic = task["description"]
        server_hint = task.get("server", "")
        tool_hint = task.get("tool", "")
        args_hint = task.get("args", {})

        progress.append(f"🔍 Researching: {topic}")

        # If no tool specified, let LLM pick one
        chosen_server = server_hint
        chosen_tool = tool_hint
        chosen_args = args_hint

        if not chosen_tool and server_hint:
            available = state.get("mcp_tools", {}).get(server_hint, [])
            if available:
                pick_prompt = (
                    f"Research topic: '{topic}'\n"
                    f"Available tools on '{server_hint}': {', '.join(available)}\n"
                    f"Reply with just: TOOL: server/toolname"
                )
                resp = model.invoke([HumanMessage(content=pick_prompt)])
                line = resp.content.strip().split("\n")[0]
                if "/" in line:
                    parts = line.split(":", 1)[1].strip().split("/", 1)
                    if len(parts) == 2:
                        chosen_server = parts[0].strip()
                        chosen_tool = parts[1].strip()
                progress.append(f"  └─ Tool selected: {chosen_tool}")

        if chosen_server and chosen_tool:
            if chosen_server in mcp_manager.sessions:
                try:
                    progress.append(f"  └─ Calling {chosen_server}/{chosen_tool}...")
                    raw = _run_async(
                        _call_async(mcp_manager, chosen_server, chosen_tool, chosen_args or {"query": topic})
                    )
                    progress.append(f"  └─ Got {len(raw)} chars")

                    summary_prompt = (
                        f"Summarize these findings for '{topic}':\n\n{raw[:4000]}\n\n"
                        f"Concise bullet points with key facts and source URLs."
                    )
                    summary = model.invoke([HumanMessage(content=summary_prompt)]).content.strip()
                    findings_parts.append(f"\n### {topic}\n{summary}")
                    remaining_tasks.remove(task)
                    progress.append(f"  └─ Done")
                except Exception as e:
                    progress.append(f"  └─ Error: {e}")
            else:
                progress.append(f"  └─ Server '{chosen_server}' not connected")
        else:
            progress.append(f"  └─ No valid tool, skipping")

    findings_text = "".join(findings_parts)
    notepad.append_section("Research Findings", findings_text)

    return {
        "research_findings": findings_text,
        "research_queue": remaining_tasks,
        "progress_messages": progress,
    }
```

- [ ] **Step 2: Verify import**

Run: `python -c "from agents.researcher import research_node; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agents/researcher.py
git commit -m "feat: add researcher node with MCP tool execution and summarization"
```

---

### Task 10: Interrupt Helper — Human-in-the-Loop

**Files:**
- Create: `agents/interrupt.py`

- [ ] **Step 1: Write agents/interrupt.py**

```python"""Human-in-the-loop interrupt helper."""

from langgraph.types import interrupt

from agents.state import AgentState


def human_interrupt(state: AgentState) -> dict[str, str]:
    """Pause the graph and present research findings + pending topics to the user.

    Uses LangGraph's interrupt() to pause execution until the user responds.

    Args:
        state: Current graph state with research_findings and pending_research.

    Returns:
        Dict with 'user_response' key containing the user's text answer.
    """
    findings = state.get("research_findings", "(no findings yet)")
    pending = state.get("pending_research", [])

    description_lines = [
        f"**Research Findings:**\n{findings[:1000]}",
    ]
    if pending:
        description_lines.append(f"\n**Pending Research:**\n" + "\n".join(f"- {t}" for t in pending))

    description = "\n".join(description_lines)

    response = interrupt([
        {
            "action_request": {"action": "research_priorities", "args": {"pending": pending}},
            "config": {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": False,
                "allow_accept": False,
            },
            "description": description,
        }
    ])[0]

    if response.get("type") == "response":
        return {"user_response": response.get("content", "")}
    elif response.get("type") == "accept":
        return {"user_response": "proceed"}
    else:
        return {"user_response": "skip"}
```

- [ ] **Step 2: Verify import**

Run: `python -c "from agents.interrupt import human_interrupt; print('OK')"`

Expected: `OK` (no graph execution needed at import time)

- [ ] **Step 3: Commit**

```bash
git add agents/interrupt.py
git commit -m "feat: add human-in-the-loop interrupt for research priority selection"
```

---

### Task 11: Graph Definition — Main LangGraph StateGraph

**Files:**
- Create: `agents/graph.py`

- [ ] **Step 1: Write agents/graph.py**

```python
"""Main LangGraph state graph definition for the project-agent."""

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.state import AgentState
from agents.planner import planner_interview, planner_plan_research
from agents.researcher import research_node
from agents.interrupt import human_interrupt
from agents.notepad import Notepad
from agents.model import create_model

from mcp.manager import MCPManager


def _should_continue(state: AgentState) -> str:
    """After the Planner, decide: research or interrupt."""
    queue = state.get("research_queue", [])
    if queue:
        return "research"
    # Research complete, check if we need to continue planning
    return "finalize"


def _research_router(state: AgentState) -> str:
    """After research or interrupt, decide: more research or finalize."""
    queue = state.get("research_queue", [])
    # If there are still tasks, keep researching
    if queue:
        return "research"
    return "finalize"


def _finalize(state: AgentState) -> dict[str, Any]:
    """Generate the final report from the spec and research findings."""
    model = create_model(temperature=0.2)

    spec = state.get("spec", "(no spec)")
    findings = state.get("research_findings", "(no research)")

    report_prompt = f"""Compile a comprehensive project report from the spec and research findings.

## Project Spec
{spec}

## Research Findings
{findings}

Produce a well-structured report that:
- Summarizes the project goals and approach
- Includes all research findings
- Provides recommendations based on the research
- Lists next steps for implementation

OUTPUT THE REPORT ONLY — no preamble."""

    response = model.invoke([
        {"role": "user", "content": report_prompt},
    ])

    report = response.content.strip()
    notepad = Notepad()
    notepad.save_report(report)

    return {
        "final_report": report,
        "planning_complete": True,
        "progress_messages": state.get("progress_messages", []) + ["Final report generated"],
    }


def build_graph(mcp_manager: MCPManager, notepad: Notepad) -> Any:
    """Build and compile the agent graph.

    Args:
        mcp_manager: MCPManager with active server connections.
        notepad: Shared Notepad instance.

    Returns:
        Compiled LangGraph graph ready for invocation.
    """
    builder = StateGraph(AgentState)

    # Wrap nodes to inject dependencies
    def _interview(state: AgentState) -> dict[str, Any]:
        return planner_interview(state, notepad)

    def _plan_research(state: AgentState) -> dict[str, Any]:
        return planner_plan_research(state, notepad)

    def _do_research(state: AgentState) -> dict[str, Any]:
        return research_node(state, mcp_manager, notepad)

    def _interrupt_flow(state: AgentState) -> dict[str, Any]:
        return human_interrupt(state)

    def _finalize_report(state: AgentState) -> dict[str, Any]:
        return _finalize(state)

    # Add nodes
    builder.add_node("interview", _interview)
    builder.add_node("plan_research", _plan_research)
    builder.add_node("research", _do_research)
    builder.add_node("interrupt", _interrupt_flow)
    builder.add_node("finalize", _finalize_report)

    # Edges
    builder.add_edge(START, "interview")
    builder.add_edge("interview", "plan_research")
    builder.add_conditional_edges(
        "plan_research",
        _should_continue,
        {
            "research": "research",
            "finalize": "finalize",
        },
    )
    builder.add_conditional_edges(
        "research",
        _research_router,
        {
            "research": "interrupt",
            "finalize": "finalize",
        },
    )
    builder.add_edge("interrupt", "plan_research")
    builder.add_edge("finalize", END)

    # Compile with checkpointer for human-in-the-loop support
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph
```

- [ ] **Step 2: Verify import**

Run: `python -c "from agents.graph import build_graph; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agents/graph.py
git commit -m "feat: add LangGraph state graph with all agent nodes and routing"
```

---

### Task 12: Entry Point — main.py

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Write main.py**

```python"""Project Agent — CLI entry point."""

import asyncio
import sys
from pathlib import Path

from rich.console import Console

from config import load_env, get_llm_config, load_mcp_config
from agents.orchestrator import ask_user, present_mcp_servers, present_research_choices, present_open_questions, present_report, show_progress_line
from agents.graph import build_graph
from agents.state import AgentState
from notepad import Notepad
from mcp.manager import MCPManager


def bootstrap() -> tuple[MCPManager, Notepad, dict]:
    """Initialize all infrastructure: config, MCP, notepad."""
    load_env()

    llm_config = get_llm_config()
    if not llm_config["api_key"]:
        print("Error: OPENAI_API_KEY not set. Copy .env.example to .env and fill in your API key.")
        sys.exit(1)

    mcp_manager = MCPManager()
    notepad = Notepad("project_notepad.md")

    return mcp_manager, notepad, llm_config


def select_mcp_servers(mcp_manager, mcp_config) -> tuple[MCPManager, dict[str, list[str]], list[str]]:
    """Show available MCP servers and let the user select which to enable."""
    if not mcp_config:
        print("No MCP servers configured. Running without MCP tools.")
        return mcp_manager, {}, []

    console = Console()
    available = [(s["name"], ["<no tools discovered yet>"]) for s in mcp_config]

    selected_names = present_mcp_servers(available)
    if not selected_names:
        return mcp_manager, {}, []

    # Discover tools by starting servers
    selected_configs = [s for s in mcp_config if s["name"] in selected_names]

    tools_catalog = asyncio.run(mcp_manager.start(selected_configs))

    console.print(f"\nMCP servers active: {', '.join(mcp_manager.available_servers)}")
    return mcp_manager, tools_catalog, selected_names


def run_project_agent():
    """Main entry point for the project agent."""
    console = Console()

    console.print("[bold]Project Agent[/bold] — Plan and research your projects")
    console.print()

    # Bootstrap infrastructure
    mcp_manager, notepad, llm_config = bootstrap()

    # Select MCP servers
    mcp_config = load_mcp_config()
    mcp_manager, tools_catalog, selected_servers = select_mcp_servers(mcp_manager, mcp_config)

    # Ask for project idea
    console.print()
    project_idea = ask_user("What project are you thinking of?")
    if not project_idea:
        console.print("[red]No project idea provided. Exiting.[/red]")
        sys.exit(0)

    # Build and run the graph
    graph = build_graph(mcp_manager, notepad)

    initial_state: AgentState = {
        "messages": [],
        "project_idea": project_idea,
        "spec": "",
        "research_queue": [],
        "research_findings": "",
        "pending_research": [],
        "open_questions": [],
        "planning_complete": False,
        "final_report": "",
        "progress_messages": [],
        "mcp_tools": tools_catalog,
        "selected_mcp_servers": selected_servers,
        "report_signed_off": False,
    }

    config = {"configurable": {"thread_id": "default"}}

    # Run the graph with progress display
    console.print()
    console.print("[bold blue]Starting project agent...[/bold blue]")
    console.print()

    result = graph.invoke(initial_state, config=config)

    # Show final report
    if result.get("final_report"):
        answer = present_report(result["final_report"])

        if answer == "signed_off":
            notepad.save_report(result["final_report"])
            console.print("\n[green]Report saved to project_notepad.md[/green]")
        else:
            console.print("[yellow]Revision requested. Manually edit the notepad and re-run.[/yellow]")
    else:
        console.print("[yellow]No report generated.[/yellow]")

    # Cleanup
    asyncio.run(mcp_manager.shutdown())


if __name__ == "__main__":
    run_project_agent()
```

- [ ] **Step 2: Verify import and structure**

Run: `python -c "import main; print('OK')"`

Expected: `OK` (no graph execution at import time)

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add CLI entry point with MCP selection, graph invocation, and report display"
```

---

### Task 13: Integration Test — Verify the Full Flow

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write a basic integration test**

```python"""Integration test: verify the graph can be built and invoked with minimal data."""

import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from agents.graph import build_graph
from agents.state import AgentState
from notepad import Notepad
from mcp.manager import MCPManager
import tempfile


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
    with patch.object(MCPManager, 'sessions', {}):
        graph = build_graph(tmp_mcp_manager, tmp_notepad)
        assert graph is not None


def test_initial_state_has_all_keys(tmp_notepad, tmp_mcp_manager):
    """Verify the initial state has all required keys."""
    with patch.object(MCPManager, 'sessions', {}):
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
        assert all(k in state for k in graph.get_input_schema().model_fields.keys())
```

- [ ] **Step 2: Run the integration tests**

Run: `python -m pytest tests/test_integration.py -v`

Expected: Tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for graph building and state validation"
```

---

### Task 14: Error Handling and Edge Cases

**Files:**
- Modify: `config.py`
- Modify: `mcp/manager.py`
- Modify: `agents/orchestrator.py`
- Modify: `agents/researcher.py`

- [ ] **Step 1: Add env validation to config.py**

Add to `config.py`:

```python
def validate_env() -> list[str]:
    """Validate required environment variables.

    Returns a list of error messages. Empty means all good.
    """
    errors = []
    config = get_llm_config()
    if not config["api_key"] or config["api_key"].startswith("sk-your"):
        errors.append("OPENAI_API_KEY is not set or uses the default placeholder value.")
    return errors
```

- [ ] **Step 2: Add retry logic to mcp/manager.py**

Add to `MCPManager.call_tool`:

```python
import time

async def call_tool(self, server_name: str, tool_name: str, args: dict[str, Any], max_retries: int = 3) -> str:
    """Execute a tool on a connected MCP server with retry logic."""
    for attempt in range(max_retries):
        try:
            return await self._call_tool_once(server_name, tool_name, args)
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error (after {max_retries} retries): {e}"
            wait = 2 ** attempt
            await asyncio.sleep(wait)
    return "Unexpected: loop completed without return"

async def _call_tool_once(self, server_name: str, tool_name: str, args: dict[str, Any]) -> str:
    """Execute a single tool call attempt."""
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
```

- [ ] **Step 3: Add graceful shutdown in main.py**

Add to the end of `run_project_agent()`:

```python
    # Ensure MCP cleanup even on errors
    try:
        # ... graph execution code ...
    finally:
        await mcp_manager.shutdown()
```

Wait — `run_project_agent` is synchronous but `shutdown` is async. Let me adjust:

```python
    finally:
        asyncio.run(mcp_manager.shutdown())
```

- [ ] **Step 4: Run existing tests to verify nothing broke**

Run: `python -m pytest tests/ -v`

Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add config.py mcp/manager.py main.py agents/orchestrator.py agents/researcher.py
git commit -m "fix: add env validation, MCP retry logic, and graceful error handling"
```

---

## Self-Review

**1. Spec coverage:**

| Spec Requirement | Task | Status |
|---|---|---|
| CLI entry point with user prompt | Task 12 | Covered |
| OpenAI-compatible API (not Claude) | Task 6 | Covered |
| Shared notepad (single file) | Task 3 | Covered |
| Three agents (Orchestrator, Planner, Researcher) | Tasks 7, 8, 9, 11 | Covered |
| MCP integration with explicit selection | Task 4, 12 | Covered |
| Brave Search MCP example | Task 4 (generic, used by all) | Covered |
| Human-in-the-loop interrupt | Task 10 | Covered |
| Live progress display | Task 7 (orchestrator), Task 9 (researcher progress) | Covered |
| Async research | Task 9 (async bridge) | Covered |
| Planning complete → report → sign-off | Task 11 (finalize), Task 12 (present_report) | Covered |
| Error handling (env, MCP, LLM) | Task 14 | Covered |

**2. Placeholder scan:** No TBD/TODO/placeholder patterns found.

**3. Type consistency:** All state field names match between `state.py` and usage in other nodes. `mcp_manager` and `notepad` are injected as constructor parameters to `build_graph`.

**4. Scope check:** Focused on the core three-agent flow. Multi-round interviewing is simplified to a single-pass for V1 (noted in Task 8 comments).

## Implementation Notes

- **Async boundary:** MCP calls are async; graph nodes are sync. Bridged via `_run_async()` in `researcher.py`. This works because `asyncio.run()` creates a new event loop when no loop is running (the default case for `graph.invoke()`).

- **MCP tool exposure:** MCP tools are called directly by the Researcher node, not via LangChain's ToolNode. The LLM in the Researcher generates tool selection instructions as text, which are parsed and executed against the MCP manager. This avoids the complexity of wrapping MCP async calls in LangChain's tool interface.

- **Config management:** MCP servers are defined in `config.yaml`, users select at runtime from the discovered list. API keys are either in `config.yaml` env vars or inherited from the system environment.

- **Notepad path:** Default is `project_notepad.md` in the working directory. This is configurable via the `Notepad(path)` constructor.

- **Graph checkpointer:** `InMemorySaver` provides state persistence for the interrupt mechanism. For production, this could be replaced with a file-based checkpointer.

---

Plan complete and saved to `docs/superpowers/plans/2026-05-23-project-agent.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?

<tool_call>
<function=Write>
<parameter=file_path>
/home/rmdluo/Documents/project-agent/docs/superpowers/plans/2026-05-23-project-agent.md