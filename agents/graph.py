"""Main LangGraph state graph definition for the project-agent."""

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.state import AgentState
from agents.planner import planner_interview, planner_plan_research
from agents.researcher import research_node
from agents.interrupt import human_interrupt
from agents.notepad import Notepad

from mcp_servers.manager import MCPManager


def _should_continue(state: AgentState) -> str:
    """After the Planner, decide: research or interrupt."""
    queue = state.get("research_queue", [])
    if queue:
        return "research"
    return "finalize"


def _research_router(state: AgentState) -> str:
    """After research or interrupt, decide: more research or finalize."""
    queue = state.get("research_queue", [])
    if queue:
        return "research"
    return "finalize"


def _finalize(state: AgentState) -> dict[str, Any]:
    """Generate the final report from the spec and research findings."""
    from agents.model import create_model

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

OUTPUT THE REPORT ONLY -- no preamble."""

    response = model.invoke([
        {"role": "user", "content": report_prompt},
    ])

    report = response.content.strip()

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
