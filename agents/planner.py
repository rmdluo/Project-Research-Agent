"""Planner node: interviews the user, writes the spec, and queues research."""

from typing import Any

from langchain_core.messages import HumanMessage

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

OUTPUT THE SPEC ONLY -- no preamble, no conversational text."""

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
