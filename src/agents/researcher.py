"""Researcher node: executes MCP tool calls and summarizes findings."""

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage

from src.agents.model import create_model
from src.agents.notepad import Notepad
from src.agents.state import AgentState


async def research_node(
    state: AgentState,
    notepad: Notepad,
) -> dict[str, Any]:
    """Execute research tasks from the queue using MCP tools.

    For each task:
    1. Use model.bind_tools() to invoke MCP tools for the topic
    2. Execute the tool calls the model generates
    3. Summarize the results

    All tasks in the queue are executed in parallel.

    Args:
        state: Current graph state.
        notepad: Notepad instance for writing findings.

    Returns:
        Updated state with research_findings appended.
    """
    model = create_model(temperature=0.3)
    queue = state.get("research_queue", [])
    progress = state.get("progress_messages", [])

    # Build tool lookup
    tools = state.get("mcp_tools", [])
    tool_map = {t.name: t for t in tools}
    bound_model = model.bind_tools(tools)

    if not queue:
        progress.append("No research tasks to execute.")
        return {
            "research_findings": "",
            "research_queue": [],
            "progress_messages": progress,
        }

    async def _run_task(task: dict[str, Any]) -> tuple[str, str | None]:
        """Run a single research task. Returns (finding, task_description) on success, or (error_msg, task_description) on failure."""
        topic = task["description"]
        progress.append(f"Researching: {topic}")

        try:
            response = await bound_model.ainvoke([
                HumanMessage(content=f"Research this topic: {topic}"),
            ])
            tool_calls = getattr(response, "tool_calls", [])

            if not tool_calls:
                progress.append(f"  No tool call generated")
                return f"\n### {topic}\n_No tool calls generated_", task["description"]

            all_results = []
            for call in tool_calls:
                name = call["name"]
                args = call["args"]
                progress.append(f"  Calling {name}...")

                if name in tool_map:
                    tool = tool_map[name]
                    raw = await tool.ainvoke(args)
                    raw = str(raw) if not isinstance(raw, str) else raw
                    all_results.append(f"{name}: {raw}")
                    progress.append(f"  Got {len(raw)} chars")
                else:
                    progress.append(f"  Tool '{name}' not found")

            if all_results:
                combined = "\n".join(all_results)
                summary_prompt = (
                    f"Summarize these findings for '{topic}':\n\n{combined[:4000]}\n\n"
                    f"Concise bullet points with key facts and source URLs."
                )
                summary = (await model.ainvoke([
                    HumanMessage(content=summary_prompt),
                ])).content.strip()
                return f"\n### {topic}\n{summary}", task["description"]
            else:
                return f"\n### {topic}\n_No results collected_", task["description"]

        except Exception as e:
            progress.append(f"  Error: {e}")
            return f"\n### {topic}\nError: {e}", task["description"]

    results = await asyncio.gather(*[_run_task(t) for t in queue], return_exceptions=True)

    findings_parts = []
    completed_descriptions = set()
    for result in results:
        if isinstance(result, Exception):
            continue
        finding, task_desc = result
        findings_parts.append(finding)
        completed_descriptions.add(task_desc)

    remaining_tasks = [t for t in queue if t["description"] not in completed_descriptions]

    findings_text = "".join(findings_parts)
    notepad.append_section("Research Findings", findings_text)

    return {
        "research_findings": findings_text,
        "research_queue": remaining_tasks,
        "progress_messages": progress,
    }
