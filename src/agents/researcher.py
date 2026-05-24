"""Researcher node: executes MCP tool calls and summarizes findings."""

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

    print(queue)

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
        progress.append(f"Researching: {topic}")

        # Let the model decide which tools to call
        try:
            response = bound_model.invoke([
                HumanMessage(content=f"Research this topic: {topic}"),
            ])
            tool_calls = getattr(response, "tool_calls", [])

            if not tool_calls:
                progress.append(f"  No tool call generated")
                continue

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
                summary = model.invoke([
                    HumanMessage(content=summary_prompt),
                ]).content.strip()
                findings_parts.append(f"\n### {topic}\n{summary}")
                remaining_tasks.remove(task)
                progress.append(f"  Done")
            else:
                progress.append(f"  No results collected")

        except Exception as e:
            progress.append(f"  Error: {e}")

    findings_text = "".join(findings_parts)
    notepad.append_section("Research Findings", findings_text)

    return {
        "research_findings": findings_text,
        "research_queue": remaining_tasks,
        "progress_messages": progress,
    }
