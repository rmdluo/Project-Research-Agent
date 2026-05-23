"""Researcher node: executes MCP tool calls and summarizes findings."""

from typing import Any

from langchain_core.messages import HumanMessage

from src.agents.model import create_model
from src.agents.notepad import Notepad
from src.agents.state import AgentState


async def research_node(state: AgentState, mcp_manager, notepad: Notepad) -> dict[str, Any]:
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

        progress.append(f"Researching: {topic}")

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
                progress.append(f"  Tool selected: {chosen_tool}")

        if chosen_server and chosen_tool:
            if chosen_server in mcp_manager.sessions:
                try:
                    progress.append(f"  Calling {chosen_server}/{chosen_tool}...")
                    raw = await mcp_manager.call_tool(
                        chosen_server, chosen_tool, chosen_args or {"query": topic}
                    )
                    progress.append(f"  Got {len(raw)} chars")

                    summary_prompt = (
                        f"Summarize these findings for '{topic}':\n\n{raw[:4000]}\n\n"
                        f"Concise bullet points with key facts and source URLs."
                    )
                    summary = model.invoke([HumanMessage(content=summary_prompt)]).content.strip()
                    findings_parts.append(f"\n### {topic}\n{summary}")
                    remaining_tasks.remove(task)
                    progress.append(f"  Done")
                except Exception as e:
                    progress.append(f"  Error: {e}")
            else:
                progress.append(f"  Server '{chosen_server}' not connected")
        else:
            progress.append(f"  No valid tool, skipping")

    findings_text = "".join(findings_parts)
    notepad.append_section("Research Findings", findings_text)

    return {
        "research_findings": findings_text,
        "research_queue": remaining_tasks,
        "progress_messages": progress,
    }
