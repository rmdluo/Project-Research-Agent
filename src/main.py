"""Project Agent - CLI entry point."""

import asyncio
import sys

from rich.console import Console

from src.config import load_env, get_llm_config, load_mcp_config
from src.agents.orchestrator import ask_user, present_report
from src.agents.graph import build_graph
from src.agents.state import AgentState
from src.agents.notepad import Notepad


def bootstrap() -> tuple[Notepad, dict]:
    """Initialize all infrastructure: config, notepad."""
    load_env()

    llm_config = get_llm_config()
    if not llm_config["api_key"]:
        print("Error: OPENAI_API_KEY not set. Copy .env.example to .env and fill in your API key.")
        sys.exit(1)

    notepad = Notepad("project_notepad.md")

    return notepad, llm_config


async def _run_project_async(notepad, project_idea, mcp_configs) -> None:
    """Run the full graph in a single async context (one event loop)."""
    from langchain_core.tools import BaseTool

    console = Console()

    # Load all MCP tools from config
    if mcp_configs:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        client = MultiServerMCPClient(mcp_configs)
        tools: list[BaseTool] = await client.get_tools()
        tool_names = [t.name for t in tools]
        console.print(f"\nLoaded {len(tool_names)} MCP tool(s): {', '.join(tool_names)}")
    else:
        console.print("\nNo MCP servers configured.")
        tools = []
        tool_names = []

    graph = build_graph(tools, notepad)

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
        "mcp_tools": tool_names,
        "selected_mcp_servers": list(mcp_configs.keys()) if mcp_configs else [],
        "report_signed_off": False,
    }

    config = {"configurable": {"thread_id": "default"}}

    result = await graph.ainvoke(initial_state, config=config)

    if result.get("final_report"):
        answer = present_report(result["final_report"])

        if answer == "signed_off":
            notepad.save_report(result["final_report"])
            console.print("\n[green]Report saved to project_notepad.md[/green]")
        else:
            console.print("[yellow]Revision requested. Manually edit the notepad and re-run.[/yellow]")
    else:
        console.print("[yellow]No report generated.[/yellow]")


def run_project_agent():
    """Main entry point for the project agent."""
    console = Console()

    console.print("[bold]Project Agent[/bold] - Plan and research your projects")
    console.print()

    console.print()
    console.print("[bold blue]Starting project agent...[/bold blue]")
    console.print()

    # Bootstrap infrastructure
    notepad, llm_config = bootstrap()

    # Load MCP config and run everything in one event loop
    mcp_configs = load_mcp_config()

    # Ask for project idea
    console.print()
    project_idea = ask_user("What project are you thinking of?")
    if not project_idea:
        console.print("[red]No project idea provided. Exiting.[/red]")
        sys.exit(0)

    asyncio.run(_run_project_async(notepad, project_idea, mcp_configs))


if __name__ == "__main__":
    run_project_agent()
