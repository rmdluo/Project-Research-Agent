"""Project Agent - CLI entry point."""

import asyncio
import sys

from rich.console import Console

from src.config import load_env, get_llm_config, load_mcp_config
from src.agents.orchestrator import ask_user, present_mcp_servers, present_report
from src.agents.graph import build_graph
from src.agents.state import AgentState
from src.notepad import Notepad
from src.mcp_servers.manager import MCPManager


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

    console.print("[bold]Project Agent[/bold] - Plan and research your projects")
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

    try:
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
    finally:
        asyncio.run(mcp_manager.shutdown())


if __name__ == "__main__":
    run_project_agent()
