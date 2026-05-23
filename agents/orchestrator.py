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


def present_mcp_servers(available: list[tuple[str, list[str]]]) -> list[str]:
    """Present available MCP servers and let the user select which to enable.

    Args:
        available: List of (server_name, tool_names) tuples from config.yaml.

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
