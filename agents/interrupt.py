"""Human-in-the-loop interrupt helper."""

from typing import Any

from langgraph.types import interrupt

from agents.state import AgentState


def human_interrupt(state: AgentState) -> dict[str, Any]:
    """Pause the graph and present research findings + pending topics to the user.

    Uses LangGraph's interrupt() to pause execution until the user responds.

    Args:
        state: Current graph state with research_findings and pending_research.

    Returns:
        Dict with 'user_response' key and cleared research_queue.
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

    user_response = "skip"
    if response.get("type") == "response":
        user_response = response.get("content", "")
    elif response.get("type") == "accept":
        user_response = "proceed"

    return {
        "user_response": user_response,
        "research_queue": [],  # Clear queue so plan_research regenerates from findings
        "pending_research": pending,
    }
