"""Human-in-the-loop interrupt helper."""

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
