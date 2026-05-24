"""Planner node: interviews the user, writes the spec, and queues research."""

from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.agents.model import create_model
from src.agents.notepad import Notepad
from src.agents.state import AgentState


class InterviewQuestion(BaseModel):
    """Output from the question generator."""
    question: str


class InterviewDecision(BaseModel):
    """Whether enough info has been gathered."""
    complete: bool


INTERVIEW_SYSTEM = """You are a Planner agent. Your job is to understand the user's
project idea by asking targeted questions and then write a detailed specification.

Ask 1 questions at a time to gather enough detail. Once you have sufficient
information, write the project spec to the notepad.

Questions should cover:
- Project goals and scope
- Target users and use cases
- Technical constraints (platform, budget, timeline)
- Must-have features vs. nice-to-have
- Existing tools or preferences
- What materials/software is required"""

RESEARCH_SYSTEM = """You are a Planner agent. Review the research findings and
decide if more research is needed.

For each research finding, identify gaps or follow-up questions.
Return a JSON array of research tasks, each with:
- description: what to research
- tool: tool name (or "" to let the Researcher pick)
- args: preferred arguments (or {} to let the Researcher decide)

Only return the JSON array, nothing else."""


def planner_interview(state: AgentState, notepad: Notepad) -> dict[str, Any]:
    """Interview the user about their project idea and write the spec.

    Multi-round loop (up to 5 rounds): generates 1 question per round,
    collects answers via interrupt, decides after each round if enough info
    was gathered. Writes the spec once satisfied.
    """
    from langgraph.types import interrupt

    questions_model = create_model(temperature=0.7).with_structured_output(InterviewQuestion)
    decision_model = create_model(temperature=0.1).with_structured_output(InterviewDecision)
    spec_model = create_model(temperature=0.7)

    project_idea = state['project_idea']
    transcript: list[dict] = []  # [{"questions": [...], "answers": [...]}]
    max_rounds = 0 # 100

    for round_num in range(max_rounds):
        # Phase 1: Decide if we have enough info, or generate new questions
        if transcript:
            decide_prompt = INTERVIEW_SYSTEM + f"""

You have already interviewed the user. Review the transcript:

Project idea:
{project_idea}

Interview transcript:
{chr(10).join(f"Round {r['round']}:\n" + f"  Q: {r["questions"]}" + "\n" + f"  A: {r["answers"]}" for r in transcript)}

Do you have enough information to write a detailed spec?"""

            decision = decision_model.invoke([HumanMessage(content=decide_prompt)]).complete
            if decision:
                break

        # Phase 2: Generate targeted question
        if transcript:
            questions_prompt = INTERVIEW_SYSTEM + f"""
Round {round_num + 1}: Based on the user's answers so far, generate exactly 1 focused follow-up question to fill remaining gaps.

Project idea:
{project_idea}

Interview transcript:
{chr(10).join(f"Round {r['round']}:\n" + f"  Q: {r["questions"]}" + "\n" + f"  A: {r["answers"]}" for r in transcript)}
Return a JSON-like object with a "questions" key containing exactly 1 questions."""
        else:
            questions_prompt = INTERVIEW_SYSTEM + f"""

Round 1: Based on the idea below, generate exactly 1 targeted question to gather enough detail for a spec.

Project idea:
{project_idea}

Return a JSON-like object with a "question" key containing exactly 1 questions."""

        question = questions_model.invoke([HumanMessage(content=questions_prompt)]).question

        # Phase 3: Present questions to user via interrupt
        try:
            response = interrupt([{
                "action_request": {"action": "interview", "args": {"questions": question}},
                "config": {
                    "allow_ignore": True,
                    "allow_respond": True,
                    "allow_edit": False,
                    "allow_accept": False,
                },
                "description": f"Round {round_num + 1} of the project interview. Please answer: " + question,
            }])[0]

            user_answer = []
            if response.get("type") == "response":
                user_answer = response.get("content", "").strip()
        except:
            print(question)
            user_answer = input("> ")
            print()

        transcript.append({"round": round_num + 1, "questions": question, "answers": user_answer})

    # Phase 4: Write the spec from the full transcript
    transcript_text = chr(10).join(
        f"Round {r['round']}:\n" + f"  Q: {r['questions']}" + "\n" + f"  A: {r['answers']}"
        for r in transcript
    )

    spec_prompt = f"""Write a detailed project specification based on this idea and the user's answers to clarifying questions:

Project idea:
{project_idea}

Interview transcript:
{transcript_text}

Include:
- Project overview and goals
- Target users
- Key features and requirements
- Technical approach and tech stack recommendations
- Timeline and milestones
- Risks and dependencies

Keep it structured and actionable.

OUTPUT THE SPEC ONLY -- no preamble, no conversational text."""

    spec_response = spec_model.invoke([HumanMessage(content=spec_prompt)])
    spec_text = spec_response.content.strip()
    notepad.set_section("Project Spec", spec_text)
    notepad.update_progress(f"Spec written after {len(transcript)} round(s) of interviewing")

    return {
        "spec": spec_text,
        "progress_messages": state.get("progress_messages", []) + ["Spec written"],
    }


def planner_plan_research(state: AgentState, notepad: Notepad) -> dict[str, Any]:
    """Review the spec and determine what research is needed.

    Returns an updated research_queue.
    """
    model = create_model(temperature=0.2)

    tool_names = state.get("mcp_tools", [])
    research_prompt = f"""Based on this project spec, identify what research is needed.
Be specific about what to look up (prices, comparisons, alternatives, etc.).

Spec:
{state['spec'][:4000]}

Available tools:
{', '.join(tool_names)}

Return a JSON array of research tasks. Example format:
[
  {{"description": "Pricing for Python ML frameworks 2026", "tool": "brave_web_search", "args": {{"query": "Python ML framework pricing comparison 2026"}}}},
  {{"description": "LangGraph vs CrewAI comparison", "tool": "brave_web_search", "args": {{"query": "LangGraph vs CrewAI comparison"}}}}
]

Return only the JSON array, nothing else."""

    response = model.invoke([HumanMessage(content=research_prompt)])
    import json
    import re
    text = response.content.strip()

    # Extract JSON from the response
    try:
        tasks = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        tasks = json.loads(match.group()) if match else []

    return {
        "research_queue": tasks,
        "progress_messages": state.get("progress_messages", []) + [f"Queued {len(tasks)} research tasks"],
    }
