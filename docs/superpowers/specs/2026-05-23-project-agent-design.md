# Project Agent - Design Specification

## Overview

A CLI-based multi-agent tool that helps users plan and research projects. It interviews the user about their project idea, researches tech stacks/pricing/compatibilities using MCP-connected tools, and produces a final report.

## Dependencies

- `langgraph` — Agent graph orchestration
- `langchain-openai` — LLM abstraction (OpenAI-compatible endpoint)
- `mcp` — MCP client to manage MCP server subprocesses
- `rich` — Terminal UI (tables, formatting, progress indicators)
- `python-dotenv` — Load `.env` for API configuration

## Environment Configuration

`.env` file with:
```
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

## Shared Notepad

A single `notepad.md` file with these sections:
- `## Project Spec` — evolving project specification
- `## Research Findings` — research results appended by the Researcher
- `## Open Questions` — questions presented to the user
- `## Decisions` — key decisions made during planning
- `## Progress` — current state and next steps

All agents read from and write to this file via a shared utility module.

## Agent Architecture

### Three Agents

1. **Orchestrator** — CLI interface. Handles all user interaction: presenting menus, collecting input, displaying live progress and final results. Does not reason — it facilitates.

2. **Planner** — Creates the project spec, determines research needs, reviews research findings, and decides if more research is needed.

3. **Researcher** — Executes research tasks using MCP tools (web search, API lookups, etc.). Reports findings back to the Planner.

### Workflow

```
Orchestrator: Greet user, select MCP servers
    ↓
Orchestrator: Ask user about project idea
    ↓
Planner: Interview user (via Orchestrator) → Write spec outline
    ↓
Planner: Determine research needs → Queue research tasks
    ↓
Researcher: Execute research tasks → Report findings to Planner
    ↓
Planner: Review findings → Generate list of "still need to research..."
    ↓
*** HUMAN-IN-THE-LOOP ***
Orchestrator: Presents findings + pending research topics to user
    User selects priorities: "Research X first, then Y. Skip Z."
    ↓ (graph resumes with user's priorities)
Researcher: Execute selected research tasks
    ↓
Planner: Review → more research needed? → Human-in-the-loop (repeat)
    ↓ No
Final spec + report
    ↓
Orchestrator: Present final report → User signs off
```

### Human-in-the-Loop

Implemented via LangGraph `interrupt`. After each research cycle, the graph pauses and the Orchestrator presents:
- Summary of findings so far
- List of pending research topics with priorities
- User selects what to research next or requests changes

The graph then resumes with the user's input.

### Live Progress Display

Each agent node yields progress messages to the graph state:
- **Planner node:** Shows "Planning..." then "Analyzing requirements...", "Writing spec...", etc.
- **Researcher node:** Shows "Researching: X" with live status like "Searching pricing...", "Comparing options..."
- **Orchestrator:** Streams these messages to the terminal as `rich` status indicators

Research tasks run concurrently (multiple MCP tool calls in parallel) within the Researcher node.

## File Structure

```
project-agent/
├── main.py                 # Entry point, CLI bootstrap
├── config.py               # Load .env, MCP config
├── mcp/
│   ├── manager.py          # MCP server lifecycle management
│   └── tools.py            # Expose MCP tools to agents
├── agents/
│   ├── graph.py            # LangGraph definition (main graph + subgraph)
│   ├── state.py            # Shared state TypedDict
│   ├── orchestrator.py     # Orchestrator node (CLI handling)
│   ├── planner.py          # Planner node
│   ├── researcher.py       # Researcher node
│   └── interrupt.py        # Human-in-the-loop interrupt logic
├── notepad.py              # Shared notepad read/write
└── .env.example            # Template for environment config
```

## MCP Integration (Example: Brave Search)

### Config format

```yaml
mcp_servers:
  - name: brave-search
    command: npx
    args: ["-y", "@brave/brave-search-mcp-server"]
    env:
      BRAVE_API_KEY: "your-api-key"
    enabled_tools: []   # empty = all tools
```

### Discovery

The MCP Manager spawns the server as a stdio subprocess, calls `list_tools()`, and exposes the tool catalog to the Orchestrator for user selection.

### Example tools (Brave Search)

| Tool | Purpose |
|---|---|
| `brave_web_search` | General web search (query, count, freshness, language, filters) |
| `brave_news_search` | Recent news articles (with breaking flag, freshness) |
| `brave_image_search` | Image search (with dimensions, confidence) |
| `brave_local_search` | Local business search (Pro plan required) |
| `brave_summarizer` | AI-generated summary (requires key from `brave_web_search` with `summary: true`) |

### How the Researcher uses Brave Search

```python
# General research
result = await mcp.call_tool(
    "brave_web_search",
    {"query": "Python ML framework pricing 2026", "count": 5, "freshness": "pm"}
)

# Time-sensitive research
result = await mcp.call_tool(
    "brave_news_search",
    {"query": "AI regulation 2026", "count": 5, "freshness": "pw"}
)

# Deep-dive (two-step)
search = await mcp.call_tool("brave_web_search", {"query": "...", "summary": True})
key = extract_summarizer_key(search)
summary = await mcp.call_tool("brave_summarizer", {"key": key, "inline_references": True})
```

### Live progress example

```
🔍 Researching: Python ML framework pricing...
   └─ Searching web (5 results, last 31 days)...
   └─ Found: LangGraph $0 (open source) vs Databricks $1000/mo
   └─ 5 results aggregated
```

## Data Flow

1. User provides project idea via CLI
2. Planner interviews user (iterative, through Orchestrator)
3. Planner writes initial spec to notepad
4. Planner queues research tasks based on spec gaps
5. Researcher executes tasks via MCP tools, writes findings to notepad
6. Planner reviews findings, determines additional research needs
7. **Interrupt:** Orchestrator asks user for research priorities
8. Graph resumes with user-selected priorities
9. Loop repeats until research is complete
10. Planner writes final report to notepad
11. Orchestrator presents report to user
12. User signs off or requests changes

## Error Handling

- Missing `.env` file: Prompt user to create it with `.env.example`
- MCP server failure: Log error, skip that server, continue with available tools
- LLM API error: Retry up to 3 times with exponential backoff, then inform user
- Interrupt cancellation: User can abort the entire process at any interrupt point
