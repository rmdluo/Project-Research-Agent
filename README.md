# Project Research Agent

Project Research Agent is a LangGraph-based agent that helps flesh out project ideas. In particular, it helps users add more specifics to their project plans by asking specific questions, uses Brave Search (or other configured MCP servers) to research the idea, and then writes up a final project report.

## Installation

Use the package manager [uv](https://docs.astral.sh/uv/getting-started/installation/) to install Project Research Agent.

```bash
uv sync
```

Using other package managers such as Poetry and pip should be possible as well (untested).

## Usage

From the commandline:
```bash
uv run src/main.py
```

or

```bash
uv run project-agent
```

From there, it will first prompt you for your project idea. It will then ask questions about your idea to further flesh it out. After that, it will synthesize your answers into a research plan, perform the research, and synthesize what it finds into a final project report. This report can be found at `project_notepad.md`.

## Agent Diagram

OBE until I get it to a spot I am decently happy with (which may be never).

## Roadmap

 - Feat: Separate the MCPs for the different agents
 - Feat: Better integrate internet lookups into the planning phase
 - Feat: Report back with research results before writing final spec, give chance for user to request more research
 - Feat: Rework the notepad to allow for list of freeform notes rather than specific sections
 - Feat: Separate the final report from the notepad.
 - UI: Vibe code UI (probably streamlit)
 - Integration: Integrate with LangFuse

## Notes on Claude Code Usage

A side goal of this project was to test the capabilities of Claude Code with local LLMs (Qwen3.6-35B-A3B-UD-Q8_K_XL running on x2 3090's through LlamaCPP). Originally, I used the obra/superpowers and context7 plugins to work with me to figure out a full spec and implementation; however, I found that to be a bit cumbersome and error-prone. After having it write out most of the project framework in that way, I switched to looking through the code myself, debugging errors by myself or with Claude Code, and passing it back to Claude Code with the fix I want. This helped get the code to a better working state. It is entirely possible that Claude Opus or Codex would be able to build a full working implementation on its own, but this workflow was decent for a quantized non-enterprise solution.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)