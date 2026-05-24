"""Configuration loader for project-agent."""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Default paths
BASE_DIR = Path(__file__).parent.parent
ENV_PATH = BASE_DIR / ".env"
CONFIG_PATH = BASE_DIR / "config.json"


def load_env() -> None:
    """Load environment variables from .env file if it exists."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)


def get_llm_config() -> dict[str, str]:
    """Return LLM configuration from environment variables."""
    return {
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
    }


def validate_env() -> list[str]:
    """Validate required environment variables.

    Returns a list of error messages. Empty means all good.
    """
    errors = []
    config = get_llm_config()
    if not config["api_key"] or config["api_key"].startswith("sk-your"):
        errors.append(
            "OPENAI_API_KEY is not set or uses the default placeholder value."
        )
    return errors


def load_mcp_config() -> dict[str, Any]:
    """Load MCP server definitions from config.json.

    Returns a dict mapping server_name -> connection_config,
    ready to pass to MultiServerMCPClient.
    """
    if not CONFIG_PATH.exists():
        return {}

    with open(CONFIG_PATH) as f:
        data = json.load(f)

    return data.get("mcp_servers", {})
