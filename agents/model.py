"""Factory for creating LLM models with the configured OpenAI-compatible endpoint."""

from langchain_openai import ChatOpenAI


def create_model(
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
) -> ChatOpenAI:
    """Create a ChatOpenAI model configured from environment variables.

    Args:
        model_name: Override OPENAI_MODEL env var.
        base_url: Override OPENAI_BASE_URL env var.
        api_key: Override OPENAI_API_KEY env var.
        temperature: Model temperature (lower = more deterministic).
    """
    from config import get_llm_config

    config = get_llm_config()
    return ChatOpenAI(
        model=model_name or config["model"],
        base_url=base_url or config["base_url"],
        api_key=api_key or config["api_key"],
        temperature=temperature,
    )
