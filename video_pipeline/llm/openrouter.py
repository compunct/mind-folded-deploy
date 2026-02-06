"""
OpenRouter LLM wrapper with query utilities.

Provides a simple interface for querying LLMs via OpenRouter.
"""

from typing import Optional
from openai import OpenAI


def query_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Send a prompt to the specified model and return the response content.

    Args:
        client: OpenAI client configured for OpenRouter
        model: Model identifier (e.g., "anthropic/claude-3.5-sonnet")
        prompt: The prompt to send
        max_tokens: Optional maximum tokens in response
        temperature: Optional temperature for generation

    Returns:
        The response content as a string
    """
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content
