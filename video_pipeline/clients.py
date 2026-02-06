"""
API client factories for the video pipeline.

Provides factory functions for creating and configuring API clients.
"""

import os
from typing import Tuple

from dotenv import load_dotenv
from openai import OpenAI
import replicate


def load_api_keys() -> Tuple[str, str, str]:
    """
    Load API keys from environment variables or .env file.

    Returns:
        Tuple of (openrouter_api_key, replicate_api_key, wan2_api_url)

    Raises:
        ValueError: If required API keys are not set
    """
    load_dotenv()

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment variables or .env file")

    replicate_key = os.getenv("REPLICATE_API_KEY")
    if not replicate_key:
        raise ValueError("REPLICATE_API_KEY not set in environment variables or .env file")

    wan2_api_url = os.getenv("WAN2_API_URL", "")

    return openrouter_key, replicate_key, wan2_api_url


def create_openrouter_client(api_key: str) -> OpenAI:
    """
    Create and return the OpenAI client configured for OpenRouter.

    Args:
        api_key: OpenRouter API key

    Returns:
        Configured OpenAI client
    """
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def create_replicate_client(api_key: str) -> replicate.Client:
    """
    Create and return a Replicate client.

    Args:
        api_key: Replicate API key

    Returns:
        Configured Replicate client
    """
    return replicate.Client(api_token=api_key)


class PipelineClients:
    """
    Container for all API clients used by the pipeline.

    Lazy-initializes clients on first access.
    """

    def __init__(self, openrouter_key: str, replicate_key: str, wan2_api_url: str = ""):
        """
        Initialize with API keys.

        Args:
            openrouter_key: OpenRouter API key
            replicate_key: Replicate API key
            wan2_api_url: Local video-api URL (optional, for local backend)
        """
        self._openrouter_key = openrouter_key
        self._replicate_key = replicate_key
        self._wan2_api_url = wan2_api_url
        self._openrouter_client = None
        self._replicate_client = None

    @classmethod
    def from_env(cls) -> "PipelineClients":
        """
        Create clients from environment variables.

        Returns:
            PipelineClients instance with keys from environment
        """
        openrouter_key, replicate_key, wan2_api_url = load_api_keys()
        return cls(openrouter_key, replicate_key, wan2_api_url)

    @property
    def openrouter(self) -> OpenAI:
        """Get the OpenRouter client (lazy-initialized)."""
        if self._openrouter_client is None:
            self._openrouter_client = create_openrouter_client(self._openrouter_key)
        return self._openrouter_client

    @property
    def replicate(self) -> replicate.Client:
        """Get the Replicate client (lazy-initialized)."""
        if self._replicate_client is None:
            self._replicate_client = create_replicate_client(self._replicate_key)
        return self._replicate_client

    @property
    def openrouter_api_key(self) -> str:
        """Get the OpenRouter API key (for litellm calls)."""
        return self._openrouter_key

    @property
    def replicate_api_key(self) -> str:
        """Get the Replicate API key."""
        return self._replicate_key

    @property
    def wan2_api_url(self) -> str:
        """Get the local video-api URL (from WAN2_API_URL env var)."""
        return self._wan2_api_url
