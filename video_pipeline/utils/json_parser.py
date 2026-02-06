"""
Robust JSON parser for LLM responses.

Handles common LLM output issues:
- Markdown code blocks (```json ... ```)
- Extra text before/after JSON
- Malformed JSON with regex fallback
"""

import json
import re
from typing import Any, Optional


def parse_llm_json(
    response: str,
    required_keys: Optional[list[str]] = None,
    fallback: Optional[Any] = None,
) -> dict:
    """
    Robust JSON parser for LLM responses.

    Args:
        response: Raw LLM response text
        required_keys: Optional list of keys that must be present in parsed JSON
        fallback: Value to return if parsing fails completely (default: raises exception)

    Returns:
        Parsed JSON as a dictionary

    Raises:
        ValueError: If parsing fails and no fallback is provided

    Examples:
        >>> parse_llm_json('{"key": "value"}')
        {'key': 'value'}

        >>> parse_llm_json('```json\\n{"key": "value"}\\n```')
        {'key': 'value'}

        >>> parse_llm_json('Here is the JSON: {"key": "value"}')
        {'key': 'value'}
    """
    if not response or not response.strip():
        if fallback is not None:
            return fallback
        raise ValueError("Empty response")

    cleaned = response.strip()

    # Step 1: Try direct json.loads()
    try:
        result = json.loads(cleaned)
        if _validate_keys(result, required_keys):
            return result
    except json.JSONDecodeError:
        pass

    # Step 2: Strip markdown code blocks
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        # Remove closing fence
        cleaned = re.sub(r'\n?```$', '', cleaned.strip())

        try:
            result = json.loads(cleaned)
            if _validate_keys(result, required_keys):
                return result
        except json.JSONDecodeError:
            pass

    # Step 3: Regex fallback to find JSON object in text
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if _validate_keys(result, required_keys):
                return result
        except json.JSONDecodeError:
            pass

    # Step 4: Try to find JSON array
    array_match = re.search(r'\[[\s\S]*\]', response)
    if array_match:
        try:
            result = json.loads(array_match.group())
            return result
        except json.JSONDecodeError:
            pass

    # All parsing attempts failed
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not parse JSON from response: {response[:200]}...")


def _validate_keys(data: Any, required_keys: Optional[list[str]]) -> bool:
    """Validate that all required keys are present in the data."""
    if required_keys is None:
        return True
    if not isinstance(data, dict):
        return False
    return all(key in data for key in required_keys)


def extract_json_field(response: str, field: str, default: Optional[str] = None) -> Optional[str]:
    """
    Extract a specific field from a JSON response.

    Args:
        response: Raw LLM response text
        field: The field name to extract
        default: Default value if extraction fails

    Returns:
        The field value or default
    """
    try:
        data = parse_llm_json(response)
        if isinstance(data, dict) and field in data:
            value = data[field]
            # Handle escaped newlines in string values
            if isinstance(value, str):
                return value.replace('\\n', '\n').strip()
            return value
    except ValueError:
        pass
    return default
