"""
Narrative ordering for the video pipeline.

Handles reordering facts into a coherent storytelling arc.
"""

import json
import re
from openai import OpenAI

from ..utils.file_io import save_to_file, read_response_from_file
from ..utils.json_parser import parse_llm_json
from ..utils.text_processing import count_numbered_facts
from ..llm.openrouter import query_llm


def reorder_facts_for_story(
    client: OpenAI,
    model: str,
    topic_file_path: str,
    facts_file_path: str,
    folder_name: str,
    timestamp: str,
) -> str:
    """
    Reorder all facts to flow like a story and save the reordered facts.

    Args:
        client: OpenRouter client
        model: Model to use for reordering (typically GPT-4o)
        topic_file_path: Path to the topic file
        facts_file_path: Path to the facts file
        folder_name: Project folder path
        timestamp: Timestamp for file naming

    Returns:
        Path to the saved reordered facts file
    """
    topic_response = read_response_from_file(topic_file_path)
    facts_response = read_response_from_file(facts_file_path)

    # Count the actual number of facts in the input
    fact_count = count_numbered_facts(facts_response)

    prompt = f"""
    I have {fact_count} facts about this topic: {topic_response}
    I plan to turn them into a video that flows like a story. So I will need to reorder the facts to flow better in a more engaging way.

    Here are the facts:
    {facts_response}

    Reorder them logically for storytelling (e.g., chronological, building tension, or thematic progression).

    CRITICAL OUTPUT INSTRUCTIONS - FOLLOW THESE EXACTLY OR THE OUTPUT WILL BE INVALID:
    - Output ONLY a valid JSON object with a single key: "reordered_facts" whose value is an array of strings, each string being one fact.
    - The array must contain exactly {fact_count} facts (no more, no less; do not add or remove any).
    - The JSON must be pure and parsable—no extra text, wrappers, explanations, or comments before or after it.
    - In each fact string, preserve the original fact content but ensure it's concise (1-2 sentences).
    - If you add ANY extra text outside the JSON, the entire response will be discarded.

    GOOD EXAMPLE (DO THIS):
    {{"reordered_facts": ["Fact about origin.", "Fact about early development.", "Fact about modern impact."]}}
    """

    response = query_llm(client, model, prompt)

    # Parse the response as JSON using robust parser
    try:
        json_data = parse_llm_json(response, required_keys=["reordered_facts"])
        if isinstance(json_data["reordered_facts"], list) and len(json_data["reordered_facts"]) == fact_count:
            response_to_save = json.dumps(json_data, indent=2)
        else:
            raise ValueError(f"Invalid JSON structure or wrong fact count (expected {fact_count}, got {len(json_data.get('reordered_facts', []))})")
    except ValueError as e:
        print(f"JSON parsing failed ({str(e)}). Using raw response as fallback.")
        response_to_save = response

    filename = f"reordered_facts_{timestamp}.txt"
    file_path = save_to_file(folder_name, filename, prompt, response_to_save)
    return file_path
