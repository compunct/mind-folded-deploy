"""
Research and fact gathering for the video pipeline.

Handles iterative fact collection using Perplexity/Sonar.
"""

import os
import re
from openai import OpenAI

from ..config import FactsConfig
from ..utils.file_io import save_to_file, read_response_from_file
from ..llm.openrouter import query_llm


def _extract_numbered_facts(response: str, max_number: int = 10) -> list[str]:
    """
    Extract numbered facts from LLM response with flexible format matching.

    Supports formats: "1.", "1)", "1:", "**1.**", "**1.**", "1 -", etc.

    Args:
        response: Raw LLM response text
        max_number: Maximum fact number to look for

    Returns:
        List of extracted fact strings
    """
    facts = []
    lines = response.strip().split("\n")

    # Pattern matches: 1. or 1) or 1: or **1.** or **1)** or 1 - at start of line
    # Also handles optional leading whitespace and markdown bold
    pattern = re.compile(
        r'^\s*\*{0,2}(\d{1,2})[.):]\*{0,2}\s+(.+)',
        re.MULTILINE
    )

    for line in lines:
        match = pattern.match(line.strip())
        if match:
            number = int(match.group(1))
            if 1 <= number <= max_number:
                facts.append(line.strip())

    return facts


def get_facts_about_topic(
    client: OpenAI,
    model: str,
    topic_file_path: str,
    folder_name: str,
    timestamp: str,
    config: FactsConfig = None,
) -> str:
    """
    Get facts about the video topic iteratively and save the final response.

    Uses iterative querying to accumulate unique facts, avoiding repetition.

    Args:
        client: OpenRouter client
        model: Model to use for fact gathering (typically Perplexity/Sonar)
        topic_file_path: Path to the topic file
        folder_name: Project folder path
        timestamp: Timestamp for file naming
        config: Facts configuration (defaults to FactsConfig defaults)

    Returns:
        Path to the saved facts file
    """
    if config is None:
        config = FactsConfig()

    topic_response = read_response_from_file(topic_file_path)

    # Create a subfolder for Perplexity logs
    perplexity_folder = os.path.join(folder_name, "perplexity_logs")
    os.makedirs(perplexity_folder, exist_ok=True)

    facts_per_iteration = config.facts_per_iteration
    num_iterations = config.num_iterations

    # Initial prompt base
    base_prompt = f"""
    Based on the following video topic (title and description), provide exactly {facts_per_iteration} new interesting, factual points about the subject. Number them 1-{facts_per_iteration} and keep each fact concise (1-2 sentences max).
    {topic_response}
    """

    accumulated_facts = []
    previous_facts_str = ""

    max_retries = 2

    for iteration in range(num_iterations):
        if iteration > 0:
            previous_facts_str = "\n\nPrevious facts (do not repeat any of these):\n" + "\n".join(accumulated_facts)

        prompt = base_prompt + previous_facts_str

        # Retry loop for failed extractions
        facts = []
        for attempt in range(max_retries + 1):
            response = query_llm(client, model, prompt)

            # Save the prompt and response for this iteration
            suffix = f"_retry{attempt}" if attempt > 0 else ""
            iter_filename = f"perplexity_iteration_{iteration + 1}{suffix}_{timestamp}.txt"
            save_to_file(perplexity_folder, iter_filename, prompt, response)

            # Extract facts using flexible pattern matching
            facts = _extract_numbered_facts(response, max_number=facts_per_iteration)

            if len(facts) > 0:
                break
            elif attempt < max_retries:
                print(f"Warning: Iteration {iteration + 1} attempt {attempt + 1} returned 0 facts. Retrying...")

        if len(facts) != facts_per_iteration:
            print(f"Warning: Iteration {iteration + 1} returned {len(facts)} facts instead of {facts_per_iteration}.")

        accumulated_facts.extend(facts)

    # Final response is all accumulated facts joined
    final_response = "\n".join(accumulated_facts)

    # Use the last prompt as the saved prompt (or base + note about iterations)
    total_facts = facts_per_iteration * num_iterations
    final_prompt = base_prompt + f"\n\n(Note: This was generated iteratively over {num_iterations} queries to accumulate {total_facts} unique facts.)"

    filename = f"facts_{timestamp}.txt"
    file_path = save_to_file(folder_name, filename, final_prompt, final_response)
    return file_path
