"""
Script generation for the video pipeline.

Handles iterative script writing and intro/outro generation.
"""

import os
import json
import re
from openai import OpenAI

from ..config import ScriptConfig
from ..utils.file_io import save_to_file, read_response_from_file
from ..utils.json_parser import parse_llm_json, extract_json_field
from ..llm.openrouter import query_llm


def generate_video_script(
    client: OpenAI,
    model: str,
    topic_file_path: str,
    reordered_facts_file_path: str,
    folder_name: str,
    timestamp: str,
    config: ScriptConfig = None,
) -> str:
    """
    Generate the video script iteratively and save the final script.

    Args:
        client: OpenRouter client
        model: Model to use for script generation (typically Claude)
        topic_file_path: Path to the topic file
        reordered_facts_file_path: Path to the reordered facts file
        folder_name: Project folder path
        timestamp: Timestamp for file naming
        config: Script configuration (defaults to ScriptConfig defaults)

    Returns:
        Path to the saved script file
    """
    if config is None:
        config = ScriptConfig()

    topic_response = read_response_from_file(topic_file_path)
    reordered_facts_response = read_response_from_file(reordered_facts_file_path)

    # Parse reordered facts as JSON
    # Strip markdown code blocks if present (LLM often wraps JSON in ```json ... ```)
    cleaned_response = reordered_facts_response.strip()
    if cleaned_response.startswith("```"):
        cleaned_response = re.sub(r'^```\w*\n?', '', cleaned_response)
        cleaned_response = re.sub(r'\n?```$', '', cleaned_response.strip())

    try:
        json_data = json.loads(cleaned_response)
        facts_list = json_data["reordered_facts"]
    except json.JSONDecodeError:
        # Fallback to old split if needed (for backward compatibility)
        facts_list = [line.strip() for line in reordered_facts_response.strip().split("\n") if line.strip()]

    print(f"Processing {len(facts_list)} facts for script generation.")

    # Create a subfolder for script logs
    script_folder = os.path.join(folder_name, "script_logs")
    os.makedirs(script_folder, exist_ok=True)

    accumulated_script = ""
    facts_per_section = config.facts_per_section
    section_word_count = config.section_word_count

    num_iterations = len(facts_list) // facts_per_section
    if len(facts_list) % facts_per_section != 0:
        num_iterations += 1

    base_prompt = f"""
    Based on the following video topic (title and description):
    {topic_response}

    You are writing a script section for a YouTube sleepy-time philosophy video.

    ############################################################
    # WORD COUNT REQUIREMENT - THIS IS THE #1 PRIORITY
    ############################################################
    This section MUST be {section_word_count}. This is NON-NEGOTIABLE.

    Count your words carefully. If the requirement says 1500-2500 words,
    you must write AT LEAST 1500 words. If it says 300-500 words, write
    at least 300 words. Match the specified range exactly.
    ############################################################

    This is a sleepy-time video designed to help people fall asleep. It requires:
    - Slow, meditative prose with gentle rhythm
    - Thoughtful elaboration on each fact
    - Philosophical reflections and deeper meaning
    - Vivid sensory imagery and metaphors
    - Smooth transitions between ideas

    Cover exactly the given {facts_per_section} facts in narrative style.
    Ensure all facts are included without omitting any.

    OUTPUT FORMAT (must follow exactly):
    - Output ONLY a valid JSON object: {{"script": "your script text here"}}
    - Use \\n for line breaks within the script string
    - No extra text before or after the JSON
    - Start directly with the first sentence (no "Here's a script...")
    - End with the last sentence (no "[Transition to next section]")
    """

    for iteration in range(num_iterations):
        start = iteration * facts_per_section
        next_facts = facts_list[start:start + facts_per_section]
        next_facts_str = "\n".join(next_facts)

        prompt = base_prompt + f"""
        Write the script section for these exact facts:\n{next_facts_str}
        Remember: Output ONLY the JSON object with "script" key. No intros, no explanations, no placeholders, no extra text. Violation invalidates the response.
        """

        response = query_llm(client, model, prompt)

        # Save the prompt and response for this iteration
        iter_filename = f"script_iteration_{iteration + 1}_{timestamp}.txt"
        save_to_file(script_folder, iter_filename, prompt, response)

        # Parse the JSON
        script_content = extract_json_field(response, "script")

        if script_content is None:
            print(f"Warning: JSON parsing failed for iteration {iteration + 1}. Attempting extraction.")
            # Extract content between {"script": and } , handling multi-line
            match = re.search(r'\{\s*"script"\s*:\s*"(.*)"\s*\}', response, re.DOTALL)
            if match:
                script_content = match.group(1)
                # Replace raw newlines with \n (but since it's text, unescape for use)
                script_content = script_content.replace('\\n', '\n').replace('\r', '').strip()
            else:
                script_content = response.strip()  # Original fallback
                print("Extraction failed; using raw response.")

        # Append the new section to the accumulated script
        if iteration > 0:
            accumulated_script += "\n\n" + script_content
        else:
            accumulated_script = script_content

    # Final response is the full accumulated script
    final_prompt = base_prompt + f"\n\n(Note: This was generated iteratively over {num_iterations} queries to build a cohesive script in sections.)"

    filename = f"script_{timestamp}.txt"
    file_path = save_to_file(folder_name, filename, final_prompt, accumulated_script)
    return file_path


def generate_intro_outro(
    client: OpenAI,
    model: str,
    topic_file_path: str,
    script_file_path: str,
    folder_name: str,
    timestamp: str,
) -> str:
    """
    Generate intro and outro for the video script and extend the script.

    Args:
        client: OpenRouter client
        model: Model to use for generation (typically Claude)
        topic_file_path: Path to the topic file
        script_file_path: Path to the script file
        folder_name: Project folder path
        timestamp: Timestamp for file naming

    Returns:
        Path to the saved extended script file
    """
    topic_response = read_response_from_file(topic_file_path)
    script_response = read_response_from_file(script_file_path)

    # Truncate the script to the first 200 words for the prompt
    words = script_response.split()
    truncated_script = ' '.join(words[:200]) + ('...' if len(words) > 200 else '')

    prompt = f"""
    Based on the following video topic (title and description):
    {topic_response}

    And this truncated portion of the existing script (first part for context):
    {truncated_script}

    Generate an engaging intro (50-100 words) that hooks the listener and sets a calming, narrative tone suitable for a sleepy time philosophy video.
    Generate a concluding outro (50-100 words) that wraps up the ideas smoothly, encourages reflection, and ends on a relaxing note.

    Ensure both are in a natural, spoken-language style with smooth transitions, vivid but serene descriptions, and no stimulating elements.

    CRITICAL OUTPUT INSTRUCTIONS - FOLLOW THESE EXACTLY OR THE OUTPUT WILL BE INVALID:
    - Output ONLY a valid JSON object with two keys: "intro" and "outro", each with the raw text as a string.
    - The JSON must be pure and parsable—no extra text, wrappers, explanations, or comments before or after it.
    - In the string values, represent line breaks or paragraphs with \\n (escaped newline).
    - Start each text directly with the first sentence (no introductions).
    - End each text directly with the last sentence (no meta-text).
    - If you add ANY extra text outside the JSON, the entire response will be discarded.

    GOOD EXAMPLE (DO THIS):
    {{"intro": "As the night settles in, let's drift into the world of ancient thoughts...", "outro": "And so, as our journey ends, let these ideas lull you to sleep..."}}
    """

    response = query_llm(client, model, prompt)

    # Parse the JSON
    try:
        json_data = parse_llm_json(response, required_keys=["intro", "outro"])
        intro = json_data["intro"].replace('\\n', '\n').strip()
        outro = json_data["outro"].replace('\\n', '\n').strip()
    except ValueError as e:
        raise RuntimeError(f"JSON parsing failed ({str(e)}). No fallback available.")

    # Extend the original script
    extended_script = intro + "\n\n" + script_response + "\n\n" + outro

    # Save the extended script
    filename = f"extended_script_{timestamp}.txt"
    file_path = save_to_file(folder_name, filename, prompt, extended_script)
    return file_path
