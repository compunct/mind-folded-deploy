"""
Text processing utilities for the video pipeline.

Provides functions for script sanitization and chunking for TTS.
"""

import re
from typing import List


def sanitize_script_for_tts(text: str) -> str:
    """
    Remove stage directions, markdown formatting, and non-spoken text.

    Prepares script text for text-to-speech by removing:
    - *[...]* stage directions
    - Standalone [bracketed] lines
    - Markdown bold and italic formatting
    - Orphaned asterisks
    - Excessive whitespace

    Args:
        text: Raw script text

    Returns:
        Sanitized text suitable for TTS
    """
    # Remove *[...]* stage directions
    text = re.sub(r'\*\[.*?\]\*', '', text)
    # Remove standalone [bracketed] lines (e.g. [End Script])
    text = re.sub(r'^\s*\[.*?\]\s*$', '', text, flags=re.MULTILINE)
    # Strip markdown bold: **text** -> text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Strip markdown italic: *text* -> text
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove orphaned asterisks
    text = re.sub(r'(?<!\w)\*(?!\w)', '', text)
    # Collapse excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)
    return text.strip()


def chunk_script(text: str, max_words: int = 150) -> List[str]:
    """
    Chunk the script text into small pieces, each with up to max_words words.

    Splits on sentence boundaries to maintain natural flow.

    Args:
        text: Script text to chunk
        max_words: Maximum words per chunk (default 150)

    Returns:
        List of text chunks
    """
    # Split into sentences using regex to handle periods better
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = len(sentence.split())
        if current_word_count + words > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = words
        else:
            current_chunk.append(sentence)
            current_word_count += words

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def count_numbered_facts(text: str) -> int:
    """
    Count the number of numbered facts in a text.

    Uses regex to identify lines starting with a number followed by a period.

    Args:
        text: Text containing numbered facts

    Returns:
        Count of numbered facts found
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    numbered_fact_pattern = re.compile(r'^\d+\.\s')
    count = len([line for line in lines if numbered_fact_pattern.match(line)])

    if count == 0:
        # Fallback: count all non-empty lines
        count = len(lines)

    return count


def extract_numbered_prompts(text: str, expected_count: int) -> List[str]:
    """
    Extract numbered prompts from text.

    Args:
        text: Text containing numbered prompts
        expected_count: Expected number of prompts

    Returns:
        List of prompt strings

    Raises:
        ValueError: If the expected number of prompts is not found
    """
    prompts = []
    lines = text.strip().split("\n")

    for line in lines:
        # Check if line starts with a number followed by a period
        if line.strip().startswith(tuple(str(i) + "." for i in range(1, expected_count + 1))):
            # Extract the prompt text after the number
            prompt_text = line.split(".", 1)[1].strip()
            prompts.append(prompt_text)

    if len(prompts) != expected_count:
        raise ValueError(f"Expected {expected_count} prompts, but found {len(prompts)}.")

    return prompts
