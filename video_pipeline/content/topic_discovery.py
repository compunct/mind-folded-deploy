"""
Topic discovery and selection for the video pipeline.

Handles sub-niche discovery and video topic selection.
"""

from openai import OpenAI

from ..utils.file_io import save_to_file, read_response_from_file
from ..llm.openrouter import query_llm


def get_sub_niches(
    client: OpenAI,
    model: str,
    folder_name: str,
    timestamp: str,
) -> str:
    """
    Query the model for philosophy sub-niches and save the response.

    Args:
        client: OpenRouter client
        model: Model to use for generation
        folder_name: Project folder path
        timestamp: Timestamp for file naming

    Returns:
        Path to the saved response file
    """
    prompt = """
    So videos on YouTube about philosophy are popular. What are some more specific sub niches that are popular within this? Can you search YouTube and find sub niche videos with this niche that have at least 500,000 views?
    """
    response = query_llm(client, model, prompt)
    filename = f"sub_niches_{timestamp}.txt"
    file_path = save_to_file(folder_name, filename, prompt, response)
    return file_path


def choose_video_topic(
    client: OpenAI,
    model: str,
    sub_niches_file_path: str,
    folder_name: str,
    timestamp: str,
) -> str:
    """
    Choose a video topic based on the sub-niches response.

    Args:
        client: OpenRouter client
        model: Model to use for generation
        sub_niches_file_path: Path to the sub-niches file
        folder_name: Project folder path
        timestamp: Timestamp for file naming

    Returns:
        Path to the saved topic file
    """
    sub_niches_response = read_response_from_file(sub_niches_file_path)
    prompt = f"""
    Based on the following list of philosophy sub-niches and popular videos from YouTube, choose one specific sub-niche that seems particularly popular and engaging for a new YouTube video. Suggest a video topic title and a brief description (1-2 sentences) for it. The title should be SEO optimized and entice the viewer to click on it.
    List of philosophy sub-niches:
    {sub_niches_response}
    """
    response = query_llm(client, model, prompt)
    filename = f"topic_{timestamp}.txt"
    file_path = save_to_file(folder_name, filename, prompt, response)
    return file_path
