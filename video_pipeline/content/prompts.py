"""
Visual prompt generation for the video pipeline.

Handles video prompt and thumbnail prompt generation.
"""

from openai import OpenAI

from ..config import VideoConfig, ThumbnailConfig
from ..utils.file_io import save_to_file, read_response_from_file
from ..llm.openrouter import query_llm


def generate_video_prompts(
    client: OpenAI,
    model: str,
    script_file_path: str,
    folder_name: str,
    timestamp: str,
    config: VideoConfig = None,
) -> str:
    """
    Generate video prompts based on the video script.

    Args:
        client: OpenRouter client
        model: Model to use for generation
        script_file_path: Path to the extended script file
        folder_name: Project folder path
        timestamp: Timestamp for file naming
        config: Video configuration (defaults to VideoConfig defaults)

    Returns:
        Path to the saved video prompts file
    """
    if config is None:
        config = VideoConfig()

    script_response = read_response_from_file(script_file_path)
    num_videos = config.num_videos

    prompt = f"""
    Based on the following video script about a philosophy topic:
    {script_response}

    Generate exactly {num_videos} video prompts that visually represent key parts or concepts from this script.

    CRITICAL RULE — ONE VISUAL THEME FOR THE ENTIRE VIDEO:
    First, choose exactly ONE visual theme category from the list below. Then ALL {num_videos} prompts must stay within that single theme. Every clip in the video should feel like it belongs to the same visual world. Vary the specific compositions, angles, and movements between clips for variety, but never leave the chosen theme.

    Visual theme categories (pick ONE for all clips):
    1. FOG / MIST — Dark fog banks, mist rolling through voids, haze drifting in darkness
    2. WATER / FLUID — Dark water surfaces, slow underwater currents, abstract fluid blending
    3. SMOKE / VAPOR — Dark smoke drifting, vapor dissolving into void, abstract dark fumes
    4. SPACE / COSMOS — Deep space nebulae, distant galaxy hazes, cosmic void with faint glows
    5. DARK LANDSCAPE — Shadowy mountain silhouettes, dim twilight horizons, dark still lakes
    6. DRONE OVER LANDSCAPE — Slow aerial view over dark terrain, dim forests, shadowy valleys
    7. DRONE THROUGH CLOUDS — Slow flight through dark cloud banks, moving through thick overcast
    8. DRONE THROUGH SPACE — Slow drift through cosmic dust, floating through dark nebula fields

    STYLE REQUIREMENTS (apply to ALL clips regardless of theme):
    - Sleepy time aesthetic: serene, dreamy, calming
    - ONLY muted, desaturated, dark tones — deep navy, charcoal, soft grey, faded indigo, dim earth tones
    - ABSOLUTELY NO vivid, bright, saturated, or neon colors (no bright blue, red, orange, green, purple, gold, cyan, magenta, or white)
    - Every prompt MUST include the phrase "muted desaturated dark color palette"
    - Use gentle lighting (dim moonlight, deep twilight), relaxing atmospheres
    - Include slow motion phrasing: 'slow motion', 'ultra slow', 'extremely slow', etc.
    - Avoid distinct shapes, lines, squiggles, tendrils, curls, particles, specks, wisps, or fragmented forms
    - Focus on seamless, uniform transitions: gentle fading, blending gradients, diffuse mists
    - No figures, people, animals, or recognizable objects

    EXAMPLE PROMPTS (one per theme):

    FOG/MIST: Infinite dark foggy abyss, thick slow-moving mist with seamless currents, barely perceptible subtle blending, extremely low visibility and contrast, deeply relaxing ambient atmosphere, muted desaturated dark color palette, cinematic soft focus, ultra slow drift.

    WATER/FLUID: Slow motion abstract dark fluid blending in deep water, gentle hypnotic seamless patterns, ultra slow diffusion, dark ambient mood, no objects or figures, muted desaturated dark color palette, cinematic depth of field.

    SMOKE/VAPOR: Ethereal dark vapor drifting in void, soft shadowy fades slowly blending and dissipating, extremely slow movement, abstract non-representational, mysterious calming atmosphere, muted desaturated dark color palette, volumetric fog, 8k.

    SPACE/COSMOS: Endless deep space nebula, slow drifting clouds of cosmic haze, faint distant glows, extremely slow gentle blending motion, abstract ethereal atmosphere, no foreground elements, muted desaturated dark color palette, cinematic ambient style, ultra slow camera drift.

    DARK LANDSCAPE: Shadowy mountain range silhouette under dim twilight, barely visible dark terrain fading into haze, extremely slow ambient drift, no bright elements, muted desaturated dark color palette, serene and meditative.

    DRONE OVER LANDSCAPE: Ultra slow aerial drone shot gliding over dark shadowy forest canopy at deep twilight, minimal detail, fading into distant haze, muted desaturated dark color palette, cinematic ambient, dreamy soft focus.

    DRONE THROUGH CLOUDS: Slow motion drone flight through thick dark overcast clouds, seamless blending of grey and charcoal cloud banks, extremely slow forward drift, muted desaturated dark color palette, ethereal and meditative atmosphere.

    DRONE THROUGH SPACE: Ultra slow drift through dark cosmic dust clouds, faint distant nebula glows, seamless blending of deep void and dim haze, muted desaturated dark color palette, abstract meditative space journey, cinematic ambient.

    FORMAT: First state the chosen theme on its own line like "CHOSEN THEME: FOG/MIST". Then number the prompts 1-{num_videos}. For each, write the theme category in brackets first, then the prompt. Keep each prompt to 1-2 sentences. All prompts must use the same theme bracket.
    Example:
    CHOSEN THEME: FOG/MIST
    1. [FOG/MIST] Infinite dark foggy abyss, thick slow-moving mist...
    2. [FOG/MIST] Dark haze drifting through an endless void...
    3. [FOG/MIST] Barely visible mist currents blending seamlessly...

    Incorporate philosophical themes from the script subtly where possible. Vary compositions between clips (different densities, movements, depths) while staying within the chosen theme.
    """

    response = query_llm(client, model, prompt)

    filename = f"video_prompts_{timestamp}.txt"
    file_path = save_to_file(folder_name, filename, prompt, response)
    return file_path


def generate_thumbnail_prompt(
    client: OpenAI,
    model: str,
    topic_file_path: str,
    folder_name: str,
    timestamp: str,
    config: ThumbnailConfig = None,
) -> str:
    """
    Generate a prompt for the thumbnail image.

    Args:
        client: OpenRouter client
        model: Model to use for generation
        topic_file_path: Path to the topic file
        folder_name: Project folder path
        timestamp: Timestamp for file naming
        config: Thumbnail configuration (defaults to ThumbnailConfig defaults)

    Returns:
        Path to the saved thumbnail prompt file
    """
    if config is None:
        config = ThumbnailConfig()

    topic_response = read_response_from_file(topic_file_path)

    prompt = f"""
    Based on the following video topic (title and description):
    {topic_response}

    Create a single, concise prompt for a text-to-image AI model (like Stable Diffusion) to generate a thumbnail image.
    The thumbnail should be visually appealing for a YouTube video: **do not include any text or text overlay**.
    Ensure it's in a {config.style} to match a calming, sleepy time philosophy video—use dreamy, abstract elements like soft gradients, misty landscapes, or symbolic icons (e.g., ancient scrolls, stars, or serene voids), avoiding anything bright or stimulating.
    Keep the prompt 1-2 sentences long, specifying resolution {config.resolution} and aspect ratio (e.g., square or 16:9).
    """

    response = query_llm(client, model, prompt)

    filename = f"thumbnail_prompt_{timestamp}.txt"
    file_path = save_to_file(folder_name, filename, prompt, response)
    return file_path
