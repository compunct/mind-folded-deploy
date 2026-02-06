"""
Thumbnail generation for the video pipeline.

Handles text-to-image thumbnail generation via Stable Diffusion.
"""

import os
import requests
import replicate

from ..config import ThumbnailConfig
from ..utils.file_io import read_response_from_file


def generate_thumbnail(
    replicate_api_key: str,
    thumbnail_prompt_file_path: str,
    folder_name: str,
    timestamp: str,
    config: ThumbnailConfig = None,
) -> str:
    """
    Generate a thumbnail image using Replicate's Stable Diffusion.

    Args:
        replicate_api_key: Replicate API key
        thumbnail_prompt_file_path: Path to the thumbnail prompt file
        folder_name: Project folder path
        timestamp: Timestamp for file naming
        config: Thumbnail configuration

    Returns:
        Path to the generated thumbnail image

    Raises:
        RuntimeError: If thumbnail generation fails
    """
    if config is None:
        config = ThumbnailConfig()

    prompt_response = read_response_from_file(thumbnail_prompt_file_path)

    client = replicate.Client(api_token=replicate_api_key)

    # Parse resolution
    width, height = map(int, config.resolution.split('x'))

    try:
        output = client.run(
            "stability-ai/stable-diffusion-3",
            input={
                "prompt": prompt_response,
                "width": width,
                "height": height,
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            }
        )

        # Output is a list with the image URL
        image_url = output[0]

        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()

        filename = f"thumbnail_{timestamp}.png"
        thumbnail_path = os.path.join(folder_name, filename)
        with open(thumbnail_path, "wb") as f:
            f.write(response.content)

        print(f"Thumbnail saved to {thumbnail_path}")
        return thumbnail_path

    except Exception as e:
        raise RuntimeError(f"Thumbnail generation failed: {str(e)}")
