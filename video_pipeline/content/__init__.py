"""
Content generation modules for the video pipeline.
"""

from .topic_discovery import get_sub_niches, choose_video_topic
from .research import get_facts_about_topic
from .narrative import reorder_facts_for_story
from .scriptwriter import generate_video_script, generate_intro_outro
from .prompts import generate_video_prompts, generate_thumbnail_prompt

__all__ = [
    "get_sub_niches",
    "choose_video_topic",
    "get_facts_about_topic",
    "reorder_facts_for_story",
    "generate_video_script",
    "generate_intro_outro",
    "generate_video_prompts",
    "generate_thumbnail_prompt",
]
