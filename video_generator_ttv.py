#!/usr/bin/env python3
"""
Video Pipeline Orchestrator

Automated YouTube video generation pipeline for sleepy-time philosophy content.
Takes a niche topic, researches it, writes a script, generates AI voiceover + video clips,
and composes a final video with thumbnail.

Usage:
    source venv/bin/activate
    python video_generator_ttv.py

Requires .env with OPENROUTER_API_KEY and REPLICATE_API_KEY.
"""

import os

from video_pipeline.config import PipelineConfig
from video_pipeline.clients import PipelineClients
from video_pipeline.utils.file_io import (
    generate_timestamp,
    move_old_projects,
    ProjectFolder,
    get_audio_duration,
    calculate_required_clips,
)

from video_pipeline.content.topic_discovery import get_sub_niches, choose_video_topic
from video_pipeline.content.research import get_facts_about_topic
from video_pipeline.content.narrative import reorder_facts_for_story
from video_pipeline.content.scriptwriter import generate_video_script, generate_intro_outro
from video_pipeline.content.prompts import generate_video_prompts, generate_thumbnail_prompt

from video_pipeline.media.thumbnail import generate_thumbnail
from video_pipeline.media.tts import generate_audio_from_script
from video_pipeline.media.video_generator import generate_videos

from video_pipeline.composition.video_composer import generate_final_video
from video_pipeline.evaluation.evaluator import run_evaluation


def main():
    """Main function to orchestrate the video generation workflow."""
    # Load configuration and initialize clients
    config = PipelineConfig.from_defaults()
    clients = PipelineClients.from_env()

    # Archive previous project folders
    move_old_projects()

    # Create new project folder
    timestamp = generate_timestamp()
    folder_name = f"project_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting Video Pipeline")
    print(f"Project folder: {folder_name}")
    print(f"{'='*60}\n")

    # Step 1: Discover sub-niches
    print(f"Step 1: Discovering philosophy sub-niches... [model: {config.models.sub_niches}]")
    sub_niches_file = get_sub_niches(
        clients.openrouter, config.models.sub_niches, folder_name, timestamp
    )

    # Step 2: Choose video topic
    print(f"Step 2: Choosing video topic... [model: {config.models.topic}]")
    topic_file = choose_video_topic(
        clients.openrouter, config.models.topic, sub_niches_file, folder_name, timestamp
    )

    # Step 2.5: Generate thumbnail
    print(f"Step 2.5: Generating thumbnail prompt... [model: {config.models.thumbnail_prompt}]")
    thumbnail_prompt_file = generate_thumbnail_prompt(
        clients.openrouter, config.models.thumbnail_prompt, topic_file,
        folder_name, timestamp, config.thumbnail
    )
    print(f"Step 2.5: Generating thumbnail image... [model: stability-ai/stable-diffusion-3]")
    thumbnail_file = generate_thumbnail(
        clients.replicate_api_key, thumbnail_prompt_file,
        folder_name, timestamp, config.thumbnail
    )

    # Step 3: Gather facts
    print(f"Step 3: Gathering facts about topic... [model: {config.models.facts}]")
    facts_file = get_facts_about_topic(
        clients.openrouter, config.models.facts, topic_file,
        folder_name, timestamp, config.facts
    )

    # Step 4: Reorder facts for narrative
    print(f"Step 4: Reordering facts for storytelling... [model: {config.models.reorder}]")
    reordered_facts_file = reorder_facts_for_story(
        clients.openrouter, config.models.reorder, topic_file,
        facts_file, folder_name, timestamp
    )

    # Step 5: Generate script
    print(f"Step 5: Generating video script... [model: {config.models.script}]")
    script_file = generate_video_script(
        clients.openrouter, config.models.script, topic_file,
        reordered_facts_file, folder_name, timestamp, config.script
    )

    # Step 5.5: Generate intro and outro
    print(f"Step 5.5: Generating intro and outro... [model: {config.models.script}]")
    extended_script_file = generate_intro_outro(
        clients.openrouter, config.models.script, topic_file,
        script_file, folder_name, timestamp
    )

    # Step 6: Generate audio
    tts_model = "jaaari/kokoro-82m" if config.tts_provider == "kokoro" else "minimax/speech-02-hd"
    print(f"Step 6: Generating audio from script... [model: {tts_model}]")
    audio_file = generate_audio_from_script(
        extended_script_file, folder_name, timestamp,
        clients.replicate_api_key, config.tts_provider,
        config.kokoro_tts, config.minimax_tts
    )

    # Step 7: Generate video prompts
    print(f"Step 7: Generating video prompts... [model: {config.models.video_prompts}]")
    video_prompts_file = generate_video_prompts(
        clients.openrouter, config.models.video_prompts,
        extended_script_file, folder_name, timestamp, config.video
    )

    # Step 8: Generate video clips
    if config.video.backend == "local":
        video_backend_label = f"local video-api ({config.local_video.api_url})"
    else:
        video_backend_label = "bytedance/seedance-1-pro-fast (Replicate)"
    print(f"Step 8: Generating video clips... [backend: {video_backend_label}, quality_gate: {config.quality_gate.eval_model}]")
    video_files = generate_videos(
        clients.replicate_api_key, video_prompts_file, folder_name, timestamp,
        clients.openrouter_api_key, config.video, config.quality_gate,
        local_video_config=config.local_video if config.video.backend == "local" else None,
    )

    # Step 9: Compose final video
    print("Step 9: Composing final video... [moviepy]")
    video_file = generate_final_video(
        video_files, audio_file, folder_name, timestamp,
        speed_factor=config.video.speed_factor,
        transition_duration=config.video.transition_duration,
    )

    # Step 10: Run quality evaluation
    print(f"Step 10: Running quality evaluation... [model: {config.models.evaluator}]")
    eval_file = run_evaluation(
        folder_name, topic_file, extended_script_file, video_prompts_file,
        clients.openrouter_api_key, config.models.evaluator,
        video_file_path=video_file,
        target_duration_minutes=config.script.target_duration_minutes
    )

    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"Project folder: {folder_name}")
    print(f"Final video: {video_file}")
    print(f"Evaluation: {eval_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
