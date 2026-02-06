"""
Quality evaluation for the video pipeline.

Evaluates generated content on 8 quality dimensions using GPT-4o,
plus objective video length scoring.
"""

import os
import json
import re
import subprocess
from typing import Dict, Any, Optional

import litellm

from ..utils.file_io import read_response_from_file


def _get_video_duration_seconds(video_path: str) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def _compute_duration_score(
    duration_seconds: float,
    target_min_minutes: int = 60,
    target_max_minutes: int = 120
) -> dict:
    """
    Compute a score for video duration based on target range.

    Returns dict with score (1-10), actual duration, target range, and justification.
    """
    duration_minutes = duration_seconds / 60
    target_min = target_min_minutes
    target_max = target_max_minutes

    # Score calculation:
    # - 10: Within target range
    # - 7-9: Within 20% of target range
    # - 4-6: Within 50% of target range
    # - 1-3: More than 50% off

    if target_min <= duration_minutes <= target_max:
        score = 10
        justification = f"Video length ({duration_minutes:.1f} min) is within target range ({target_min}-{target_max} min)."
    elif duration_minutes < target_min:
        shortfall_pct = (target_min - duration_minutes) / target_min
        if shortfall_pct <= 0.2:
            score = 8
        elif shortfall_pct <= 0.5:
            score = 5
        else:
            score = max(1, int(3 - shortfall_pct * 2))
        justification = f"Video length ({duration_minutes:.1f} min) is {shortfall_pct*100:.0f}% below target minimum ({target_min} min)."
    else:  # duration_minutes > target_max
        overage_pct = (duration_minutes - target_max) / target_max
        if overage_pct <= 0.2:
            score = 8
        elif overage_pct <= 0.5:
            score = 6
        else:
            score = max(4, int(7 - overage_pct * 2))
        justification = f"Video length ({duration_minutes:.1f} min) is {overage_pct*100:.0f}% above target maximum ({target_max} min)."

    return {
        "score": score,
        "justification": justification,
        "actual_minutes": round(duration_minutes, 1),
        "target_range_minutes": [target_min, target_max]
    }


def run_evaluation(
    folder_name: str,
    topic_file_path: str,
    extended_script_file_path: str,
    video_prompts_file_path: str,
    api_key: str,
    model: str = "openai/gpt-4o",
    video_file_path: Optional[str] = None,
    target_duration_minutes: tuple[int, int] = (60, 120),
) -> str:
    """
    Run the quality evaluation on all text outputs.

    Evaluates 8 subjective dimensions via LLM:
    - Topic: niche_originality, sleepy_time_fit
    - Script: hook_strength, narrative_flow, tone_consistency, fact_integration
    - Visuals: visual_prompt_quality
    - Overall: rewatch_sleep_value

    Plus 1 objective dimension computed directly:
    - video_length: How close the video is to target duration

    Args:
        folder_name: Project folder path
        topic_file_path: Path to the topic file
        extended_script_file_path: Path to the extended script file
        video_prompts_file_path: Path to the video prompts file
        api_key: OpenRouter API key
        model: Model to use for evaluation
        video_file_path: Path to the final video file (for duration scoring)
        target_duration_minutes: (min, max) target video length in minutes

    Returns:
        Path to the evaluation JSON file
    """
    print("\n" + "=" * 60)
    print("Running Quality Evaluation")
    print("=" * 60 + "\n")

    # Read the artifacts
    topic_text = read_response_from_file(topic_file_path)
    script_text = read_response_from_file(extended_script_file_path)
    video_prompts_text = read_response_from_file(video_prompts_file_path)

    # Truncate script for context window
    script_preview = script_text[:3000] + "\n\n[... script continues ...]" if len(script_text) > 3000 else script_text

    evaluation_prompt = f"""Evaluate ALL text outputs from this video pipeline run. Score each dimension 1-10 with a concrete justification.

The video topic is:
{topic_text}

Script preview (full script is {len(script_text.split())} words):
{script_preview}

Video prompts:
{video_prompts_text}

RUBRIC (score each dimension):

TOPIC:
- niche_originality: How fresh/unexplored is this angle? (1=overdone, 10=never seen before)
- sleepy_time_fit: Suitability for calming, meditative format (1=stimulating/jarring, 10=perfect lullaby material)

SCRIPT:
- hook_strength: Do the first 2-3 sentences create genuine curiosity? (1=generic opener, 10=impossible to stop listening)
- narrative_flow: Natural transitions, clear arc, not just a list of facts (1=disconnected list, 10=seamless journey)
- tone_consistency: Calming tone maintained, no jarring shifts or cliches (1=inconsistent/jarring, 10=perfectly sustained)
- fact_integration: Are facts woven into narrative or just dropped in? (1=dry recitation, 10=invisible integration)

VISUALS:
- visual_prompt_quality: Specific, evocative, aligned with sleepy-time aesthetic (1=vague/generic, 10=cinematically precise)

OVERALL:
- rewatch_sleep_value: Would someone fall asleep to this and come back? (1=forgettable, 10=new bedtime ritual)

Output ONLY a valid JSON object with this exact structure:
{{
    "scores": {{
        "niche_originality": {{"score": N, "justification": "..."}},
        "sleepy_time_fit": {{"score": N, "justification": "..."}},
        "hook_strength": {{"score": N, "justification": "..."}},
        "narrative_flow": {{"score": N, "justification": "..."}},
        "tone_consistency": {{"score": N, "justification": "..."}},
        "fact_integration": {{"score": N, "justification": "..."}},
        "visual_prompt_quality": {{"score": N, "justification": "..."}},
        "rewatch_sleep_value": {{"score": N, "justification": "..."}}
    }},
    "top_3_improvements": ["...", "...", "..."],
    "what_worked_well": ["...", "...", "..."]
}}

Be specific in justifications. Reference actual content from the outputs."""

    try:
        response = litellm.completion(
            model=f"openrouter/{model}",
            messages=[{"role": "user", "content": evaluation_prompt}],
            max_tokens=2000,
            api_key=api_key,
        )
        raw_output = response.choices[0].message.content.strip()

        # Parse JSON
        try:
            eval_data = json.loads(raw_output)
        except json.JSONDecodeError:
            # Regex fallback: find JSON within text
            json_match = re.search(r'\{[\s\S]*\}', raw_output)
            if json_match:
                eval_data = json.loads(json_match.group())
            else:
                eval_data = {"raw_output": raw_output, "parse_error": "Could not extract JSON"}

    except Exception as e:
        print(f"Evaluation failed: {e}")
        eval_data = {"error": str(e)}

    # Compute video length score if video file is provided
    if video_file_path and os.path.exists(video_file_path):
        duration = _get_video_duration_seconds(video_file_path)
        if duration:
            duration_score = _compute_duration_score(
                duration,
                target_min_minutes=target_duration_minutes[0],
                target_max_minutes=target_duration_minutes[1]
            )
            if "scores" in eval_data:
                eval_data["scores"]["video_length"] = duration_score
            else:
                eval_data["video_length"] = duration_score

    # Merge video quality summary if it exists
    quality_summary_path = os.path.join(folder_name, "video_quality_summary.json")
    if os.path.exists(quality_summary_path):
        try:
            with open(quality_summary_path, "r") as qf:
                eval_data["video_quality"] = json.load(qf)
        except Exception as qe:
            print(f"Warning: Could not merge video quality data: {qe}")

    # Save evaluation
    eval_path = os.path.join(folder_name, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    print(f"Evaluation saved to {eval_path}")

    # Print summary
    if "scores" in eval_data:
        print("\nEvaluation Scores:")
        for dim, data in eval_data["scores"].items():
            score = data.get("score", "?") if isinstance(data, dict) else data
            print(f"  {dim}: {score}/10")

    return eval_path
