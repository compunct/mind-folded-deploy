"""
Video generation and quality gate for the video pipeline.

Handles text-to-video generation with quality evaluation and retry logic.
"""

import os
import json
import re
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional

import requests
import replicate
import litellm
from PIL import Image
from moviepy import VideoFileClip

from ..config import VideoConfig, QualityGateConfig, LocalVideoConfig
from ..utils.file_io import save_to_file, read_response_from_file
from ..utils.text_processing import extract_numbered_prompts


def extract_frames_as_base64(
    video_path: str,
    folder_name: str,
    clip_index: int,
    attempt: int,
    num_frames: int = None,
) -> List[Dict[str, str]]:
    """
    Extract frames from a video clip at evenly spaced intervals and return as base64 JPEG.

    Args:
        video_path: Path to the video file
        folder_name: Project folder to save extracted frame images
        clip_index: Index of the clip (for naming)
        attempt: Attempt number (for naming)
        num_frames: Number of frames to extract (default 3)

    Returns:
        List of dicts with "base64" and "path" keys
    """
    if num_frames is None:
        num_frames = 3

    clip = VideoFileClip(video_path)
    frames = []
    # Generate evenly spaced percentages based on num_frames
    percentages = [(i + 1) / (num_frames + 1) for i in range(num_frames)]

    try:
        for pct in percentages:
            t = clip.duration * pct
            frame_array = clip.get_frame(t)
            img = Image.fromarray(frame_array)

            # Save frame to disk
            pct_label = int(pct * 100)
            frame_filename = f"frame_{clip_index}_attempt{attempt}_{pct_label}pct.jpg"
            frame_path = os.path.join(folder_name, frame_filename)
            img.save(frame_path, "JPEG", quality=85)

            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            frames.append({"base64": b64, "path": frame_path})
    finally:
        clip.close()

    return frames


def evaluate_video_clip(
    frames_b64_list: List[Dict[str, str]],
    original_prompt: str,
    api_key: str,
    config: QualityGateConfig = None,
) -> Dict[str, Any]:
    """
    Evaluate a video clip's quality using vision LLM on extracted frames.

    Args:
        frames_b64_list: List of dicts with "base64" keys (from extract_frames_as_base64)
        original_prompt: The text-to-video prompt used to generate the clip
        api_key: OpenRouter API key
        config: Quality gate configuration

    Returns:
        Dict with keys: visual_coherence, artifact_absence, aesthetic_fit,
        motion_smoothness (each 1-10), composite_score, verdict, rejection_reason
    """
    if config is None:
        config = QualityGateConfig()

    num_frames = len(frames_b64_list)
    image_content = []
    for i, frame in enumerate(frames_b64_list):
        image_content.append({
            "type": "text",
            "text": f"Frame {i+1} of {num_frames}:"
        })
        image_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame['base64']}"
            }
        })

    rubric_text = f"""You are evaluating frames extracted from an AI-generated video clip.

Original prompt: "{original_prompt}"

Score each dimension 1-10:
- visual_coherence: Do the frames look like a coherent scene? No warped/melted objects, consistent perspective. (1=nonsensical, 10=photorealistic consistency)
- artifact_absence: Free from visual glitches, banding, color blotches, face distortions. (1=heavy artifacts, 10=clean)
- aesthetic_fit: Matches the sleepy-time aesthetic — serene, dreamy, calming with ONLY muted/desaturated/dark tones. Any vivid, bright, saturated, or neon colors = automatic 1. (1=vivid colors or jarring, 10=perfectly dark and meditative)
- motion_smoothness: Do sequential frames suggest smooth motion? No teleporting objects or flickering. (1=chaotic, 10=silk-smooth)

Output ONLY valid JSON:
{{"visual_coherence": N, "artifact_absence": N, "aesthetic_fit": N, "motion_smoothness": N, "rejection_reason": "brief reason if any dimension <= 3, else null"}}"""

    messages = [
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": rubric_text}]
        }
    ]

    default_scores = {
        "visual_coherence": 5,
        "artifact_absence": 5,
        "aesthetic_fit": 5,
        "motion_smoothness": 5,
        "rejection_reason": None,
    }

    try:
        response = litellm.completion(
            model=config.eval_model,
            messages=messages,
            max_tokens=300,
            api_key=api_key,
        )
        raw_text = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            scores = json.loads(raw_text)
        except json.JSONDecodeError:
            # Regex fallback: find JSON within text
            json_match = re.search(r'\{[\s\S]*\}', raw_text)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                print(f"  [Quality Gate] Could not parse evaluation response, auto-accepting")
                scores = default_scores

    except Exception as e:
        print(f"  [Quality Gate] Evaluation call failed ({e}), auto-accepting")
        scores = default_scores

    # Ensure all keys exist with valid values
    for key in ["visual_coherence", "artifact_absence", "aesthetic_fit", "motion_smoothness"]:
        if key not in scores or not isinstance(scores[key], (int, float)):
            scores[key] = 5

    # Calculate weighted composite
    composite = (
        scores["visual_coherence"] * 0.3
        + scores["artifact_absence"] * 0.3
        + scores["aesthetic_fit"] * 0.2
        + scores["motion_smoothness"] * 0.2
    )
    scores["composite_score"] = round(composite, 2)

    # Determine verdict
    threshold = config.threshold
    if composite >= threshold:
        scores["verdict"] = "accept"
    else:
        scores["verdict"] = "reject"
        if not scores.get("rejection_reason"):
            worst_dim = min(
                ["visual_coherence", "artifact_absence", "aesthetic_fit", "motion_smoothness"],
                key=lambda k: scores[k]
            )
            scores["rejection_reason"] = f"Low {worst_dim} ({scores[worst_dim]})"

    return scores


def _build_quality_summary(quality_log: List[Dict], total_requested: int) -> Dict[str, Any]:
    """
    Aggregate per-clip evaluation log entries into a summary dict.

    Args:
        quality_log: List of per-evaluation log entry dicts
        total_requested: Number of clips originally requested

    Returns:
        Summary dict with aggregate stats and per-clip details
    """
    total_attempts = len(quality_log)
    total_rejections = sum(1 for entry in quality_log if entry.get("verdict") == "reject")
    composites = [entry["composite_score"] for entry in quality_log if "composite_score" in entry]
    avg_composite = round(sum(composites) / len(composites), 2) if composites else 0.0

    # Group by clip index
    per_clip = {}
    for entry in quality_log:
        idx = entry.get("clip_index", 0)
        if idx not in per_clip:
            per_clip[idx] = {"attempts": [], "accepted_attempt": None}
        per_clip[idx]["attempts"].append({
            "attempt": entry.get("attempt", 0),
            "composite_score": entry.get("composite_score", 0),
            "verdict": entry.get("verdict", "unknown"),
            "rejection_reason": entry.get("rejection_reason"),
            "scores": {
                k: entry.get(k) for k in
                ["visual_coherence", "artifact_absence", "aesthetic_fit", "motion_smoothness"]
            },
        })
        if entry.get("accepted", False):
            per_clip[idx]["accepted_attempt"] = entry.get("attempt", 0)

    return {
        "total_clips_requested": total_requested,
        "total_generation_attempts": total_attempts,
        "total_rejections": total_rejections,
        "rejection_rate": round(total_rejections / total_attempts, 2) if total_attempts else 0.0,
        "avg_composite_score": avg_composite,
        "per_clip": per_clip,
    }


def _generate_video_local(prompt: str, config: VideoConfig,
                          local_config: LocalVideoConfig) -> bytes:
    """
    Generate a video clip via the local video-api server.

    Posts a generation request and downloads the resulting MP4.

    Args:
        prompt: Text prompt for video generation
        config: Video configuration (resolution, fps, duration)
        local_config: Local API configuration (url, steps, guidance)

    Returns:
        Raw bytes of the generated MP4 file

    Raises:
        requests.exceptions.RequestException: On network/API errors
    """
    # Map resolution string to pixel dimensions
    resolution_map = {
        "480p": (480, 832),
        "720p": (720, 1280),
        "1080p": (1080, 1920),
    }
    height, width = resolution_map.get(config.resolution, (720, 1280))

    # Calculate num_frames: (num_frames - 1) % 4 == 0
    raw_frames = config.duration_seconds * config.fps
    num_frames = ((raw_frames - 1) // 4) * 4 + 1

    payload = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "fps": config.fps,
        "num_inference_steps": local_config.num_inference_steps,
        "guidance_scale": local_config.guidance_scale,
    }

    api_url = local_config.api_url.rstrip("/")

    # POST generation request
    resp = requests.post(
        f"{api_url}/generate",
        json=payload,
        timeout=local_config.timeout_seconds,
    )
    resp.raise_for_status()
    result = resp.json()

    # Download the video file
    video_url = result["video_url"]
    video_resp = requests.get(
        f"{api_url}{video_url}",
        timeout=60,
    )
    video_resp.raise_for_status()
    return video_resp.content


def _generate_video_replicate(client, prompt: str, config: VideoConfig) -> bytes:
    """
    Generate a video clip via Replicate's Seedance model.

    Args:
        client: Replicate client instance
        prompt: Text prompt for video generation
        config: Video configuration (resolution, fps, duration)

    Returns:
        Raw bytes of the generated MP4 file
    """
    output = client.run(
        "bytedance/seedance-1-pro-fast",
        input={
            "fps": config.fps,
            "prompt": prompt,
            "duration": config.duration_seconds,
            "resolution": config.resolution,
            "aspect_ratio": config.aspect_ratio,
            "camera_fixed": False,
        }
    )
    video_url = output.url
    response = requests.get(video_url)
    response.raise_for_status()
    return response.content


def generate_videos(
    replicate_api_key: str,
    video_prompts_file_path: str,
    folder_name: str,
    timestamp: str,
    api_key: str = None,
    config: VideoConfig = None,
    quality_config: QualityGateConfig = None,
    local_video_config: LocalVideoConfig = None,
) -> List[str]:
    """
    Generate videos based on the video prompts using the configured backend.

    Supports two backends:
    - "local": POST to a local video-api server (WAN 2.2, etc.)
    - "replicate": Replicate's Seedance model (original behavior)

    If quality gate is enabled, each clip is evaluated after generation.
    Clips below threshold are regenerated up to max_retries times.

    Args:
        replicate_api_key: Replicate API key
        video_prompts_file_path: Path to the video prompts file
        folder_name: Project folder path
        timestamp: Timestamp for file naming
        api_key: OpenRouter API key (for quality evaluation)
        config: Video configuration
        quality_config: Quality gate configuration
        local_video_config: Local video API configuration (required if backend="local")

    Returns:
        List of paths to generated video files
    """
    if config is None:
        config = VideoConfig()
    if quality_config is None:
        quality_config = QualityGateConfig()

    use_local = config.backend == "local"

    if use_local and local_video_config is None:
        local_video_config = LocalVideoConfig()

    prompts_response = read_response_from_file(video_prompts_file_path)

    # Extract the prompts
    prompts = extract_numbered_prompts(prompts_response, config.num_videos)

    # Only init Replicate client when needed
    client = None
    if not use_local:
        client = replicate.Client(api_token=replicate_api_key)

    quality_gate_enabled = quality_config.enabled
    max_retries = quality_config.max_retries
    quality_log = []
    quality_log_path = os.path.join(folder_name, "video_quality_log.jsonl")

    video_paths = []
    for idx, original_prompt in enumerate(prompts):
        final_path = os.path.join(folder_name, f"video_{idx + 1}_{timestamp}.mp4")
        # Track best attempt: [current_path, composite_score, attempt_num, rejection_reason]
        # current_path is updated after rename to rejected path
        best_attempt = None

        clip_skipped = False
        clip_accepted = False
        for attempt in range(max_retries + 1):
            # Modify prompt on retry with rejection feedback
            if attempt == 0:
                prompt = original_prompt
            else:
                prev_reason = best_attempt[3] if best_attempt and len(best_attempt) > 3 else "low quality"
                prompt = f"{original_prompt} [IMPORTANT: Avoid: {prev_reason}]"

            print(f"Generating video {idx + 1}/{len(prompts)} (attempt {attempt + 1}/{max_retries + 1})")

            try:
                if use_local:
                    video_bytes = _generate_video_local(prompt, config, local_video_config)
                else:
                    video_bytes = _generate_video_replicate(client, prompt, config)
            except requests.exceptions.RequestException as e:
                if use_local:
                    print(f"  [Local API] Request failed: {e}")
                    raise
                else:
                    raise
            except Exception as e:
                error_msg = str(e)
                if not use_local and ("flagged as sensitive" in error_msg or "content moderation" in error_msg.lower()):
                    print(f"  [Content Filter] Prompt flagged by moderation, skipping clip {idx + 1}")
                    print(f"  [Content Filter] Error: {error_msg[:100]}...")
                    clip_skipped = True
                    break
                elif not use_local and ("spend limit" in error_msg.lower() or "402" in error_msg):
                    print(f"  [Spend Limit] Monthly spend limit reached. Stopping video generation.")
                    print(f"  [Spend Limit] Generated {len(video_paths)} clips before limit was hit.")
                    # Break both loops by returning early with what we have
                    if quality_gate_enabled and quality_log:
                        summary = _build_quality_summary(quality_log, len(prompts))
                        summary_path = os.path.join(folder_name, "video_quality_summary.json")
                        with open(summary_path, "w") as f:
                            json.dump(summary, f, indent=2)
                        print(f"Video quality summary saved: {summary_path}")
                    details = f"Generated {len(video_paths)} videos (spend limit reached):\n" + "\n".join(video_paths)
                    details_filename = f"videos_details_{timestamp}.txt"
                    save_to_file(folder_name, details_filename, "Generated videos from prompts.", details)
                    return video_paths
                else:
                    # Re-raise other errors
                    raise

            # Save with attempt suffix (will rename to final path if accepted)
            attempt_path = os.path.join(
                folder_name, f"video_{idx + 1}_attempt{attempt}_{timestamp}.mp4"
            )
            with open(attempt_path, "wb") as f:
                f.write(video_bytes)

            # If quality gate is disabled, accept immediately
            if not quality_gate_enabled:
                os.rename(attempt_path, final_path)
                video_paths.append(final_path)
                print(f"Video saved: {final_path} (quality gate disabled)")
                break

            # Extract frames and evaluate
            print(f"  [Quality Gate] Extracting frames and evaluating...")
            frame_paths = []
            try:
                frames = extract_frames_as_base64(
                    attempt_path, folder_name, idx + 1, attempt,
                    num_frames=quality_config.num_frames
                )
                frame_paths = [fr["path"] for fr in frames]
                scores = evaluate_video_clip(frames, original_prompt, api_key, quality_config)
            except Exception as e:
                print(f"  [Quality Gate] Frame extraction/eval failed ({e}), auto-accepting")
                scores = {
                    "visual_coherence": 5, "artifact_absence": 5,
                    "aesthetic_fit": 5, "motion_smoothness": 5,
                    "composite_score": 5.0, "verdict": "accept",
                    "rejection_reason": None,
                }

            # Log entry (written immediately)
            log_entry = {
                "clip_index": idx + 1,
                "attempt": attempt,
                "prompt_used": prompt,
                "original_prompt": original_prompt,
                **scores,
                "accepted": False,
                "video_path": attempt_path,
                "frame_paths": frame_paths,
            }

            # Track best-scoring attempt (path will be updated after rename if rejected)
            composite = scores.get("composite_score", 0)
            rejection_reason = scores.get("rejection_reason")
            if best_attempt is None or composite > best_attempt[1]:
                best_attempt = [attempt_path, composite, attempt, rejection_reason]  # Use list for mutability

            if scores["verdict"] == "accept":
                log_entry["accepted"] = True
                quality_log.append(log_entry)
                with open(quality_log_path, "a") as f:
                    f.write(json.dumps(log_entry, default=str) + "\n")

                # Rename to final path
                os.rename(attempt_path, final_path)
                video_paths.append(final_path)
                print(f"  [Quality Gate] ACCEPTED (composite: {composite}) -> {final_path}")
                clip_accepted = True
                break
            else:
                quality_log.append(log_entry)
                with open(quality_log_path, "a") as f:
                    f.write(json.dumps(log_entry, default=str) + "\n")

                # Rename to rejected
                rejected_path = os.path.join(
                    folder_name, f"video_{idx + 1}_rejected_attempt{attempt}_{timestamp}.mp4"
                )
                os.rename(attempt_path, rejected_path)
                # Update best_attempt path if this was the best so far
                if best_attempt and best_attempt[2] == attempt:
                    best_attempt[0] = rejected_path
                print(f"  [Quality Gate] REJECTED (composite: {composite}, reason: {rejection_reason})")

        # If clip was skipped due to content filter or accepted, continue to next clip
        if clip_skipped or clip_accepted:
            continue

        # All retries exhausted — use best-scoring attempt
        if best_attempt:
            best_path, best_score, best_attempt_num, _ = best_attempt
            # best_path is already updated to the rejected path after rename
            if os.path.exists(best_path):
                os.rename(best_path, final_path)
                video_paths.append(final_path)
                print(f"  [Quality Gate] All retries exhausted. Using best attempt {best_attempt_num} (composite: {best_score}) -> {final_path}")
            else:
                print(f"  [Quality Gate] ERROR: Best attempt file not found: {best_path}")

    # Write quality summary
    if quality_gate_enabled and quality_log:
        summary = _build_quality_summary(quality_log, len(prompts))
        summary_path = os.path.join(folder_name, "video_quality_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Video quality summary saved: {summary_path}")

    # Save details to a file
    backend_label = "local video-api" if use_local else "Replicate Seedance"
    details = f"Generated {len(video_paths)} videos using {backend_label}:\n" + "\n".join(video_paths)
    details_filename = f"videos_details_{timestamp}.txt"
    save_to_file(folder_name, details_filename, "Generated videos from prompts.", details)

    return video_paths
