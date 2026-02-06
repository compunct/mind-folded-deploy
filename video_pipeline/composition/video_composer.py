"""
Video composition for the video pipeline.

Handles looping, crossfading, and audio overlay for final video assembly.
"""

import os
from typing import List

import numpy as np
from pydub import AudioSegment
from moviepy import VideoFileClip, AudioFileClip
from moviepy.video.VideoClip import VideoClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip


def generate_final_video(
    video_files: List[str],
    audio_file: str,
    folder_name: str,
    timestamp: str,
    speed_factor: float = 0.5,
    transition_duration: float = 1.0,
    output_fps: int = 30,
) -> str:
    """
    Generate a video by looping the videos with transitions until the audio finishes.

    Video clips are slowed down, looped with crossfade transitions to fill the
    audio duration, then the audio track is overlaid.

    Args:
        video_files: List of paths to video clip files
        audio_file: Path to the audio file
        folder_name: Project folder path
        timestamp: Timestamp for file naming
        speed_factor: Speed factor for video clips (0.5 = half speed)
        transition_duration: Duration of crossfade transitions in seconds
        output_fps: Output video frame rate

    Returns:
        Path to the final video file
    """
    # Load audio to get duration
    audio = AudioSegment.from_file(audio_file)
    duration_sec = len(audio) / 1000.0

    # Load the video clips and slow them down, remove original audio
    clips = [VideoFileClip(f).with_speed_scaled(speed_factor).without_audio() for f in video_files]

    # Assume all clips have the same size; use the first one's size
    clip_size = clips[0].size if clips else None

    # Get durations after slowing
    clip_durs = [c.duration for c in clips]
    sum_per_cycle = sum(clip_durs)
    num_per_cycle = len(clips)

    # Find minimal num_cycles such that total duration >= audio duration
    n = 1
    trans_dur = transition_duration
    while True:
        total_sum = n * sum_per_cycle
        num_trans = n * num_per_cycle - 1
        est_dur = total_sum - trans_dur * num_trans if num_trans > 0 else total_sum
        if est_dur >= duration_sec:
            break
        n += 1
    num_cycles = n

    # Create full sequence of clips
    full_sequence = clips * num_cycles

    # Build the composed clip with crossfade transitions
    composed_clips = []
    current_time = 0.0
    for i, clip in enumerate(full_sequence):
        if i > 0:
            # Create fade-in mask
            def opacity_frame(t, td=trans_dur):
                return np.full((clip.h, clip.w), min(t / td, 1.0), dtype=float)

            fade_mask = VideoClip(frame_function=opacity_frame, duration=clip.duration, is_mask=True)
            clip = clip.with_mask(fade_mask)

            # Start overlapping
            clip = clip.with_start(current_time - trans_dur)

        composed_clips.append(clip)
        # Update current_time to the end
        current_time = clip.start + clip.duration

    full_clip = CompositeVideoClip(composed_clips, size=clip_size)

    # Trim the video to match audio duration
    video_clip = full_clip.subclipped(0, duration_sec)

    # Load audio clip
    audio_clip = AudioFileClip(audio_file)

    # Set audio to video
    final_clip = video_clip.with_audio(audio_clip)

    # Output path
    video_filename = f"final_video_{timestamp}.mp4"
    video_path = os.path.join(folder_name, video_filename)

    # Write the video file
    final_clip.write_videofile(video_path, fps=output_fps, codec='libx264', audio_codec='aac')

    print(f"Video saved to {video_path}")
    return video_path
