"""
File I/O utilities for the video pipeline.

Provides standardized file operations for saving and reading pipeline artifacts.
"""

import os
import json
import math
import datetime
import shutil
from typing import Optional, Any

from pydub import AudioSegment


def generate_timestamp() -> str:
    """Generate a timestamp string for folder and file naming."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_to_file(folder_name: str, filename: str, prompt: str, response: str) -> str:
    """
    Save the prompt and response to a file in the specified folder.

    Args:
        folder_name: Target folder path
        filename: Name of the file to create
        prompt: The prompt used to generate the response
        response: The response to save

    Returns:
        Full path to the saved file
    """
    file_path = os.path.join(folder_name, filename)
    with open(file_path, "w") as file:
        file.write(f"Prompt:\n{prompt}\n\n")
        file.write(f"Response:\n{response}\n")
    print(f"File saved to {file_path}")
    return file_path


def read_response_from_file(file_path: str) -> str:
    """
    Read the response part from a saved file.

    Args:
        file_path: Path to the file to read

    Returns:
        The response portion of the file content

    Raises:
        ValueError: If the response cannot be parsed from the file
    """
    with open(file_path, "r") as file:
        file_content = file.read()
    try:
        response_part = file_content.split("Response:\n")[1].strip()
    except IndexError:
        raise ValueError("Could not parse response from the file.")
    return response_part


class ProjectFolder:
    """
    Manages a project folder for pipeline artifacts.

    Provides methods for saving and reading various artifact types
    with consistent naming conventions.
    """

    def __init__(self, folder_path: str, timestamp: Optional[str] = None):
        """
        Initialize a project folder.

        Args:
            folder_path: Path to the project folder
            timestamp: Optional timestamp for file naming (auto-generated if not provided)
        """
        self.folder_path = folder_path
        self.timestamp = timestamp or generate_timestamp()
        os.makedirs(folder_path, exist_ok=True)

    @classmethod
    def create(cls, base_path: str = ".", timestamp: Optional[str] = None) -> "ProjectFolder":
        """
        Create a new project folder with timestamp-based naming.

        Args:
            base_path: Base directory to create the folder in
            timestamp: Optional timestamp (auto-generated if not provided)

        Returns:
            New ProjectFolder instance
        """
        ts = timestamp or generate_timestamp()
        folder_path = os.path.join(base_path, f"project_{ts}")
        return cls(folder_path, ts)

    def save_artifact(
        self,
        name: str,
        content: str,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Save a text artifact to the project folder.

        Args:
            name: Base name for the artifact (e.g., "script", "facts")
            content: The content to save
            prompt: Optional prompt that generated the content

        Returns:
            Full path to the saved file
        """
        filename = f"{name}_{self.timestamp}.txt"
        if prompt:
            return save_to_file(self.folder_path, filename, prompt, content)
        else:
            file_path = os.path.join(self.folder_path, filename)
            with open(file_path, "w") as f:
                f.write(content)
            print(f"File saved to {file_path}")
            return file_path

    def read_artifact(self, name: str) -> str:
        """
        Read an artifact's response content.

        Args:
            name: Base name for the artifact (e.g., "script", "facts")

        Returns:
            The response content from the artifact file
        """
        filename = f"{name}_{self.timestamp}.txt"
        file_path = os.path.join(self.folder_path, filename)
        return read_response_from_file(file_path)

    def save_json(self, name: str, data: Any) -> str:
        """
        Save JSON data to the project folder.

        Args:
            name: Base name for the file (e.g., "evaluation")
            data: Data to serialize as JSON

        Returns:
            Full path to the saved file
        """
        filename = f"{name}.json"
        file_path = os.path.join(self.folder_path, filename)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"JSON saved to {file_path}")
        return file_path

    def read_json(self, name: str) -> Any:
        """
        Read JSON data from the project folder.

        Args:
            name: Base name for the file (e.g., "evaluation")

        Returns:
            Parsed JSON data
        """
        filename = f"{name}.json"
        file_path = os.path.join(self.folder_path, filename)
        with open(file_path, "r") as f:
            return json.load(f)

    def append_jsonl(self, name: str, entry: dict) -> str:
        """
        Append a JSON entry to a JSONL file.

        Args:
            name: Base name for the file (e.g., "video_quality_log")
            entry: Dictionary to append as a JSON line

        Returns:
            Full path to the file
        """
        filename = f"{name}.jsonl"
        file_path = os.path.join(self.folder_path, filename)
        with open(file_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return file_path

    def create_subfolder(self, name: str) -> str:
        """
        Create a subfolder within the project folder.

        Args:
            name: Name of the subfolder

        Returns:
            Full path to the subfolder
        """
        subfolder_path = os.path.join(self.folder_path, name)
        os.makedirs(subfolder_path, exist_ok=True)
        return subfolder_path

    def get_path(self, filename: str) -> str:
        """
        Get the full path for a file in the project folder.

        Args:
            filename: Name of the file

        Returns:
            Full path to the file
        """
        return os.path.join(self.folder_path, filename)

    def save_binary(self, filename: str, content: bytes) -> str:
        """
        Save binary content to the project folder.

        Args:
            filename: Name of the file
            content: Binary content to save

        Returns:
            Full path to the saved file
        """
        file_path = os.path.join(self.folder_path, filename)
        with open(file_path, "wb") as f:
            f.write(content)
        print(f"Binary file saved to {file_path}")
        return file_path


def move_old_projects(base_path: str = ".", archive_dir: str = "old_projects") -> None:
    """
    Move all existing project folders to the archive directory.

    Args:
        base_path: Base directory to search for project folders
        archive_dir: Name of the archive directory
    """
    archive_path = os.path.join(base_path, archive_dir)
    os.makedirs(archive_path, exist_ok=True)

    for folder in os.listdir(base_path):
        if folder.startswith("project_") and os.path.isdir(os.path.join(base_path, folder)):
            src = os.path.join(base_path, folder)
            dest = os.path.join(archive_path, folder)
            if os.path.exists(dest):
                print(f"Destination {dest} already exists. Skipping move for {folder}.")
            else:
                shutil.move(src, dest)
                print(f"Moved {folder} to {dest}")


def get_audio_duration(audio_file: str) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        audio_file: Path to the audio file

    Returns:
        Duration in seconds
    """
    audio = AudioSegment.from_file(audio_file)
    return len(audio) / 1000.0


def calculate_required_clips(
    audio_duration_seconds: float,
    clip_duration: int = 6,
    speed_factor: float = 0.5,
    transition_duration: float = 1.0,
    max_loops: int = 5,
    min_clips: int = 5,
    max_clips: int = 20,
) -> int:
    """
    Calculate the number of video clips needed to cover the audio duration.

    The formula accounts for:
    - Clips being slowed down by speed_factor (0.5 = half speed, so 6s clip becomes 12s)
    - Crossfade transitions that overlap clips
    - A maximum number of loops through the clip pool

    For N clips looped L times with effective duration E and transition T:
    total_duration = N * L * (E - T) + T

    Args:
        audio_duration_seconds: Target audio duration to cover
        clip_duration: Raw duration of each generated clip in seconds
        speed_factor: Playback speed factor (0.5 = half speed)
        transition_duration: Crossfade overlap duration in seconds
        max_loops: Maximum times to loop through all clips
        min_clips: Minimum number of clips for variety
        max_clips: Maximum number of clips (cost constraint)

    Returns:
        Number of clips to generate
    """
    # Effective duration per clip after slowdown
    effective_duration = clip_duration / speed_factor

    # Duration added per clip accounting for transition overlap
    duration_per_clip = effective_duration - transition_duration

    # Calculate clips needed for max_loops repetitions
    # total = N * max_loops * duration_per_clip + transition_duration
    # Solving for N: N = (total - transition_duration) / (max_loops * duration_per_clip)
    required = (audio_duration_seconds - transition_duration) / (max_loops * duration_per_clip)
    num_clips = math.ceil(required)

    # Clamp to min/max bounds
    num_clips = max(min_clips, min(max_clips, num_clips))

    # Calculate actual loop count with this many clips
    single_pass_duration = num_clips * duration_per_clip + transition_duration
    estimated_loops = math.ceil(audio_duration_seconds / single_pass_duration)

    print(f"[Clip Calculator] Audio duration: {audio_duration_seconds:.1f}s ({audio_duration_seconds/60:.1f} min)")
    print(f"[Clip Calculator] Effective clip duration: {effective_duration:.1f}s (raw {clip_duration}s at {speed_factor}x)")
    print(f"[Clip Calculator] Clips needed: {num_clips} (min={min_clips}, max={max_clips})")
    print(f"[Clip Calculator] Estimated loops: {estimated_loops}x through {num_clips} clips")

    return num_clips
