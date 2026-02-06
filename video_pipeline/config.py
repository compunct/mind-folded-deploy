"""
Configuration dataclasses for the video pipeline.

All tunable parameters are centralized here as dataclass-based configuration.
"""

from dataclasses import dataclass, field


@dataclass
class VideoConfig:
    """Configuration for video generation."""
    # About $5.00 per minute at 1080p
    num_videos: int = 9 # Can be overridden dynamically based on audio duration
    duration_seconds: int = 12  # between 2 and 12 for bytedance fast
    resolution: str = "720p"  # options: "480p", "720p", "1080p"
    fps: int = 24
    aspect_ratio: str = "16:9"
    # Dynamic clip calculation parameters
    max_loops: int = 5  # Maximum times to loop through all clips
    speed_factor: float = 0.75  # Playback speed in final video
    transition_duration: float = 1.0  # Crossfade duration in seconds
    backend: str = "local"  # "local" (WAN 2.2 via video-api) or "replicate" (Seedance)


@dataclass
class LocalVideoConfig:
    """Configuration for the local video generation API."""
    api_url: str = "http://localhost:8000"
    num_inference_steps: int = 40
    guidance_scale: float = 4.0
    timeout_seconds: int = 600       # 10 min per clip
    health_check_timeout: int = 900  # 15 min for model startup
    health_check_interval: int = 10


@dataclass
class QualityGateConfig:
    """Configuration for video quality gate."""
    enabled: bool = True
    threshold: float = 5.0  # Composite score 1-10
    max_retries: int = 2
    eval_model: str = "openrouter/google/gemini-2.0-flash-001"
    num_frames: int = 3  # Number of frames to extract for evaluation


@dataclass
class FactsConfig:
    """Configuration for fact gathering."""
    facts_per_iteration: int = 10 #10
    num_iterations: int = 3 #3


@dataclass
class ScriptConfig:
    """Configuration for script generation."""
    # 10,000 words = roughly 60 minutes of video
    facts_per_section: int = 3 # 3
    section_word_count: str = "900-1500 words" # 900-1500 words
    target_duration_minutes: tuple[int, int] = (60, 120)  # (min, max) target video length


@dataclass
class MinimaxTTSConfig:
    """Configuration for Minimax HD TTS."""
    voice_id: str = "English_ManWithDeepVoice"
    pitch: int = 0
    speed: float = 0.8
    volume: float = 1.0
    bitrate: int = 128000
    channel: str = "mono"
    emotion: str = "calm"
    sample_rate: int = 32000
    audio_format: str = "mp3"
    language_boost: str = "English"
    subtitle_enable: bool = False
    english_normalization: bool = True


@dataclass
class KokoroTTSConfig:
    """Configuration for Kokoro TTS."""
    voice: str = "am_onyx"  # other good voices: am_echo
    speed: float = 0.85  # 0.1 to 5.0; default 1.0


@dataclass
class ThumbnailConfig:
    """Configuration for thumbnail generation."""
    resolution: str = "1280x720"
    style: str = "serene, abstract, philosophical theme with soft colors and no text overlay"


@dataclass
class ModelConfig:
    """LLM model configuration for each pipeline step."""
    sub_niches: str = "google/gemini-2.0-flash-001"
    topic: str = "google/gemini-2.0-flash-001"
    facts: str = "perplexity/sonar"
    reorder: str = "google/gemini-2.0-flash-001"
    script: str = "anthropic/claude-haiku-4.5"
    video_prompts: str = "google/gemini-2.0-flash-001"
    thumbnail_prompt: str = "google/gemini-2.0-flash-001"
    evaluator: str = "google/gemini-2.0-flash-001"


@dataclass
class PipelineConfig:
    """Main configuration for the entire pipeline."""
    video: VideoConfig = field(default_factory=VideoConfig)
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    local_video: LocalVideoConfig = field(default_factory=LocalVideoConfig)
    facts: FactsConfig = field(default_factory=FactsConfig)
    script: ScriptConfig = field(default_factory=ScriptConfig)
    minimax_tts: MinimaxTTSConfig = field(default_factory=MinimaxTTSConfig)
    kokoro_tts: KokoroTTSConfig = field(default_factory=KokoroTTSConfig)
    thumbnail: ThumbnailConfig = field(default_factory=ThumbnailConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    tts_provider: str = "kokoro"  # Options: "kokoro", "minimax"

    @classmethod
    def from_defaults(cls) -> "PipelineConfig":
        """Create configuration with all defaults."""
        return cls()
