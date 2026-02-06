"""
Text-to-speech generation for the video pipeline.

Supports multiple TTS providers: Kokoro and Minimax HD.
"""

import os
import re
import requests
import replicate
from pydub import AudioSegment

from ..config import KokoroTTSConfig, MinimaxTTSConfig
from ..utils.file_io import save_to_file, read_response_from_file
from ..utils.text_processing import sanitize_script_for_tts, chunk_script


def _chunk_by_paragraphs(text: str, max_words: int = 5000) -> list[str]:
    """Split text into chunks at paragraph boundaries (double newlines).

    Keeps paragraph structure intact so TTS preserves natural pauses
    between script sections. Falls back to sentence splitting only if
    a single paragraph exceeds max_words.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_words = len(para.split())

        if current_word_count + para_words > max_words and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_word_count = para_words
        else:
            current_chunk.append(para)
            current_word_count += para_words

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def _kokoro_generate_chunk(client, text: str, config: KokoroTTSConfig) -> bytes:
    """Generate audio bytes for a single text chunk via Kokoro."""
    output = client.run(
        "jaaari/kokoro-82m:f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13",
        input={
            "text": text,
            "voice": config.voice,
            "speed": config.speed,
        }
    )

    if isinstance(output, str):
        audio_url = output
    elif hasattr(output, 'url'):
        audio_url = output.url
    elif isinstance(output, dict) and 'url' in output:
        audio_url = output['url']
    else:
        raise ValueError(f"Unexpected output format: {output}")

    response = requests.get(audio_url)
    response.raise_for_status()
    return response.content


def generate_audio_kokoro(
    script_file_path: str,
    folder_name: str,
    timestamp: str,
    replicate_api_key: str,
    config: KokoroTTSConfig = None,
) -> str:
    """
    Generate audio using Kokoro TTS via Replicate.

    For short scripts (<50K chars), sends the full text in one call.
    For longer scripts, chunks into ~5000-word segments to avoid HTTP timeouts,
    then stitches the audio together.

    Args:
        script_file_path: Path to the script file
        folder_name: Project folder path
        timestamp: Timestamp for file naming
        replicate_api_key: Replicate API key
        config: Kokoro TTS configuration (defaults to KokoroTTSConfig defaults)

    Returns:
        Path to the generated audio file
    """
    if config is None:
        config = KokoroTTSConfig()

    script_response = read_response_from_file(script_file_path)
    script_response = sanitize_script_for_tts(script_response)

    client = replicate.Client(api_token=replicate_api_key)

    print(f"Generating audio with Kokoro ({len(script_response)} chars, voice: {config.voice})")

    # For large scripts, chunk on paragraph boundaries to avoid Replicate HTTP timeouts
    # while preserving natural pauses between sections
    CHUNK_THRESHOLD = 50_000  # chars
    if len(script_response) <= CHUNK_THRESHOLD:
        chunks = [script_response]
    else:
        chunks = _chunk_by_paragraphs(script_response, max_words=5000)
        print(f"  Script too large for single call, split into {len(chunks)} chunks")

    combined_audio = AudioSegment.empty()

    for idx, chunk_text in enumerate(chunks):
        if len(chunks) > 1:
            print(f"  Generating chunk {idx + 1}/{len(chunks)} ({len(chunk_text)} chars)")

        audio_bytes = _kokoro_generate_chunk(client, chunk_text, config)

        chunk_path = os.path.join(folder_name, f"audio_chunk_{idx}_{timestamp}.wav")
        with open(chunk_path, "wb") as f:
            f.write(audio_bytes)

        combined_audio += AudioSegment.from_file(chunk_path)

    audio_path = os.path.join(folder_name, f"full_audio_{timestamp}.mp3")
    combined_audio.export(audio_path, format="mp3")

    print(f"Full audio saved to {audio_path} ({len(combined_audio)/1000:.1f}s)")
    return audio_path


def generate_audio_minimax(
    script_file_path: str,
    folder_name: str,
    timestamp: str,
    replicate_api_key: str,
    config: MinimaxTTSConfig = None,
) -> str:
    """
    Generate audio using Minimax HD TTS via Replicate.

    Chunks the script and stitches audio together.

    Args:
        script_file_path: Path to the script file
        folder_name: Project folder path
        timestamp: Timestamp for file naming
        replicate_api_key: Replicate API key
        config: Minimax TTS configuration (defaults to MinimaxTTSConfig defaults)

    Returns:
        Path to the generated audio file
    """
    if config is None:
        config = MinimaxTTSConfig()

    script_response = read_response_from_file(script_file_path)
    script_response = sanitize_script_for_tts(script_response)

    # Chunk the script into small pieces
    script_chunks = chunk_script(script_response)

    audio_chunks_paths = []
    total_duration = 0.0

    client = replicate.Client(api_token=replicate_api_key)

    # Generate audio for each chunk using Replicate Minimax HD
    for idx, chunk_text in enumerate(script_chunks):
        print(f"Generating audio for chunk {idx + 1} (text length: {len(chunk_text)} chars)")

        output = client.run(
            "minimax/speech-02-hd",
            input={
                "text": chunk_text,
                "pitch": config.pitch,
                "speed": config.speed,
                "volume": config.volume,
                "bitrate": config.bitrate,
                "channel": config.channel,
                "emotion": config.emotion,
                "voice_id": config.voice_id,
                "sample_rate": config.sample_rate,
                "audio_format": config.audio_format,
                "language_boost": config.language_boost,
                "subtitle_enable": config.subtitle_enable,
                "english_normalization": config.english_normalization
            }
        )

        # Get audio URL
        if isinstance(output, str):
            audio_url = output
        elif hasattr(output, 'url'):
            audio_url = output.url
        elif isinstance(output, dict) and 'url' in output:
            audio_url = output['url']
        else:
            raise ValueError(f"Unexpected output format: {output}")

        # Download
        response = requests.get(audio_url)
        response.raise_for_status()

        filename = f"audio_chunk_{idx}_{timestamp}.mp3"
        chunk_path = os.path.join(folder_name, filename)
        with open(chunk_path, "wb") as f:
            f.write(response.content)

        file_size = os.path.getsize(chunk_path)
        print(f"Audio chunk {idx + 1} saved to {chunk_path} (size: {file_size} bytes)")

        # Calculate duration
        audio_segment = AudioSegment.from_file(chunk_path)
        duration_seconds = len(audio_segment) / 1000.0
        total_duration += duration_seconds

        audio_chunks_paths.append(chunk_path)

    # Stitch all chunks together
    combined_audio = AudioSegment.empty()
    for chunk_path in audio_chunks_paths:
        chunk_audio = AudioSegment.from_file(chunk_path)
        combined_audio += chunk_audio

    audio_filename = f"full_audio_{timestamp}.mp3"
    audio_path = os.path.join(folder_name, audio_filename)
    combined_audio.export(audio_path, format="mp3")

    # Save details to a file
    details = f"Total chunks: {len(script_chunks)}\nTotal duration: {total_duration:.2f} seconds\nAudio file: {audio_path}\n\nChunks:\n"
    for idx, chunk in enumerate(script_chunks):
        details += f"Chunk {idx}: {chunk[:100]}... (words: {len(chunk.split())})\n"

    details_filename = f"audio_details_{timestamp}.txt"
    save_to_file(folder_name, details_filename, "Generated audio from script chunks using Replicate Minimax HD.", details)

    print(f"Full audio saved to {audio_path}")
    return audio_path


def generate_audio_from_script(
    script_file_path: str,
    folder_name: str,
    timestamp: str,
    replicate_api_key: str,
    tts_provider: str = "kokoro",
    kokoro_config: KokoroTTSConfig = None,
    minimax_config: MinimaxTTSConfig = None,
) -> str:
    """
    Generate audio from script using the specified TTS provider.

    Args:
        script_file_path: Path to the script file
        folder_name: Project folder path
        timestamp: Timestamp for file naming
        replicate_api_key: Replicate API key
        tts_provider: TTS provider to use ("kokoro" or "minimax")
        kokoro_config: Kokoro TTS configuration
        minimax_config: Minimax TTS configuration

    Returns:
        Path to the generated audio file

    Raises:
        ValueError: If an unknown TTS provider is specified
    """
    if tts_provider == "kokoro":
        return generate_audio_kokoro(
            script_file_path, folder_name, timestamp,
            replicate_api_key, kokoro_config
        )
    elif tts_provider == "minimax":
        return generate_audio_minimax(
            script_file_path, folder_name, timestamp,
            replicate_api_key, minimax_config
        )
    else:
        raise ValueError(f"Unknown TTS provider: {tts_provider}")
