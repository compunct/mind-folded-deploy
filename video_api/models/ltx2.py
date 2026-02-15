"""
LTX-2 19B Distilled model loader and generator.

Loads the Lightricks LTX-2 19B distilled FP8 model via the diffusers library.
Uses single-stage generation (no upsampling) for simplicity and speed.
"""

import time
import torch


MODEL_ID = "rootonchair/LTX-2-19b-distilled"
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "low quality, deformed, watermark, text overlay, overexposed"
)


def load():
    """
    Load the LTX-2 19B distilled pipeline (text-to-video + image-to-video).

    Returns:
        Dict with "t2v" and "i2v" pipeline instances (shared weights, zero extra RAM).
    """
    from diffusers import LTX2Pipeline, LTX2ImageToVideoPipeline

    print("[LTX-2] Loading text-to-video pipeline from HuggingFace...")
    t2v_pipe = LTX2Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    # Offload MUST happen before from_pipe() — frees enough RAM for it to succeed.
    # Sequential offload hooks are registered on the module objects themselves,
    # so they fire for both t2v and i2v (shared via from_pipe). No swap needed.
    print("[LTX-2] Enabling sequential CPU offload (t2v)...")
    t2v_pipe.enable_sequential_cpu_offload(device="cuda")

    # Decode VAE in chunks to avoid 32-bit tensor index overflow on long videos
    print("[LTX-2] Enabling VAE tiling for long video support...")
    t2v_pipe.vae.enable_tiling(
        tile_sample_min_num_frames=16,
        tile_sample_stride_num_frames=8,
    )

    # Create i2v pipe AFTER offload — shares model weights (zero extra RAM).
    # The sequential offload hooks live on the shared module objects, so both
    # pipelines use them automatically. No runtime hook swapping needed.
    print("[LTX-2] Creating image-to-video pipeline (shared weights)...")
    i2v_pipe = LTX2ImageToVideoPipeline.from_pipe(t2v_pipe)

    print("[LTX-2] Model loaded successfully.")
    return {"t2v": t2v_pipe, "i2v": i2v_pipe}


def generate(pipe, prompt, height=512, width=768, num_frames=121,
             fps=24, num_inference_steps=8, guidance_scale=1.0,
             negative_prompt=None, seed=None, image=None):
    """
    Generate video frames using the loaded LTX-2 pipeline.

    Args:
        pipe: Dict with "t2v" and "i2v" pipeline instances
        prompt: Text prompt for generation
        height: Video height in pixels (divisible by 32)
        width: Video width in pixels (divisible by 32)
        num_frames: Number of frames. Must satisfy (num_frames - 1) % 8 == 0.
                    121 frames at 24fps = ~5 seconds.
        fps: Frames per second
        num_inference_steps: Number of denoising steps (8 for distilled)
        guidance_scale: Guidance scale (1.0 for distilled model)
        negative_prompt: Negative prompt (uses default if None)
        seed: Random seed for reproducibility
        image: Optional PIL Image for image-to-video generation

    Returns:
        (frames, elapsed): List of PIL Image frames and generation time in seconds.
    """
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
    import numpy as np
    from PIL import Image as PILImage

    if negative_prompt is None:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    # Validate frame count constraint: (num_frames - 1) % 8 == 0
    if (num_frames - 1) % 8 != 0:
        num_frames = ((num_frames - 1) // 8) * 8 + 1
        print(f"[LTX-2] Adjusted num_frames to {num_frames} (must satisfy (n-1) % 8 == 0)")

    # Validate resolution: divisible by 32
    if height % 32 != 0:
        height = (height // 32) * 32
        print(f"[LTX-2] Adjusted height to {height} (must be divisible by 32)")
    if width % 32 != 0:
        width = (width // 32) * 32
        print(f"[LTX-2] Adjusted width to {width} (must be divisible by 32)")

    # Select pipeline: image-to-video or text-to-video
    if image is not None:
        active_key = "i2v"
        mode = "img2vid"
        # Resize image to target resolution
        image = image.convert("RGB").resize((width, height), PILImage.LANCZOS)
        print(f"[LTX-2] Image resized to {width}x{height}")
    else:
        active_key = "t2v"
        mode = "txt2vid"

    active_pipe = pipe[active_key]

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"[LTX-2] Generating ({mode}): {width}x{height}, {num_frames} frames, "
          f"{num_inference_steps} steps, guidance={guidance_scale}")

    # Use pipeline's built-in decode for short videos (< 750 frames),
    # chunked latent decode only for long videos that exceed 32-bit index limit.
    use_chunked = num_frames > 750

    # Build common kwargs
    pipe_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=float(fps),
        num_inference_steps=num_inference_steps,
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent" if use_chunked else "np",
        return_dict=False,
    )
    if image is not None:
        pipe_kwargs["image"] = image
        # Add noise to image conditioning so the model has room to create motion.
        # Without this (default 0.0), early frames are static copies of the input.
        pipe_kwargs["noise_scale"] = 0.04

    start = time.time()
    result, audio = active_pipe(**pipe_kwargs)
    diff_elapsed = time.time() - start

    if use_chunked:
        print(f"[LTX-2] Diffusion complete in {diff_elapsed:.1f}s, chunked decode...")
        latents = result
        if latents.ndim == 3:
            latents = active_pipe._unpack_latents(latents, num_frames, height, width)
        latents = latents.to(active_pipe.vae.dtype)
        frames = _decode_latents_chunked(active_pipe, latents, chunk_frames=8)
    else:
        print(f"[LTX-2] Generation complete in {diff_elapsed:.1f}s")
        video_np = result[0]
        video_uint8 = (video_np * 255).clip(0, 255).astype(np.uint8)
        frames = [PILImage.fromarray(frame) for frame in video_uint8]

    elapsed = time.time() - start
    print(f"[LTX-2] Total: {elapsed:.1f}s ({len(frames)} frames)")

    return frames, elapsed


def _decode_latents_chunked(pipe, latents, chunk_frames=8):
    """Decode latent tensor to PIL images in temporal chunks.

    The VAE decodes the full latent at once which can exceed PyTorch's
    32-bit index limit for long videos.  This function slices along the
    temporal dimension of the latent, decodes each slice, and converts
    to PIL images incrementally.
    """
    import numpy as np
    from PIL import Image

    vae = pipe.vae
    # latents shape: (batch, channels, num_latent_frames, h, w)
    # LTX-2 VAE temporal compression is 8x, so latent frames = ceil(num_frames/8)
    num_latent_frames = latents.shape[2]

    all_frames = []
    for i in range(0, num_latent_frames, chunk_frames):
        chunk = latents[:, :, i:i + chunk_frames, :, :]
        with torch.no_grad():
            decoded = vae.decode(chunk, return_dict=False)[0]
        # decoded shape: (batch, 3, temporal_frames, H, W) — float tensor
        video_chunk = decoded[0]  # (3, T, H, W)
        video_chunk = video_chunk.permute(1, 2, 3, 0)  # (T, H, W, 3)
        video_chunk = (video_chunk.float() / 2 + 0.5).clamp(0, 1).cpu().numpy()
        video_uint8 = (video_chunk * 255).astype(np.uint8)
        for frame in video_uint8:
            all_frames.append(Image.fromarray(frame))
        print(f"[LTX-2] Decoded frames {len(all_frames)}...")

    return all_frames
