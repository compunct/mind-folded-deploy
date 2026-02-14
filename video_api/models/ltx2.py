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
    Load the LTX-2 19B distilled pipeline.

    Returns:
        Loaded diffusers LTX2Pipeline ready for generation.
    """
    from diffusers import LTX2Pipeline

    print("[LTX-2] Loading pipeline from HuggingFace...")
    pipe = LTX2Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    print("[LTX-2] Enabling sequential CPU offload...")
    pipe.enable_sequential_cpu_offload(device="cuda")

    # Decode VAE in chunks to avoid 32-bit tensor index overflow on long videos
    print("[LTX-2] Enabling VAE tiling for long video support...")
    pipe.vae.enable_tiling(
        tile_sample_min_num_frames=16,
        tile_sample_stride_num_frames=8,
    )

    print("[LTX-2] Model loaded successfully.")
    return pipe


def generate(pipe, prompt, height=512, width=768, num_frames=121,
             fps=24, num_inference_steps=8, guidance_scale=1.0,
             negative_prompt=None, seed=None):
    """
    Generate video frames using the loaded LTX-2 pipeline.

    Args:
        pipe: Loaded LTX-2 pipeline
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

    Returns:
        (frames, elapsed): List of PIL Image frames and generation time in seconds.
    """
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
    import numpy as np
    from PIL import Image

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

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"[LTX-2] Generating: {width}x{height}, {num_frames} frames, "
          f"{num_inference_steps} steps, guidance={guidance_scale}")

    start = time.time()
    latents, audio = pipe(
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
        output_type="latent",
        return_dict=False,
    )
    diff_elapsed = time.time() - start
    print(f"[LTX-2] Diffusion complete in {diff_elapsed:.1f}s, decoding frames...")

    # The pipeline returns packed, normalized latents when output_type="latent".
    # We must unpack and denormalize before VAE decode, then decode in chunks
    # to avoid PyTorch's 32-bit index overflow on long videos.
    latents = pipe._unpack_latents(latents, num_frames, height, width)
    latents = pipe._denormalize_latents(
        latents, pipe.vae.latents_mean, pipe.vae.latents_std,
        pipe.vae.config.scaling_factor,
    )
    latents = latents.to(pipe.vae.dtype)
    frames = _decode_latents_chunked(pipe, latents, chunk_frames=8)

    elapsed = time.time() - start
    print(f"[LTX-2] Total generation complete in {elapsed:.1f}s ({len(frames)} frames)")

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
        video_chunk = video_chunk.float().clamp(0, 1).cpu().numpy()
        video_uint8 = (video_chunk * 255).astype(np.uint8)
        for frame in video_uint8:
            all_frames.append(Image.fromarray(frame))
        print(f"[LTX-2] Decoded frames {len(all_frames)}...")

    return all_frames
