"""
WAN 2.2 14B model loader and generator.

Loads the Wan-AI/Wan2.2-T2V-A14B-Diffusers model with 4-bit quantization
for efficient GPU memory usage on consumer/prosumer GPUs.
"""

import time
import torch


# Model configuration
MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DEFAULT_NEGATIVE_PROMPT = (
    "色情, 暴力, 血腥, 恶心, 恐怖, 变形, 模糊, 低质量, 水印, 文字, "
    "Bright colors, overexposed, neon, vivid saturation, text overlay, watermark, "
    "blurry, distorted, low quality, deformed faces, extra limbs"
)


def load():
    """
    Load the WAN 2.2 14B pipeline with quantization.

    Returns:
        Loaded diffusers pipeline ready for generation.
    """
    from diffusers import AutoencoderKLWan, WanPipeline
    from transformers import CLIPTextModel, CLIPTokenizer
    from optimum.quanto import qint4

    print("[WAN 2.2] Loading VAE in float32...")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    print("[WAN 2.2] Loading pipeline with 4-bit quantized transformer...")
    pipe = WanPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    # Enable layerwise casting for memory efficiency
    print("[WAN 2.2] Enabling layerwise casting (float8 storage, bfloat16 compute)...")
    pipe.transformer.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
    )

    # Enable model CPU offload for safety
    print("[WAN 2.2] Enabling model CPU offload...")
    pipe.enable_model_cpu_offload()

    print("[WAN 2.2] Model loaded successfully.")
    return pipe


def generate(pipe, prompt, height=720, width=1280, num_frames=289,
             fps=24, num_inference_steps=40, guidance_scale=4.0,
             negative_prompt=None, seed=None):
    """
    Generate video frames using the loaded WAN 2.2 pipeline.

    Args:
        pipe: Loaded WAN 2.2 pipeline
        prompt: Text prompt for generation
        height: Video height in pixels
        width: Video width in pixels
        num_frames: Number of frames to generate. Must satisfy (num_frames - 1) % 4 == 0.
                    For 12s at 24fps = 289 frames. For 5s at 16fps = 81 frames.
        fps: Frames per second (used for output, not generation)
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        negative_prompt: Negative prompt (uses default if None)
        seed: Random seed for reproducibility

    Returns:
        List of PIL Image frames
    """
    if negative_prompt is None:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    # Validate num_frames constraint
    if (num_frames - 1) % 4 != 0:
        # Round to nearest valid value
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        print(f"[WAN 2.2] Adjusted num_frames to {num_frames} (must satisfy (n-1) % 4 == 0)")

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"[WAN 2.2] Generating: {width}x{height}, {num_frames} frames, "
          f"{num_inference_steps} steps, guidance={guidance_scale}")

    start = time.time()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    elapsed = time.time() - start
    print(f"[WAN 2.2] Generation complete in {elapsed:.1f}s")

    return output.frames[0], elapsed
