"""
WAN 2.2 14B model loader and generator (text-to-video + image-to-video).

Loads both the T2V and I2V pipelines with shared VAE and text encoder.
Both transformers use float8 layerwise casting + model CPU offload.
"""

import time
import torch


T2V_MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
I2V_MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurry, low quality, deformed, "
    "text, watermark, distorted faces, extra limbs"
)


def load():
    """
    Load WAN 2.2 14B pipelines (text-to-video + image-to-video).

    Shares VAE and text encoder between pipelines.  Each transformer
    gets float8 layerwise casting and model CPU offload.

    Returns:
        Dict with "t2v" and "i2v" pipeline instances.
    """
    from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline
    from transformers import CLIPVisionModel

    # --- Shared components ---
    print("[WAN 2.2] Loading shared VAE (float32)...")
    vae = AutoencoderKLWan.from_pretrained(
        T2V_MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    # --- Text-to-video pipeline ---
    print("[WAN 2.2] Loading T2V pipeline...")
    t2v_pipe = WanPipeline.from_pretrained(
        T2V_MODEL_ID,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    print("[WAN 2.2] T2V: enabling layerwise casting (float8 storage, bfloat16 compute)...")
    t2v_pipe.transformer.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
    )

    print("[WAN 2.2] T2V: enabling model CPU offload...")
    t2v_pipe.enable_model_cpu_offload()

    # --- Image-to-video pipeline ---
    print("[WAN 2.2] Loading I2V pipeline...")
    image_encoder = CLIPVisionModel.from_pretrained(
        I2V_MODEL_ID,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    )

    i2v_pipe = WanImageToVideoPipeline.from_pretrained(
        I2V_MODEL_ID,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch.bfloat16,
    )

    print("[WAN 2.2] I2V: enabling layerwise casting (float8 storage, bfloat16 compute)...")
    i2v_pipe.transformer.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
    )

    print("[WAN 2.2] I2V: enabling model CPU offload...")
    i2v_pipe.enable_model_cpu_offload()

    print("[WAN 2.2] Both pipelines loaded successfully.")
    return {"t2v": t2v_pipe, "i2v": i2v_pipe}


def generate(pipe, prompt, height=480, width=832, num_frames=161,
             fps=16, num_inference_steps=40, guidance_scale=4.0,
             negative_prompt=None, seed=None, image=None):
    """
    Generate video frames using the loaded WAN 2.2 pipeline.

    Args:
        pipe: Dict with "t2v" and "i2v" pipeline instances
        prompt: Text prompt for generation
        height: Video height in pixels
        width: Video width in pixels
        num_frames: Number of frames. Must satisfy (num_frames - 1) % 4 == 0.
                    161 frames at 16fps = ~10 seconds.
        fps: Frames per second (used for output, not generation)
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale (4.0 for t2v, 3.5 for i2v)
        negative_prompt: Negative prompt (uses default if None)
        seed: Random seed for reproducibility
        image: Optional PIL Image for image-to-video generation

    Returns:
        (frames, elapsed): List of PIL Image frames and generation time in seconds.
    """
    from PIL import Image as PILImage

    if negative_prompt is None:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    # Validate num_frames constraint
    if (num_frames - 1) % 4 != 0:
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        print(f"[WAN 2.2] Adjusted num_frames to {num_frames} (must satisfy (n-1) % 4 == 0)")

    # Select pipeline
    if image is not None:
        active_pipe = pipe["i2v"]
        mode = "img2vid"
        # Auto-adjust guidance for i2v
        guidance_scale = 3.5
        # Resize image to target resolution
        image = image.convert("RGB").resize((width, height), PILImage.LANCZOS)
        print(f"[WAN 2.2] Image resized to {width}x{height}")
    else:
        active_pipe = pipe["t2v"]
        mode = "txt2vid"

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"[WAN 2.2] Generating ({mode}): {width}x{height}, {num_frames} frames, "
          f"{num_inference_steps} steps, guidance={guidance_scale}")

    # Build kwargs
    pipe_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    if image is not None:
        pipe_kwargs["image"] = image

    start = time.time()
    output = active_pipe(**pipe_kwargs)
    elapsed = time.time() - start
    print(f"[WAN 2.2] Generation complete in {elapsed:.1f}s")

    return output.frames[0], elapsed
