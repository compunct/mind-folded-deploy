"""
Local video generation FastAPI server.

Pluggable model backend selected via VIDEO_MODEL environment variable.
Loads the model into GPU memory once at startup and serves generation requests.
"""

import os
import time
import uuid
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "wan2.2-14b")
OUTPUT_DIR = Path("/app/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "wan2.2-14b": {
        "load": "video_api.models.wan22:load",
        "generate": "video_api.models.wan22:generate",
    },
    "ltx2-distilled": {
        "load": "video_api.models.ltx2:load",
        "generate": "video_api.models.ltx2:generate",
    },
}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="Video Generation API")
pipe = None
model_status = "loading"  # "loading" | "ready" | "error"
generation_lock = threading.Lock()


def _import_func(dotted_path: str):
    """Import a function from 'module.path:func_name' notation."""
    module_path, func_name = dotted_path.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def load_model():
    global pipe, model_status

    if VIDEO_MODEL not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        print(f"[Server] ERROR: Unknown model '{VIDEO_MODEL}'. Supported: {supported}")
        print(f"[Server] Starting in placeholder mode — /generate will return errors.")
        model_status = "error"
        return

    entry = MODEL_REGISTRY[VIDEO_MODEL]
    load_fn = _import_func(entry["load"])

    print(f"[Server] Loading model: {VIDEO_MODEL}")
    try:
        pipe = load_fn()
        model_status = "ready"
        print(f"[Server] Model '{VIDEO_MODEL}' loaded and ready.")
    except Exception as e:
        model_status = "error"
        print(f"[Server] Failed to load model: {e}")
        raise


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    height: int = 720
    width: int = 1280
    num_frames: int = 289  # 12s at 24fps
    fps: int = 24
    num_inference_steps: int = 40
    guidance_scale: float = 4.0
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    video_url: str
    filename: str
    generation_time_seconds: float
    model: str
    parameters: dict


class HealthResponse(BaseModel):
    status: str
    model: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status=model_status, model=VIDEO_MODEL)


@app.post("/generate", response_model=GenerateResponse)
def generate_video(req: GenerateRequest):
    if model_status != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready (status: {model_status}). "
                   f"Check /health and wait for status 'ready'."
        )

    entry = MODEL_REGISTRY[VIDEO_MODEL]
    generate_fn = _import_func(entry["generate"])

    # Only one generation at a time (GPU bottleneck)
    acquired = generation_lock.acquire(timeout=5)
    if not acquired:
        raise HTTPException(
            status_code=429,
            detail="Another generation is in progress. Try again later."
        )

    try:
        frames, elapsed = generate_fn(
            pipe,
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            fps=req.fps,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            negative_prompt=req.negative_prompt,
            seed=req.seed,
        )

        # Export frames to MP4
        filename = f"{uuid.uuid4().hex}.mp4"
        output_path = OUTPUT_DIR / filename
        _export_frames_to_mp4(frames, output_path, req.fps)

        video_url = f"/videos/{filename}"

        return GenerateResponse(
            video_url=video_url,
            filename=filename,
            generation_time_seconds=round(elapsed, 2),
            model=VIDEO_MODEL,
            parameters={
                "prompt": req.prompt,
                "height": req.height,
                "width": req.width,
                "num_frames": req.num_frames,
                "fps": req.fps,
                "num_inference_steps": req.num_inference_steps,
                "guidance_scale": req.guidance_scale,
                "seed": req.seed,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()


@app.get("/videos/{filename}")
def serve_video(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(str(file_path), media_type="video/mp4")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _export_frames_to_mp4(frames, output_path: Path, fps: int):
    """Export a list of PIL Image frames to an MP4 file using imageio."""
    import imageio
    import numpy as np

    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p"])
    for frame in frames:
        if hasattr(frame, "numpy"):
            # torch tensor
            arr = frame.numpy()
        elif hasattr(frame, "__array__"):
            arr = np.asarray(frame)
        else:
            arr = np.array(frame)

        # Handle float [0,1] range
        if arr.dtype in (np.float32, np.float64, np.float16):
            arr = (arr * 255).clip(0, 255).astype(np.uint8)

        writer.append_data(arr)
    writer.close()
    print(f"[Server] Video saved: {output_path} ({len(frames)} frames @ {fps}fps)")
