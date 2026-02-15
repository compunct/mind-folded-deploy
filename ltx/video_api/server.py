"""
LTX-2 video generation FastAPI server.

Loads the LTX-2 19B Distilled model at startup and serves generation requests.
Jobs are processed asynchronously — POST /generate returns immediately with a
job_id, and clients poll GET /jobs/{job_id} for status.
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

from video_api.models.ltx2 import load, generate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "ltx2-distilled"
OUTPUT_DIR = Path("/app/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="Video Generation API")
pipe = None
model_status = "loading"  # "loading" | "ready" | "error"
generation_lock = threading.Lock()
jobs = {}  # job_id -> {status, result, error, request, created_at}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def load_model():
    global pipe, model_status

    print(f"[Server] Loading model: {MODEL_NAME}")
    try:
        pipe = load()
        model_status = "ready"
        print(f"[Server] Model '{MODEL_NAME}' loaded and ready.")
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
    height: int = 512
    width: int = 768
    num_frames: int = 121  # ~5s at 24fps
    fps: int = 24
    num_inference_steps: int = 8
    guidance_scale: float = 1.0
    seed: Optional[int] = None
    image_base64: Optional[str] = None  # Base64-encoded image for img2vid


class JobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # queued | running | completed | failed
    video_url: Optional[str] = None
    filename: Optional[str] = None
    generation_time_seconds: Optional[float] = None
    model: Optional[str] = None
    parameters: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status=model_status, model=MODEL_NAME)


@app.post("/generate", response_model=JobResponse)
def generate_video(req: GenerateRequest):
    if model_status != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready (status: {model_status}). "
                   f"Check /health and wait for status 'ready'."
        )

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "status": "queued",
        "result": None,
        "error": None,
        "request": req,
        "created_at": time.time(),
    }

    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    thread.start()

    return JobResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    resp = JobStatusResponse(job_id=job_id, status=job["status"])

    if job["status"] == "completed" and job["result"]:
        resp.video_url = job["result"]["video_url"]
        resp.filename = job["result"]["filename"]
        resp.generation_time_seconds = job["result"]["generation_time_seconds"]
        resp.model = job["result"]["model"]
        resp.parameters = job["result"]["parameters"]
    elif job["status"] == "failed":
        resp.error = job["error"]

    return resp


@app.get("/videos/{filename}")
def serve_video(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(str(file_path), media_type="video/mp4")


# ---------------------------------------------------------------------------
# Background job worker
# ---------------------------------------------------------------------------

def _run_job(job_id: str):
    """Run video generation in a background thread."""
    import base64
    import io
    from PIL import Image

    job = jobs[job_id]
    req = job["request"]

    # Decode base64 image if provided
    input_image = None
    if req.image_base64:
        image_bytes = base64.b64decode(req.image_base64)
        input_image = Image.open(io.BytesIO(image_bytes))
        print(f"[Server] Decoded input image: {input_image.size} {input_image.mode}")

    mode = "img2vid" if input_image else "txt2vid"

    # Block until the GPU is free — job stays "queued" while waiting
    with generation_lock:
        job["status"] = "running"
        print(f"[Server] Job {job_id} running ({mode}): {req.prompt!r}")

        try:
            frames, elapsed = generate(
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
                image=input_image,
            )

            filename = f"{uuid.uuid4().hex}.mp4"
            output_path = OUTPUT_DIR / filename
            _export_frames_to_mp4(frames, output_path, req.fps)

            job["status"] = "completed"
            job["result"] = {
                "video_url": f"/videos/{filename}",
                "filename": filename,
                "generation_time_seconds": round(elapsed, 2),
                "model": MODEL_NAME,
                "parameters": {
                    "mode": mode,
                    "prompt": req.prompt,
                    "height": req.height,
                    "width": req.width,
                    "num_frames": req.num_frames,
                    "fps": req.fps,
                    "num_inference_steps": req.num_inference_steps,
                    "guidance_scale": req.guidance_scale,
                    "seed": req.seed,
                },
            }
            print(f"[Server] Job {job_id} completed in {elapsed:.1f}s")
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            print(f"[Server] Job {job_id} failed: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _export_frames_to_mp4(frames, output_path: Path, fps: int):
    """Export a list of PIL Image frames to an MP4 file using imageio."""
    import imageio
    import numpy as np

    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p",
                                               "-crf", "18"])
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
