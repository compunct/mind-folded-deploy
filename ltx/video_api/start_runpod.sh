#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Video API Server - RunPod Startup"
echo "============================================"

# --- SSH Setup ---
# RunPod injects PUBLIC_KEY env var from your account settings
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
    echo "[RunPod] SSH key configured"
fi

# Start SSH daemon (for remote development)
service ssh start 2>/dev/null || /usr/sbin/sshd 2>/dev/null || echo "[RunPod] WARNING: Could not start SSH"

# --- Persistent storage ---
# RunPod mounts persistent volume at /workspace
# Use it for model cache so weights survive pod restarts
export HF_HOME=/workspace/model_cache
mkdir -p "$HF_HOME" /app/outputs

echo "[RunPod] HF_HOME=$HF_HOME"
echo "[RunPod] GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPU detected)"
echo ""

# --- Start API server ---
echo "[RunPod] Starting video API server on port 8000..."
echo "[RunPod] SSH into this pod to edit code in /app/video_api/"
echo "[RunPod] To restart the server: kill the uvicorn process and re-run this script"
echo ""

exec uvicorn video_api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --timeout-keep-alive 600
