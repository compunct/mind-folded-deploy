#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Video API Server - Deploy"
echo "============================================"
echo ""

# Parse model argument
MODEL="${1:-}"
if [ -z "$MODEL" ]; then
    echo "Usage: ./deploy.sh <wan|ltx>"
    echo ""
    echo "  wan  - WAN 2.2 14B (requires ~60GB RAM, 32GB VRAM)"
    echo "  ltx  - LTX-2 19B Distilled FP8 (requires ~40GB RAM, 24GB VRAM)"
    echo ""
    exit 1
fi

case "$MODEL" in
    wan) SERVICE="video-api-wan"; PROFILE="wan" ;;
    ltx) SERVICE="video-api-ltx"; PROFILE="ltx" ;;
    *)
        echo "ERROR: Unknown model '$MODEL'. Use 'wan' or 'ltx'."
        exit 1
        ;;
esac

# Check for NVIDIA Container Toolkit
echo "[1/3] Checking NVIDIA GPU support..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    if ! dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then
        echo "WARNING: NVIDIA Container Toolkit may not be installed."
        echo "Install it with: sudo apt install nvidia-container-toolkit"
        echo "Then restart Docker: sudo systemctl restart docker"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Build container
echo "[2/3] Building $SERVICE container..."
docker compose --profile "$PROFILE" build "$SERVICE"
echo ""

# Start and wait for health
echo "[3/3] Starting $SERVICE..."
echo "This will download model weights on first run. Be patient."
echo ""
docker compose --profile "$PROFILE" up -d "$SERVICE"

echo "Waiting for video API to become healthy..."
echo "(Model loading can take 5-15 minutes on first run)"
echo ""

MAX_WAIT=900
ELAPSED=0
INTERVAL=10

while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS=$(curl -s http://localhost:8000/health 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
    if [ "$STATUS" = "ready" ]; then
        echo ""
        echo "Video API is READY!"
        echo ""
        echo "============================================"
        echo "  Deployment Complete ($SERVICE)"
        echo "============================================"
        echo ""
        echo "Usage:"
        echo "  # Test video generation:"
        echo "  curl -X POST http://localhost:8000/generate \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"prompt\": \"dark foggy abyss, muted colors\", \"height\": 480, \"width\": 832, \"num_frames\": 81}'"
        echo ""
        echo "  # Check health:"
        echo "  curl http://localhost:8000/health"
        echo ""
        echo "  # View logs:"
        echo "  docker compose --profile $PROFILE logs -f $SERVICE"
        echo ""
        echo "  # Stop:"
        echo "  docker compose --profile $PROFILE down"
        echo ""
        exit 0
    fi
    printf "  [%ds] Status: %s\r" $ELAPSED "${STATUS:-connecting...}"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""
echo "ERROR: Video API did not become ready within ${MAX_WAIT}s."
echo "Check logs: docker compose --profile $PROFILE logs $SERVICE"
exit 1
