#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Deploy Mind Folded - Video Pipeline"
echo "============================================"
echo ""

# Check for NVIDIA Container Toolkit
echo "[1/4] Checking NVIDIA GPU support..."
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

# Check for .env file
echo "[2/4] Checking environment..."
if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Copy .env.example and fill in your API keys:"
    echo "  cp .env.example .env"
    echo "  nano .env"
    exit 1
fi

echo ".env file found."
echo ""

# Build containers
echo "[3/4] Building containers..."
docker compose build
echo ""

# Start video-api and wait for health
echo "[4/4] Starting video-api service..."
echo "This will download model weights on first run (~28GB). Be patient."
echo ""
docker compose up -d video-api

echo "Waiting for video-api to become healthy..."
echo "(Model loading can take 5-15 minutes on first run)"
echo ""

# Poll health endpoint
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
        echo "  Deployment Complete"
        echo "============================================"
        echo ""
        echo "Usage:"
        echo "  # Run the full pipeline:"
        echo "  docker compose run pipeline"
        echo ""
        echo "  # Test video generation:"
        echo "  curl -X POST http://localhost:8000/generate \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"prompt\": \"dark foggy abyss, muted colors\", \"height\": 480, \"width\": 832, \"num_frames\": 81}'"
        echo ""
        echo "  # Check health:"
        echo "  curl http://localhost:8000/health"
        echo ""
        echo "  # View logs:"
        echo "  docker compose logs -f video-api"
        echo ""
        echo "  # Stop everything:"
        echo "  docker compose down"
        echo ""
        exit 0
    fi
    printf "  [%ds] Status: %s\r" $ELAPSED "${STATUS:-connecting...}"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""
echo "ERROR: Video API did not become ready within ${MAX_WAIT}s."
echo "Check logs: docker compose logs video-api"
exit 1
