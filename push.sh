#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: ./push.sh <target>"
    echo ""
    echo "Targets:"
    echo "  ltx          Build and push compunct/video-api:ltx2 (app code only, fast)"
    echo "  ltx-base     Build and push compunct/video-api-base:ltx2 (deps, slow, rarely needed)"
    echo "  wan          Build and push compunct/video-api:wan22 + :latest (app code only, fast)"
    echo "  wan-base     Build and push compunct/video-api-base:wan22 (deps, slow, rarely needed)"
    echo "  runpod       Build and push compunct/video-api:runpod (LTX-2 + SSH for RunPod pods)"
    echo ""
    echo "Base images contain OS + Python + pip packages. Rebuild only when requirements change."
    echo "App images just copy your code on top — pushes are tiny and fast."
    exit 1
}

TARGET="${1:-}"
[ -z "$TARGET" ] && usage

case "$TARGET" in
    ltx)
        echo "Building compunct/video-api:ltx2 ..."
        docker build -t compunct/video-api:ltx2 -f video_api/Dockerfile.ltx2 video_api/
        echo "Pushing compunct/video-api:ltx2 ..."
        docker push compunct/video-api:ltx2
        ;;
    ltx-base)
        echo "Building compunct/video-api-base:ltx2 ..."
        docker build -t compunct/video-api-base:ltx2 -f video_api/Dockerfile.ltx2-base video_api/
        echo "Pushing compunct/video-api-base:ltx2 ..."
        docker push compunct/video-api-base:ltx2
        ;;
    wan)
        echo "Building compunct/video-api:wan22 ..."
        docker build -t compunct/video-api:wan22 -t compunct/video-api:latest -f video_api/Dockerfile.wan22 video_api/
        echo "Pushing compunct/video-api:wan22 ..."
        docker push compunct/video-api:wan22
        docker push compunct/video-api:latest
        ;;
    wan-base)
        echo "Building compunct/video-api-base:wan22 ..."
        docker build -t compunct/video-api-base:wan22 -f video_api/Dockerfile.wan22-base video_api/
        echo "Pushing compunct/video-api-base:wan22 ..."
        docker push compunct/video-api-base:wan22
        ;;
    runpod)
        echo "Building compunct/video-api:runpod ..."
        docker build -t compunct/video-api:runpod -f video_api/Dockerfile.runpod video_api/
        echo "Pushing compunct/video-api:runpod ..."
        docker push compunct/video-api:runpod
        ;;
    *)
        echo "Unknown target: $TARGET"
        usage
        ;;
esac

echo "Done!"
