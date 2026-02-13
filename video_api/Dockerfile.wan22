FROM compunct/video-api-base:wan22

# Copy application code (this is the only layer that changes for code updates)
COPY . /app/video_api/

EXPOSE 8000

CMD ["uvicorn", "video_api.server:app", "--host", "::", "--port", "8000", "--timeout-keep-alive", "600"]
