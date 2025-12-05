#!/bin/bash
# start.sh – Memory-optimized FastAPI startup for Render

# Set Python to use less memory
export PYTHONUNBUFFERED=1
export MALLOC_TRIM_THRESHOLD_=100000

# Start server with optimized workers
# Use 1 worker to minimize memory usage on free tier
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port ${PORT:-5000} \
  --workers 1 \
  --timeout-keep-alive 30 \
  --log-level info
