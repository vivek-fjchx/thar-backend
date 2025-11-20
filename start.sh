#!/bin/bash
# start.sh – correct script for FastAPI on Render

uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-5000}