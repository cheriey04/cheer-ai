"""
server.py — FastAPI web server for Cheer AI form analysis.

Start with:  uvicorn server:app --reload
Then visit: http://localhost:8000
"""

import os
import sys
import tempfile
import traceback

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure cheer_ai is importable from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cheer_ai.pipeline import analyze_video

# ── App setup ──────────────────────────────────────────────────────────

app = FastAPI(title="Cheer AI", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


# ── Routes ─────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    """Upload a cheerleading video, get form analysis back as JSON."""
    # Validate file type
    if not video.filename or not video.filename.lower().endswith('.mp4'):
        return JSONResponse(
            status_code=400,
            content={"error": "Only .mp4 files are supported."},
        )

    # Save uploaded video to a temporary file
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, dir=BASE_DIR
        ) as tmp:
            contents = await video.read()
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to save uploaded file."},
        )

    # Run the full analysis pipeline
    try:
        result = analyze_video(tmp_path, MODEL_DIR)
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Analysis failed. See server logs for details."},
        )
    finally:
        # Always clean up the temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return result


@app.get("/health")
async def health():
    """Quick health check — returns OK if the server is alive."""
    return {"status": "ok"}


# ── Serve frontend (must be last so routes take priority) ─────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main frontend page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
