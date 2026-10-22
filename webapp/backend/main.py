"""
Backend: serves the frontend and the API on the same origin.
App setup and routes only: HTTP helpers are in api_helpers.py, the work happens in pipeline.py.

- GET  /api/health          ready check and implemented versions
- POST /api/analyse/text    typed text (JSON)
- POST /api/analyse/media   an uploaded file or microphone recording (multipart form)

Both analyse routes stream the run as NDJSON events (see pipeline.py).
Try them at http://localhost:8000/docs

Run from the repository root:  uv run fastapi dev webapp/backend/main.py
Then open http://localhost:8000/
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from api_helpers import require_version, stream
from pipeline import Pipeline
from schemas import RunMetadata, TextRequest
from versions import VERSIONS

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# Pipeline logs (windows, segments, claims) appear in the server output, next to uvicorn's
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
for noisy in ("primp", "urllib3", "httpx", "watchfiles"):   # one line per HTTP request or file change
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("main")

state = {}


@asynccontextmanager
async def lifespan(app):
    # Load the models once at startup, not for every request
    log.info("loading models...")
    state["pipeline"] = Pipeline()
    log.info("models loaded, ready")
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
async def health():
    """Polled by the frontend's connection status. The models are loaded before the server
    accepts requests, so any answer means the backend is ready."""
    return {"status": "ok", "versions": list(VERSIONS)}


@app.post("/api/analyse/text")
async def analyse_text(body: TextRequest, request: Request):
    """Check typed text: it is split into sentences and each checkable one is verified."""
    version = require_version(body.version_id)
    log.info("run %s (%s): text, %d characters", body.run_id, version.id, len(body.text))
    return stream(request, body, version, state["pipeline"].run_text(version, body.run_id, body.text))


@app.post("/api/analyse/media")
async def analyse_media(media: UploadFile, metadata: Annotated[str, Form()], request: Request):
    """Transcribe and check a complete file or recording (WebM/Opus, MP4, Ogg, mp3...).
    metadata is a JSON string with run_id, version_id and config."""
    try:
        body = RunMetadata.model_validate_json(metadata)
    except ValidationError as error:
        raise HTTPException(422, error.errors(include_url=False))
    if (media.size or 0) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Files are limited to 100 MB")
    version = require_version(body.version_id)
    log.info("run %s (%s): media %r, %s, %d bytes", body.run_id, version.id,
             media.filename, media.content_type, media.size or 0)
    events = state["pipeline"].run_media(version, body.run_id, media.file, media.filename or "recording")
    return stream(request, body, version, events)


# Mounted last so /api routes take priority
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
