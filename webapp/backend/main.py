"""
Backend: serves the frontend and POST /api/analyse on the same origin.
Routes only: the work happens in pipeline.py.

Run from the repository root:  uv run fastapi dev webapp/backend/main.py
Then open http://localhost:8000/
"""

import logging
from contextlib import asynccontextmanager

import anyio
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.datastructures import UploadFile   # what request.form() returns

from pipeline import Pipeline
from schemas import AnalyseRequest, Event
from versions import VERSIONS, get_version

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


@app.post("/api/analyse")
async def analyse(request: Request):
    """Validate the request, then stream the run as NDJSON events (see pipeline.py)."""
    try:
        if request.headers.get("content-type", "").startswith("multipart/"):
            form = await request.form(max_part_size=MAX_UPLOAD_BYTES)
            media, metadata = form.get("media"), form.get("metadata")
            if not isinstance(media, UploadFile) or not isinstance(metadata, str):
                raise HTTPException(400, "Expected a 'media' file and a 'metadata' JSON field")
            body = AnalyseRequest.model_validate_json(metadata)
        else:
            body = AnalyseRequest.model_validate_json(await request.body())
            media = None
    except ValidationError as error:
        log.warning("rejected request: %s", error.errors(include_url=False))
        raise HTTPException(422, error.errors(include_url=False))

    version = get_version(body.version_id)
    if version is None:
        log.warning("rejected request: version %r is not implemented", body.version_id)
        raise HTTPException(422, f"Version {body.version_id!r} is not implemented yet")

    pipeline = state["pipeline"]
    if media is not None:
        log.info("run %s (%s): media %r, %s, %d bytes", body.run_id, version.id,
                 media.filename, media.content_type, media.size or 0)
        events = pipeline.run_media(version, body.run_id, media.file, media.filename or "recording")
    elif body.text and body.text.strip():
        log.info("run %s (%s): text, %d characters", body.run_id, version.id, len(body.text))
        events = pipeline.run_text(version, body.run_id, body.text)
    else:
        log.warning("rejected request: no text and no media")
        raise HTTPException(422, "Send either text or a media file")

    def line(type_, data):
        return Event(run_id=body.run_id, version_id=version.id, type=type_, data=data).model_dump_json() + "\n"

    async def stream():
        """One JSON event per line, sent as soon as the pipeline yields it. Each step runs in a
        worker thread, so Whisper, web search and NLI don't block the server. If the browser
        goes away (new run, closed tab), the pipeline is closed after its current step."""
        try:
            while (event := await run_in_threadpool(lambda: next(events, None))) is not None:
                if await request.is_disconnected():
                    break
                yield line(*event)
        except Exception as error:   # the HTTP status is already sent: report the failure as an event
            yield line("run.error", {"message": repr(error)})
        finally:
            # On disconnect this task is cancelled; shield the cleanup so it still runs.
            # close() marks the source failed and closes the DB connection (a no-op if the run finished).
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(events.close)

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# Mounted last so /api routes take priority
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
