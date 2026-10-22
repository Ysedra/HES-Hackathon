"""
HTTP helpers shared by the routes in main.py: turning versions and pipeline runs into HTTP
errors and streamed responses. The rest of the backend (pipeline, versions, services) stays
HTTP-free, so notebooks can import it.
"""

import logging

import anyio
from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from pipeline import Run
from schemas import Event, RunMetadata
from versions import Version, get_version

log = logging.getLogger("main")


def require_version(version_id) -> Version:
    """The implemented version with this id, or HTTP 422: never fall back to the baseline."""
    version = get_version(version_id)
    if version is None:
        log.warning("rejected request: version %r is not implemented", version_id)
        raise HTTPException(422, f"Version {version_id!r} is not implemented yet")
    return version


def stream(request: Request, body: RunMetadata, version: Version, events: Run) -> StreamingResponse:
    """Send the run's events as NDJSON, one per line, as soon as the pipeline yields them.
    Each step runs in a worker thread, so Whisper, web search and NLI don't block the server.
    If the browser goes away (new run, closed tab), the pipeline is closed after its current step."""

    def line(type_, data):
        return Event(run_id=body.run_id, version_id=version.id, type=type_, data=data).model_dump_json() + "\n"

    def next_event() -> tuple[str, dict] | None:
        return next(events, None)   # None when the run is over

    async def lines():
        try:
            while (event := await run_in_threadpool(next_event)) is not None:
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

    return StreamingResponse(lines(), media_type="application/x-ndjson")
