"""
Request and response shapes of the /api/analyse routes, matching the frontend contract
(webapp/frontend/README.md). Pydantic validates them, so a malformed request gets a clear error.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RunMetadata(BaseModel):
    """Identifies a run: the 'metadata' field of a media upload."""
    run_id: str
    version_id: str
    config: dict = {}       # the frontend's copy of the version config; versions.py is the reference


class TextRequest(RunMetadata):
    """The JSON body of /api/analyse/text."""
    text: str = Field(min_length=1, pattern=r"\S")   # at least one non-space character


class LiveRequest(RunMetadata):
    """The JSON body of /api/analyse/live."""
    source: Literal["loopback", "microphone"] = "loopback"


class StopRequest(BaseModel):
    """The JSON body of /api/analyse/live/stop."""
    run_id: str


class SegmentOut(BaseModel):
    id: str
    start_ms: int
    end_ms: int
    text: str
    is_stable: bool = True
    extraction_complete: bool = True


class EvidenceOut(BaseModel):
    text: str
    url: str
    source_type: str        # "page"; search snippets are never accepted as verdict evidence
    scores: dict[str, float]   # NLI entailment, contradiction, neutral


class ClaimOut(BaseModel):
    claim_id: str
    segment_ids: list[str]
    claim_text: str
    status: str             # not_checked, checking, supported, contradicted or uncertain
    path: str | None = None   # fresh, claims_cache or evidence_cache
    confidence: float | None = None
    explanation: str | None = None
    latency_ms: int | None = None   # from the sentence becoming complete to its verdict
    timings_ms: dict[str, float] = {}
    evidence: list[EvidenceOut] = []
    errors: list[str] = []


class Event(BaseModel):
    """One line of the streamed response (NDJSON): the frontend's event envelope."""
    run_id: str
    version_id: str
    type: str               # transcript.segment, claim.update, metrics.update, run.complete or run.error
    data: dict
