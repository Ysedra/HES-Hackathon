"""
The pipeline: runs the services in order for one request (v0, from pipeline_v0.ipynb).

    input -> transcript segments -> sentences -> checkable claims -> verify -> save

Every input becomes a stream of transcript segments, so the rest of the pipeline is the same:
- media (an uploaded file or a microphone recording from the browser): transcribed by Whisper
- text: already a transcript, so it is one segment

run_*() are generators of (type, data) events, in the frontend's event format, so each segment
and claim reaches the browser as soon as it is ready:
- transcript.segment: a new transcript segment
- claim.update: a claim, first as "checking", then again with its verdict
- metrics.update: latency so far
- run.complete: the end of the run

v0 is sequential: each complete sentence is checked before the next segments are read, so a
claim's latency includes the time it waits behind earlier claims.
"""

import logging
import math
import statistics
import time
from collections import Counter, deque
from typing import BinaryIO, Generator, Iterable
from urllib.parse import urlparse

from schemas import ClaimOut, EvidenceOut, SegmentOut
from services import extraction, storage, transcription
from services.verification import EVIDENCE_STRENGTH_THRESHOLD, MODEL_VERSION, Verifier, passage_stance
from versions import Version

log = logging.getLogger("pipeline")

Event = tuple[str, dict]
Run = Generator[Event, None, None]


class Pipeline:
    """Holds the models, loaded once at startup. The run_*() generators block for seconds
    between events: iterate them from a worker thread."""

    def __init__(self):
        self.whisper = transcription.load_model()
        self.verifier = Verifier()
        storage.init_db()

    def run_media(self, version: Version, run_id: str, media: BinaryIO, name: str) -> Run:
        segments = transcription.transcribe_file(self.whisper, media)
        return self._run(version, run_id, name, segments)

    def run_text(self, version: Version, run_id: str, text: str) -> Run:
        return self._run(version, run_id, "text input", [transcription.Segment(0, 0, text.strip())])

    def _run(self, version, run_id, source_url, segments: Iterable[transcription.Segment]) -> Run:
        started = time.perf_counter()
        con = storage.connect()
        n_segments, claims = 0, []
        assembler = extraction.SentenceAssembler()
        recent = deque(maxlen=5)   # skip repeats of recent sentences, e.g. six "The Great Depression." in a row
        source_id = storage.create_source(con, source_url)
        log.info("run %s: source %d (%s)", run_id, source_id, source_url)

        def check(sentences):
            stable_at = storage.now_ms()   # the sentences are complete now: the latency stopwatch starts
            for sentence in sentences:
                key = sentence.text.lower()
                if key in recent:
                    log.info("skipped (repeat):         %s", sentence.text)
                    continue
                recent.append(key)
                if not extraction.is_checkable(sentence.text):
                    log.info("skipped (not checkable):  %s", sentence.text)
                    continue   # not a claim: nothing stored, the passage stays "Not checked"
                claim_id = storage.insert_checking_claim(con, source_id, sentence, stable_at)
                claim = ClaimOut(claim_id=f"claim-{claim_id}", claim_text=sentence.text, status="checking",
                                 segment_ids=[f"segment-{i}" for i in dict.fromkeys(sentence.segment_ids)])
                yield "claim.update", claim.model_dump()   # shown as "Checking" while the search runs
                self._check(con, claim_id, claim, sentence, stable_at)
                claims.append(claim)
                yield "claim.update", claim.model_dump()
                yield "metrics.update", latency_metrics(claims)

        try:
            for segment in segments:   # media: Whisper runs here, window by window
                segment_id = storage.save_segment(con, source_id, segment)
                n_segments += 1
                log.info("segment %d [%d-%d ms]: %s", segment_id, segment.start_ms, segment.end_ms, segment.text)
                yield "transcript.segment", SegmentOut(id=f"segment-{segment_id}", start_ms=segment.start_ms,
                                                       end_ms=segment.end_ms, text=segment.text).model_dump()
                yield from check(assembler.add(segment_id, segment.start_ms, segment.end_ms, segment.text))
            yield from check(assembler.flush())   # the last sentence may have no closing punctuation
            storage.set_source_status(con, source_id, "done")
        except GeneratorExit:   # the browser closed the stream (new run, closed tab)
            log.warning("run %s cancelled by the client (source %d marked failed)", run_id, source_id)
            storage.set_source_status(con, source_id, "failed")
            raise
        except Exception:
            log.exception("run %s failed (source %d marked failed)", run_id, source_id)
            storage.set_source_status(con, source_id, "failed")   # keep what was saved
            raise
        finally:
            con.close()

        metrics = latency_metrics(claims)
        log.info("run %s done in %.1f s: %d segments, %d claims %s, latency p50 %s ms, p95 %s ms",
                 run_id, time.perf_counter() - started, n_segments, len(claims),
                 dict(Counter(c.status for c in claims)), metrics.get("p50_ms"), metrics.get("p95_ms"))
        if not n_segments:
            log.warning("run %s: no speech was transcribed", run_id)
        yield "run.complete", {}

    def _check(self, con, claim_id, claim: ClaimOut, sentence, stable_at):
        """Check one sentence, store the result and fill in claim. A failed check doesn't stop the run."""
        try:
            result = self.verifier.check_claim(sentence.text)
            verdict_at = storage.save_result(con, claim_id, result, MODEL_VERSION, passage_stance)
        except Exception as error:
            log.exception("claim %d failed: %s", claim_id, sentence.text)
            storage.mark_claim_failed(con, claim_id)
            claim.status = "not_checked"
            claim.explanation = "The check failed before a verdict was reached."
            claim.errors = [repr(error)]
            return

        claim.status = result["verdict"] if result["verdict"] != "not_checkable" else "not_checked"
        claim.path = "fresh" if result["verdict"] != "not_checkable" else None
        claim.latency_ms = verdict_at - stable_at
        log.info("claim %d: %-13s%7.0f ms  %s", claim_id, result["verdict"],
                 result["timings_ms"]["total"], sentence.text)
        for error in result["errors"]:
            log.warning("claim %d retrieval: %s", claim_id, error)
        claim.timings_ms = result["timings_ms"]
        claim.errors = result["errors"]
        claim.explanation = explain(result)
        claim.evidence = [EvidenceOut(text=p["text"], url=p["url"], source_type=p["source_type"],
                                      scores={k: round(v, 4) for k, v in p["scores"].items()})
                          for p in result["evidence"]]


def explain(result):
    """A short assessment built from the evidence. The NLI model gives scores, not reasons."""
    verdict, evidence = result["verdict"], result["evidence"]
    if verdict == "supported":
        p = evidence[0]
        return (f"A passage from {urlparse(p['url']).netloc} entails the claim "
                f"(support strength {p['support_strength']:.2f}, threshold {EVIDENCE_STRENGTH_THRESHOLD}).")
    if verdict == "contradicted":
        p = evidence[0]
        return (f"A passage from {urlparse(p['url']).netloc} contradicts the claim "
                f"(contradiction strength {p['contradiction_strength']:.2f}, threshold {EVIDENCE_STRENGTH_THRESHOLD}).")
    if verdict == "not_checkable":
        return "Not a checkable factual claim."
    if not evidence:
        return "No relevant evidence passage was found." + (" Retrieval had errors." if result["errors"] else "")
    if len(evidence) == 2 and all(passage_stance(p) != "neutral" for p in evidence):
        return "Passages both support and contradict the claim with similar strength."
    return f"No passage reached the evidence threshold of {EVIDENCE_STRENGTH_THRESHOLD}."


def latency_metrics(claims):
    """p50/p95 over the claims that got a verdict so far (not over skipped sentences)."""
    latencies = sorted(c.latency_ms for c in claims if c.latency_ms is not None)
    if not latencies:
        return {}
    return {"p50_ms": statistics.median(latencies),
            "p95_ms": latencies[math.ceil(0.95 * len(latencies)) - 1]}
