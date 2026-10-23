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

V0 and V1 are sequential. The V4 prototype runs the same current V1 work with up to three
claims in flight, isolating the effect of concurrency before V2/V3 are connected.
"""

import logging
import math
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import BinaryIO, Generator, Iterable
from urllib.parse import urlparse

from schemas import ClaimOut, EvidenceOut, SegmentOut
from services import extraction, storage, transcription
from services.verification import EVIDENCE_STRENGTH_THRESHOLD, MODEL_VERSION, Verifier, passage_stance
from versions import Version

log = logging.getLogger("pipeline")
CACHE_MODEL_VERSION = f"v1-exact:web-r2:{MODEL_VERSION}"
PARALLEL_WORKERS = 3

Event = tuple[str, dict]
Run = Generator[Event, None, None]


class Pipeline:
    """Holds the models, loaded once at startup. The run_*() generators block for seconds
    between events: iterate them from a worker thread."""

    def __init__(self):
        # Intel Macs must keep CTranslate2 and PyTorch in separate processes because their
        # wheels bundle competing copies of libiomp5. The child is started only for media.
        self.whisper = None if transcription.needs_isolated_worker() else transcription.load_model()
        self.verifier = Verifier()
        storage.init_db()
        storage.init_claim_cache()

    def run_media(self, version: Version, run_id: str, media: BinaryIO, name: str) -> Run:
        segments = (transcription.transcribe_file_isolated(media, name)
                    if self.whisper is None else transcription.transcribe_file(self.whisper, media))
        return self._run(version, run_id, name, segments)

    def run_text(self, version: Version, run_id: str, text: str) -> Run:
        return self._run(version, run_id, "text input", [transcription.Segment(0, 0, text.strip())])

    def _run(self, version, run_id, source_url, segments: Iterable[transcription.Segment]) -> Run:
        started = time.perf_counter()
        con = storage.connect()
        cache_con = (storage.connect(storage.CACHE_DB_PATH)
                     if "claims" in version.caches and not version.parallel else None)
        executor = ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) if version.parallel else None
        pending = {}
        n_segments, claims = 0, []
        assembler = extraction.SentenceAssembler()
        seen_claims = set()   # one verification per normalized sentence for the entire run
        source_id = storage.create_source(con, source_url)
        log.info("run %s: source %d (%s)", run_id, source_id, source_url)

        def check(sentences):
            stable_at = storage.now_ms()   # every sentence in this completed batch is now stable
            for sentence in sentences:
                key = " ".join(sentence.text.casefold().split())
                if key in seen_claims:
                    log.info("skipped (repeat):         %s", sentence.text)
                    continue
                seen_claims.add(key)
                if not extraction.is_checkable(sentence.text):
                    log.info("skipped (not checkable):  %s", sentence.text)
                    continue   # not a claim: nothing stored, the passage stays "Not checked"
                claim_id = storage.insert_checking_claim(con, source_id, sentence, stable_at)
                claim = ClaimOut(claim_id=f"claim-{claim_id}", claim_text=sentence.text, status="checking",
                                 segment_ids=[f"segment-{i}" for i in dict.fromkeys(sentence.segment_ids)])
                yield "claim.update", claim.model_dump()   # shown as "Checking" while the search runs
                if executor is not None:
                    future = executor.submit(
                        self._check_in_worker, version, claim_id, claim, sentence, stable_at
                    )
                    pending[future] = claim
                else:
                    self._check(con, cache_con, claim_id, claim, sentence, stable_at,
                                version.source_policy)
                    claims.append(claim)
                    yield "claim.update", claim.model_dump()
                    yield "metrics.update", latency_metrics(
                        claims, cache_con is not None, queue_size=0
                    )

        def drain(finish=False):
            ready = as_completed(list(pending)) if finish else [future for future in pending if future.done()]
            for future in ready:
                claim = pending.pop(future)
                future.result()   # unexpected worker errors fail the run instead of disappearing
                claims.append(claim)
                yield "claim.update", claim.model_dump()
                yield "metrics.update", latency_metrics(
                    claims, "claims" in version.caches, queue_size=len(pending)
                )

        try:
            for segment in segments:   # media: Whisper runs here, window by window
                segment_id = storage.save_segment(con, source_id, segment)
                n_segments += 1
                log.info("segment %d [%d-%d ms]: %s", segment_id, segment.start_ms, segment.end_ms, segment.text)
                yield "transcript.segment", SegmentOut(id=f"segment-{segment_id}", start_ms=segment.start_ms,
                                                       end_ms=segment.end_ms, text=segment.text).model_dump()
                yield from check(assembler.add(segment_id, segment.start_ms, segment.end_ms, segment.text))
                if executor is not None:
                    yield from drain()
            yield from check(assembler.flush())   # the last sentence may have no closing punctuation
            if executor is not None:
                yield from drain(finish=True)
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
            if cache_con is not None:
                cache_con.close()
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

        metrics = latency_metrics(claims, "claims" in version.caches)
        log.info("run %s done in %.1f s: %d segments, %d claims %s, latency p50 %s ms, p95 %s ms",
                 run_id, time.perf_counter() - started, n_segments, len(claims),
                 dict(Counter(c.status for c in claims)), metrics.get("p50_ms"), metrics.get("p95_ms"))
        if not n_segments:
            log.warning("run %s: no speech was transcribed", run_id)
        yield "run.complete", {}

    def _check_in_worker(self, version, claim_id, claim, sentence, stable_at):
        """Give each parallel task its own SQLite connections; connections are never shared."""
        con = storage.connect()
        cache_con = storage.connect(storage.CACHE_DB_PATH) if "claims" in version.caches else None
        try:
            self._check(con, cache_con, claim_id, claim, sentence, stable_at,
                        version.source_policy)
        finally:
            con.close()
            if cache_con is not None:
                cache_con.close()

    def _check(self, con, cache_con, claim_id, claim: ClaimOut, sentence, stable_at,
               source_policy="wikipedia_first"):
        """Check one sentence, store the result and fill in claim. A failed check doesn't stop the run."""
        started = time.perf_counter()
        cache_lookup_ms = 0.0
        try:
            result = None
            if cache_con is not None:
                lookup_started = time.perf_counter()
                try:
                    result = storage.lookup_cached_claim(
                        cache_con, sentence.text, CACHE_MODEL_VERSION
                    )
                except Exception as error:
                    log.warning("claim %d cache lookup failed: %s", claim_id, error)
                cache_lookup_ms = (time.perf_counter() - lookup_started) * 1000

            if result is not None:
                result["timings_ms"] = {
                    "cache_lookup": round(cache_lookup_ms, 2),
                    "total": round((time.perf_counter() - started) * 1000, 2),
                }
                result["explanation"] = result.get("explanation") or explain(result)
                verdict_at = storage.save_result(
                    con, claim_id, result, MODEL_VERSION, passage_stance, path="claims_cache"
                )
                path = "claims_cache"
            else:
                result = self.verifier.check_claim(sentence.text, source_policy)
                result["explanation"] = explain(result)
                if cache_con is not None:
                    result["timings_ms"]["cache_lookup"] = round(cache_lookup_ms, 2)
                    write_started = time.perf_counter()
                    try:
                        storage.save_cached_claim(
                            cache_con, sentence.text, result, CACHE_MODEL_VERSION
                        )
                    except Exception as error:
                        log.warning("claim %d cache write failed: %s", claim_id, error)
                        result["errors"].append(f"Cache write failed: {error}")
                    result["timings_ms"]["cache_write"] = round(
                        (time.perf_counter() - write_started) * 1000, 2
                    )
                    result["timings_ms"]["total"] = round(
                        (time.perf_counter() - started) * 1000, 2
                    )
                verdict_at = storage.save_result(
                    con, claim_id, result, MODEL_VERSION, passage_stance, path="fresh"
                )
                path = "fresh"
        except Exception as error:
            log.exception("claim %d failed: %s", claim_id, sentence.text)
            storage.mark_claim_failed(con, claim_id)
            claim.status = "not_checked"
            claim.explanation = "The check failed before a verdict was reached."
            claim.errors = [repr(error)]
            return

        no_verdict = result["verdict"] in {"not_checkable", "retrieval_failed"}
        claim.status = "not_checked" if no_verdict else result["verdict"]
        claim.path = None if result["verdict"] == "not_checkable" else path
        claim.latency_ms = verdict_at - stable_at
        log.info("claim %d: %-13s%7.0f ms  %-12s %s", claim_id, result["verdict"],
                 result["timings_ms"]["total"], path, sentence.text)
        for error in result["errors"]:
            log.warning("claim %d retrieval: %s", claim_id, error)
        claim.timings_ms = result["timings_ms"]
        claim.errors = result["errors"]
        claim.explanation = result["explanation"]
        claim.confidence = confidence(result)
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
    if verdict == "retrieval_failed":
        if any(error.startswith("Search failed:") for error in result["errors"]):
            return ("Not checked: the configured search providers returned no usable results "
                    "or failed, so the verifier had no evidence to score.")
        return ("Not checked: search results existed, but no readable, relevant prose passage "
                "was retrieved. Titles, table rows and snippets cannot produce a verdict.")
    if not evidence:
        if any(error.startswith("Search failed:") for error in result["errors"]):
            return ("The primary and fallback searches returned no usable results, so there "
                    "was no evidence from which to reach a verdict.")
        if result["errors"]:
            return ("Search results were found, but their pages could not provide a readable, "
                    "relevant passage. Without evidence, the claim remains uncertain.")
        return ("Search completed, but no passage was relevant enough to assess this claim. "
                "Without evidence, the claim remains uncertain.")
    if len(evidence) == 2 and all(passage_stance(p) != "neutral" for p in evidence):
        return "Passages both support and contradict the claim with similar strength."
    support = max(p["support_strength"] for p in evidence)
    contradiction = max(p["contradiction_strength"] for p in evidence)
    return (f"Evidence was found, but its strongest support ({support:.2f}) and contradiction "
            f"({contradiction:.2f}) stayed below the {EVIDENCE_STRENGTH_THRESHOLD:.2f} threshold.")


def confidence(result):
    """Expose the winning uncalibrated evidence strength, including for uncertain results."""
    if not result["evidence"]:
        return None
    if result["verdict"] == "supported":
        return max(p["support_strength"] for p in result["evidence"])
    if result["verdict"] == "contradicted":
        return max(p["contradiction_strength"] for p in result["evidence"])
    return max(max(p["support_strength"], p["contradiction_strength"])
               for p in result["evidence"])


def latency_metrics(claims, cache_enabled=False, queue_size=0):
    """p50/p95 over the claims that got a verdict so far (not over skipped sentences)."""
    latencies = sorted(c.latency_ms for c in claims if c.latency_ms is not None)
    metrics = {"queue_size": queue_size}
    if latencies:
        metrics.update(p50_ms=statistics.median(latencies),
                       p95_ms=latencies[math.ceil(0.95 * len(latencies)) - 1])
    if cache_enabled and claims:
        hits = sum(claim.path == "claims_cache" for claim in claims)
        metrics.update(cache_hits=hits, cache_misses=len(claims) - hits,
                       cache_hit_rate=100 * hits / len(claims))
    return metrics
