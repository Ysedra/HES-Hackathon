"""
The pipeline: runs the services in order for one request (v0, from pipeline_v0.ipynb).

    input -> transcript segments -> sentences -> checkable claims -> verify -> save

Every input becomes a stream of transcript segments, so the rest of the pipeline is the same:
- media (an uploaded file or a microphone recording from the browser): transcribed by Whisper
- live (the sound of the computer, loopback): transcribed by Kroko while it plays, until stopped
- text: already a transcript, so it is one segment

run_*() are generators of (type, data) events, in the frontend's event format, so each segment
and claim reaches the browser as soon as it is ready:
- transcript.segment: a new transcript segment
- claim.update: a claim, first as "checking", then again with its verdict
- metrics.update: latency so far
- run.complete: the end of the run

V0 and V1 are sequential: each claim is checked before the next segment is transcribed. The V4
prototype runs the same V1 work concurrently, isolating the effect of concurrency before V2/V3 are
connected: transcription and cache lookups in a producer thread, fresh checks in a pool of workers
(chosen per run, within versions.MAX_WORKERS), NLI still one call at a time.
"""

import logging
import math
import queue
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import BinaryIO, Generator, Iterable
from urllib.parse import urlparse

from schemas import ClaimOut, EvidenceOut, SegmentOut
from services import extraction, storage, transcription
from services.verification import (EVIDENCE_STRENGTH_THRESHOLD, MODEL_VERSION, Verifier, content_words,
                                   passage_stance)
from versions import Version

log = logging.getLogger("pipeline")
# web-r3: Wikipedia is searched through its own API (verification.search_wikipedia), so verdicts
# cached with the older search (web-r2) are not reused. The evidence cache keeps web-r2: its
# rows are page sentences, which don't depend on how the page was found.
CACHE_MODEL_VERSION = f"v1-exact:web-r3:{MODEL_VERSION}"
EVIDENCE_CACHE_VERSION = f"v3-fts:web-r2:{MODEL_VERSION}"

Event = tuple[str, dict]
Run = Generator[Event, None, None]


class EvidenceCache:
    """V3's evidence cache (storage.py), in the shape the verifier's evidence_cache hook expects.
    Each call opens its own connection, so one instance can be used from any thread."""

    def candidates(self, claim):
        con = storage.connect(storage.EVIDENCE_DB_PATH)
        try:
            return storage.lookup_cached_evidence(con, sorted(content_words(claim)), EVIDENCE_CACHE_VERSION)
        finally:
            con.close()

    def save_page(self, url, sentences):
        con = storage.connect(storage.EVIDENCE_DB_PATH)
        try:
            storage.save_cached_page(con, url, sentences, EVIDENCE_CACHE_VERSION)
        finally:
            con.close()


class Pipeline:
    """Holds the models, loaded once at startup. The run_*() generators block for seconds
    between events: iterate them from a worker thread."""

    def __init__(self):
        # Intel Macs must keep CTranslate2 and PyTorch in separate processes because their
        # wheels bundle competing copies of libiomp5. The child is started only for media.
        self.whisper = None if transcription.needs_isolated_worker() else transcription.load_model()
        self.kroko = None   # the live model, loaded at the first live run
        self._kroko_lock = threading.Lock()
        self.verifier = Verifier()
        storage.init_db()
        storage.init_claim_cache()
        storage.init_evidence_cache()

    def run_media(self, version: Version, run_id: str, media: BinaryIO, name: str) -> Run:
        segments = (transcription.transcribe_file_isolated(media, name)
                    if self.whisper is None else transcription.transcribe_file(self.whisper, media))
        return self._run(version, run_id, name, segments)

    def run_text(self, version: Version, run_id: str, text: str) -> Run:
        return self._run(version, run_id, "text input", [transcription.Segment(0, 0, text.strip())])

    def run_live(self, version: Version, run_id: str, stop: threading.Event, source="loopback",
                 partials=False) -> Run:
        """Transcribe the computer's sound with Kroko and check its claims until stop is set.
        partials=True also sends the unfinished sentence ("transcript.partial", for live.html).
        stop is also set when the run ends for any other reason (error, closed tab), so the
        capture never keeps the audio device open. V4 suits it best: the sequential versions
        stop reading the transcript while they check a claim, so the text falls behind."""
        try:
            with self._kroko_lock:
                if self.kroko is None:
                    log.info("loading Kroko...")
                    self.kroko = transcription.load_kroko()
            segments = transcription.transcribe_live_kroko(self.kroko, stop, source, partials)
            yield from self._run(version, run_id, f"live {source}", segments)
        finally:
            stop.set()

    def _run(self, version, run_id, source_url, segments: Iterable[transcription.Segment]) -> Run:
        started = time.perf_counter()
        con = storage.connect()
        claims = []
        source_id = storage.create_source(con, source_url)
        log.info("run %s: source %d (%s), %s", run_id, source_id, source_url,
                 f"{version.workers} workers" if version.parallel else "sequential")
        run = self._run_parallel if version.parallel else self._run_sequential
        try:
            n_segments = yield from run(version, con, source_id, segments, claims)
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

        metrics = latency_metrics(claims, "claims" in version.caches)
        log.info("run %s done in %.1f s: %d segments, %d claims %s, latency p50 %s ms, p95 %s ms",
                 run_id, time.perf_counter() - started, n_segments, len(claims),
                 dict(Counter(c.status for c in claims)), metrics.get("p50_ms"), metrics.get("p95_ms"))
        if not n_segments:
            log.warning("run %s: no speech was transcribed", run_id)
        yield "run.complete", {}

    def _claims(self, con, source_id, segments):
        """Save each segment and find the checkable sentences it completes. Yields the
        ("transcript.segment", data) events, ("claim", (claim_id, claim, sentence, stable_at))
        for each new claim, already stored as "checking", ("pause", None) for each live Pause,
        and ("transcript.partial", data) for each live Partial."""
        assembler = extraction.SentenceAssembler()
        seen_claims = set()   # one verification per normalized sentence for the entire run

        def claims_in(sentences):
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
                yield "claim", (claim_id, claim, sentence, stable_at)

        for segment in segments:   # media: Whisper runs here, window by window; live: Kroko
            if isinstance(segment, transcription.Pause):
                # Live: the sentence is finished or the speaker paused. pysbd keeps the last
                # sentence until the next one starts, which can take many seconds live.
                yield from claims_in(assembler.flush())
                yield "pause", None
                continue
            if isinstance(segment, transcription.Partial):   # not stored: only shown until the sentence ends
                yield "transcript.partial", {"start_ms": segment.start_ms, "text": segment.text}
                continue
            segment_id = storage.save_segment(con, source_id, segment)
            log.info("segment %d [%d-%d ms]: %s", segment_id, segment.start_ms, segment.end_ms, segment.text)
            yield "transcript.segment", SegmentOut(id=f"segment-{segment_id}", start_ms=segment.start_ms,
                                                   end_ms=segment.end_ms, text=segment.text).model_dump()
            yield from claims_in(assembler.add(segment_id, segment.start_ms, segment.end_ms, segment.text))
        yield from claims_in(assembler.flush())   # the last sentence may have no closing punctuation

    def _run_sequential(self, version, con, source_id, segments, claims):
        """V0/V1: each claim is checked as soon as it is complete, before the next segment.
        Returns the number of segments."""
        cache_con = storage.connect(storage.CACHE_DB_PATH) if "claims" in version.caches else None
        evidence_cache = EvidenceCache() if "evidence" in version.caches else None
        n_segments = 0
        try:
            for type_, data in self._claims(con, source_id, segments):
                if type_ in ("transcript.segment", "transcript.partial"):
                    n_segments += type_ == "transcript.segment"
                    yield type_, data
                    continue
                if type_ == "pause":   # live: an event now and then, so a closed browser is noticed
                    yield "metrics.update", latency_metrics(claims, cache_con is not None,
                                                            evidence_enabled=evidence_cache is not None)
                    continue
                claim_id, claim, sentence, stable_at = data
                yield "claim.update", claim.model_dump()   # shown as "Checking" while the search runs
                self._check(con, cache_con, claim_id, claim, sentence, stable_at, version.source_policy,
                            evidence_cache)
                claims.append(claim)
                yield "claim.update", claim.model_dump()
                yield "metrics.update", latency_metrics(claims, cache_con is not None,
                                                        evidence_enabled=evidence_cache is not None)
        finally:
            if cache_con is not None:
                cache_con.close()
        return n_segments

    def _run_parallel(self, version, con, source_id, segments, claims):
        """V4: a producer thread transcribes, splits sentences and answers cache hits at once;
        cache misses go to a pool of version.workers threads for the fresh check. Both put their
        results on one queue, which this generator empties, so each verdict is sent as soon as it
        is ready, even while Whisper works on the next window. Returns the number of segments."""
        results = queue.Queue()   # (kind, data); kinds: event, tick, hit, submitted, checked, produced, error
        stop = threading.Event()   # set when the run ends early: the producer stops after its current step
        executor = ThreadPoolExecutor(max_workers=version.workers)

        def fresh_check(claim_id, claim, sentence, stable_at, lookup_ms):
            try:
                self._check_in_worker(version, claim_id, claim, sentence, stable_at, lookup_ms)
                results.put(("checked", claim))
            except BaseException as error:   # failed checks are handled in _check_fresh: this is a bug
                results.put(("error", error))

        def produce():
            # Its own connections: the generator thread still uses con, for the source status
            producer_con = storage.connect()
            cache_con = storage.connect(storage.CACHE_DB_PATH) if "claims" in version.caches else None
            try:
                for type_, data in self._claims(producer_con, source_id, segments):
                    if stop.is_set():
                        return
                    if type_ in ("transcript.segment", "transcript.partial"):
                        results.put(("event", (type_, data)))
                        continue
                    if type_ == "pause":
                        results.put(("tick", None))
                        continue
                    claim_id, claim, sentence, stable_at = data
                    results.put(("event", ("claim.update", claim.model_dump())))   # "Checking"
                    done, lookup_ms = self._check_cached(producer_con, cache_con, claim_id, claim,
                                                         sentence, stable_at)
                    if done:
                        results.put(("hit", claim))
                    else:
                        results.put(("submitted", None))   # before submit, so it's counted before its result
                        executor.submit(fresh_check, claim_id, claim, sentence, stable_at, lookup_ms)
                results.put(("produced", None))
            except BaseException as error:
                results.put(("error", error))
            finally:
                producer_con.close()
                if cache_con is not None:
                    cache_con.close()

        threading.Thread(target=produce, name=f"producer-{source_id}", daemon=True).start()
        n_segments, pending, produced = 0, 0, False
        try:
            while not (produced and pending == 0):
                kind, data = results.get()
                if kind == "error":
                    raise data   # unexpected errors fail the run instead of disappearing
                if kind == "produced":
                    produced = True
                elif kind == "submitted":
                    pending += 1
                elif kind == "event":
                    n_segments += data[0] == "transcript.segment"
                    yield data
                elif kind == "tick":   # live Pause: an event now and then, so a closed browser is noticed
                    yield "metrics.update", latency_metrics(claims, "claims" in version.caches,
                                                            queue_size=pending)
                else:   # a verdict: "hit" from the cache or "checked" by a worker
                    pending -= kind == "checked"
                    claims.append(data)
                    yield "claim.update", data.model_dump()
                    yield "metrics.update", latency_metrics(
                        claims, "claims" in version.caches, queue_size=pending
                    )
        finally:
            stop.set()
            executor.shutdown(wait=False, cancel_futures=True)
        return n_segments

    def _check_in_worker(self, version, claim_id, claim, sentence, stable_at, lookup_ms):
        """A fresh check in a pool thread, with its own SQLite connections; connections are never shared."""
        con = storage.connect()
        cache_con = storage.connect(storage.CACHE_DB_PATH) if "claims" in version.caches else None
        try:
            self._check_fresh(con, cache_con, claim_id, claim, sentence, stable_at,
                              version.source_policy, lookup_ms)
        finally:
            con.close()
            if cache_con is not None:
                cache_con.close()

    def _check(self, con, cache_con, claim_id, claim: ClaimOut, sentence, stable_at,
               source_policy="wikipedia_first", evidence_cache=None):
        """Check one sentence in this thread: the claim cache first, then a fresh check on a miss."""
        done, lookup_ms = self._check_cached(con, cache_con, claim_id, claim, sentence, stable_at)
        if not done:
            self._check_fresh(con, cache_con, claim_id, claim, sentence, stable_at, source_policy, lookup_ms,
                              evidence_cache)

    def _check_cached(self, con, cache_con, claim_id, claim: ClaimOut, sentence, stable_at):
        """Answer from the claim cache if it holds this sentence: store the result and fill in claim.
        Returns (done, lookup time in ms); done is False on a miss or without a cache."""
        if cache_con is None:
            return False, 0.0
        started = time.perf_counter()
        try:
            result = storage.lookup_cached_claim(cache_con, sentence.text, CACHE_MODEL_VERSION)
        except Exception as error:
            log.warning("claim %d cache lookup failed: %s", claim_id, error)
            result = None
        lookup_ms = (time.perf_counter() - started) * 1000
        if result is None:
            return False, lookup_ms
        try:
            result["timings_ms"] = {"cache_lookup": round(lookup_ms, 2),
                                    "total": round((time.perf_counter() - started) * 1000, 2)}
            result["explanation"] = result.get("explanation") or explain(result)
            verdict_at = storage.save_result(
                con, claim_id, result, MODEL_VERSION, passage_stance, path="claims_cache"
            )
        except Exception as error:
            self._fail(con, claim_id, claim, sentence, error)
        else:
            self._fill(claim_id, claim, sentence, result, "claims_cache", verdict_at - stable_at)
        return True, lookup_ms

    def _check_fresh(self, con, cache_con, claim_id, claim: ClaimOut, sentence, stable_at,
                     source_policy="wikipedia_first", lookup_ms=0.0, evidence_cache=None):
        """Verify with the evidence cache (V3) or the web and NLI, store the result (and cache it)
        and fill in claim. A failed check doesn't stop the run."""
        started = time.perf_counter()
        try:
            result = self.verifier.check_claim(sentence.text, source_policy, evidence_cache)
            path = result.get("retrieval", "fresh")   # "evidence_cache" when cached evidence decided it
            result["explanation"] = explain(result)
            if cache_con is not None:
                result["timings_ms"]["cache_lookup"] = round(lookup_ms, 2)
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
                    lookup_ms + (time.perf_counter() - started) * 1000, 2
                )
            verdict_at = storage.save_result(
                con, claim_id, result, MODEL_VERSION, passage_stance, path=path
            )
        except Exception as error:
            self._fail(con, claim_id, claim, sentence, error)
            return
        self._fill(claim_id, claim, sentence, result, path, verdict_at - stable_at)

    def _fail(self, con, claim_id, claim: ClaimOut, sentence, error):
        log.exception("claim %d failed: %s", claim_id, sentence.text)
        storage.mark_claim_failed(con, claim_id)
        claim.status = "not_checked"
        claim.explanation = "The check failed before a verdict was reached."
        claim.errors = [repr(error)]

    def _fill(self, claim_id, claim: ClaimOut, sentence, result, path, latency_ms):
        no_verdict = result["verdict"] in {"not_checkable", "retrieval_failed"}
        claim.status = "not_checked" if no_verdict else result["verdict"]
        claim.path = None if result["verdict"] == "not_checkable" else path
        claim.latency_ms = latency_ms
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


def latency_metrics(claims, cache_enabled=False, queue_size=0, evidence_enabled=False):
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
    if evidence_enabled and claims:
        metrics.update(evidence_hits=sum(claim.path == "evidence_cache" for claim in claims))
    return metrics
