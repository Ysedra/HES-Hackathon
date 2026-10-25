"""
Claim verification (from the V0 notebooks, now shared by V0/V1/V3/V4): open web search, evidence
passages from the result pages, then the NLI model scores each passage against the claim.

Retrieval failure gives "retrieval_failed"; only usable but inconclusive evidence gives
"uncertain". The pipeline displays retrieval failures as "Not checked".
"""

import html
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from urllib.parse import quote, unquote

import requests
import torch
from ddgs import DDGS
from trafilatura import extract, fetch_url
from trafilatura.settings import use_config
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from services.extraction import is_checkable

NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
MAX_RESULTS = 3
MAX_PASSAGES = 5
MIN_CACHED_PASSAGES = 2   # V3: fewer relevant cached passages than this is an evidence-cache miss
EVIDENCE_STRENGTH_THRESHOLD = 0.60
CONFLICT_MARGIN = 0.10
MODEL_VERSION = f"{NLI_MODEL_NAME}+rules-v0.3"   # bump when evidence/verdict rules change
SEARCH_PROVIDERS = ("wikipedia", "duckduckgo", "brave")
SOURCE_POLICIES = {
    "wikipedia_first": SEARCH_PROVIDERS,
    "wikipedia_only": ("wikipedia",),
}
# Wikipedia is searched through its own API: free, no key, and full-text (ddgs's "wikipedia"
# backend only matches page titles, so a whole sentence found nothing). Wikipedia asks API
# clients to name themselves in the User-Agent.
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_HEADERS = {"User-Agent": "HES-Hackathon-Veriflow/0.1 (student fact-checking prototype; HES-SO)"}
WIKIPEDIA_TIMEOUT_S = 5
# The providers are asked at the same time; one that hasn't answered after this long is skipped
# (a blocked provider can otherwise hold every check for its whole 10 s timeout).
SEARCH_DEADLINE_S = 4
PAGE_DOWNLOAD_CONFIG = use_config()
PAGE_DOWNLOAD_CONFIG.set("DEFAULT", "DOWNLOAD_TIMEOUT", "8")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "has", "have", "in", "is", "it", "of", "on",
    "that", "the", "to", "was", "were", "with"
}
ENTERTAINMENT_SOURCE_MARKERS = {
    "song", "film", "movie", "album", "novel", "tv series", "television series"
}
ENTERTAINMENT_CLAIM_MARKERS = ENTERTAINMENT_SOURCE_MARKERS | {
    "released", "directed", "director", "actor", "actress", "stars", "starring",
    "singer", "band", "recorded", "performed", "written by", "premiered", "produced",
    "screenplay", "sequel", "soundtrack", "episode", "season", "runtime", "box office",
    "academy award", "oscar", "best picture", "music chart", "charts"
}


class Verifier:
    """Holds the NLI model, loaded once."""

    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        assert tokenizer is not None   # from_pretrained only returns None on failure; this narrows the type
        self.tokenizer = tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        self.model.eval()
        self.lock = threading.Lock()   # fast tokenizers raise errors when two threads use them at once

    def nli(self, premise, hypothesis):
        with self.lock:
            inputs = self.tokenizer(
                premise, hypothesis, return_tensors="pt",
                truncation="only_first", max_length=512,
            )
            with torch.inference_mode():
                scores = self.model(**inputs).logits.softmax(dim=-1)[0].tolist()
        return {self.model.config.id2label[i].lower(): score for i, score in enumerate(scores)}

    def check_claim(self, text, source_policy="wikipedia_first", evidence_cache=None):
        """Verify one claim. evidence_cache (V3, optional) has candidates(claim) -> [(text, url)]
        and save_page(url, sentences): cached page sentences are tried before searching, and
        are used only when they give a clear verdict; downloaded pages are saved to it."""
        text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text.strip(), flags=re.IGNORECASE)
        started = time.perf_counter()
        result = {"claim": text, "verdict": "not_checkable", "evidence": [],
                  "errors": [], "timings_ms": {}}
        timings = result["timings_ms"]
        checkable = is_checkable(text)
        timings["checkability"] = (time.perf_counter() - started) * 1000

        if checkable and evidence_cache is not None and self._check_cached_evidence(text, result, evidence_cache):
            result["retrieval"] = "evidence_cache"
        elif checkable:
            result["verdict"] = "uncertain"
            stage = time.perf_counter()
            try:
                hits, search_notes = search_web(text, source_policy)
                result["errors"].extend(search_notes)
            except Exception as error:
                hits = []
                result["errors"].append(f"Search failed: {error}")
            timings["search"] = (time.perf_counter() - stage) * 1000

            def save_page(url, sentences):   # a failed cache write must not fail the check
                saving = time.perf_counter()
                try:
                    evidence_cache.save_page(url, sentences)
                except Exception as error:
                    result["errors"].append(f"Evidence cache write failed: {error}")
                timings["evidence_save"] = timings.get("evidence_save", 0.0) + (time.perf_counter() - saving) * 1000

            stage = time.perf_counter()
            passages, errors = collect_evidence(text, hits, save_page if evidence_cache is not None else None)
            result["errors"].extend(errors)
            timings["retrieval"] = (time.perf_counter() - stage) * 1000

            # A verdict must be backed by prose extracted from a page. Search snippets never enter
            # the candidate set: they are often titles, table rows or truncated text.
            stage = time.perf_counter()
            self._score(text, [p for p in passages if p["source_type"] == "page"], result)
            timings["verification"] = (time.perf_counter() - stage) * 1000

        timings["total"] = (time.perf_counter() - started) * 1000
        result["timings_ms"] = {name: round(ms, 2) for name, ms in timings.items()}
        return result

    def _check_cached_evidence(self, text, result, evidence_cache):
        """Try the evidence cache. True when its passages give a clear verdict (stored in result);
        False on a miss or an unclear verdict, and the claim is then searched as usual."""
        timings = result["timings_ms"]
        stage = time.perf_counter()
        try:
            candidates = evidence_cache.candidates(text)
        except Exception as error:
            result["errors"].append(f"Evidence cache lookup failed: {error}")
            candidates = []
        passages, _ = rank_passages(text, [(passage, url, "page") for passage, url in candidates])
        timings["evidence_lookup"] = (time.perf_counter() - stage) * 1000
        if len(passages) < MIN_CACHED_PASSAGES:
            return False
        stage = time.perf_counter()
        trial = {"verdict": "uncertain", "evidence": []}
        self._score(text, passages, trial)
        timings["cached_verification"] = (time.perf_counter() - stage) * 1000
        if trial["verdict"] not in {"supported", "contradicted"}:
            return False
        result.update(verdict=trial["verdict"], evidence=trial["evidence"])
        return True

    def _score(self, text, page_passages, result):
        """Score page passages with NLI and set result's verdict and evidence."""
        for passage in page_passages:
            passage["scores"] = self.nli(passage["text"], text)
            passage["support_strength"] = round(
                passage["scores"]["entailment"] * passage["relevance"], 4
            )
            passage["contradiction_strength"] = round(
                passage["scores"]["contradiction"] * passage["relevance"], 4
            )
        if not page_passages:
            result["verdict"] = "retrieval_failed"
            return
        support = max(page_passages, key=lambda p: p["support_strength"])
        against = max(page_passages, key=lambda p: p["contradiction_strength"])
        support_score = support["support_strength"]
        contradiction_score = against["contradiction_strength"]
        supported = support_score >= EVIDENCE_STRENGTH_THRESHOLD
        contradicted = contradiction_score >= EVIDENCE_STRENGTH_THRESHOLD
        close_conflict = abs(support_score - contradiction_score) < CONFLICT_MARGIN
        if supported and contradicted and close_conflict:
            result["evidence"] = [support, against]
        elif supported and support_score > contradiction_score:
            result.update(verdict="supported", evidence=[support])
        elif contradicted:
            result.update(verdict="contradicted", evidence=[against])
        else:
            result["evidence"] = page_passages

def content_words(text):
    return set(re.findall(r"\w+", text.lower())) - STOPWORDS


def search_wikipedia(query, max_results=2):
    """Full-text search of English Wikipedia, in the result shape of ddgs (title, href, body)."""
    response = requests.get(WIKIPEDIA_API, headers=WIKIPEDIA_HEADERS, timeout=WIKIPEDIA_TIMEOUT_S, params={
        "action": "query", "list": "search", "srsearch": query, "srlimit": max_results,
        "srprop": "snippet", "format": "json", "formatversion": 2,
    })
    response.raise_for_status()
    results = []
    for page in response.json().get("query", {}).get("search", []):
        title = page["title"]
        snippet = html.unescape(re.sub(r"<[^>]+>", "", page.get("snippet", "")))   # drops <span class="searchmatch">
        results.append({"title": title, "body": snippet,
                        "href": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))})
    if not results:
        raise RuntimeError("No results found.")
    return results


def search_web(claim, source_policy="wikipedia_first"):
    """Search the providers (all at once) and return diverse, de-duplicated results in a fixed order.

    One result is taken from each provider before a second result is considered. This stops
    the first provider from silently consuming the whole result budget and makes runs easier
    to explain. ``wikipedia_only`` is available for a controlled source-limited experiment.
    """
    words = re.findall(r"\w+", claim.lower())
    simplified = " ".join(word for word in words if word not in STOPWORDS)
    queries = [claim]
    if simplified and simplified.lower() != claim.lower():
        queries.append(simplified)

    providers = SOURCE_POLICIES.get(source_policy)
    if providers is None:
        raise ValueError(f"Unknown source policy: {source_policy}")

    def ask(provider, query):
        if provider == "wikipedia":
            return search_wikipedia(query)
        return list(DDGS(timeout=10).text(query, backend=provider, max_results=2) or [])

    hits_by_provider = {provider: [] for provider in providers}
    notes = []
    for query in queries:
        # All providers at once: the search takes as long as the slowest answer, not the sum
        pool = ThreadPoolExecutor(max_workers=len(providers))
        futures = {provider: pool.submit(ask, provider, query) for provider in providers}
        wait(futures.values(), timeout=SEARCH_DEADLINE_S)
        pool.shutdown(wait=False)   # a provider still waiting finishes in the background, unused
        for provider, future in futures.items():   # the fixed provider order is kept
            if not future.done():
                notes.append(f"{provider} search timed out for {query!r} after {SEARCH_DEADLINE_S} s")
            elif future.exception() is not None:
                notes.append(f"{provider} search failed for {query!r}: {future.exception()}")
            else:
                hits_by_provider[provider].extend(future.result())
        if any(hits_by_provider.values()):
            break

    hits = []
    seen_urls = set()
    for result_index in range(2):
        for provider in providers:
            provider_hits = hits_by_provider[provider]
            if result_index >= len(provider_hits):
                continue
            result = provider_hits[result_index]
            url = result.get("href") or result.get("url")
            key = (url or repr(sorted(result.items()))).lower()
            if key in seen_urls:
                continue
            seen_urls.add(key)
            hits.append({**result, "search_provider": provider})
            if len(hits) == MAX_RESULTS:
                return hits, notes

    if hits:
        return hits, notes
    if not notes:
        notes.append("All configured search providers returned no results.")
    raise RuntimeError("; ".join(notes))


def rank_passages(claim, candidates):
    """Clean, filter and rank candidate passages, given as (text, url, source_type).
    Keeps prose passages sharing at least half the claim's words, best first. Returns
    (passages, rejected), rejected counting the title, table or non-prose candidates."""
    claim_words = content_words(claim)
    passages = []
    seen = set()
    rejected = 0
    for text, url, source_type in candidates:
        passage = clean_passage_text(text)[:700]
        key = passage.lower()
        if not passage or key in seen:
            continue
        if not is_prose_passage(passage):
            rejected += 1
            continue
        overlap = len(claim_words & content_words(passage))
        relevance = overlap / max(len(claim_words), 1)
        if relevance >= 0.5:
            seen.add(key)
            passages.append({
                "text": passage, "url": url, "source_type": source_type,
                "overlap": overlap, "relevance": round(relevance, 3),
            })
    passages.sort(
        key=lambda passage: (passage["relevance"], passage["source_type"] == "page"),
        reverse=True,
    )
    return passages[:MAX_PASSAGES], rejected


def collect_evidence(claim, search_results, on_page=None):
    """Download the search results' pages and rank their passages. on_page(url, sentences),
    if given, receives each downloaded page's prose sentences (for the V3 evidence cache)."""
    candidates = []
    errors = []
    for result in search_results:
        url = result.get("href") or result.get("url")
        if not url:
            continue
        if not source_context_matches(claim, result):
            errors.append(f"Skipped title-matched entertainment source without matching claim context: {url}")
            continue
        try:
            html = fetch_url(url, config=PAGE_DOWNLOAD_CONFIG)
            text = extract(html, include_comments=False) if html else None
        except Exception as error:
            errors.append(f"{url}: {error}")
            continue
        if not text:
            errors.append(f"{url}: no readable text")
            continue
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))
        candidates.extend((sentence, url, "page") for sentence in sentences)
        if on_page is not None:
            prose = (clean_passage_text(sentence)[:700] for sentence in sentences)
            on_page(url, [passage for passage in prose if passage and is_prose_passage(passage)])
    passages, rejected = rank_passages(claim, candidates)
    if rejected:
        errors.append(f"Filtered {rejected} title, table or non-prose evidence passage(s).")
    return passages, errors

def is_prose_passage(text):
    """Reject title/table noise without excluding facts *about* songs or films.

    A sentence such as 'Love Blind is a song by Jamiroquai' is prose and remains eligible.
    A search row such as '| Love Blind | 3:35 | 7 |' is not evidence of that relationship.
    """
    if text.count("|") >= 2 or "\t" in text:
        return False
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][\w'’-]*", text)
    if len(words) < 6:
        return False
    separators = len(re.findall(r"[|•·]", text))
    return separators <= max(1, len(words) // 8)


def source_context_matches(claim, search_result):
    """Avoid title collisions while retaining genuine film/song claims.

    Search can map the aphorism ``Love is blind`` to a song with the same title. We reject
    that page unless the claim itself signals an entertainment work or relation. This is a
    conservative V0/V1 heuristic; V2 can provide an explicit entity type.
    """
    url = unquote(search_result.get("href") or search_result.get("url") or "")
    descriptor = f"{search_result.get('title', '')} {url}".replace("_", " ").lower()
    claim_lower = claim.lower()
    entertainment_source = any(marker in descriptor for marker in ENTERTAINMENT_SOURCE_MARKERS)
    entertainment_claim = any(marker in claim_lower for marker in ENTERTAINMENT_CLAIM_MARKERS)
    return not entertainment_source or entertainment_claim


def clean_passage_text(text, max_prefix_words=8):
    """Normalize evidence and remove a duplicated title at the start of a search snippet.

    Search results sometimes begin with text such as ``Flat Earth Flat Earth is ...``.
    That repeated claim-like title can dominate a small NLI model and reverse its verdict.
    """
    words = re.sub(r"\s+", " ", text).strip().split()

    def normalized(part):
        return [re.sub(r"\W+", "", word).lower() for word in part]

    while len(words) >= 2:
        repeated = False
        for width in range(min(max_prefix_words, len(words) // 2), 0, -1):
            if normalized(words[:width]) == normalized(words[width:2 * width]):
                words = words[width:]
                repeated = True
                break
        if not repeated:
            break
    return " ".join(words)


def passage_stance(passage):
    """Stance of one evidence passage, using the same threshold as check_claim."""
    support, against = passage["support_strength"], passage["contradiction_strength"]
    if support >= EVIDENCE_STRENGTH_THRESHOLD and support > against:
        return "supports"
    if against >= EVIDENCE_STRENGTH_THRESHOLD:
        return "contradicts"
    return "neutral"
