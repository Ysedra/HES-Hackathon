"""
Claim verification (v0, from ai_agent.ipynb via pipeline_v0.ipynb): open web search, evidence
passages from the result pages, then the NLI model scores each passage against the claim.

Missing or weak evidence gives "uncertain", never a guessed verdict.
"""

import re
import threading
import time

import torch
from ddgs import DDGS
from trafilatura import extract, fetch_url
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from services.extraction import is_checkable

NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
MAX_RESULTS = 3
MAX_PASSAGES = 5
EVIDENCE_STRENGTH_THRESHOLD = 0.60
CONFLICT_MARGIN = 0.10
MODEL_VERSION = f"{NLI_MODEL_NAME}+rules-v0"   # stored with each verdict, to tell cached verdicts apart later

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "has", "have", "in", "is", "it", "of", "on",
    "that", "the", "to", "was", "were", "with"
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

    def check_claim(self, text):
        text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text.strip(), flags=re.IGNORECASE)
        started = time.perf_counter()
        result = {"claim": text, "verdict": "not_checkable", "evidence": [],
                  "errors": [], "timings_ms": {}}
        timings = result["timings_ms"]
        checkable = is_checkable(text)
        timings["checkability"] = (time.perf_counter() - started) * 1000

        if checkable:
            result["verdict"] = "uncertain"
            stage = time.perf_counter()
            try:
                hits = search_web(text)
            except Exception as error:
                hits = []
                result["errors"].append(f"Search failed: {error}")
            timings["search"] = (time.perf_counter() - stage) * 1000

            stage = time.perf_counter()
            passages, errors = collect_evidence(text, hits)
            result["errors"].extend(errors)
            timings["retrieval"] = (time.perf_counter() - stage) * 1000

            stage = time.perf_counter()
            for passage in passages:
                passage["scores"] = self.nli(passage["text"], text)
                passage["support_strength"] = round(
                    passage["scores"]["entailment"] * passage["relevance"], 4
                )
                passage["contradiction_strength"] = round(
                    passage["scores"]["contradiction"] * passage["relevance"], 4
                )
            if passages:
                support = max(passages, key=lambda p: p["support_strength"])
                against = max(passages, key=lambda p: p["contradiction_strength"])
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
                    result["evidence"] = passages
            timings["verification"] = (time.perf_counter() - stage) * 1000

        timings["total"] = (time.perf_counter() - started) * 1000
        result["timings_ms"] = {name: round(ms, 2) for name, ms in timings.items()}
        return result


def content_words(text):
    return set(re.findall(r"\w+", text.lower())) - STOPWORDS


def search_web(claim):
    return list(DDGS(timeout=10).text(claim, max_results=MAX_RESULTS))


def collect_evidence(claim, search_results):
    claim_words = content_words(claim)
    passages = []
    errors = []
    seen = set()

    def add_passage(text, url, source_type):
        passage = re.sub(r"\s+", " ", text).strip()[:700]
        key = passage.lower()
        if not passage or key in seen:
            return
        overlap = len(claim_words & content_words(passage))
        relevance = overlap / max(len(claim_words), 1)
        if relevance >= 0.5:
            seen.add(key)
            passages.append({
                "text": passage, "url": url, "source_type": source_type,
                "overlap": overlap, "relevance": round(relevance, 3),
            })

    for result in search_results:
        url = result.get("href") or result.get("url")
        if not url:
            continue
        snippet = result.get("body") or result.get("snippet") or ""
        if snippet:
            add_passage(snippet, url, "search_snippet")
        try:
            html = fetch_url(url)
            text = extract(html, include_comments=False) if html else None
        except Exception as error:
            errors.append(f"{url}: {error}")
            continue
        if not text:
            errors.append(f"{url}: no readable text")
            continue
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))
        for sentence in sentences:
            add_passage(sentence, url, "page")
    passages.sort(
        key=lambda passage: (passage["relevance"], passage["source_type"] == "page"),
        reverse=True,
    )
    return passages[:MAX_PASSAGES], errors


def passage_stance(passage):
    """Stance of one evidence passage, using the same threshold as check_claim."""
    support, against = passage["support_strength"], passage["contradiction_strength"]
    if support >= EVIDENCE_STRENGTH_THRESHOLD and support > against:
        return "supports"
    if against >= EVIDENCE_STRENGTH_THRESHOLD:
        return "contradicts"
    return "neutral"
