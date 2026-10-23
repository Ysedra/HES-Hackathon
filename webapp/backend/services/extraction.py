"""
Claim extraction, rule-based (v0, from pipeline_v0.ipynb and ai_agent.ipynb).

Transcript segments are regrouped into complete sentences; each checkable sentence is one claim.
Model-based extraction (claim_extraction.ipynb) will replace this for the later versions.
"""

import re
import warnings
from dataclasses import dataclass
from typing import cast

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")   # harmless warnings from an old package
import pysbd
from pysbd.utils import TextSpan

_segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)


@dataclass
class Sentence:
    text: str
    start_ms: int
    end_ms: int
    segment_ids: list


def sentence_cuts(text):
    """Character positions where one sentence ends and the next begins.
    The last sentence is left out: it may still be unfinished."""
    spans = cast(list[TextSpan], _segmenter.segment(text))   # char_span=True returns TextSpans
    return [span.start for span in spans[1:]]


class SentenceAssembler:
    """Regroup transcript segments into complete sentences.

    Feed segments in order with add(); it returns the sentences they complete and keeps
    the unfinished tail until a later segment ends it. Call flush() at the end of the source."""

    def __init__(self, max_words=60):
        self.max_words = max_words   # cut anyway if no punctuation arrives for this long
        self.text = ""
        self.spans = []              # (start_char, end_char, segment_id, start_ms, end_ms)

    def add(self, segment_id, start_ms, end_ms, text):
        if self.text:
            self.text += " "
        start = len(self.text)
        self.text += text.strip()
        self.spans.append((start, len(self.text), segment_id, start_ms, end_ms))

        sentences, prev = [], 0
        for cut in sentence_cuts(self.text):
            sentences.append(self._make(prev, cut))
            prev = cut
        if len(self.text[prev:].split()) > self.max_words:
            sentences.append(self._make(prev, len(self.text)))
            prev = len(self.text)
        self._drop(prev)
        return sentences

    def flush(self):
        if not self.text.strip():
            return []
        sentences = [self._make(0, len(self.text))]
        self._drop(len(self.text))
        return sentences

    def _make(self, a, b):
        parts = [sp for sp in self.spans if sp[0] < b and sp[1] > a]
        return Sentence(self.text[a:b].strip(), parts[0][3], parts[-1][4], [p[2] for p in parts])

    def _drop(self, upto):
        """Forget the text before `upto`, keeping the unfinished tail."""
        self.text = self.text[upto:]
        self.spans = [(max(s - upto, 0), e - upto, sid, a, b)
                      for s, e, sid, a, b in self.spans if e > upto]


QUESTION_STARTS = {
    "who", "what", "when", "where", "why", "how", "which",
    "is", "are", "do", "does", "did", "can", "could", "would", "should"
}
OPINION_MARKERS = {
    "i think", "i believe", "i feel", "in my opinion", "i prefer",
    "best", "worst", "better than", "worse than", "superior to", "inferior to",
    "beautiful", "ugly", "delicious", "boring", "amazing", "favorite"
}
META_TEXT = re.compile(r"\b(?:is|are)\s+(?:a|an|the)?\s*(?:statement|sentence|phrase|word|claim)\b")
COMMAND_STARTS = {
    "please", "tell", "give", "show", "explain", "close", "open",
    "stop", "go", "make", "write", "run", "click", "select"
}


def is_checkable(text):
    clean = text.strip()
    words = re.findall(r"[a-z']+", clean.lower())
    first_word = words[0] if words else ""

    if len(words) < 3:
        return False
    if clean.endswith("?") or first_word in QUESTION_STARTS:
        return False
    if first_word in COMMAND_STARTS:
        return False
    if any(marker in clean.lower() for marker in OPINION_MARKERS):
        return False
    if META_TEXT.search(clean.lower()):
        return False
    return True
