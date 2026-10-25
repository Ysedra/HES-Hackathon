"""
Storage service: the SQLite database from database.ipynb.

The schema is created on startup if it is missing (CREATE ... IF NOT EXISTS keeps existing data).
Each run opens its own connection with connect(); connections must never be used by two threads at once.
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[3] / "database" / "app.db"
CACHE_DB_PATH = DB_PATH.with_name("v1_claim_cache.db")
EVIDENCE_DB_PATH = DB_PATH.with_name("v3_evidence_cache.db")
VERDICT_TTL_MS = 7 * 24 * 60 * 60 * 1000   # how long a verdict stays reusable in verified_claims
CACHE_TTL_MS = 60 * 60 * 1000
MAX_CACHE_ENTRIES = 1000
EVIDENCE_TTL_MS = 24 * 60 * 60 * 1000   # page text changes slowly
MAX_CACHED_PAGES = 500
MAX_SENTENCES_PER_PAGE = 2000

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
  id          INTEGER PRIMARY KEY,
  source_url  TEXT    NOT NULL,
  status      TEXT    NOT NULL DEFAULT 'queued'
              CHECK (status IN ('queued','transcribing','processing','done','failed')),
  created_at  INTEGER NOT NULL
              DEFAULT (CAST((julianday('now') - 2440587.5) * 86400000 AS INTEGER))
) STRICT;

-- Cache 1: verified claims (reusable verdicts)
CREATE TABLE IF NOT EXISTS verified_claims (
  id             INTEGER PRIMARY KEY,
  claim_text     TEXT    NOT NULL,
  embedding      BLOB,
  entities       TEXT,                       -- JSON
  numbers        TEXT,                       -- JSON
  polarity       INTEGER NOT NULL DEFAULT 1  -- 1 assertion, 0 negation
                 CHECK (polarity IN (0,1)),
  verdict        TEXT    NOT NULL
                 CHECK (verdict IN ('supported','contradicted','uncertain')),
  explanation    TEXT,
  verified_at    INTEGER NOT NULL,
  expires_at     INTEGER NOT NULL,
  model_version  TEXT    NOT NULL,
  last_used_at   INTEGER,
  hit_count      INTEGER NOT NULL DEFAULT 0
) STRICT;

-- Cache 2: evidence passages (the RAG index)
CREATE TABLE IF NOT EXISTS evidence (
  id             INTEGER PRIMARY KEY,
  evidence_text  TEXT    NOT NULL,
  source_url     TEXT    NOT NULL,
  source_date    TEXT,
  embedding      BLOB,
  fetched_at     INTEGER NOT NULL,
  last_used_at   INTEGER
) STRICT;

-- Which passages back which verdict
CREATE TABLE IF NOT EXISTS verdict_evidence (
  verified_claim_id  INTEGER NOT NULL
                     REFERENCES verified_claims(id) ON DELETE CASCADE,
  evidence_id        INTEGER NOT NULL
                     REFERENCES evidence(id) ON DELETE CASCADE,
  stance             TEXT    NOT NULL
                     CHECK (stance IN ('supports','contradicts','neutral')),
  scores_json        TEXT,                -- JSON: relevance, entailment, contradiction
  PRIMARY KEY (verified_claim_id, evidence_id)
) STRICT;

CREATE TABLE IF NOT EXISTS transcript_segments (
  id         INTEGER PRIMARY KEY,
  video_id   INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  start_ms   INTEGER NOT NULL,
  end_ms     INTEGER NOT NULL CHECK (end_ms >= start_ms),
  text       TEXT    NOT NULL,
  is_stable  INTEGER NOT NULL DEFAULT 0 CHECK (is_stable IN (0,1))
) STRICT;

CREATE TABLE IF NOT EXISTS claims (
  id                 INTEGER PRIMARY KEY,
  video_id           INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  verified_claim_id  INTEGER REFERENCES verified_claims(id) ON DELETE SET NULL,
  claim_text         TEXT    NOT NULL,
  start_ms           INTEGER NOT NULL,
  end_ms             INTEGER NOT NULL CHECK (end_ms >= start_ms),
  status             TEXT    NOT NULL DEFAULT 'not_checked'
                     CHECK (status IN ('not_checked','checking','supported',
                                       'contradicted','uncertain')),
  path               TEXT
                     CHECK (path IS NULL OR
                            path IN ('fresh','claims_cache','evidence_cache')),
  stable_at          INTEGER NOT NULL,   -- latency stopwatch starts
  verdict_at         INTEGER,            -- latency stopwatch stops
  timings_json       TEXT                -- JSON: per-stage timings in ms
) STRICT;

CREATE TABLE IF NOT EXISTS claim_segments (
  claim_id    INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
  segment_id  INTEGER NOT NULL REFERENCES transcript_segments(id) ON DELETE CASCADE,
  PRIMARY KEY (claim_id, segment_id)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_segments_video ON transcript_segments (video_id, start_ms);
CREATE INDEX IF NOT EXISTS idx_claims_video ON claims (video_id, start_ms);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims (status);
CREATE INDEX IF NOT EXISTS idx_claims_verified ON claims (verified_claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_segments_segment ON claim_segments (segment_id);
CREATE INDEX IF NOT EXISTS idx_verified_expires ON verified_claims (expires_at);
CREATE INDEX IF NOT EXISTS idx_verified_lru ON verified_claims (last_used_at);
CREATE INDEX IF NOT EXISTS idx_evidence_lru ON evidence (last_used_at);
CREATE INDEX IF NOT EXISTS idx_evidence_url ON evidence (source_url);
CREATE INDEX IF NOT EXISTS idx_verdict_evidence_evidence ON verdict_evidence (evidence_id);
CREATE INDEX IF NOT EXISTS idx_verified_claim_text ON verified_claims (claim_text);
CREATE INDEX IF NOT EXISTS idx_evidence_fetched ON evidence (fetched_at);
"""

# V3's evidence cache: the prose sentences of every downloaded page, kept in the evidence table
# (cache 2) of its own file and searched through a full-text index over that table, so a claim
# about an already-fetched topic can skip search and download. One page = its rows with that
# source_url; expiry uses fetched_at, and least-recently-used pruning uses last_used_at.
EVIDENCE_SEARCH_INDEX = """
CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts
  USING fts5(evidence_text, content='evidence', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS evidence_fts_insert AFTER INSERT ON evidence BEGIN
  INSERT INTO evidence_fts (rowid, evidence_text) VALUES (new.id, new.evidence_text);
END;
CREATE TRIGGER IF NOT EXISTS evidence_fts_delete AFTER DELETE ON evidence BEGIN
  INSERT INTO evidence_fts (evidence_fts, rowid, evidence_text) VALUES ('delete', old.id, old.evidence_text);
END;
"""


def now_ms():
    """The shared clock from database.ipynb: epoch milliseconds."""
    return time.time_ns() // 1_000_000


def connect(path=DB_PATH) -> sqlite3.Connection:
    """Open the database with the settings every connection should use.

    check_same_thread=False: a streamed run is iterated by different worker threads, one step
    at a time, so the connection moves between threads but is never used by two at once."""
    con = sqlite3.connect(path, timeout=5, check_same_thread=False)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA synchronous = NORMAL")
    con.execute("PRAGMA busy_timeout = 5000")
    return con


def init_db(path=DB_PATH):
    """Create the database file and any missing tables."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = connect(path)
    try:
        con.execute("PRAGMA journal_mode = WAL")   # stored in the file, so set once
        con.executescript(SCHEMA)
    finally:
        con.close()


def init_claim_cache():
    """Create/migrate V1's separate verified-claim cache."""
    init_db(CACHE_DB_PATH)
    con = connect(CACHE_DB_PATH)
    try:
        columns = {row[1] for row in con.execute("PRAGMA table_info(verified_claims)")}
        if "warnings_json" not in columns:
            with con:
                con.execute(
                    "ALTER TABLE verified_claims ADD COLUMN warnings_json TEXT NOT NULL DEFAULT '[]'"
                )
    finally:
        con.close()


def create_source(con, source_url):
    with con:  # create the source row first, so the DB shows it straight away
        cur = con.execute("INSERT INTO sources (source_url, status) VALUES (?, 'transcribing')",
                          (source_url,))
    return cur.lastrowid


def set_source_status(con, source_id, status):
    with con:
        con.execute("UPDATE sources SET status = ? WHERE id = ?", (status, source_id))


def save_segment(con, source_id, segment):
    with con:  # commit now, so readers see it while the run continues
        cur = con.execute(
            """INSERT INTO transcript_segments (video_id, start_ms, end_ms, text, is_stable)
               VALUES (?, ?, ?, ?, 1)""",
            (source_id, segment.start_ms, segment.end_ms, segment.text),
        )
    return cur.lastrowid


def insert_checking_claim(con, source_id, sentence, stable_at):
    """Store a sentence as a claim being checked, linked to its transcript segments."""
    with con:
        cur = con.execute(
            """INSERT INTO claims (video_id, claim_text, start_ms, end_ms, status, stable_at)
               VALUES (?, ?, ?, ?, 'checking', ?)""",
            (source_id, sentence.text, sentence.start_ms, sentence.end_ms, stable_at),
        )
        claim_id = cur.lastrowid
        con.executemany(
            "INSERT INTO claim_segments (claim_id, segment_id) VALUES (?, ?)",
            [(claim_id, segment_id) for segment_id in dict.fromkeys(sentence.segment_ids)],
        )
    return claim_id


def save_result(con, claim_id, result, model_version, stance, path="fresh"):
    """Store check_claim's verdict, evidence and timings; this stops the latency stopwatch.
    stance(passage) gives each passage's stance. Returns verdict_at."""
    verdict_at = now_ms()
    timings_json = json.dumps(result["timings_ms"])
    with con:
        if result["verdict"] in {"not_checkable", "retrieval_failed"}:
            con.execute(
                "UPDATE claims SET status = 'not_checked', verdict_at = ?, timings_json = ? WHERE id = ?",
                (verdict_at, timings_json, claim_id),
            )
            return verdict_at

        cur = con.execute(
            """INSERT INTO verified_claims
               (claim_text, verdict, explanation, verified_at, expires_at, model_version)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (result["claim"], result["verdict"], result.get("explanation"), verdict_at,
             verdict_at + VERDICT_TTL_MS, model_version),
        )
        verified_claim_id = cur.lastrowid

        for passage in result["evidence"]:
            cur = con.execute(
                "INSERT INTO evidence (evidence_text, source_url, fetched_at) VALUES (?, ?, ?)",
                (passage["text"], passage["url"], verdict_at),   # fetched during this check
            )
            scores = {
                "relevance": passage["relevance"],
                "source_type": passage.get("source_type", "page"),
                **passage["scores"],                              # entailment, neutral, contradiction
                "support_strength": passage["support_strength"],
                "contradiction_strength": passage["contradiction_strength"],
            }
            con.execute(
                """INSERT INTO verdict_evidence (verified_claim_id, evidence_id, stance, scores_json)
                   VALUES (?, ?, ?, ?)""",
                (verified_claim_id, cur.lastrowid, stance(passage), json.dumps(scores)),
            )

        con.execute(
            """UPDATE claims
               SET status = ?, verified_claim_id = ?, path = ?, verdict_at = ?, timings_json = ?
               WHERE id = ?""",
            (result["verdict"], verified_claim_id, path, verdict_at, timings_json, claim_id),
        )
    return verdict_at


def normalise_claim(text):
    """V1's deliberately conservative key: exact text after whitespace normalisation."""
    return " ".join(text.split()).strip()


def _delete_cached_claims(con, claim_ids):
    for cached_id in claim_ids:
        evidence_ids = con.execute(
            "SELECT evidence_id FROM verdict_evidence WHERE verified_claim_id = ?",
            (cached_id,),
        ).fetchall()
        con.execute("DELETE FROM verified_claims WHERE id = ?", (cached_id,))
        for (evidence_id,) in evidence_ids:
            con.execute(
                """DELETE FROM evidence WHERE id = ?
                   AND NOT EXISTS (
                     SELECT 1 FROM verdict_evidence WHERE evidence_id = evidence.id
                   )""",
                (evidence_id,),
            )


def prune_claim_cache(con):
    """Remove expired V1 entries, then apply the notebook's LRU capacity."""
    expired = con.execute(
        """SELECT id FROM verified_claims
           WHERE model_version LIKE 'v1-exact:%' AND expires_at <= ?""",
        (now_ms(),),
    ).fetchall()
    _delete_cached_claims(con, [row[0] for row in expired])
    overflow = con.execute(
        """SELECT id FROM verified_claims
           WHERE model_version LIKE 'v1-exact:%'
           ORDER BY COALESCE(last_used_at, verified_at) DESC, id DESC
           LIMIT -1 OFFSET ?""",
        (MAX_CACHE_ENTRIES,),
    ).fetchall()
    _delete_cached_claims(con, [row[0] for row in overflow])


def lookup_cached_claim(con, text, cache_model_version):
    """Return a still-valid exact V1 verdict together with its original citations."""
    row = con.execute(
        """SELECT id, verdict, explanation, warnings_json
           FROM verified_claims
           WHERE claim_text = ? AND model_version = ? AND expires_at > ?
             AND verdict IN ('supported', 'contradicted')
           ORDER BY verified_at DESC, id DESC LIMIT 1""",
        (normalise_claim(text), cache_model_version, now_ms()),
    ).fetchone()
    if row is None:
        return None

    cached_id, verdict, explanation, warnings_json = row
    rows = con.execute(
        """SELECT e.evidence_text, e.source_url, ve.scores_json
           FROM verdict_evidence AS ve
           JOIN evidence AS e ON e.id = ve.evidence_id
           WHERE ve.verified_claim_id = ? ORDER BY e.id""",
        (cached_id,),
    ).fetchall()
    if not rows:
        return None

    evidence = []
    for passage, url, metadata_json in rows:
        metadata = json.loads(metadata_json or "{}")
        evidence.append({"text": passage, "url": url, **metadata})
    with con:
        con.execute(
            """UPDATE verified_claims
               SET last_used_at = ?, hit_count = hit_count + 1 WHERE id = ?""",
            (now_ms(), cached_id),
        )
    return {
        "claim": text,
        "verdict": verdict,
        "explanation": explanation,
        "evidence": evidence,
        "errors": json.loads(warnings_json or "[]"),
    }


def save_cached_claim(con, text, result, cache_model_version):
    """Cache only cited page-backed verdicts; uncertain/retrieval failures remain misses."""
    if result["verdict"] not in {"supported", "contradicted"} or not result["evidence"]:
        return False
    if not all(passage.get("text") and passage.get("url") for passage in result["evidence"]):
        return False
    if not all(passage.get("source_type") == "page" for passage in result["evidence"]):
        return False

    verified_at = now_ms()
    claim_text = normalise_claim(text)
    with con:
        previous = con.execute(
            "SELECT id FROM verified_claims WHERE claim_text = ? AND model_version = ?",
            (claim_text, cache_model_version),
        ).fetchall()
        _delete_cached_claims(con, [row[0] for row in previous])
        cur = con.execute(
            """INSERT INTO verified_claims
               (claim_text, verdict, explanation, verified_at, expires_at,
                model_version, last_used_at, warnings_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (claim_text, result["verdict"], result.get("explanation"), verified_at,
             verified_at + CACHE_TTL_MS, cache_model_version, verified_at,
             json.dumps(result.get("errors", []))),
        )
        cached_id = cur.lastrowid
        for passage in result["evidence"]:
            evidence_id = con.execute(
                "INSERT INTO evidence (evidence_text, source_url, fetched_at) VALUES (?, ?, ?)",
                (passage["text"], passage["url"], verified_at),
            ).lastrowid
            metadata = {key: value for key, value in passage.items() if key not in {"text", "url"}}
            con.execute(
                """INSERT INTO verdict_evidence
                   (verified_claim_id, evidence_id, stance, scores_json)
                   VALUES (?, ?, ?, ?)""",
                (cached_id, evidence_id,
                 "supports" if result["verdict"] == "supported" else "contradicts",
                 json.dumps(metadata)),
            )
        prune_claim_cache(con)
    return True


def clear_claim_cache():
    """Clear only V1 experiment-cache verdicts and their unreferenced evidence."""
    con = connect(CACHE_DB_PATH)
    try:
        with con:
            rows = con.execute(
                "SELECT id FROM verified_claims WHERE model_version LIKE 'v1-exact:%'"
            ).fetchall()
            _delete_cached_claims(con, [row[0] for row in rows])
        return len(rows)
    finally:
        con.close()


def init_evidence_cache():
    """Create/migrate V3's separate evidence cache: the evidence table plus its full-text index."""
    init_db(EVIDENCE_DB_PATH)
    con = connect(EVIDENCE_DB_PATH)
    try:
        columns = {row[1] for row in con.execute("PRAGMA table_info(evidence)")}
        with con:
            if "cache_version" not in columns:
                con.execute("ALTER TABLE evidence ADD COLUMN cache_version TEXT NOT NULL DEFAULT ''")
            con.executescript(EVIDENCE_SEARCH_INDEX)
    finally:
        con.close()


def _delete_cached_pages(con, urls):
    con.executemany("DELETE FROM evidence WHERE source_url = ?", [(url,) for url in urls])


def prune_evidence_cache(con):
    """Remove expired pages, then the least recently used beyond MAX_CACHED_PAGES."""
    con.execute("DELETE FROM evidence WHERE fetched_at <= ?", (now_ms() - EVIDENCE_TTL_MS,))
    overflow = con.execute(
        """SELECT source_url FROM evidence GROUP BY source_url
           ORDER BY MAX(COALESCE(last_used_at, fetched_at)) DESC LIMIT -1 OFFSET ?""",
        (MAX_CACHED_PAGES,),
    ).fetchall()
    _delete_cached_pages(con, [row[0] for row in overflow])


def save_cached_page(con, url, sentences, cache_version):
    """Store a downloaded page's prose sentences, replacing an older copy of the same page."""
    fetched_at = now_ms()
    with con:
        _delete_cached_pages(con, [url])
        con.executemany(
            """INSERT INTO evidence (evidence_text, source_url, fetched_at, last_used_at, cache_version)
               VALUES (?, ?, ?, ?, ?)""",
            [(sentence, url, fetched_at, fetched_at, cache_version)
             for sentence in sentences[:MAX_SENTENCES_PER_PAGE]],
        )
        prune_evidence_cache(con)


def lookup_cached_evidence(con, words, cache_version, limit=50):
    """Cached sentences sharing words with a claim, best full-text matches first, from pages
    that are still valid. Returns [(text, url)]; the caller decides which are relevant."""
    words = [word for word in words if word.strip()]
    if not words:
        return []
    query = " OR ".join('"' + word.replace('"', '""') + '"' for word in words)
    rows = con.execute(
        """SELECT e.evidence_text, e.source_url FROM evidence_fts
           JOIN evidence AS e ON e.id = evidence_fts.rowid
           WHERE evidence_fts MATCH ? AND e.cache_version = ? AND e.fetched_at > ?
           ORDER BY bm25(evidence_fts) LIMIT ?""",
        (query, cache_version, now_ms() - EVIDENCE_TTL_MS, limit),
    ).fetchall()
    if rows:
        with con:
            con.executemany("UPDATE evidence SET last_used_at = ? WHERE source_url = ?",
                            [(now_ms(), url) for url in {url for _, url in rows}])
    return rows


def clear_evidence_cache():
    """Clear V3's evidence cache; returns the number of pages removed."""
    con = connect(EVIDENCE_DB_PATH)
    try:
        with con:
            pages = con.execute("SELECT COUNT(DISTINCT source_url) FROM evidence").fetchone()[0]
            con.execute("DELETE FROM evidence")
        return pages
    finally:
        con.close()


def mark_claim_failed(con, claim_id):
    with con:
        con.execute("UPDATE claims SET status = 'not_checked' WHERE id = ?", (claim_id,))
