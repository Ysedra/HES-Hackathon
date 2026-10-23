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
VERDICT_TTL_MS = 7 * 24 * 60 * 60 * 1000   # how long a verdict stays reusable in verified_claims

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


def save_result(con, claim_id, result, model_version, stance):
    """Store check_claim's verdict, evidence and timings; this stops the latency stopwatch.
    stance(passage) gives each passage's stance. Returns verdict_at."""
    verdict_at = now_ms()
    timings_json = json.dumps(result["timings_ms"])
    with con:
        if result["verdict"] == "not_checkable":
            con.execute(
                "UPDATE claims SET status = 'not_checked', verdict_at = ?, timings_json = ? WHERE id = ?",
                (verdict_at, timings_json, claim_id),
            )
            return verdict_at

        cur = con.execute(
            """INSERT INTO verified_claims (claim_text, verdict, verified_at, expires_at, model_version)
               VALUES (?, ?, ?, ?, ?)""",
            (result["claim"], result["verdict"], verdict_at, verdict_at + VERDICT_TTL_MS, model_version),
        )
        verified_claim_id = cur.lastrowid

        for passage in result["evidence"]:
            cur = con.execute(
                "INSERT INTO evidence (evidence_text, source_url, fetched_at) VALUES (?, ?, ?)",
                (passage["text"], passage["url"], verdict_at),   # fetched during this check
            )
            scores = {
                "relevance": passage["relevance"],
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
               SET status = ?, verified_claim_id = ?, path = 'fresh', verdict_at = ?, timings_json = ?
               WHERE id = ?""",
            (result["verdict"], verified_claim_id, verdict_at, timings_json, claim_id),
        )
    return verdict_at


def mark_claim_failed(con, claim_id):
    with con:
        con.execute("UPDATE claims SET status = 'not_checked' WHERE id = ?", (claim_id,))
