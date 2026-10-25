# Live Video Transcription and Fact Checking

A HES-SO 301/302 hackathon project: a local web app that transcribes speech, picks out factual claims and checks them against evidence from the web, all with models running on a laptop.

You give it text, an audio/video file or a microphone recording. The transcript streams into the page segment by segment, and each claim is marked as it is checked:

| State | Meaning |
| --- | --- |
| Not checked | Not a checkable claim, or retrieval found no usable evidence |
| Checking | Search or verification in progress |
| Supported | A retrieved passage entails the claim |
| Contradicted | A retrieved passage contradicts the claim |
| Uncertain | Evidence was found, but it was weak or conflicting |

Selecting a claim shows the explanation, the evidence passages with their source URLs, the NLI scores and per-stage timings. The research question is how far verification latency can be reduced (caching, parallelism, smaller models) without losing quality, so the dashboard also reports p50/p95 latency and cache hit rate.

The backend implements three pipeline versions, selectable in the UI:

| Version | ID | What it adds |
| --- | --- | --- |
| V0 | `baseline` | Rule-based claim extraction, web search, NLI verification, sequential |
| V1 | `claim-cache` | Exact-match cache of verified claims (1 h TTL, max 1000 entries) |
| V4 | `parallel` | V1 with three claims checked at once |

V2 (model-based claim extraction) and V3 (evidence cache) are planned and shown as unavailable.

## Repository layout

```
HES-Hackathon/
├── webapp/
│   ├── backend/
│   │   ├── main.py               # FastAPI routes; also serves the frontend
│   │   ├── api_helpers.py        # version check, streamed NDJSON responses
│   │   ├── schemas.py            # request/response models (frontend contract)
│   │   ├── versions.py           # pipeline versions and the steps each enables
│   │   ├── pipeline.py           # runs the services for one request
│   │   └── services/
│   │       ├── transcription.py  # faster-whisper, file and live capture
│   │       ├── extraction.py     # sentence assembly and checkability rules
│   │       ├── verification.py   # web search, passage extraction, NLI verdict
│   │       └── storage.py        # SQLite schema, run data and claim cache
│   └── frontend/                 # static dashboard (index.html, app.js, styles.css)
│                                 # README.md describes the API contract
├── database/                     # created on startup, not tracked
│   ├── app.db                    # run data: sources, segments, claims, verdicts
│   └── v1_claim_cache.db         # V1 verified-claim cache
├── asr/                          # speech-recognition experiments (notebooks)
├── benchmarks/                   # saved V1 benchmark results (JSON)
├── samples/                      # test audio and its reference transcript
├── ai_agent.ipynb                # V0 verifier prototype
├── ai_agent_v1.ipynb             # V1 claim-cache prototype
├── claim_extraction.ipynb        # LLM claim-extraction experiment (Qwen2.5)
├── database.ipynb                # database schema design
├── pipeline_v0.ipynb             # end-to-end V0 pipeline prototype
├── audio_capture.py              # Windows loopback capture test
├── live_transcribe.py            # live transcription with faster-whisper
├── live_transcribe_kroko.py      # live transcription with sherpa-onnx
├── pyproject.toml                # dependencies
└── uv.lock                       # locked environment
```

The services don't depend on HTTP, so the notebooks can import them for benchmarks.

## How to run

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). Internet access is needed for web search and, on the first run, to download the models.

From the repository root:

```bash
uv sync
uv run fastapi dev webapp/backend/main.py
```

Wait for `Application startup complete` (Whisper and the NLI model load first), then open <http://localhost:8000/>. The API docs are at <http://localhost:8000/docs>.

- The status dot in the top bar is grey while the backend is loading or down, green when the selected version is implemented, and amber when it isn't.
- `fastapi dev` reloads on backend changes. After editing the frontend, refresh the page (Ctrl+F5 if the browser caches `app.js`).
- Don't open `webapp/frontend/index.html` directly or serve it with `python -m http.server`: the page then has no API and shows the backend as disconnected.
- Run `uv sync` again after pulling changes.
- **Clear V1 claim cache** in the dashboard empties `v1_claim_cache.db` only; run history in `app.db` is kept.

For the notebooks, select `.venv` as the kernel and run them from the repository root.

## Dependencies
### Models

All inference runs locally on the CPU. Models are downloaded from Hugging Face on first use.

| Use | Model | Where |
| --- | --- | --- |
| Speech to text | Whisper `small` via [faster-whisper](https://github.com/SYSTRAN/faster-whisper), int8, English | `services/transcription.py` |
| Claim verification (NLI) | [`cross-encoder/nli-MiniLM2-L6-H768`](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768) | `services/verification.py` |
| Claim extraction (experiment only) | Qwen2.5 0.5B / 1.5B Instruct | `claim_extraction.ipynb` |
| ASR comparisons (experiments only) | Whisper large-v3-turbo, sherpa-onnx models | `asr/`, `live_transcribe_kroko.py` |

Claim extraction in the web app is rule-based (`services/extraction.py`), using pySBD for sentence splitting.

### Retrieval

Search goes through [ddgs](https://pypi.org/project/ddgs/) with a fixed provider order: Wikipedia, then DuckDuckGo, then Brave. No API keys are needed. Result pages are downloaded and reduced to readable prose with [trafilatura](https://trafilatura.readthedocs.io/); a verdict needs at least one passage taken from a page, never just a search snippet.

### Databases

SQLite, created automatically in `database/` (WAL mode, foreign keys on). Both files use the schema in `services/storage.py`:

| File | Contents |
| --- | --- |
| `app.db` | `sources`, `transcript_segments`, `claims`, `claim_segments`, plus each verdict with its evidence (`verified_claims`, `evidence`, `verdict_evidence`) |
| `v1_claim_cache.db` | V1 cache: reusable `verified_claims` with their cited `evidence`. Entries are keyed by normalised claim text and verifier version, so changing the model or rules invalidates them |

### Python packages

Declared in `pyproject.toml`, locked in `uv.lock`. The main ones:

- **Web:** `fastapi[standard]`
- **Speech:** `faster-whisper`, `av` (audio decoding), `soundcard` / `sounddevice` (live capture, Windows), `sherpa-onnx`
- **NLP:** `torch`, `transformers`, `pysbd`
- **Retrieval:** `ddgs`, `trafilatura`
- **Storage:** `sqlite-vec` (for the planned embedding search)
- **Evaluation:** `jiwer` (word error rate), `datasets`

Intel Macs get pinned older versions of `torch`, `transformers`, `numpy` and `onnxruntime`, and Whisper runs in a separate process there to avoid an OpenMP library conflict.
