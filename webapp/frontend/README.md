# Veriflow frontend

A dependency-free local UI for the hackathon. It does not run models, caches or transcription by itself.

## Preview

From the repository root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open <http://localhost:8000/frontend/>. Use **Try interface demo** to see a compound snippet become two individually selectable claims. All demo verdicts, source excerpts and model scores are labelled illustrative. No demo timings enter the experiment log.

A static server supports the interface and local microphone recording, but cannot handle `POST /api/analyse`. Real output needs the Python backend served on the same origin (or a same-origin proxy). Missing endpoints produce an explicit error, never synthetic verdicts. The media player is a local preview, not live transcription.

## Planned versions

| ID | Caches | AI stages | Processing |
| --- | --- | --- | --- |
| `baseline` | None | Verification; rule-based extraction | Sequential |
| `claim-cache` | Verified claims | Verification; rule-based extraction | Sequential |
| `two-ai` | Verified claims | Model-based extraction + verification | Sequential |
| `two-caches` | Verified claims + evidence | Model-based extraction + verification | Sequential |
| `parallel` | Verified claims + evidence | Model-based extraction + verification | Bounded parallel |

These are provisional definitions. Change `VERSIONS` in `app.js` if the team chooses a different role for the second AI stage or first cache. A version selection passes configuration to the backend; it does **not** implement the optimisation. Unsupported configurations must be rejected by the backend (for example HTTP 422), not silently run as the baseline.

Transcription is common to all versions and is not included in the AI-stage count. Loading the same model twice does not by itself constitute a distinct AI stage.

## Microphone and media

The microphone uses `getUserMedia` and `MediaRecorder`, not the browser's cloud speech-recognition API. It requests permission only after a click. Recording stops when clicked again or after 60 seconds. All audio tracks are released on stop, errors, and page exit.

For this first integration, the browser sends one complete recording after stopping, not independent encoded fragments. Arbitrary MediaRecorder chunks are not guaranteed independently decodable. The Python adapter must decode the actual MIME type (WebM/Opus, MP4 or Ogg) before local ASR; do not assume incoming audio is already 16 kHz PCM. Streaming transport/chunking can be added with the concurrency work.

Recordings stay in browser memory until submission. **Stop & analyse** submits them to the local backend. Local file uploads are previewed first and submitted only when **Analyse recording** is clicked. Failed requests retain the recording for saving or retrying. Limit input files to 100 MB.

Use localhost or HTTPS for microphone permission. No real microphone recording or end-to-end model run has been tested automatically in this environment.

## Local HTTP adapter

Implement `POST /api/analyse` on the same origin as the frontend. A text request is JSON:

```json
{
  "run_id": "<browser-generated UUID>",
  "version_id": "two-caches",
  "config": {
    "caches": ["claims", "evidence"],
    "extraction": "model",
    "ai_count": 2,
    "parallel": false
  },
  "text": "Paris is in France and Berlin is in Germany."
}
```

Audio/video requests are `multipart/form-data` with:
- `media`: the complete file/recording;
- `metadata`: a JSON string containing the same `run_id`, `version_id`, and `config`.

Validate the version against a backend-owned configuration. Return HTTP errors for unavailable models or versions. The frontend imposes a 120-second request timeout and allows cancellation with **New run**.

Return JSON with both IDs echoed, and separate arrays for transcript segments and claims:

```json
{
  "run_id": "<same browser-generated UUID>",
  "version_id": "two-caches",
  "segments": [
    {
      "id": "segment-42",
      "start_ms": 0,
      "end_ms": 4000,
      "text": "Paris is in France and Berlin is in Germany.",
      "is_stable": true,
      "extraction_complete": true
    }
  ],
  "claims": [
    {
      "claim_id": "claim-100",
      "segment_ids": ["segment-42"],
      "claim_text": "Paris is in France.",
      "status": "supported",
      "path": "fresh",
      "confidence": null,
      "explanation": "The cited passage directly locates Paris in France.",
      "latency_ms": 842,
      "timings_ms": {"checkability": 2, "search": 210, "retrieval": 366, "verification": 264},
      "evidence": [
        {
          "text": "<actual retrieved passage>",
          "url": "<actual HTTP(S) source URL>",
          "source_type": "page",
          "scores": {"entailment": 0.97, "contradiction": 0.01, "neutral": 0.02}
        }
      ]
    },
    {
      "claim_id": "claim-101",
      "segment_ids": ["segment-42"],
      "claim_text": "Berlin is in Germany.",
      "status": "checking",
      "evidence": []
    }
  ],
  "metrics": {"p50_ms": 842, "p95_ms": 1200, "cache_hit_rate": 0, "queue_size": 1}
}
```

Numbers above illustrate the contract; they are not benchmark results. Use `null` or omit unknown values. `confidence` must be a model score in [0, 1], not an invented truth probability; evidence NLI scores are displayed individually. `cache_hit_rate` is a percentage in [0, 100]. Do not mix non-checkable inputs, ASR time and verification latencies in p50/p95 without defining the population.

An NLI model does not generate a reasoning trace. Supply a short evidence-based assessment, e.g. which passage supports/contradicts the claim and why an uncertain result was returned.

The frontend supports `not_checked`, `not_checkable`, `checking`, `supported`, `contradicted`, and `uncertain`. The current database does not have a `not_checkable` claim status: either keep those as segments with `extraction_complete: true` and no claims, or agree a schema change. Do not insert that value into the existing CHECK constraint.

## Transcript segments are not claims

Keep transcript segments unchanged. The AI/orchestration side buffers stable snippets with recent context, extracts atomic claims, resolves references where possible, and decides checkability. Never split blindly on every “and”: “Trinidad and Tobago is a country” is one claim.

One segment can produce zero, one, or many claims. One claim can also span multiple segments. Use `claim_segments(claim_id, segment_id)`, already in the database, to preserve both directions. Do not overwrite a source segment with a claim or key results solely by segment ID.

The existing `check_claim()` notebook currently expects a single explicit claim. Compound-claim extraction is **not implemented** by this frontend. The backend must provide each atomic claim and its provenance.

## Asynchronous events

For streaming integration, use `window.veriflow.push(event)` or SSE:

```js
// Start a real run (separate from interface-demo fixtures).
window.veriflow.newRun();
const context = window.veriflow.context();
// Send context.run_id / version_id / config to the Python worker.

window.veriflow.push({
  ...context,
  type: "transcript.segment",
  data: { id: "s1", text: "Paris is in France and Berlin is in Germany.", start_ms: 0, is_stable: true }
});
window.veriflow.push({
  ...context,
  type: "claim.update",
  data: { claim_id: "c1", segment_ids: ["s1"], claim_text: "Paris is in France.", status: "checking" }
});
window.veriflow.push({
  ...context,
  type: "claim.update",
  data: { claim_id: "c2", segment_ids: ["s1"], claim_text: "Berlin is in Germany.", status: "checking" }
});
// Partial updates preserve previously supplied claim text and provenance.
window.veriflow.push({
  ...context,
  type: "claim.update",
  data: { claim_id: "c1", status: "supported", explanation: "Actual evidence-based assessment", evidence: [] }
});
window.veriflow.push({ ...context, type: "metrics.update", data: { p50_ms: 842 } });
window.veriflow.push({ ...context, type: "run.complete" });
```

Each real event **must** include matching `run_id` and `version_id`. Events for old runs or other versions are rejected. Claims are keyed by claim ID, not segment ID, and may arrive before their segment without being dropped. Updates do not steal the user's current claim selection.

For SSE, `window.veriflow.connect("/events")` adds the current IDs as query parameters. Emit ordinary `data:` JSON messages using the same event envelope. The HTTP adapter is not required for manually pushed/SSE events. Cross-origin streams and arbitrary `postMessage` injection are not supported.

## Measurements and integration status

The UI records completed real runs separately by version and can export their inputs, claims, configuration and backend metrics as JSON. Exports therefore contain what the professor said: treat them as local test data, not telemetry to upload automatically. Demo fixtures are excluded.

Still to connect:
1. The local HTTP or SSE adapter.
2. Local transcription and atomic-claim extraction.
3. Supported version configurations and cache policies.
4. Real provenance, assessments, scores and benchmark measurements.

No additional Python or frontend packages are required for the static UI.
