/* Dependency-free UI. Models, caches and claim extraction belong to the Python backend. */
const VERSIONS = [
  { id: "baseline", stage: "BASELINE", label: "Baseline", spec: "0 caches · 1 AI · sequential", description: "Fresh, deterministic Wikipedia-first retrieval and verification for every claim.", config: { caches: [], extraction: "rules", ai_count: 1, parallel: false, source_policy: "wikipedia_first" } },
  { id: "claim-cache", stage: "IMPLEMENTED", label: "Claim cache", spec: "1 cache · 1 AI · sequential", description: "Reuse only page-backed verdicts; failed and uncertain checks are never cached.", config: { caches: ["claims"], extraction: "rules", ai_count: 1, parallel: false, source_policy: "wikipedia_first" } },
  { id: "two-ai", stage: "PLANNED", label: "Two AI stages", spec: "1 cache · 2 AI · sequential", description: "Model-based claim extraction plus verification, with Wikipedia-first retrieval and web fallbacks.", config: { caches: ["claims"], extraction: "model", ai_count: 2, parallel: false, source_policy: "wikipedia_first" } },
  { id: "two-caches", stage: "IMPLEMENTED", label: "Two cache layers", spec: "2 caches · 1 AI · sequential", description: "V1 plus an evidence cache: sentences from pages already downloaded are searched first, so claims on a known topic skip web search. Used only when it gives a clear verdict.", config: { caches: ["claims", "evidence"], extraction: "rules", ai_count: 1, parallel: false, source_policy: "wikipedia_first" } },
  { id: "parallel", stage: "PROTOTYPE", label: "Parallel pipeline", spec: "Current V1 · adjustable workers", description: "Runs the current V1 components concurrently to isolate the speedup. V2 and V3 will be added to this V4 target later.", config: { caches: ["claims"], extraction: "rules", ai_count: 1, parallel: true, source_policy: "wikipedia_first" } }
];
const STATUS = { not_checked: "Not checked", not_checkable: "Not checkable", checking: "Checking", supported: "Supported", contradicted: "Contradicted", uncertain: "Uncertain" };
const ROUTES = { fresh: "Fresh retrieval", claims_cache: "Claim cache", evidence_cache: "Evidence cache" };
const $ = id => document.getElementById(id);
const state = {
  version: "baseline", runs: new Map(), history: [], timers: [], busy: false,
  controller: null, eventSource: null, recorder: null, stream: null,
  requestingMic: false, micTimer: null, media: null, mediaUrl: null, backend: null, workers: null
};
const version = () => VERSIONS.find(item => item.id === state.version);
const activeRun = () => state.runs.get(state.version);
// Worker threads for the parallel version: the backend reports its bounds in /api/health.
const workerBounds = () => ({ min: 1, max: 8, default: 3, ...state.backend?.workers });
const workers = () => { const b = workerBounds(); return Math.min(b.max, Math.max(b.min, state.workers ?? b.default)); };
const finite = value => typeof value === "number" && Number.isFinite(value);
const esc = value => String(value ?? "").replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
const formatMs = value => finite(value) && value >= 0 ? `${value < 1000 ? Math.round(value) : (value / 1000).toFixed(2)} ${value < 1000 ? "ms" : "s"}` : "—";
function clock(ms) {
  const seconds = Math.floor(Math.max(0, ms || 0) / 1000);
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}
function safeUrl(value) {
  try { const url = new URL(value); return ["https:", "http:"].includes(url.protocol) ? url.href : null; }
  catch { return null; }
}
function empty(title, copy, icon = "≡") {
  return `<div class="empty-state"><span class="empty-icon" aria-hidden="true">${icon}</span><strong>${esc(title)}</strong><p>${esc(copy)}</p></div>`;
}
function notify(text, kind = "") {
  $("runNotice").textContent = text;
  $("runNotice").className = `run-notice ${kind}`;
}
function stopEvents() {
  state.timers.forEach(clearTimeout);
  state.timers = [];
  state.eventSource?.close();
  state.eventSource = null;
  state.controller?.abort();
  state.controller = null;
  state.busy = false;
}
function beginRun(preview = false) {
  stopEvents();
  const config = structuredClone(version().config);
  if (config.parallel) config.workers = workers();
  const run = {
    id: crypto.randomUUID(), version_id: state.version, config,
    preview, segments: new Map(), claims: new Map(), selected: null, metrics: {},
    started_at: new Date().toISOString(), completed: false,
    notice: preview ? "INTERFACE DEMO · Illustrative verdicts, scores and passages. No models or searches have run." : "Ready for input. Live transcription and verification need the local backend.",
    noticeKind: preview ? "demo" : ""
  };
  state.runs.set(state.version, run);
  render();
  return run;
}
function renderVersions() {
  $("versionList").innerHTML = VERSIONS.map((item, index) => `
    <button type="button" class="version-card" data-version="${item.id}" aria-pressed="${item.id === state.version}" ${state.busy || state.recorder || state.requestingMic ? "disabled" : ""}>
      <span class="version-top"><span>VERSION ${String(index).padStart(2, "0")} · ${item.stage}</span><i class="version-radio" aria-hidden="true"></i></span>
      <strong>${item.label}</strong><span class="spec">${item.spec}</span>
    </button>`).join("");
  $("versionList").querySelectorAll("[data-version]").forEach(button => button.addEventListener("click", () => selectVersion(button.dataset.version)));
  $("versionDescription").textContent = version().description;
  $("pipelineSummary").innerHTML = [
    version().config.extraction === "model" ? "AI extraction" : "Rule extraction",
    ...version().config.caches.map(cache => cache === "claims" ? "Claim cache" : "Evidence cache"),
    version().config.source_policy === "wikipedia_only" ? "Wikipedia only" : "Wikipedia first + web fallback",
    "NLI verification", version().config.parallel ? `${workers()} parallel workers` : "Sequential"
  ].map(label => `<span>${label}</span>`).join("");
  const bounds = workerBounds(), slider = $("workersInput");
  $("workersControl").hidden = !version().config.parallel;
  Object.assign(slider, { min: bounds.min, max: bounds.max, value: workers(), disabled: state.busy || Boolean(state.recorder) || state.requestingMic });
  $("workersValue").textContent = workers();
}
function selectVersion(id) {
  if (!VERSIONS.some(item => item.id === id) || state.busy || state.recorder || state.requestingMic) return false;
  stopEvents();
  state.version = id;
  if (!activeRun()) beginRun();
  else render();
  renderConnection();
  return true;
}
// Poll GET /api/health: the backend answers only once its models are loaded, and lists the versions it implements.
async function checkBackend() {
  let health = null;
  try {
    const response = await fetch("/api/health", { cache: "no-store", signal: AbortSignal.timeout(3000) });
    if (response.ok && response.headers.get("content-type")?.includes("application/json")) health = await response.json();
  } catch {}
  state.backend = health;
  renderConnection();
}
function renderConnection() {
  const status = $("connectionStatus");
  const available = Array.isArray(state.backend?.versions) && state.backend.versions.includes(state.version);
  status.textContent = !state.backend ? "Backend not connected"
    : available ? `Backend connected · ${version().label}` : `Backend connected · ${version().label} not implemented`;
  status.classList.toggle("connected", !!state.backend && available);
  status.classList.toggle("partial", !!state.backend && !available);
}
function renderTranscript() {
  const run = activeRun();
  const segments = [...run.segments.values()].sort((a, b) => (a.start_ms || 0) - (b.start_ms || 0));
  const filtered = segments.filter(segment => !$("claimsOnly").checked || [...run.claims.values()].some(claim => claim.segment_ids.includes(segment.id)));
  $("claimCount").textContent = `${run.claims.size} claim${run.claims.size === 1 ? "" : "s"} · ${run.segments.size} segment${run.segments.size === 1 ? "" : "s"}`;
  $("transcriptStream").innerHTML = filtered.length ? filtered.map(segment => {
    const claims = [...run.claims.values()].filter(claim => claim.segment_ids.includes(segment.id));
    return `<article class="segment">
      <div class="segment-meta"><span>${esc(segment.speaker || "Speaker")} · ${clock(segment.start_ms)}</span><span>${segment.is_stable === false ? "Interim transcript" : "Stable segment"}</span></div>
      <p class="segment-text ${segment.is_stable === false ? "unstable" : ""}">${esc(segment.text)}</p>
      <div class="segment-claims">${claims.map(claim => `
        <button type="button" class="claim-button" data-claim="${esc(claim.id)}" aria-pressed="${run.selected === claim.id}">
          <i class="claim-dot ${claim.status}" aria-hidden="true"></i>
          <span class="claim-text">${esc(claim.claim_text)}</span><span class="claim-status ${claim.status}">${STATUS[claim.status]}</span>
        </button>`).join("")}</div>
      ${!claims.length ? '<p class="segment-pending">' + (segment.extraction_complete ? "No checkable claims extracted." : "Waiting for claim extraction…") + "</p>" : ""}
    </article>`;
  }).join("") : empty($("claimsOnly").checked ? "No extracted claims yet" : "Your words will appear here", "Stable transcript snippets appear first. Extracted claims appear underneath; select one to inspect its evidence.");
  $("transcriptStream").querySelectorAll("[data-claim]").forEach(button => button.addEventListener("click", () => {
    run.selected = button.dataset.claim;
    renderTranscript();
    renderInspector();
  }));
}
function renderInspector() {
  const run = activeRun();
  const claim = run.claims.get(run.selected);
  if (!claim) {
    $("inspectorBody").innerHTML = empty("Every claim has a trail", "Select a claim to see the assessment, model score, source passages and verification timings.", "↗");
    return;
  }
  const confidence = finite(claim.confidence) && claim.confidence >= 0 && claim.confidence <= 1 ? claim.confidence : null;
  const evidence = Array.isArray(claim.evidence) ? claim.evidence : [];
  const timings = Object.entries(claim.timings_ms || {}).filter(([key, value]) => key !== "total" && finite(value) && value >= 0);
  const maximum = Math.max(1, ...timings.map(([, value]) => value));
  $("inspectorBody").innerHTML = `
    <div class="detail-main">
      <span class="verdict-badge ${claim.status}">● ${STATUS[claim.status]}</span>
      <p class="detail-claim">${esc(claim.claim_text)}</p>
      <div class="detail-meta"><span>${esc(ROUTES[claim.path] || "Route pending")}</span><span>${formatMs(claim.latency_ms)}</span><span>${claim.segment_ids.length} source segment(s)</span></div>
    </div>
    <section class="inspector-section"><h3>${claim.status === "uncertain" ? "Why uncertain" : claim.status === "not_checked" ? "Why not checked" : "Assessment"}</h3><p>${esc(claim.explanation || (claim.status === "checking" ? "The pipeline is checking this claim." : "The backend has not provided an assessment."))}</p></section>
    <section class="inspector-section">
      <div class="score-header"><h3>Model confidence score</h3><strong>${confidence === null ? "—" : Math.round(confidence * 100) + "%"}</strong></div>
      <div class="score-track"><span style="width:${(confidence || 0) * 100}%"></span></div>
      <p class="score-note">${run.preview ? "Illustrative score only. " : ""}An uncalibrated model score, not a probability that this claim is true. Missing scores stay blank.</p>
    </section>
    <section class="inspector-section"><h3>Sources &amp; evidence · ${evidence.length}</h3>
      ${evidence.length ? evidence.map(item => {
        const url = safeUrl(item.url || item.source_url);
        const scores = Object.entries(item.scores || {}).filter(([, value]) => finite(value));
        return `<article class="evidence-card">
          ${url ? '<a href="' + esc(url) + '" target="_blank" rel="noopener noreferrer">' + esc(item.source || new URL(url).hostname) + ' ↗</a>' : "<strong>" + esc(item.source || "No source URL supplied") + "</strong>"}
          <blockquote>${esc(item.text || item.evidence_text)}</blockquote>
          <small>${run.preview ? "Illustrative passage · not fetched or verified" : esc(item.source_type || "Source passage")}</small>
          ${scores.length ? '<div class="evidence-scores">' + scores.map(([name, score]) => esc(name) + ": " + score.toFixed(3)).join(" · ") + "</div>" : ""}
        </article>`;
      }).join("") : "<p>No cited evidence returned yet.</p>"}
    </section>
    <section class="inspector-section"><h3>Verification stages</h3>
      ${timings.length ? timings.map(([name, value]) => `<div class="timing-row"><span>${esc(name.replaceAll("_", " "))}</span><div class="timing-track"><span style="width:${Math.max(1, value / maximum * 100)}%"></span></div><strong>${formatMs(value)}</strong></div>`).join("") : "<p>No measured stage timings yet.</p>"}
    </section>
    ${claim.errors?.length ? '<section class="inspector-section"><h3>Retrieval notes</h3><p>' + claim.errors.map(esc).join("<br>") + "</p></section>" : ""}`;
}
function metricCache(metrics) {
  const rate = finite(metrics.cache_hit_rate) && metrics.cache_hit_rate >= 0 && metrics.cache_hit_rate <= 100 ? Math.round(metrics.cache_hit_rate) + "%" : "—";
  return finite(metrics.evidence_hits) ? `${rate} · ${metrics.evidence_hits} evidence` : rate;
}
function renderMetrics() {
  const run = activeRun();
  const metrics = run.preview ? {} : run.metrics;
  $("metricP50").textContent = formatMs(metrics.p50_ms);
  $("metricP95").textContent = formatMs(metrics.p95_ms);
  $("metricCache").textContent = metricCache(metrics);
  $("metricQueue").textContent = finite(metrics.queue_size) && metrics.queue_size >= 0 ? metrics.queue_size : "—";
  $("metricKind").textContent = run.preview ? "PREVIEW · NOT MEASURED" : run.completed ? "BACKEND RESULTS" : state.busy ? "RUNNING" : "NO MEASUREMENTS";
}
function renderComparison() {
  $("comparisonBody").innerHTML = VERSIONS.map(item => {
    const run = state.history.findLast(row => row.version_id === item.id);
    const label = run?.config.workers ? `${item.label} ×${run.config.workers}` : item.label;
    return `<tr><td>${label}</td><td>${run ? esc(new Date(run.started_at).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"})) : "Not run"}</td><td>${run ? run.claims.length : "—"}</td><td>${formatMs(run?.metrics.p50_ms)}</td><td>${formatMs(run?.metrics.p95_ms)}</td><td>${run ? metricCache(run.metrics) : "—"}</td></tr>`;
  }).join("");
  $("exportButton").disabled = !state.history.length;
}
function render() {
  renderVersions();
  renderTranscript();
  renderInspector();
  renderMetrics();
  renderComparison();
  notify(activeRun().notice, activeRun().noticeKind);
  const locked = state.busy || Boolean(state.recorder) || state.requestingMic;
  $("analyseButton").disabled = locked;
  $("sendMediaButton").disabled = locked;
  $("uploadButton").disabled = locked;
  $("demoButton").disabled = locked;
  $("clearButton").disabled = Boolean(state.recorder) || state.requestingMic;
  $("clearCacheButton").disabled = locked || !state.backend;
  $("micButton").disabled = state.requestingMic || (state.busy && !state.recorder);
}
function setRunNotice(run, text, kind = "") {
  run.notice = text; run.noticeKind = kind;
  if (run === activeRun()) notify(text, kind);
}
function completeRun(run) {
  if (run.preview || run.completed) return;
  run.completed = true;
  state.history.push({
    run_id: run.id, version_id: run.version_id, config: run.config, started_at: run.started_at,
    completed_at: new Date().toISOString(), metrics: { ...run.metrics },
    segments: [...run.segments.values()], claims: [...run.claims.values()]
  });
}
function applyEvent(event) {
  const run = activeRun();
  // Scope every real event: an old run or a different pipeline must never overwrite this run.
  if (!event || !run || run.preview || event.run_id !== run.id || event.version_id !== run.version_id) return false;
  const data = event.data || {};
  if (event.type === "transcript.segment") {
    if (data.id == null) return false;
    const id = String(data.id);
    run.segments.set(id, { ...run.segments.get(id), ...data, id });
  } else if (event.type === "claim.update") {
    const id = data.claim_id ?? data.id;
    if (id == null) return false;
    const previous = run.claims.get(String(id)) || {};
    const segmentIds = data.segment_ids || (data.segment_id != null ? [data.segment_id] : previous.segment_ids);
    if (!segmentIds?.length) return false;
    const status = data.status || data.verdict || previous.status || "checking";
    if (!Object.hasOwn(STATUS, status)) return false;
    run.claims.set(String(id), {
      ...previous, ...data, id: String(id), status,
      claim_text: data.claim_text ?? data.claim ?? previous.claim_text ?? "",
      segment_ids: segmentIds.map(String)
    });
    if (!run.selected) run.selected = String(id);
  } else if (event.type === "metrics.update") {
    run.metrics = { ...run.metrics, ...data };
  } else if (event.type === "run.complete") {
    completeRun(run);
    state.busy = false;
    setRunNotice(run, "Run complete. Select a claim to inspect its evidence.");
  } else return false;
  render();
  return true;
}
async function analyse(input) {
  if (state.busy || state.recorder || state.requestingMic) return;
  const run = beginRun();
  state.busy = true;
  const controller = new AbortController();
  state.controller = controller;
  // Time out when the pipeline stops making progress, not after a fixed total: long recordings are fine.
  let timedOut = false, timeout;
  const resetTimeout = () => { clearTimeout(timeout); timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 120000); };
  resetTimeout();
  setRunNotice(run, input.text ? "Sending text for claim extraction and verification…" : "Sending audio/video to local transcription and verification…");
  render();
  try {
    let url, body, headers = {};
    const metadata = { run_id: run.id, version_id: run.version_id, config: run.config };
    if (input.text) {
      url = "/api/analyse/text";
      body = JSON.stringify({ ...metadata, text: input.text });
      headers["Content-Type"] = "application/json";
    } else {
      url = "/api/analyse/media";
      body = new FormData();
      body.append("media", input.media, input.filename || "microphone.webm");
      body.append("metadata", JSON.stringify(metadata));
    }
    const response = await fetch(url, { method: "POST", headers, body, signal: controller.signal });
    if (!response.ok) {
      if ([404, 405, 501].includes(response.status)) throw new Error(`The local ${url} backend is not connected yet. Input capture works, but live transcription and verdicts need the Python adapter. Your recording is available to save or retry.`);
      throw new Error(`Backend returned HTTP ${response.status}. No verdicts were fabricated.`);
    }
    if (!response.headers.get("content-type")?.includes("application/x-ndjson")) throw new Error("Expected a streamed pipeline response, not a static page.");
    setRunNotice(run, input.text ? "Checking claims… results appear as each one is verified." : "Transcribing and checking claims… results appear as each one is verified.");
    // One JSON event per line, applied as soon as it arrives
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "", complete = false;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      resetTimeout();
      buffer += value;
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines.filter(line => line.trim())) {
        const event = JSON.parse(line);
        if (event.type === "run.error") throw new Error(`The pipeline failed: ${event.data?.message || "unknown error"}. Results so far are kept.`);
        if (!applyEvent(event)) throw new Error("Invalid pipeline event from backend.");
        if (event.type === "run.complete") complete = true;
      }
    }
    if (!complete) throw new Error("The pipeline stream ended before the run completed. Results so far are kept.");
  } catch (error) {
    if (error.name === "AbortError" && !timedOut) return;
    const message = timedOut ? "The local pipeline made no progress for 120 seconds. Results so far are kept." : error.message;
    if (activeRun() === run) {
      setRunNotice(run, message || "Could not reach the local backend.", "error");
      checkBackend();   // a failed run may mean the backend went away
    }
  } finally {
    clearTimeout(timeout);
    if (state.controller === controller) { state.controller = null; state.busy = false; render(); }
  }
}
function setMedia(blob, name, download = false) {
  for (const id of ["videoPlayer", "audioPlayer"]) { $(id).pause(); $(id).removeAttribute("src"); $(id).hidden = true; }
  if (state.mediaUrl) URL.revokeObjectURL(state.mediaUrl);
  state.media = { blob, name };
  state.mediaUrl = URL.createObjectURL(blob);
  const player = blob.type.startsWith("video/") ? $("videoPlayer") : $("audioPlayer");
  player.src = state.mediaUrl; player.hidden = false;
  $("fileName").textContent = name;
  $("mediaSubmit").hidden = false;
  $("downloadRecording").href = state.mediaUrl;
  $("downloadRecording").download = name;
  $("downloadRecording").hidden = !download;
}
function resetMic() {
  clearInterval(state.micTimer);
  state.stream?.getTracks().forEach(track => track.stop());
  state.stream = null; state.recorder = null; state.requestingMic = false;
  $("micButton").disabled = false;
  $("micButton").setAttribute("aria-pressed", "false");
  $("micButton").setAttribute("aria-label", "Start microphone recording");
  $("recordingClock").hidden = true;
  $("micTitle").textContent = "Have something to fact-check?";
  $("micHint").textContent = "Click the microphone, speak a few sentences, then stop to send them to the local pipeline.";
  render();
}
async function toggleMic() {
  if (state.recorder?.state === "recording") {
    $("micButton").disabled = true;
    state.recorder.stop();
    // Release the device immediately. The final dataavailable event still precedes onstop.
    state.stream?.getTracks().forEach(track => track.stop());
    return;
  }
  if (state.busy || state.requestingMic || state.recorder) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    notify("Microphone recording needs a supported browser on localhost or HTTPS. You can still use text or upload a recording.", "error");
    return;
  }
  state.requestingMic = true;
  $("micButton").disabled = true;
  render();
  let chunks = [];
  let failed = false;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus"].find(type => MediaRecorder.isTypeSupported(type));
    const recorder = new MediaRecorder(state.stream, mimeType ? { mimeType } : {});
    state.recorder = recorder;
    recorder.ondataavailable = event => { if (event.data.size > 0) chunks.push(event.data); };
    recorder.onerror = () => {
      failed = true;
      setRunNotice(activeRun(), "Recording failed. Please retry or use a file.", "error");
      if (recorder.state !== "inactive") recorder.stop();
      else resetMic();
    };
    recorder.onstop = () => {
      const type = recorder.mimeType || chunks[0]?.type || "audio/webm";
      const blob = new Blob(chunks, { type });
      chunks = [];
      resetMic();
      if (failed || !blob.size) { notify("No usable recording was captured. Please try again.", "error"); return; }
      const extension = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
      setMedia(blob, `microphone.${extension}`, true);
      analyse({ media: blob, filename: `microphone.${extension}` });
    };
    beginRun();
    recorder.start();
    state.requestingMic = false;
    $("micButton").disabled = false;
    $("micButton").setAttribute("aria-pressed", "true");
    $("micButton").setAttribute("aria-label", "Stop recording and analyse");
    $("micTitle").textContent = "Listening. Click to stop & analyse.";
    $("micHint").textContent = "Speak naturally. Stops automatically after 60 seconds.";
    $("recordingClock").hidden = false;
    $("recordingClock").textContent = "00:00 / 01:00";
    const started = Date.now();
    state.micTimer = setInterval(() => {
      const elapsed = Date.now() - started;
      $("recordingClock").textContent = `${clock(elapsed)} / 01:00`;
      if (elapsed >= 60000 && recorder.state === "recording") toggleMic();
    }, 250);
    render();
  } catch (error) {
    resetMic();
    const message = error.name === "NotAllowedError" ? "Microphone permission was denied. Allow access in the browser or use text input."
      : error.name === "NotFoundError" ? "No microphone was found. Connect one or use text input."
      : "Could not start the microphone. Try a supported browser or upload an audio file.";
    setRunNotice(activeRun(), message, "error");
  }
}
function demo() {
  if (state.busy || state.recorder || state.requestingMic) return;
  const run = beginRun(true);
  run.segments.set("s1", { id: "s1", start_ms: 0, end_ms: 4100, is_stable: true, speaker: "Example speaker", text: "Paris is in France and Berlin is in Germany.", extraction_complete: true });
  run.segments.set("s2", { id: "s2", start_ms: 4200, end_ms: 6300, is_stable: true, speaker: "Example speaker", text: "Paris is the capital of Germany.", extraction_complete: true });
  run.segments.set("s3", { id: "s3", start_ms: 6400, end_ms: 8000, is_stable: true, speaker: "Example speaker", text: "I think Paris is beautiful.", extraction_complete: true });
  const claims = [
    { id: "c1", segment_ids: ["s1"], claim_text: "Paris is in France.", status: "supported", confidence: .97, explanation: "Illustration: a passage locating Paris in France supports this atomic claim.", evidence: [{ source: "Paris · example source", url: "https://en.wikipedia.org/wiki/Paris", text: "Example passage: Paris is the capital of France." }] },
    { id: "c2", segment_ids: ["s1"], claim_text: "Berlin is in Germany.", status: "supported", confidence: .96, explanation: "Illustration: the second claim is assessed separately, even though it came from the same transcript snippet.", evidence: [{ source: "Berlin · example source", url: "https://en.wikipedia.org/wiki/Berlin", text: "Example passage: Berlin is the capital of Germany." }] },
    { id: "c3", segment_ids: ["s2"], claim_text: "Paris is the capital of Germany.", status: "contradicted", confidence: .98, explanation: "Illustration: evidence identifying Berlin, rather than Paris, as Germany's capital contradicts the claim.", evidence: [{ source: "Berlin · example source", url: "https://en.wikipedia.org/wiki/Berlin", text: "Example passage: Berlin is the capital of Germany." }] }
  ];
  for (const claim of claims) run.claims.set(claim.id, claim);
  run.selected = "c1";
  render();
}
function connect(url = "/events") {
  const run = activeRun();
  if (run.preview) throw new Error("Start a new live run before connecting backend events.");
  const endpoint = new URL(url, location.href);
  if (endpoint.origin !== location.origin) throw new Error("Use the same local origin for pipeline events.");
  endpoint.searchParams.set("run_id", run.id);
  endpoint.searchParams.set("version_id", run.version_id);
  state.eventSource?.close();
  const source = new EventSource(endpoint);
  state.eventSource = source;
  source.onmessage = event => {
    try { applyEvent(JSON.parse(event.data)); } catch { setRunNotice(run, "Invalid pipeline event received.", "error"); }
  };
  source.onopen = () => { $("connectionStatus").textContent = "Local event stream connected"; $("connectionStatus").classList.add("connected"); };
  source.onerror = () => { $("connectionStatus").textContent = "Event stream reconnecting…"; $("connectionStatus").classList.remove("connected"); };
  return source;
}
async function clearCaches() {
  if (state.busy || !state.backend) return;
  if (!window.confirm("Clear the V1 claim cache and the V3 evidence cache? Transcript runs and the main database will not be changed.")) return;
  $("clearCacheButton").disabled = true;
  try {
    const [claims, evidence] = await Promise.all(["claims", "evidence"].map(async cache => {
      const response = await fetch(`/api/cache/${cache}/clear`, { method: "POST" });
      if (!response.ok) throw new Error(`Backend returned HTTP ${response.status}.`);
      return response.json();
    }));
    setRunNotice(activeRun(), `Caches cleared (${claims.cleared ?? 0} claims, ${evidence.cleared ?? 0} evidence pages). Your run history was kept.`);
  } catch (error) {
    setRunNotice(activeRun(), `Could not clear the caches: ${error.message}`, "error");
  } finally {
    render();
  }
}
$("textForm").addEventListener("submit", event => {
  event.preventDefault();
  const text = $("testText").value.trim();
  if (!text) { $("testText").focus(); notify("Enter a few sentences first.", "error"); return; }
  analyse({ text });
});
$("micButton").addEventListener("click", toggleMic);
$("demoButton").addEventListener("click", demo);
$("clearButton").addEventListener("click", () => beginRun());
$("clearCacheButton").addEventListener("click", clearCaches);
$("claimsOnly").addEventListener("change", renderTranscript);
$("workersInput").addEventListener("input", event => { state.workers = Number(event.target.value); renderVersions(); });
$("uploadButton").addEventListener("click", () => $("mediaInput").click());
$("mediaInput").addEventListener("change", event => {
  const file = event.target.files[0];
  if (!file) return;
  if (!/^(audio|video)\//.test(file.type)) { notify("Choose an audio or video file.", "error"); return; }
  if (file.size > 100 * 1024 * 1024) { notify("Use a short clip smaller than 100 MB for this prototype.", "error"); return; }
  setMedia(file, file.name);
  $("mediaInput").value = "";
});
$("sendMediaButton").addEventListener("click", () => {
  if (state.media) analyse({ media: state.media.blob, filename: state.media.name });
});
$("exportButton").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(state.history, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a"); link.href = url; link.download = "veriflow-runs.json"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener("beforeunload", () => {
  stopEvents();
  clearInterval(state.micTimer);
  state.stream?.getTracks().forEach(track => track.stop());
  if (state.mediaUrl) URL.revokeObjectURL(state.mediaUrl);
});
window.veriflow = {
  push: applyEvent, connect, demo,
  newRun: () => { if (!state.recorder && !state.requestingMic) return beginRun().id; },
  selectVersion,
  context: () => { const run = activeRun(); return { run_id: run.id, version_id: run.version_id, config: structuredClone(run.config) }; }
};
beginRun();
checkBackend();
setInterval(checkBackend, 5000);
window.addEventListener("focus", checkBackend);
