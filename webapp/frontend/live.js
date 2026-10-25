/* The end-user page. One live run at a time: the backend listens to the computer's sound (Kroko),
   streams each sentence and each claim's verdict, and this page colours the claims in the text.
   It always uses the parallel version (V4), which keeps the transcript moving while claims are checked. */
const VERSION_ID = "parallel";
const WORKERS = 4;   // checks running at the same time; the backend caps it (versions.MAX_WORKERS)
const STATUS = { checking: "Checking", supported: "Supported", contradicted: "Contradicted", uncertain: "Uncertain", not_checked: "Could not check" };
// One sentence can hold several claims: show the one that matters most
const PRIORITY = ["checking", "contradicted", "uncertain", "supported", "not_checked"];
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const state = { backend: false, running: false, stopping: false, runId: null, controller: null,
                segments: new Map(), claims: new Map(), partial: "", selected: null, started: 0, clock: null,
                source: null };   // from /api/live/source: what the backend will record

function claimsOf(segmentId) {
  return [...state.claims.values()].filter(claim => claim.segment_ids.includes(segmentId));
}
function segmentStatus(segmentId) {
  const statuses = claimsOf(segmentId).map(claim => claim.status);
  return PRIORITY.find(status => statuses.includes(status)) || null;
}
function host(url) {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return null; }
}
function seconds(ms) {
  if (!Number.isFinite(ms)) return "";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function renderTranscript() {
  const box = $("transcript");
  const follow = box.scrollHeight - box.scrollTop - box.clientHeight < 80;   // keep following the end
  if (!state.segments.size && !state.partial) {
    box.innerHTML = state.running
      ? `<div class="empty"><strong>Listening…</strong><span>Play a video or a podcast on this computer. The words appear here while they are spoken.</span></div>`
      : `<div class="empty"><strong>Fact-check what you are watching.</strong><span>Press <b>Start listening</b>, then play a video or a podcast in English on this computer. Sentences appear as they are spoken; factual claims are coloured when their sources are checked. Click a coloured sentence to see the source.</span></div>`;
    return;
  }
  const parts = [...state.segments.values()].map(segment => {
    const status = segmentStatus(segment.id);
    if (!status) return `<span class="s">${esc(segment.text)}</span>`;
    return `<span class="s claim ${status}${state.selected === segment.id ? " selected" : ""}" role="button" tabindex="0" data-id="${esc(segment.id)}" title="${STATUS[status]}">${esc(segment.text)}</span>`;
  });
  if (state.partial) parts.push(`<span class="partial">${esc(state.partial)}</span>`);
  box.innerHTML = `<p>${parts.join(" ")}</p>`;
  if (follow) box.scrollTop = box.scrollHeight;
}

function renderDetail() {
  const panel = $("detail");
  const claims = state.selected ? claimsOf(state.selected) : [];
  if (!claims.length) {
    panel.innerHTML = `<h2>Sources</h2><p>Coloured sentences are factual claims. Click one to see what the sources say. Other sentences stay black: they are opinions, questions, or not a fact to check.</p>`;
    return;
  }
  panel.innerHTML = `<h2>Sources</h2>` + claims.map(claim => {
    const evidence = (claim.evidence || [])[0];
    const site = evidence && host(evidence.url);
    const body = claim.status === "checking"
      ? `<p>Searching the sources…</p>`
      : `<p>${esc(claim.explanation || "")}</p>` +
        (evidence ? `${site ? `<a class="source" href="${esc(evidence.url)}" target="_blank" rel="noopener noreferrer">${esc(site)} ↗</a>` : ""}
                     <blockquote>${esc(evidence.text.length > 360 ? evidence.text.slice(0, 360) + "…" : evidence.text)}</blockquote>` : "") +
        (claim.latency_ms != null ? `<span class="meta">Checked in ${seconds(claim.latency_ms)}${claim.path === "claims_cache" ? " · already checked before" : ""}</span>` : "");
    return `<div class="card"><span class="badge ${claim.status}">${STATUS[claim.status] || claim.status}</span>
      <span class="claim-text">${esc(claim.claim_text)}</span>${body}</div>`;
  }).join("");
}

function renderBar() {
  const status = $("status"), toggle = $("toggle");
  status.className = "status" + (state.running ? " live" : state.backend ? " ready" : "");
  const elapsed = Math.floor((Date.now() - state.started) / 1000);
  const clock = `${Math.floor(elapsed / 60)}:${String(elapsed % 60).padStart(2, "0")}`;
  const via = state.source?.source === "microphone" ? "through the microphone" : "to this computer";
  status.textContent = state.stopping ? "Finishing the last checks…" : state.running ? `Listening ${via} · ${clock}`
    : state.backend ? "Ready" : "Backend not connected";
  toggle.textContent = state.running ? (state.stopping ? "Stopping…" : "Stop") : "Start listening";
  toggle.classList.toggle("stop", state.running);
  toggle.disabled = state.stopping || (!state.running && !state.backend);
  const claims = [...state.claims.values()];
  const checking = claims.filter(claim => claim.status === "checking").length;
  $("counts").hidden = !claims.length;
  $("counts").textContent = `${claims.length - checking} checked${checking ? ` · ${checking} checking` : ""}`;
}

// The computer's own sound (loopback) needs no note. Without it (a Mac without BlackHole) the
// backend records the microphone, which also hears the room: say so before the user starts.
function renderSource() {
  const note = $("sourceNote"), source = state.source;
  note.hidden = !source || source.source === "loopback";
  if (note.hidden) return;
  note.classList.toggle("error", !source.source);
  if (!source.source) {
    note.innerHTML = `<strong>No audio input found</strong>Veriflow can't record this computer's sound or a microphone (${esc(source.error || "no device")}). Connect a microphone or allow microphone access, then reload this page.`;
    return;
  }
  const mac = source.platform === "darwin";
  note.innerHTML = `<strong>Listening through the microphone</strong>` +
    `This computer can't record its own sound${mac ? " (on a Mac this needs BlackHole, which is not installed)" : ""}, so Veriflow will listen through the microphone (${esc(source.device)}). ` +
    `Voices and noise in the room are picked up too, which can make the transcript and the checks less accurate. ` +
    `Play the video through the speakers, not headphones, and keep the room quiet. ` +
    `Some laptops filter their own speaker sound out of the microphone (echo cancellation): if no words appear, play the video on another device or through external speakers.` +
    (mac ? ` The app that runs Veriflow (for example Terminal) needs microphone access: System Settings → Privacy &amp; Security → Microphone.` : "");
}

async function checkSource() {
  try {
    const response = await fetch("/api/live/source", { cache: "no-store", signal: AbortSignal.timeout(5000) });
    if (response.ok) state.source = await response.json();
  } catch {}
  renderSource();
  renderBar();
}

function render() { renderTranscript(); renderDetail(); renderBar(); }

function notice(text) {
  $("notice").textContent = text || "";
  $("notice").hidden = !text;
}

function apply(event) {
  if (event.run_id !== state.runId) return;
  const data = event.data || {};
  if (event.type === "transcript.segment") {
    state.segments.set(data.id, data);
    state.partial = "";
  } else if (event.type === "transcript.partial") {
    state.partial = data.text || "";
  } else if (event.type === "claim.update") {
    state.claims.set(data.claim_id, { ...state.claims.get(data.claim_id), ...data });
  } else if (event.type === "run.error") {
    notice(`Something went wrong: ${data.message || "unknown error"}. The text so far is kept.`);
  } else return;   // metrics.update and run.complete change nothing on this page
  if (event.type !== "transcript.partial") renderDetail();
  renderTranscript();
  renderBar();
}

async function start() {
  Object.assign(state, { running: true, stopping: false, runId: crypto.randomUUID(), segments: new Map(),
                         claims: new Map(), partial: "", selected: null, started: Date.now() });
  state.controller = new AbortController();
  state.clock = setInterval(renderBar, 1000);
  notice("");
  render();
  try {
    const response = await fetch("/api/analyse/live", {
      method: "POST", headers: { "Content-Type": "application/json" }, signal: state.controller.signal,
      // "loopback" lets the backend use the microphone itself when there is no loopback device
      body: JSON.stringify({ run_id: state.runId, version_id: VERSION_ID, source: "loopback",
                             config: { workers: Math.min(WORKERS, state.maxWorkers || WORKERS), partials: true } })
    });
    if (!response.ok) throw new Error(`the backend answered HTTP ${response.status}`);
    // One JSON event per line, applied as soon as it arrives
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) if (line.trim()) apply(JSON.parse(line));
    }
  } catch (error) {
    if (error.name !== "AbortError") notice(`Listening stopped: ${error.message}.`);
  } finally {
    clearInterval(state.clock);
    Object.assign(state, { running: false, stopping: false, controller: null, partial: "" });
    render();
  }
}

async function stop() {
  state.stopping = true;
  renderBar();
  try {
    const response = await fetch("/api/analyse/live/stop", { method: "POST", headers: { "Content-Type": "application/json" },
                                                            body: JSON.stringify({ run_id: state.runId }) });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch {
    state.controller?.abort();   // the run is gone: stop reading it
  }
}

async function checkBackend() {
  try {
    const response = await fetch("/api/health", { cache: "no-store", signal: AbortSignal.timeout(3000) });
    const health = response.ok ? await response.json() : null;
    state.backend = Boolean(health?.versions?.includes(VERSION_ID));
    state.maxWorkers = health?.workers?.max;
  } catch { state.backend = false; }
  renderBar();
}

$("toggle").addEventListener("click", () => state.running ? stop() : start());
function select(target) {
  const sentence = target.closest(".s.claim");
  if (!sentence) return;
  state.selected = sentence.dataset.id;
  renderTranscript();
  renderDetail();
}
$("transcript").addEventListener("click", event => select(event.target));
$("transcript").addEventListener("keydown", event => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(event.target); }
});
window.addEventListener("beforeunload", () => state.controller?.abort());
render();
checkBackend().then(checkSource);
setInterval(checkBackend, 5000);
