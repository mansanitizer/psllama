// PS5 Lab core: helpers, the proxy API, PS5 status, the models and settings sheets, tabs.
'use strict';

const $ = (id) => document.getElementById(id);

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) el.append(c);
  return el;
}

function toast(text, bad) {
  const el = h('div', { class: 'toast' + (bad ? ' bad' : ''), text });
  $('toasts').append(el);
  setTimeout(() => el.remove(), bad ? 6000 : 3000);
}

// ---- Settings (localStorage) ----
const DEFAULTS = {
  system: 'You are a helpful assistant running locally on a PlayStation 5.',
  maxTokens: 512,
  temp: 0.7,
  tools: false,
  voiceSystem: 'You are a helpful voice assistant. Answer in one to three short spoken sentences. Plain words only: no lists, markdown, emoji or code.',
  voiceLlm: 'Qwen3-4B-Q4_K_M.gguf',
  voiceMax: 200,
  handsFree: false,
  ctx: '4096',
  kv: 'q4_0',
  tab: 'chat',
};
const S = Object.assign({}, DEFAULTS, readStore('ps5lab.settings', {}));
function readStore(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function writeStore(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode or full */ }
}
function saveSettings() { writeStore('ps5lab.settings', S); }

// ---- API ----
function errorMessage(body, status) {
  try {
    const j = typeof body === 'string' ? JSON.parse(body) : body;
    const e = j && j.error;
    if (e) return typeof e === 'string' ? e : (e.message || JSON.stringify(e));
  } catch { /* not JSON */ }
  return (typeof body === 'string' && body.trim().slice(0, 200)) || `HTTP ${status}`;
}

// POSTs JSON and calls onEvent for every JSON object of the reply, whether the
// reply is a server-sent event stream or a single JSON document.
async function postStream(url, body, onEvent, signal) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream, application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new Error(errorMessage(await res.text(), res.status));
  if (!(res.headers.get('content-type') || '').includes('event-stream')) {
    onEvent(await res.json());
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '', finished = false;
  const feed = (line) => {
    line = line.trim();
    if (!line.startsWith('data:')) return;
    const data = line.slice(5).trim();
    // The proxy can hold the connection open after the last event, so the
    // [DONE] marker ends the stream rather than the socket closing.
    if (data === '[DONE]') { finished = true; return; }
    if (!data || finished) return;
    let event;
    try { event = JSON.parse(data); } catch { return; }
    onEvent(event);
  };
  try {
    while (!finished) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      lines.forEach(feed);
    }
    if (!finished) feed(buffer);
  } finally {
    reader.cancel().catch(() => {});
  }
}

async function postJSON(url, body, signal) {
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal });
  if (!res.ok) throw new Error(errorMessage(await res.text(), res.status));
  return res;
}

// ---- One job at a time ----
// The PS5 serves one request at a time, so the page runs one job at a time
// and says which.
let busyLabel = null;
function setBusy(label) {
  busyLabel = label;
  document.querySelectorAll('[data-run]').forEach((b) => { b.disabled = !!label; });
  renderStatus();
}
async function exclusive(label, job) {
  if (busyLabel) { toast(`The PS5 is busy: ${busyLabel}`); return undefined; }
  setBusy(label);
  try { return await job(); }
  finally { setBusy(null); refreshHealth(); }
}

function shortModel(name) {
  return String(name || '').split('/').pop().replace(/\.(gguf|bin)$/i, '').replace(/-0000\d-of-0000\d$/, '');
}

// Text for the load and generation phases the PS5 streams.
function phaseText(ev) {
  const model = shortModel(ev.model);
  switch (ev.status) {
    case 'unloading': return `Unloading ${model}`;
    case 'loading': return `Loading ${model}` + (ev.progress != null ? ` ${ev.progress}%` : '');
    case 'creating_context': return `Starting ${model}`;
    case 'restoring_previous': return `Load failed, restoring ${model}`;
    case 'preparing': return 'Reading the prompt';
    case 'generating':
      // Counts in the hundreds are model weights streaming in, not sampling steps.
      if (ev.steps > 100) return `Loading weights ${Math.round(100 * ev.step / ev.steps)}%`;
      return ev.step ? `Step ${ev.step} of ${ev.steps}` : 'Generating';
    case 'decoding': return 'Decoding';
    default: return ev.status || '';
  }
}

function setProgress(el, fraction, text) {
  el.hidden = fraction == null;
  if (fraction == null) return;
  el.querySelector('i').style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
  el.querySelector('span').textContent = text || '';
}

// ---- Audio helpers ----
let sharedAudio = null;
// Call from a tap handler: browsers only start audio after a user gesture.
function audioContext() {
  if (!sharedAudio) sharedAudio = new (window.AudioContext || window.webkitAudioContext)();
  if (sharedAudio.state === 'suspended') sharedAudio.resume();
  return sharedAudio;
}

function bytesToBase64(bytes) {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
  return btoa(binary);
}
function base64ToBytes(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function encodeWav(samples, rate) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const text = (at, s) => { for (let i = 0; i < s.length; i++) view.setUint8(at + i, s.charCodeAt(i)); };
  text(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(view.buffer);
}

// Any audio the browser can decode -> base64 mono 16-bit WAV at `rate`.
async function toWavBase64(blob, rate) {
  const decoded = await audioContext().decodeAudioData(await blob.arrayBuffer());
  const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * rate)), rate);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return { b64: bytesToBase64(encodeWav(rendered.getChannelData(0), rate)), seconds: decoded.duration };
}

const micAvailable = () => !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
const micHint = () => (window.isSecureContext ? 'This browser cannot record audio.' : 'The microphone needs HTTPS: open the https:// address of this page (port 8443).');

// Microphone recorder with a level meter and an optional "you stopped talking" callback.
class Recorder {
  async start({ onLevel, onSilence } = {}) {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    this.chunks = [];
    this.rec = new MediaRecorder(this.stream);
    this.rec.ondataavailable = (e) => { if (e.data.size) this.chunks.push(e.data); };
    this.rec.start();
    this.active = true;
    const ctx = audioContext();
    this.source = ctx.createMediaStreamSource(this.stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    this.source.connect(analyser);
    const data = new Float32Array(analyser.fftSize);
    const started = performance.now();
    let spokeAt = 0, quietSince = 0, noise = 0.01;
    const tick = () => {
      if (!this.active) return;
      analyser.getFloatTimeDomainData(data);
      let sum = 0;
      for (const v of data) sum += v * v;
      const level = Math.sqrt(sum / data.length);
      const now = performance.now();
      if (onLevel) onLevel(Math.min(1, level * 6));
      const loud = level > Math.max(0.02, noise * 3);
      if (!loud) noise = noise * 0.95 + level * 0.05;
      if (loud) { if (!spokeAt) spokeAt = now; quietSince = 0; }
      else if (spokeAt && !quietSince) quietSince = now;
      const pausedAfterSpeech = spokeAt && quietSince && now - quietSince > 1100 && quietSince - spokeAt > 250;
      if (onSilence && (pausedAfterSpeech || now - started > 30000)) { onSilence(); return; }
      this.raf = requestAnimationFrame(tick);
    };
    tick();
  }
  stop() {
    this.active = false;
    cancelAnimationFrame(this.raf);
    return new Promise((resolve) => {
      const finish = () => {
        this.stream.getTracks().forEach((t) => t.stop());
        try { this.source.disconnect(); } catch { /* already gone */ }
        resolve(new Blob(this.chunks, { type: this.rec.mimeType || 'audio/webm' }));
      };
      if (this.rec.state === 'inactive') finish();
      else { this.rec.onstop = finish; this.rec.stop(); }
    });
  }
}

function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  h('a', { href: url, download: name }).click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

// ---- PS5 status ----
let health = null;
async function refreshHealth() {
  if (busyLabel) return;
  try {
    const res = await fetch('/api/health');
    health = await res.json();
  } catch { health = { online: false, proxyDown: true }; }
  renderStatus();
}

function loadedParts() {
  if (!health || !health.online) return [];
  const parts = [];
  if (health.model_loaded && health.model_file && health.model_file !== 'unknown') parts.push(['Chat', shortModel(health.model_file)]);
  if (health.image_model) parts.push(['Image / video', health.image_model]);
  if (health.stt_model) parts.push(['Speech to text', shortModel(health.stt_model)]);
  if (health.tts_model) parts.push(['Text to speech', shortModel(health.tts_model)]);
  return parts;
}
const voiceReady = () => !!(health && health.online && health.model_loaded && health.stt_model && health.tts_model);

function renderStatus() {
  const dot = $('statusDot'), text = $('statusText');
  if (busyLabel) { dot.className = 'dot busy'; text.textContent = busyLabel; }
  else if (!health) { dot.className = 'dot'; text.textContent = 'Connecting…'; }
  else if (health.proxyDown) { dot.className = 'dot bad'; text.textContent = 'This page lost its server'; }
  else if (!health.online) { dot.className = 'dot bad'; text.textContent = 'PS5 offline'; }
  else if (health.busy) { dot.className = 'dot busy'; text.textContent = 'PS5 busy'; }
  else {
    dot.className = 'dot ok';
    const names = [];
    if (health.model_loaded && health.model_file !== 'unknown') names.push(shortModel(health.model_file));
    if (health.image_model) names.push(health.image_model);
    if (health.stt_model && health.tts_model) names.push('voice');
    text.textContent = names.join(' · ') || 'Nothing loaded';
  }
  const summary = $('loadedSummary');
  summary.replaceChildren(...loadedParts().map(([k, v]) => h('span', { class: 'pill' }, h('b', { text: k }), v)));
  if (!summary.children.length) summary.append(h('span', { class: 'hint', text: health && health.online ? 'Nothing is loaded.' : 'The PS5 is not answering.' }));
  document.querySelectorAll('#modelList .item').forEach((el) => {
    el.classList.toggle('on', !!health && health.model_loaded && el.dataset.id === health.model_file);
  });
}

// ---- Models sheet ----
let llmModels = [];
const modelSeesImages = () => !!(health && health.model_loaded && llmModels.some((m) => m.vision && m.id === health.model_file));
async function refreshModels() {
  try {
    const data = await (await fetch('/api/models')).json();
    // Hide aliases and the later parts of split files.
    // The PS5 lists the loaded model under the alias "ps5-local"; its file name is the real id.
    const all = (data.data || []).map((m) => (m.filename ? { ...m, id: m.filename } : m));
    // While the PS5 is busy the proxy answers with a placeholder list; keep the last real one.
    if (!all.some((m) => m.object)) throw new Error('placeholder list');
    // A model can take images when its vision projector sits next to it.
    const projectors = new Set(all.filter((m) => m.id.endsWith('.mmproj.gguf')).map((m) => m.id.replace(/\.mmproj\.gguf$/, '.gguf')));
    llmModels = all.filter((m) => m.id.endsWith('.gguf') && !m.id.endsWith('.mmproj.gguf') && m.id !== 'model.gguf' && !/-0000[2-9]-of-/.test(m.id))
      .sort((a, b) => (a.size_bytes || 0) - (b.size_bytes || 0));
    llmModels.forEach((m) => { m.vision = projectors.has(m.id); });
  } catch { /* keep the list we had */ }
  $('modelList').replaceChildren(...llmModels.map((m) => h('button', { class: 'item', type: 'button', 'data-id': m.id, 'data-run': true, onclick: () => loadModel(m.id) },
    h('span', { class: 'name', text: shortModel(m.id) }),
    m.vision ? h('span', { class: 'tag', text: 'sees images' }) : null,
    h('span', { class: 'size', text: m.size_formatted || '' }))));
  if (!llmModels.length) $('modelList').append(h('p', { class: 'hint', text: 'Could not read the model list from the PS5.' }));
  // Voice needs a model small enough to share the GPU with the speech models.
  const small = llmModels.filter((m) => (m.size_bytes || 0) < 5e9);
  $('setVoiceLlm').replaceChildren(...small.map((m) => h('option', { value: m.id, text: `${shortModel(m.id)} (${m.size_formatted})` })));
  if (small.some((m) => m.id === S.voiceLlm)) $('setVoiceLlm').value = S.voiceLlm;
  renderStatus();
}

// Qwen2.5-VL produces garbage with a 4-bit KV cache on the PS5; 8-bit is fine.
const kvFor = (id) => (/qwen2\.5-vl/i.test(id) && S.kv === 'q4_0' ? 'q8_0' : S.kv);

async function loadModel(id) {
  const bar = $('loadProgress');
  await exclusive(`Loading ${shortModel(id)}`, async () => {
    let failure = null;
    setProgress(bar, 0, 'Connecting');
    try {
      await postStream('/api/models/load', {
        model: id, context_tokens: Number(S.ctx), max_output_tokens: 4096,
        cache_type_k: kvFor(id), cache_type_v: kvFor(id), flash_attention: 1, gpu_layers: 999,
        threads: 4, threads_batch: 4, batch_tokens: 128, stream: true,
      }, (ev) => {
        if (ev.error) failure = errorMessage(ev, 0);
        else if (ev.http_status >= 400) failure = `The PS5 refused the load (${ev.http_status})`;
        if (ev.status) setProgress(bar, ev.status === 'loading' ? (ev.progress || 0) / 100 : 1, phaseText(ev));
      });
      if (failure) throw new Error(failure);
      toast(`${shortModel(id)} is ready`);
    } catch (e) {
      toast(`Load failed: ${e.message}`, true);
    } finally {
      setProgress(bar, null);
    }
  });
}

// ---- Sheets, settings, tabs ----
function openSheet(dialog) {
  dialog.showModal();
}
document.querySelectorAll('dialog.sheet').forEach((d) => {
  d.querySelector('[data-close]').addEventListener('click', () => d.close());
  d.addEventListener('click', (e) => { if (e.target === d) d.close(); });
});

$('statusBtn').addEventListener('click', () => { openSheet($('modelsSheet')); refreshHealth(); refreshModels(); });
$('settingsBtn').addEventListener('click', () => openSheet($('settingsSheet')));

function bindSetting(id, key, parse) {
  const el = $(id);
  el.value = S[key];
  el.addEventListener('change', () => {
    const v = parse ? parse(el.value) : el.value;
    if (v === '' || (typeof v === 'number' && Number.isNaN(v))) { el.value = S[key]; return; }
    S[key] = v;
    saveSettings();
  });
}
const clampInt = (lo, hi) => (v) => Math.max(lo, Math.min(hi, parseInt(v, 10)));
bindSetting('setSystem', 'system');
bindSetting('setMaxTokens', 'maxTokens', clampInt(16, 4096));
bindSetting('setTemp', 'temp', (v) => Math.max(0, Math.min(1.5, parseFloat(v))));
bindSetting('setVoiceSystem', 'voiceSystem');
bindSetting('setVoiceMax', 'voiceMax', clampInt(16, 1024));
bindSetting('setVoiceLlm', 'voiceLlm');
bindSetting('optCtx', 'ctx');
bindSetting('optKv', 'kv');

function showTab(name) {
  if (!$(`view-${name}`)) name = 'chat';
  document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${name}`));
  document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  S.tab = name;
  saveSettings();
}
$('tabs').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-tab]');
  if (b) showTab(b.dataset.tab);
});
showTab(S.tab);

refreshHealth();
refreshModels();
setInterval(refreshHealth, 10000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) refreshHealth(); });
