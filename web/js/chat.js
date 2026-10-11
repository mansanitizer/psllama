// Chat: streaming replies, optional local tools, dictation and read-aloud.
'use strict';

const TOOLS_PROMPT = `# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "calculator", "description": "Evaluates a mathematical expression and returns the exact number.", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "The math expression to evaluate, e.g. '38947 * 4928'"}}, "required": ["expression"]}}}
{"type": "function", "function": {"name": "get_current_time", "description": "Returns the current local and UTC date and time.", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "python_interpreter", "description": "Executes a Python 3 code snippet and returns stdout/stderr.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code snippet to execute"}}, "required": ["code"]}}}
{"type": "function", "function": {"name": "fetch_web", "description": "Fetches web page text content from a URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The URL to fetch content from"}}, "required": ["url"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "<function-name>", "arguments": <args-json-object>}
</tool_call>`;

const SUGGESTIONS = [
  'Explain how a PS5 can run a language model.',
  'Write a haiku about a console that learned to talk.',
  'What is 38947 × 4928? Use the calculator.',
];

let chat = readStore('ps5lab.chat', []);
let chatAbort = null;

const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function renderMarkdown(raw) {
  let text = raw.replace(/<tool_call>[\s\S]*?(<\/tool_call>|$)/g, '').trim();
  let thinking = '';
  text = text.replace(/<think>([\s\S]*?)(<\/think>|$)/g, (_, t) => { thinking += t.trim(); return ''; }).trim();
  let html = '';
  if (thinking) html += `<details><summary>Thinking</summary><div style="white-space:pre-wrap">${escapeHtml(thinking)}</div></details>`;
  if (window.marked && window.DOMPurify) html += DOMPurify.sanitize(marked.parse(text, { breaks: true, gfm: true }));
  else html += `<div style="white-space:pre-wrap">${escapeHtml(text)}</div>`;
  return html;
}
function highlight(el) {
  if (window.hljs) el.querySelectorAll('pre code:not(.hljs)').forEach((c) => hljs.highlightElement(c));
}
// The spoken form of a reply: no markup, code or reasoning.
function plainText(raw) {
  return raw.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').replace(/<tool_call>[\s\S]*?(<\/tool_call>|$)/g, '')
    .replace(/```[\s\S]*?```/g, ' ').replace(/[*_#`>|]/g, '').replace(/\s+/g, ' ').trim();
}

function toolLine(t) {
  const result = t.ok ? JSON.stringify(t.result) : `failed: ${t.error}`;
  return h('div', { class: 'tool', text: `${t.name}(${JSON.stringify(t.args)}) → ${String(result).slice(0, 300)}` });
}

function messageEl(m, index) {
  if (m.role === 'user') {
    return h('div', { class: 'msg user' }, h('div', { class: 'bubble' },
      m.image ? h('img', { class: 'attached', src: m.image, alt: 'Attached image' }) : null, m.content));
  }
  const el = h('div', { class: 'msg assistant' + (m.error ? ' error' : '') });
  (m.tools || []).forEach((t) => el.append(toolLine(t)));
  const bubble = h('div', { class: 'bubble' + (m.pending ? ' typing' : ''), html: m.error ? escapeHtml(m.error) : renderMarkdown(m.content) });
  highlight(bubble);
  el.append(bubble);
  if (!m.pending && !m.error) {
    el.append(h('div', { class: 'meta' },
      m.stats ? h('span', { text: m.stats }) : null,
      h('button', { type: 'button', text: 'Copy', onclick: () => navigator.clipboard.writeText(m.content).then(() => toast('Copied')) }),
      h('button', { type: 'button', text: 'Listen', onclick: (e) => speakMessage(m, e.target) }),
      index === chat.length - 1 ? h('button', { type: 'button', text: 'Retry', onclick: retryLast }) : null));
  }
  return el;
}

function renderChat() {
  const list = $('chatList');
  if (!chat.length) {
    list.replaceChildren(h('div', { class: 'empty' },
      h('h1', { text: 'Ask the PS5' }),
      h('div', { text: 'Everything runs on the console’s GPU. Nothing leaves your network.' }),
      h('div', { class: 'suggest' }, SUGGESTIONS.map((s) => h('button', { class: 'chip', type: 'button', text: s, onclick: () => sendChat(s) })))));
    return;
  }
  list.replaceChildren(...chat.map(messageEl));
  scrollChat(true);
}
function scrollChat(force) {
  const sc = $('chatScroll');
  if (force || sc.scrollHeight - sc.scrollTop - sc.clientHeight < 160) sc.scrollTop = sc.scrollHeight;
}
function saveChat() {
  // Attached images are too big for localStorage; they last until the page reloads.
  writeStore('ps5lab.chat', chat.filter((m) => !m.pending).slice(-60).map((m) => (m.image ? { ...m, image: undefined } : m)));
}

async function streamCompletion(messages, onText, signal) {
  let text = '', tokens = 0, firstAt = 0;
  const startedAt = performance.now();
  await postStream('/api/chat', {
    model: 'ps5-local', messages, max_tokens: S.maxTokens, temperature: S.temp, top_p: 1.0, repeat_penalty: 1.1, stream: true,
  }, (ev) => {
    if (ev.error) throw new Error(errorMessage(ev, 0));
    const delta = ev.choices && ev.choices[0] && ev.choices[0].delta && ev.choices[0].delta.content;
    if (!delta) return;
    if (!firstAt) firstAt = performance.now();
    tokens++;
    text += delta;
    onText(text);
  }, signal);
  const seconds = (performance.now() - (firstAt || startedAt)) / 1000;
  return { text, tokens, rate: tokens > 1 && seconds > 0 ? (tokens - 1) / seconds : 0, firstS: firstAt ? (firstAt - startedAt) / 1000 : 0 };
}

async function runTool(call) {
  const record = { name: call.name, args: call.arguments || {}, ok: false };
  try {
    const res = await fetch('/api/execute_tool', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: record.name, arguments: record.args }) });
    const data = await res.json();
    record.ok = !!data.success;
    record.result = data.result;
    record.error = data.error;
  } catch (e) { record.error = e.message; }
  return record;
}

// An image attached to the next message, as a JPEG data URL (at most 896 px a side).
let chatImage = null;
function setChatImage(dataUrl) {
  chatImage = dataUrl;
  $('chatImageThumb').hidden = !dataUrl;
  if (dataUrl) $('chatImageThumb').src = dataUrl;
}
$('chatImage').addEventListener('click', () => $('chatImageFile').click());
$('chatImageThumb').addEventListener('click', () => setChatImage(null));
$('chatImageFile').addEventListener('change', () => {
  const f = $('chatImageFile').files[0];
  $('chatImageFile').value = '';
  if (!f) return;
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const img = await loadImage(reader.result);
      const scale = Math.min(1, 896 / Math.max(img.width, img.height));
      const canvas = h('canvas', { width: Math.round(img.width * scale), height: Math.round(img.height * scale) });
      canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
      setChatImage(canvas.toDataURL('image/jpeg', 0.9));
      if (!modelSeesImages()) toast('The loaded model cannot see images. Load one marked “sees images” from the status menu.');
    } catch (e) { toast(e.message, true); }
  };
  reader.readAsDataURL(f);
});

async function sendChat(text) {
  text = text.trim();
  if (!text && chatImage) text = 'Describe this image.';
  if (!text) return;
  if (busyLabel) { toast(`The PS5 is busy: ${busyLabel}`); return; }
  $('chatInput').value = '';
  autosize();
  const message = { role: 'user', content: text };
  if (chatImage) message.image = chatImage;
  setChatImage(null);
  chat.push(message);
  await answer();
}

async function retryLast() {
  if (busyLabel) return;
  while (chat.length && chat[chat.length - 1].role === 'assistant') chat.pop();
  if (chat.length) await answer();
  else renderChat();
}

// Answers the conversation so far (which ends with a user message).
async function answer() {
  const reply = { role: 'assistant', content: '', pending: true, tools: [] };
  const messages = [{ role: 'system', content: S.tools ? `${S.system}\n\n${TOOLS_PROMPT}` : S.system }]
    .concat(chat.filter((m) => !m.error).map((m, i, all) => {
      // Only the newest message carries its image: each one costs a few
      // hundred tokens of context, and older ones are already described.
      if (m.image && i === all.length - 1) {
        return { role: m.role, content: [{ type: 'image_url', image_url: { url: m.image } }, { type: 'text', text: m.content }] };
      }
      return { role: m.role, content: m.image ? `[an image was attached earlier] ${m.content}` : m.content };
    }));
  chat.push(reply);
  renderChat();
  const bubble = $('chatList').lastElementChild.querySelector('.bubble');
  let frame = 0;
  const paint = (text) => {
    reply.content = text;
    if (frame) return;
    frame = requestAnimationFrame(() => { frame = 0; bubble.innerHTML = renderMarkdown(reply.content); scrollChat(); });
  };
  chatAbort = new AbortController();
  $('chatSend').classList.add('stop');
  $('chatSend').innerHTML = '<svg class="ico" viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor"/></svg>';
  setBusy('Replying');
  try {
    for (let round = 0; ; round++) {
      const out = await streamCompletion(messages, paint, chatAbort.signal);
      reply.stats = `${out.tokens} tokens · ${out.rate.toFixed(1)} tok/s · first in ${out.firstS.toFixed(1)}s`;
      const match = S.tools && round < 3 && out.text.match(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/);
      if (!match) break;
      let call;
      try { call = JSON.parse(match[1]); } catch { break; }
      const record = await runTool(call);
      reply.tools.push(record);
      bubble.before(toolLine(record));
      messages.push({ role: 'assistant', content: out.text },
        { role: 'user', content: `<tool_response>\n${JSON.stringify(record.ok ? record.result : { error: record.error })}\n</tool_response>` });
      paint('');
    }
    if (!reply.content.trim()) throw new Error('The model returned an empty reply.');
  } catch (e) {
    if (e.name === 'AbortError') {
      if (!reply.content.trim()) chat.pop();
    } else if (reply.content.trim()) {
      toast(`The reply was cut short: ${e.message}`, true);
    } else {
      reply.error = e.message + (health && health.online && !health.model_loaded ? ' — no chat model is loaded; pick one from the status menu at the top.' : '');
    }
  } finally {
    cancelAnimationFrame(frame);
    reply.pending = false;
    chatAbort = null;
    $('chatSend').classList.remove('stop');
    $('chatSend').innerHTML = '<svg class="ico" viewBox="0 0 24 24"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
    setBusy(null);
    saveChat();
    renderChat();
    refreshHealth();
  }
}

// ---- Read aloud ----
let speaking = null;
async function speakMessage(m, button) {
  if (speaking) { speaking.pause(); speaking = null; return; }
  const text = plainText(m.content).slice(0, 1000);
  if (!text) return;
  await exclusive('Speaking', async () => {
    button.textContent = 'Preparing…';
    try {
      const res = await postJSON('/api/audio/speech', { input: text, model: 'qwen3-tts-1.7b', language: 'en', response_format: 'wav' });
      const audio = new Audio(URL.createObjectURL(await res.blob()));
      speaking = audio;
      audio.onended = audio.onpause = () => { speaking = null; button.textContent = 'Listen'; };
      button.textContent = 'Stop';
      await audio.play();
    } catch (e) {
      button.textContent = 'Listen';
      toast(`Could not speak: ${e.message}`, true);
    }
  });
}

// ---- Dictation ----
let dictation = null;
async function transcribe(blob) {
  const wav = await toWavBase64(blob, 16000);
  const res = await postJSON('/api/audio/transcriptions', { audio_b64: wav.b64, model: 'whisper/ggml-large-v3-turbo-q5_0.bin', language: 'en', translate: false });
  const data = await res.json();
  return { text: (data.text || '').trim(), seconds: wav.seconds };
}
$('chatMic').addEventListener('click', async () => {
  const button = $('chatMic');
  if (dictation) {
    const rec = dictation;
    dictation = null;
    button.classList.remove('on');
    const blob = await rec.stop();
    await exclusive('Transcribing', async () => {
      try {
        const { text } = await transcribe(blob);
        if (!text) { toast('No speech heard'); return; }
        const input = $('chatInput');
        input.value = (input.value.trim() + ' ' + text).trim();
        autosize();
        input.focus();
      } catch (e) { toast(`Could not transcribe: ${e.message}`, true); }
    });
    return;
  }
  if (!micAvailable()) { toast(micHint(), true); return; }
  if (busyLabel) { toast(`The PS5 is busy: ${busyLabel}`); return; }
  try {
    const rec = new Recorder();
    await rec.start({});
    dictation = rec;
    button.classList.add('on');
  } catch (e) { toast(`Microphone: ${e.message}`, true); }
});

// ---- Composer ----
function autosize() {
  const input = $('chatInput');
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, window.innerHeight * 0.3)}px`;
}
$('chatInput').addEventListener('input', autosize);
$('chatInput').addEventListener('keydown', (e) => {
  // Enter sends with a keyboard; on touch screens Enter stays a new line.
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && !matchMedia('(pointer: coarse)').matches) {
    e.preventDefault();
    $('chatForm').requestSubmit();
  }
});
$('chatForm').addEventListener('submit', (e) => {
  e.preventDefault();
  if (chatAbort) chatAbort.abort();
  else sendChat($('chatInput').value);
});
$('chatClear').addEventListener('click', () => {
  if (chatAbort) chatAbort.abort();
  chat = [];
  saveChat();
  renderChat();
});
$('toolsToggle').setAttribute('aria-pressed', String(S.tools));
$('toolsToggle').addEventListener('click', () => {
  S.tools = !S.tools;
  saveSettings();
  $('toolsToggle').setAttribute('aria-pressed', String(S.tools));
  toast(S.tools ? 'Tools on: calculator, clock, Python, web fetch' : 'Tools off');
});

renderChat();
