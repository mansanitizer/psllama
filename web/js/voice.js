// Live voice (speech in, streamed speech out) and the speech tools tab.
'use strict';

// ---- Sentence player: plays each WAV the PS5 sends, back to back ----
const player = {
  nextAt: 0, sources: new Set(), pending: 0, onIdle: null,
  async enqueue(b64) {
    const ctx = audioContext();
    this.pending++;
    try {
      const buffer = await ctx.decodeAudioData(base64ToBytes(b64).buffer);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      const at = Math.max(ctx.currentTime + 0.02, this.nextAt);
      this.nextAt = at + buffer.duration;
      this.sources.add(source);
      source.onended = () => { this.sources.delete(source); this.checkIdle(); };
      source.start(at);
    } catch { /* an undecodable chunk is skipped */ }
    this.pending--;
    this.checkIdle();
  },
  get playing() { return this.sources.size > 0 || this.pending > 0; },
  checkIdle() { if (!this.playing && this.onIdle) { const f = this.onIdle; this.onIdle = null; f(); } },
  stop() {
    this.onIdle = null;
    this.sources.forEach((s) => { s.onended = null; try { s.stop(); } catch { /* not started */ } });
    this.sources.clear();
    this.nextAt = 0;
  },
};

// ---- Live voice ----
let voiceTurns = [];          // {role, content} history sent with each turn
let voiceState = 'idle';      // idle | listening | thinking | speaking
let voiceRecorder = null;
let voiceAbort = null;
let voiceRef = null;          // base64 WAV of a reference voice, this session only

function setVoiceState(state, statusText) {
  voiceState = state;
  $('voiceMic').className = 'mic' + (state === 'idle' ? '' : ` ${state}`);
  if (statusText != null) $('voiceStatus').textContent = statusText;
  if (state !== 'listening') $('voiceLevel').style.transform = 'scale(1)';
}

function voiceBubble(role, text) {
  const bubble = h('div', { class: 'bubble', text });
  const el = h('div', { class: `msg ${role}` }, bubble);
  $('voiceList').querySelector('.empty')?.remove();
  $('voiceList').append(el);
  $('voiceScroll').scrollTop = $('voiceScroll').scrollHeight;
  return { el, bubble };
}

function renderVoiceEmpty() {
  $('voiceList').replaceChildren(h('div', { class: 'empty' },
    h('h1', { text: 'Talk to the PS5' }),
    h('div', { text: 'Whisper hears you, a small model answers, and the reply is spoken sentence by sentence.' })));
}

async function startListening() {
  if (busyLabel) { toast(`The PS5 is busy: ${busyLabel}`); return; }
  if (!micAvailable()) { pickRecording(); return; }
  player.stop();
  try {
    voiceRecorder = new Recorder();
    await voiceRecorder.start({
      onLevel: (v) => { $('voiceLevel').style.transform = `scale(${1 + v * 0.6})`; },
      onSilence: S.handsFree ? stopListening : null,
    });
    setVoiceState('listening', S.handsFree ? 'Listening… pause when you are done' : 'Listening… tap to send');
  } catch (e) {
    voiceRecorder = null;
    setVoiceState('idle', 'Tap the mic and talk');
    toast(`Microphone: ${e.message}`, true);
  }
}

async function stopListening() {
  if (!voiceRecorder) return;
  const rec = voiceRecorder;
  voiceRecorder = null;
  setVoiceState('thinking', 'Sending…');
  await voiceTurn(await rec.stop());
}

// Without microphone access (plain HTTP on a phone), fall back to the system recorder.
function pickRecording() {
  const input = h('input', { type: 'file', accept: 'audio/*', capture: 'user' });
  input.addEventListener('change', () => { if (input.files[0]) { setVoiceState('thinking', 'Sending…'); voiceTurn(input.files[0]); } });
  input.click();
}

async function voiceTurn(blob) {
  let wav;
  try { wav = await toWavBase64(blob, 16000); }
  catch { setVoiceState('idle', 'That recording could not be read. Try again.'); return; }
  if (wav.seconds < 0.4) { setVoiceState('idle', 'Too short. Tap and speak a little longer.'); return; }
  if (busyLabel) { setVoiceState('idle', `The PS5 is busy: ${busyLabel}`); return; }

  const cold = !voiceReady();
  setVoiceState('thinking', cold ? 'Waking the voice models. The first turn takes about a minute.' : 'Thinking…');
  setBusy('Voice turn');
  voiceAbort = new AbortController();
  const user = voiceBubble('user', '…');
  let reply = null, replyText = '', transcript = '', timings = null;
  const body = {
    audio_b64: wav.b64, system: S.voiceSystem, llm: S.voiceLlm, max_tokens: S.voiceMax, language: 'en',
    messages: voiceTurns.slice(-20),
  };
  if (voiceRef) body.speaker_b64 = voiceRef;

  try {
    await postStream('/api/voice/chat', body, (ev) => {
      if (ev.type === 'error' || ev.error) throw new Error(ev.message || errorMessage(ev, 0));
      if (ev.status) { $('voiceStatus').textContent = `${phaseText(ev)}…`; return; }
      if (ev.type === 'transcript') {
        transcript = (ev.text || '').trim();
        user.bubble.textContent = transcript || '(nothing heard)';
        $('voiceStatus').textContent = 'Thinking…';
      } else if (ev.type === 'text' && ev.delta) {
        if (!reply) reply = voiceBubble('assistant', '');
        replyText += ev.delta;
        reply.bubble.textContent = replyText;
        $('voiceScroll').scrollTop = $('voiceScroll').scrollHeight;
      } else if (ev.type === 'audio' && ev.audio_b64) {
        if (voiceState !== 'speaking') setVoiceState('speaking', 'Speaking… tap to stop');
        player.enqueue(ev.audio_b64);
      } else if (ev.type === 'done') {
        timings = ev.timings;
        if (!replyText && ev.reply) { replyText = ev.reply; (reply || (reply = voiceBubble('assistant', ''))).bubble.textContent = replyText; }
      }
    }, voiceAbort.signal);

    if (transcript) voiceTurns.push({ role: 'user', content: transcript });
    if (replyText.trim()) voiceTurns.push({ role: 'assistant', content: replyText.trim() });
    if (reply && timings) {
      reply.el.append(h('div', { class: 'meta' }, h('span', {
        text: `first sound ${timings.first_audio_s.toFixed(1)}s · ${timings.speech_s.toFixed(1)}s of speech in ${timings.total_s.toFixed(1)}s`,
      })));
    }
    if (!transcript) setVoiceState('idle', 'Nothing was heard. Tap and try again.');
  } catch (e) {
    player.stop();
    if (e.name === 'AbortError') {
      setVoiceState('idle', 'Stopped');
    } else {
      if (user.bubble.textContent === '…') user.el.remove();
      const failed = voiceBubble('assistant', e.message);
      failed.el.classList.add('error');
      setVoiceState('idle', 'That turn failed. Tap to try again.');
    }
  } finally {
    voiceAbort = null;
    setBusy(null);
    refreshHealth();
  }

  if (voiceState === 'idle') return;   // failed, stopped or nothing heard
  const finished = () => {
    setVoiceState('idle', 'Tap the mic and talk');
    if (S.handsFree && S.tab === 'voice' && !document.hidden) startListening();
  };
  if (player.playing) player.onIdle = finished;
  else finished();
}

$('voiceMic').addEventListener('click', () => {
  audioContext();   // unlock audio inside the tap
  if (voiceState === 'idle') startListening();
  else if (voiceState === 'listening') stopListening();
  else {
    if (voiceAbort) voiceAbort.abort();
    player.stop();
    setVoiceState('idle', 'Stopped');
  }
});
$('voiceClear').addEventListener('click', () => {
  voiceTurns = [];
  renderVoiceEmpty();
  toast('Started a new conversation');
});
$('voiceHandsFree').setAttribute('aria-pressed', String(S.handsFree));
$('voiceHandsFree').addEventListener('click', () => {
  S.handsFree = !S.handsFree;
  saveSettings();
  $('voiceHandsFree').setAttribute('aria-pressed', String(S.handsFree));
  toast(S.handsFree ? 'Hands-free on: it sends when you pause and listens again after replying' : 'Hands-free off');
});
if (!micAvailable()) $('voiceHint').textContent = `${micHint()} Tapping the mic opens your phone’s recorder instead.`;
renderVoiceEmpty();

// ---- Reference voice pickers (live voice in Settings, and the TTS card) ----
function bindVoicePicker(buttonId, fileId, infoId, onChange) {
  const button = $(buttonId), file = $(fileId), info = $(infoId);
  let active = false;
  button.addEventListener('click', () => {
    if (active) { active = false; onChange(null); info.textContent = 'Default voice'; button.textContent = 'Clone a voice'; return; }
    file.click();
  });
  file.addEventListener('change', async () => {
    const f = file.files[0];
    file.value = '';
    if (!f) return;
    try {
      const wav = await toWavBase64(f, 24000);
      active = true;
      onChange(wav.b64);
      info.textContent = `Copying the voice in ${f.name} (${wav.seconds.toFixed(0)}s)`;
      button.textContent = 'Use default voice';
    } catch { toast('That audio file could not be read', true); }
  });
}
bindVoicePicker('setVoiceRefBtn', 'setVoiceRefFile', 'setVoiceRefInfo', (b64) => { voiceRef = b64; });
let ttsRef = null;
bindVoicePicker('ttsVoiceBtn', 'ttsVoiceFile', 'ttsInfo', (b64) => { ttsRef = b64; });

// ---- Audio tab: speech to text ----
let sttRecorder = null;
async function sttRun(blob) {
  await exclusive('Transcribing', async () => {
    $('sttInfo').textContent = 'Transcribing…';
    const startedAt = performance.now();
    try {
      const out = await transcribe(blob);
      $('sttText').value = out.text;
      $('sttInfo').textContent = `${out.seconds.toFixed(1)}s of audio in ${((performance.now() - startedAt) / 1000).toFixed(1)}s`;
    } catch (e) {
      $('sttInfo').textContent = '';
      toast(`Could not transcribe: ${e.message}`, true);
    }
  });
}
$('sttRecord').addEventListener('click', async () => {
  const button = $('sttRecord');
  if (sttRecorder) {
    const rec = sttRecorder;
    sttRecorder = null;
    button.textContent = 'Record';
    button.classList.replace('danger', 'primary');
    sttRun(await rec.stop());
    return;
  }
  if (!micAvailable()) { toast(micHint(), true); return; }
  try {
    const rec = new Recorder();
    await rec.start({});
    sttRecorder = rec;
    button.textContent = 'Stop and transcribe';
    button.classList.replace('primary', 'danger');
    $('sttInfo').textContent = 'Recording…';
  } catch (e) { toast(`Microphone: ${e.message}`, true); }
});
$('sttUpload').addEventListener('click', () => $('sttFile').click());
$('sttFile').addEventListener('change', () => {
  const f = $('sttFile').files[0];
  $('sttFile').value = '';
  if (f) { audioContext(); sttRun(f); }
});
$('sttCopy').addEventListener('click', () => navigator.clipboard.writeText($('sttText').value).then(() => toast('Copied')));
$('sttToChat').addEventListener('click', () => {
  const text = $('sttText').value.trim();
  if (!text) return;
  $('chatInput').value = text;
  showTab('chat');
  autosize();
  $('chatInput').focus();
});

// ---- Audio tab: text to speech ----
$('ttsGo').addEventListener('click', async () => {
  const text = $('ttsText').value.trim();
  if (!text) { toast('Type something to speak'); return; }
  await exclusive('Speaking', async () => {
    const startedAt = performance.now();
    try {
      const body = { input: text, model: 'qwen3-tts-1.7b', language: 'en', response_format: 'wav' };
      if (ttsRef) body.speaker_b64 = ttsRef;
      const blob = await (await postJSON('/api/audio/speech', body)).blob();
      const url = URL.createObjectURL(blob);
      const audio = h('audio', { controls: true, src: url });
      const seconds = ((performance.now() - startedAt) / 1000).toFixed(1);
      $('ttsOut').prepend(h('div', { class: 'stack' }, audio,
        h('div', { class: 'row' },
          h('span', { class: 'hint grow', text: `${text.slice(0, 60)}${text.length > 60 ? '…' : ''} · ${seconds}s` }),
          h('button', { class: 'chip', type: 'button', text: 'Save', onclick: () => downloadBlob(blob, 'ps5-speech.wav') }))));
      audio.play().catch(() => {});
    } catch (e) { toast(`Could not speak: ${e.message}`, true); }
  });
});
