// Image and video generation.
'use strict';

// ---- A small IndexedDB gallery so images survive a reload ----
const gallery = {
  db: null,
  open() {
    return new Promise((resolve) => {
      if (!window.indexedDB) { resolve(null); return; }
      const req = indexedDB.open('ps5lab', 1);
      req.onupgradeneeded = () => req.result.createObjectStore('images', { keyPath: 'id' });
      req.onsuccess = () => { this.db = req.result; resolve(this.db); };
      req.onerror = () => resolve(null);
    });
  },
  run(mode, fn) {
    return new Promise((resolve) => {
      if (!this.db) { resolve(null); return; }
      const req = fn(this.db.transaction('images', mode).objectStore('images'));
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    });
  },
  all() { return this.run('readonly', (s) => s.getAll()).then((rows) => (rows || []).sort((a, b) => b.id - a.id)); },
  put(item) { return this.run('readwrite', (s) => s.put(item)); },
  remove(id) { return this.run('readwrite', (s) => s.delete(id)); },
};

// Progress through unload, load, prompt, sampling and decode as one 0..1 bar.
function generationProgress(ev) {
  switch (ev.status) {
    case 'unloading': return 0.04;
    case 'loading': return 0.05 + 0.35 * (ev.progress || 0) / 100;
    case 'preparing': return 0.45;
    case 'generating': return ev.steps > 100 ? 0.47 : 0.5 + 0.42 * (ev.step || 0) / (ev.steps || 1);
    case 'decoding': return 0.95;
    default: return null;
  }
}

async function generate(url, body, bar) {
  let result = null, failure = null;
  const startedAt = performance.now();
  let label = 'Connecting';
  const timer = setInterval(() => {
    bar.querySelector('span').textContent = `${label} · ${((performance.now() - startedAt) / 1000).toFixed(0)}s`;
  }, 500);
  setProgress(bar, 0.02, label);
  try {
    await postStream(url, body, (ev) => {
      if (ev.error) failure = errorMessage(ev, 0);
      if (Array.isArray(ev.data) && ev.data.length) result = ev;
      const p = generationProgress(ev);
      if (p != null) { label = phaseText(ev); setProgress(bar, p, label); }
    });
  } finally {
    clearInterval(timer);
    setProgress(bar, null);
  }
  if (!result) throw new Error(failure || 'The PS5 finished without returning a result.');
  return result;
}

// ================= Images =================
const IMAGE_SIZES = ['512x512', '768x768', '1024x1024', '768x1024', '1024x768', '576x1024', '1024x576'];
let imageModels = [];

// Models that edit from a reference image ("the same fox, wearing a scarf").
// The others can only redraw over a start image by a chosen amount.
const REFERENCE_EDIT_MODELS = ['flux2-klein-4b'];
let sourceImage = null;   // data URL of the image being edited

function renderImageHint() {
  const m = imageModels.find((x) => x.id === $('imgModel').value);
  if (!m) return;
  const byReference = REFERENCE_EDIT_MODELS.includes(m.id);
  $('imgStrength').hidden = !sourceImage || byReference;
  $('imgPrompt').placeholder = sourceImage ? (byReference ? 'Describe the change, e.g. “the same fox wearing a blue scarf, at night”' : 'Describe the redrawn image…') : 'Describe the image…';
  $('imgGo').textContent = sourceImage ? 'Edit' : 'Generate';
  if (sourceImage) $('imgModelHint').textContent = byReference ? 'Keeps the subject of your image and applies the change you describe.' : 'This model redraws over your image; for true edits use FLUX.2-klein.';
  else $('imgModelHint').textContent = `${m.badge ? m.badge + ' · ' : ''}about ${Math.round(m.warm_time_s) || 1}s once loaded`;
}

function setSourceImage(dataUrl) {
  sourceImage = dataUrl;
  $('imgSourceThumb').hidden = !dataUrl;
  if (dataUrl) $('imgSourceThumb').src = dataUrl;
  $('imgSourceBtn').textContent = dataUrl ? 'Stop editing' : 'Edit an image';
  renderImageHint();
}

function imageModelChanged() {
  const m = imageModels.find((x) => x.id === $('imgModel').value);
  if (!m) return;
  $('imgSize').value = `${m.default_width}x${m.default_height}`;
  $('imgSteps').value = m.default_steps;
  renderImageHint();
}

async function initImages() {
  $('imgSize').replaceChildren(...IMAGE_SIZES.map((s) => h('option', { value: s, text: s.replace('x', ' × ') })));
  try { imageModels = (await (await fetch('/api/images/models')).json()).data || []; } catch { imageModels = []; }
  $('imgModel').replaceChildren(...imageModels.map((m) => h('option', { value: m.id, text: m.name })));
  const saved = readStore('ps5lab.image', {});
  if (imageModels.some((m) => m.id === saved.model)) $('imgModel').value = saved.model;
  imageModelChanged();
  if (saved.prompt) $('imgPrompt').value = saved.prompt;
  $('imgModel').addEventListener('change', imageModelChanged);
  await gallery.open();
  (await gallery.all()).forEach((item) => $('imgGrid').append(imageTile(item)));
}

const imageUrl = (item) => `data:image/png;base64,${item.b64}`;

function imageTile(item) {
  return h('div', { class: 'tile', 'data-id': item.id },
    h('img', { src: imageUrl(item), alt: item.prompt, loading: 'lazy', onclick: () => openImage(item) }));
}

function openImage(item) {
  const close = () => box.remove();
  const box = h('div', { class: 'lightbox', onclick: (e) => { if (e.target === box) close(); } },
    h('div', { class: 'tile wide', style: 'max-width:720px;max-height:100%;overflow:auto' },
      h('img', { src: imageUrl(item), alt: item.prompt }),
      h('div', { class: 'cap' },
        h('p', { text: item.prompt }),
        h('span', { text: `${item.model} · ${item.width} × ${item.height} · ${item.steps} step${item.steps === 1 ? '' : 's'} · seed ${item.seed} · ${item.seconds.toFixed(1)}s` }),
        h('div', { class: 'row wrap' },
          h('button', { class: 'chip', type: 'button', text: 'Save', onclick: () => downloadBlob(new Blob([base64ToBytes(item.b64)], { type: 'image/png' }), `ps5-${item.id}.png`) }),
          h('button', { class: 'chip', type: 'button', text: 'Reuse prompt', onclick: () => { $('imgPrompt').value = item.prompt; $('imgSeed').value = item.seed; close(); } }),
          h('button', { class: 'chip', type: 'button', text: 'Edit', onclick: () => {
            if (imageModels.some((m) => m.id === REFERENCE_EDIT_MODELS[0])) { $('imgModel').value = REFERENCE_EDIT_MODELS[0]; imageModelChanged(); }
            setSourceImage(imageUrl(item)); $('imgPrompt').value = ''; close(); $('imgPrompt').focus();
          } }),
          h('button', { class: 'chip', type: 'button', text: 'Animate', onclick: () => { setFirstFrame(imageUrl(item)); $('vidPrompt').value = $('vidPrompt').value || item.prompt; close(); showTab('video'); } }),
          h('button', { class: 'chip', type: 'button', text: 'Delete', onclick: async () => { await gallery.remove(item.id); document.querySelector(`.tile[data-id="${item.id}"]`)?.remove(); close(); } }),
          h('span', { class: 'grow' }),
          h('button', { class: 'chip', type: 'button', text: 'Close', onclick: close })))));
  document.body.append(box);
}

$('imgSourceBtn').addEventListener('click', () => { if (sourceImage) setSourceImage(null); else $('imgSourceFile').click(); });
$('imgSourceFile').addEventListener('change', () => {
  const f = $('imgSourceFile').files[0];
  $('imgSourceFile').value = '';
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => setSourceImage(reader.result);
  reader.readAsDataURL(f);
});

$('imgGo').addEventListener('click', async () => {
  const prompt = $('imgPrompt').value.trim();
  if (!prompt) { toast('Describe the image first'); return; }
  const [width, height] = $('imgSize').value.split('x').map(Number);
  const model = $('imgModel').value;
  const info = imageModels.find((m) => m.id === model) || {};
  const body = { model, prompt, width, height, size: `${width}x${height}`, steps: Number($('imgSteps').value) || info.default_steps || 4, cfg_scale: info.default_cfg || 1.0, stream: true };
  if ($('imgNegative').value.trim()) body.negative_prompt = $('imgNegative').value.trim();
  if ($('imgSeed').value !== '') body.seed = Number($('imgSeed').value);
  if (!sourceImage) writeStore('ps5lab.image', { model, prompt });
  await exclusive(sourceImage ? 'Editing an image' : 'Generating an image', async () => {
    try {
      if (sourceImage && REFERENCE_EDIT_MODELS.includes(model)) {
        body.ref_images_b64 = [await imageBase64(sourceImage, { maxSide: 1024 })];
      } else if (sourceImage) {
        body.init_image_b64 = await imageBase64(sourceImage, { width, height });
        body.strength = Number($('imgStrength').value);
      }
      const ev = await generate('/api/images/generate', body, $('imgProgress'));
      const item = {
        id: Date.now(), b64: ev.data[0].b64_json, prompt, model,
        width: ev.width || width, height: ev.height || height, steps: ev.steps || body.steps,
        seed: ev.seed ?? '?', seconds: (ev.timings && ev.timings.generate_s) || 0,
      };
      await gallery.put(item);
      $('imgGrid').prepend(imageTile(item));
      openImage(item);
    } catch (e) { toast(`Image failed: ${e.message}`, true); }
  });
});

// ================= Video =================
const VIDEO_SIZES = ['416x240', '416x224', '640x352', '832x480'];
let videoModels = [];
let firstFrame = null;   // data URL of the optional first frame

function setFirstFrame(dataUrl) {
  firstFrame = dataUrl;
  $('vidImageThumb').hidden = !dataUrl;
  if (dataUrl) $('vidImageThumb').src = dataUrl;
  $('vidImageBtn').textContent = dataUrl ? 'Remove first frame' : 'Add first frame';
}

function videoModelChanged() {
  const m = videoModels.find((x) => x.id === $('vidModel').value);
  if (!m) return;
  const size = `${m.default_width}x${m.default_height}`;
  const sizes = VIDEO_SIZES.includes(size) ? VIDEO_SIZES : [size, ...VIDEO_SIZES];
  $('vidSize').replaceChildren(...sizes.map((s) => h('option', { value: s, text: s.replace('x', ' × ') })));
  $('vidSize').value = size;
  $('vidFrames').value = String(m.default_frames);
  $('vidSteps').value = m.default_steps;
  const needsImage = (m.modes || [])[0] === 'i2v';
  $('vidModelHint').textContent = `${needsImage ? 'Needs a first frame · ' : ''}about ${Math.round(m.warm_time_s)}s at the default size`;
}

async function initVideo() {
  $('vidFrames').replaceChildren(...[9, 17, 25, 33].map((n) => h('option', { value: n, text: `${n} (${(n / 16).toFixed(1)}s)` })));
  try { videoModels = (await (await fetch('/api/videos/models')).json()).data || []; } catch { videoModels = []; }
  $('vidModel').replaceChildren(...videoModels.map((m) => h('option', { value: m.id, text: m.name })));
  videoModelChanged();
  $('vidModel').addEventListener('change', videoModelChanged);
}

$('vidImageBtn').addEventListener('click', () => { if (firstFrame) setFirstFrame(null); else $('vidImageFile').click(); });
$('vidImageFile').addEventListener('change', () => {
  const f = $('vidImageFile').files[0];
  $('vidImageFile').value = '';
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => setFirstFrame(reader.result);
  reader.readAsDataURL(f);
});

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('That image could not be read'));
    img.src = src;
  });
}

// An image as base64 PNG: cropped to fill width x height, or scaled down so
// its longer side is at most maxSide.
async function imageBase64(src, { width, height, maxSide }) {
  const img = await loadImage(src);
  if (!width) {
    const shrink = Math.min(1, maxSide / Math.max(img.width, img.height));
    width = Math.round(img.width * shrink);
    height = Math.round(img.height * shrink);
  }
  const canvas = h('canvas', { width, height });
  const scale = Math.max(width / img.width, height / img.height);
  const w = img.width * scale, hgt = img.height * scale;
  canvas.getContext('2d').drawImage(img, (width - w) / 2, (height - hgt) / 2, w, hgt);
  return canvas.toDataURL('image/png').split(',')[1];
}

function clipCard(frames, ev, prompt) {
  const canvas = h('canvas', { width: frames[0].width, height: frames[0].height });
  const ctx = canvas.getContext('2d');
  const fps = ev.fps || 16;
  let playing = true, index = 0, last = 0;
  const draw = (now) => {
    if (!canvas.isConnected) return;
    if (playing && now - last >= 1000 / fps) { last = now; ctx.drawImage(frames[index], 0, 0); index = (index + 1) % frames.length; }
    requestAnimationFrame(draw);
  };
  ctx.drawImage(frames[0], 0, 0);
  setTimeout(() => requestAnimationFrame(draw), 0);
  canvas.addEventListener('click', () => { playing = !playing; });

  const save = async (button) => {
    const type = ['video/mp4', 'video/webm;codecs=vp9', 'video/webm'].find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
    if (!type || !canvas.captureStream) { toast('This browser cannot export video', true); return; }
    button.disabled = true;
    const rec = new MediaRecorder(canvas.captureStream(fps), { mimeType: type, videoBitsPerSecond: 6e6 });
    const chunks = [];
    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onstop = () => { downloadBlob(new Blob(chunks, { type }), `ps5-clip.${type.includes('mp4') ? 'mp4' : 'webm'}`); button.disabled = false; };
    index = 0; playing = true;
    rec.start();
    setTimeout(() => rec.stop(), (frames.length / fps) * 2 * 1000 + 150);   // two loops
  };

  return h('div', { class: 'card clip' }, canvas,
    h('div', { class: 'hint', text: prompt }),
    h('div', { class: 'row wrap' },
      h('span', { class: 'hint grow', text: `${ev.model} · ${ev.width} × ${ev.height} · ${frames.length} frames · ${((ev.timings && ev.timings.generate_s) || 0).toFixed(0)}s · tap the clip to pause` }),
      h('button', { class: 'chip', type: 'button', text: 'Save video', onclick: (e) => save(e.target) })));
}

$('vidGo').addEventListener('click', async () => {
  const prompt = $('vidPrompt').value.trim();
  if (!prompt) { toast('Describe the clip first'); return; }
  const model = $('vidModel').value;
  const info = videoModels.find((m) => m.id === model) || {};
  if ((info.modes || [])[0] === 'i2v' && !firstFrame) { toast('This model needs a first frame'); return; }
  const [width, height] = $('vidSize').value.split('x').map(Number);
  const body = {
    model, prompt, width, height, size: `${width}x${height}`, frames: Number($('vidFrames').value), fps: info.default_fps || 16,
    steps: Number($('vidSteps').value) || info.default_steps || 20, cfg_scale: info.default_cfg || 6.0, stream: true,
  };
  await exclusive('Generating a video', async () => {
    try {
      if (firstFrame) body.init_image_b64 = await imageBase64(firstFrame, { width, height });
      const ev = await generate('/api/videos/generate', body, $('vidProgress'));
      const frames = await Promise.all(ev.data.map((f) => loadImage(`data:image/png;base64,${f.b64_png || f.b64_json}`)));
      $('vidResults').prepend(clipCard(frames, Object.assign({ model, width, height }, ev), prompt));
    } catch (e) { toast(`Video failed: ${e.message}`, true); }
  });
});

initImages();
initVideo();
