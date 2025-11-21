// Simple local annotation tool
// - Load input folder of JSON files with shape like { players: [{username, role}], chats: [{userName, chat}] }
// - Load categories JSONL with { ss_technique, ss_definition?, ss_example? }
// - Pick number of files (alphabetical)
// - Show target message with 5 before and 5 after
// - Select techniques (multi) or None (clear)
// - Save annotations to JSON

const els = {
  setupForm: document.getElementById('setup-form'),
  inputFile: document.getElementById('inputFile'),
  categoriesFile: document.getElementById('categoriesFile'),
  outputSuffix: document.getElementById('outputSuffix'),
  sampleCount: document.getElementById('sampleCount'),
  seedInput: document.getElementById('seedInput'),
  startBtn: document.getElementById('startBtn'),

  annotator: document.getElementById('annotator'),
  progress: document.getElementById('progress'),
  techniques: document.getElementById('techniques'),
  techSearch: document.getElementById('techSearch'),
  noneBtn: document.getElementById('noneBtn'),
  context: document.getElementById('context'),

  prevBtn: document.getElementById('prevBtn'),
  nextBtn: document.getElementById('nextBtn'),
  finishBtn: document.getElementById('finishBtn'),
};

/** State */
const state = {
  // Files selected from the folder
  files: [], // [{ name, content, chats: ["user: msg"], rawChats: [{userName, chat}], roles: Map(username->role) }]
  fileIndex: 0,
  chats: [], // pointer to current file chats
  rawChats: [], // pointer to current file raw chats
  roles: new Map(), // pointer to current file roles

  categories: [], // [{ name, def, example }]
  samples: [], // [{ fileIndex, messageIndex }]
  sampleIndex: 0, // current sample pointer (0..samples.length-1)
  selections: new Map(), // key: `${fileIndex}:${messageIndex}` -> Set of technique names
  sampleLimit: 0,
  isCialdini: false,
  savedFiles: new Set(), // track which files have already been saved
};

function assert(predicate, msg) { if (!predicate) throw new Error(msg); }

// Utilities
function readTextFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

async function parseCategoriesJSONL(txt) {
  const lines = txt.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const seen = new Set();
  const cats = [];
  let sawCialdiniShape = false;
  for (const line of lines) {
    try {
      const obj = JSON.parse(line);
      const name = obj.ss_technique?.toString().trim();
      if (!name || seen.has(name)) continue;
      seen.add(name);
      const def = obj.ss_definition || obj.ss_description || '';
      if (obj.ss_description) sawCialdiniShape = true;
      cats.push({ name, def, example: obj.ss_example || '' });
    } catch (e) {
      console.warn('Skipping invalid JSONL line:', line);
    }
  }
  const sorted = cats.sort((a, b) => a.name.localeCompare(b.name));
  // If the JSONL appears to be Cialdini, mark state later
  state.isCialdini = sawCialdiniShape;
  return sorted;
}

// Attempt to locate an array of chat messages anywhere in the object tree
function findChatArray(obj, depth = 0) {
  if (!obj || depth > 6) return null;
  // If it's already an array of chat-like objects, return it
  if (Array.isArray(obj)) {
    const looksLikeChats = obj.some(
      (el) => el && typeof el === 'object' && ('chat' in el || 'message' in el)
    );
    if (looksLikeChats) return obj;
    // Otherwise, search within each element (e.g., xhr_data root arrays)
    for (const el of obj) {
      const res = findChatArray(el, depth + 1);
      if (res) return res;
    }
    return null;
  }
  if (typeof obj === 'object') {
    // Prefer an explicit 'chats' property if present
    if (Array.isArray(obj.chats)) return obj.chats;
    for (const key of Object.keys(obj)) {
      const res = findChatArray(obj[key], depth + 1);
      if (res) return res;
    }
  }
  return null;
}

function normalizeFileContent(raw) {
  // Unwrap array-wrapped payloads like ["replayGameData", { ... }]
  let root = raw;
  if (Array.isArray(root)) {
    const withChats = root.find((el) => el && typeof el === 'object' && Array.isArray(el.chats));
    root = withChats || root.find((el) => el && typeof el === 'object') || {};
  }
  // Support multiple shapes: prefer top-level chats, else recursively find a chat array
  const chatsArr = Array.isArray(root?.chats) ? root.chats : (findChatArray(root) || []);
  const normalized = [];
  const originals = [];
  for (const item of chatsArr) {
    if (!item || typeof item !== 'object') continue;
    const userName = item.userName ?? item.username ?? item.user ?? 'Unknown';
    const chat = item.chat;
    // Skip non-user messages and claims if indicated by flags
    if (item.gameChat === true) continue;
    if (item.isClaim === true) continue;
    if (typeof chat !== 'string') continue; // skip non-string chats
    const text = `${userName}: ${chat}`;
    normalized.push(text);
    originals.push({ userName, chat });
  }
  // Build roles map from players array if available
  const roles = new Map();
  const players = Array.isArray(root?.players) ? root.players : [];
  for (const p of players) {
    const name = p.username ?? p.userName ?? p.user;
    const role = p.role ?? 'unknown';
    if (name) roles.set(name, role);
  }
  // Also map roles from winningPlayers/losingPlayers if present
  const winners = Array.isArray(root?.winningPlayers) ? root.winningPlayers : [];
  const losers = Array.isArray(root?.losingPlayers) ? root.losingPlayers : [];
  for (const p of [...winners, ...losers]) {
    const name = p.userName ?? p.username ?? p.user;
    const role = p.role ?? p.team ?? 'unknown';
    if (name && !roles.has(name)) roles.set(name, role);
  }
  return { normalized, originals, roles };
}

function sampleIndices(n, max) {
  // pick n evenly spaced indices across 0..max-1 to reduce clustering
  if (n >= max) return Array.from({ length: max }, (_, i) => i);
  const step = (max - 1) / (n - 1);
  return Array.from({ length: n }, (_, i) => Math.floor(i * step));
}

function getContextForIndex(idx, arr, before = 2, after = 2) {
  const start = Math.max(0, idx - before);
  const end = Math.min(arr.length, idx + after + 1);
  const context = [];
  for (let i = start; i < end; i++) {
    context.push({ index: i, text: arr[i], isTarget: i === idx, rel: i < idx ? 'before' : (i === idx ? 'target' : 'after') });
  }
  return context;
}

function renderTechniques(filter = '') {
  const q = filter.trim().toLowerCase();
  const techniques = state.categories.filter(c => !q || c.name.toLowerCase().includes(q));
  const selected = currentSelection();
  els.techniques.innerHTML = '';
  const inline = document.createElement('div');
  inline.className = 'tech-inline-controls';
  // Inline Previous button
  const prev = document.createElement('button');
  prev.type = 'button';
  prev.textContent = 'Previous';
  prev.disabled = state.sampleIndex <= 0;
  prev.addEventListener('click', () => prevSample());
  inline.appendChild(prev);
  const hotkeysActive = state.categories.length < 10; // mirrors keyboard handler condition
  techniques.forEach((t, idx) => {
    const active = selected.has(t.name);
    const btn = document.createElement('button');
    btn.className = 'tech-btn' + (active ? ' active' : '');
    btn.type = 'button';
    btn.title = [t.def, t.example].filter(Boolean).join('\n\nExample: ');
    btn.dataset.name = t.name;
    const label = techniqueLabelWithEmoji(t.name);
    const keycap = hotkeysActive && idx < 6 ? `<span class="keycap">${idx + 1}</span>` : '';
    btn.innerHTML = `${keycap}<span class="name">${escapeHtml(label)}</span>`;
    btn.addEventListener('click', () => toggleTechnique(t.name));
    inline.appendChild(btn);
  });
  const saveNext = document.createElement('button');
  saveNext.className = 'primary';
  saveNext.type = 'button';
  saveNext.textContent = (state.sampleIndex >= state.samples.length - 1) ? 'Save & Finish' : 'Save & Next';
  saveNext.addEventListener('click', () => {
    if (state.sampleIndex >= state.samples.length - 1) {
      saveAllPendingAndFinish();
    } else {
      nextSample();
    }
  });
  inline.appendChild(saveNext);
  els.techniques.appendChild(inline);
}

function isLastSample() {
  return state.sampleIndex >= state.samples.length - 1;
}

// Keyboard shortcuts: if techniques < 10, map 1..6 to first 6, Enter -> next
document.addEventListener('keydown', (e) => {
  // Avoid when typing in inputs/textareas or when setup is visible
  const tag = (document.activeElement?.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea') return;
  const annotatorHidden = document.getElementById('annotator')?.classList.contains('hidden');
  if (annotatorHidden) return;

  const totalTechs = state.categories.length;
  if (totalTechs >= 10) {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (isLastSample()) { saveAllPendingAndFinish(); } else { nextSample(); }
    }
    return;
  }

  // Hotkeys 1..6
  if (/^[1-6]$/.test(e.key)) {
    const idx = parseInt(e.key, 10) - 1;
    const q = (document.getElementById('techSearch')?.value || '').trim().toLowerCase();
    const techniques = state.categories.filter(c => !q || c.name.toLowerCase().includes(q));
    if (idx < techniques.length) {
      e.preventDefault();
      toggleTechnique(techniques[idx].name);
    }
    return;
  }

  // Enter -> Save & Next
  if (e.key === 'Enter') {
    e.preventDefault();
    if (isLastSample()) { saveAllPendingAndFinish(); } else { nextSample(); }
  }
});

function techniqueLabelWithEmoji(name) {
  if (!state.isCialdini) return name;
  // Hardcoded emoji mapping for supplied Cialdini set
  const map = new Map([
    ['Reciprocation', '🤝 Reciprocation'],
    ['Social Validation', '👥 Social Validation'],
    ['Consistency', '📏 Consistency'],
    ['Friendship/Liking', '💖 Friendship/Liking'],
    ['Scarcity', '⏳ Scarcity'],
    ['Authority', '🏛️ Authority'],
  ]);
  return map.get(name) || name;
}

function renderContext() {
  // Ensure current file pointers reflect the sample's file
  const { fileIndex, messageIndex } = state.samples[state.sampleIndex];
  const file = state.files[fileIndex];
  state.chats = file.chats;
  state.rawChats = file.rawChats;
  state.roles = file.roles;

  const ctx = getContextForIndex(messageIndex, state.chats, 2, 2);
  els.context.innerHTML = '';
  for (const c of ctx) {
    const div = document.createElement('div');
    div.className = `msg ${c.isTarget ? 'target' : (c.rel === 'before' ? 'context-before' : 'context-after')}`;
    const [who, ...rest] = c.text.split(': ');
    const text = rest.join(': ');
    const role = (state.roles.get(who) || 'unknown').toLowerCase();
    const roleClass = role === 'liberal' ? 'liberal' : (role === 'fascist' ? 'fascist' : (role === 'hitler' ? 'hitler' : ''));
    let annBadgesHtml = '';
    if (!c.isTarget && c.rel === 'before') {
      const key = `${fileIndex}:${c.index}`;
      const ann = Array.from(state.selections.get(key) || []);
      if (ann.length) {
        const badges = ann.map(a => `<span class="badge"><span class="dot"></span>${escapeHtml(a)}</span>`).join(' ');
        annBadgesHtml = `<div class="tags">${badges}</div>`;
      }
    }
    const roleBadgeHtml = roleClass ? `<span class="tags"><span class="badge ${roleClass}"><span class="dot"></span>${escapeHtml(role)}</span></span>` : '';
    div.innerHTML = `
      <div class="who">#${c.index} • <strong>${escapeHtml(who)}</strong> ${roleBadgeHtml}</div>
      <div class="text">${escapeHtml(text)}</div>
      ${annBadgesHtml}
    `;
    els.context.appendChild(div);
  }
}

function updateProgressAndControls() {
  const total = state.samples.length;
  const pos = state.sampleIndex + 1;
  const current = state.samples[state.sampleIndex];
  const fname = state.files[current.fileIndex]?.name || '';
  // Compute file counters
  const totalFiles = state.files.length;
  const currentFileIndex = current.fileIndex + 1;
  const fileSamplesTotal = state.files[current.fileIndex]?.chats.length || 0;
  // Find how many samples within this file up to current sample
  let currentFileSampleIndex = 0;
  for (let i = 0; i <= state.sampleIndex; i++) {
    if (state.samples[i].fileIndex === current.fileIndex) currentFileSampleIndex++;
  }
  els.progress.textContent = `Sample ${pos} / ${total} • File ${currentFileIndex}/${totalFiles} • In file ${currentFileSampleIndex}/${fileSamplesTotal} • ${fname}`;
  els.prevBtn.disabled = state.sampleIndex <= 0;
  els.nextBtn.disabled = false;
  els.finishBtn.disabled = false;
}

function keyForCurrent() {
  const { fileIndex, messageIndex } = state.samples[state.sampleIndex];
  return `${fileIndex}:${messageIndex}`;
}

function toggleTechnique(name) {
  const key = keyForCurrent();
  if (!state.selections.has(key)) state.selections.set(key, new Set());
  const set = state.selections.get(key);
  if (set.has(name)) set.delete(name); else set.add(name);
  renderTechniques(els.techSearch.value);
}

function currentSelection() {
  const key = keyForCurrent();
  if (!state.selections.has(key)) state.selections.set(key, new Set());
  return state.selections.get(key);
}

function noneSelection() {
  const key = keyForCurrent();
  state.selections.set(key, new Set());
  renderTechniques(els.techSearch.value);
}

function nextSample() {
  if (state.sampleIndex < state.samples.length - 1) {
    // Save file when moving from last message of a file to the first of the next
    const cur = state.samples[state.sampleIndex];
    const nxt = state.samples[state.sampleIndex + 1];
    const willCross = cur.fileIndex !== nxt.fileIndex;
    const curFileIdx = cur.fileIndex;
    state.sampleIndex++;
    renderContext();
    renderTechniques(els.techSearch.value);
    updateProgressAndControls();
    if (willCross) {
      saveFileAnnotations(curFileIdx);
    }
  }
}
function prevSample() {
  if (state.sampleIndex > 0) {
    state.sampleIndex--;
    renderContext();
    renderTechniques(els.techSearch.value);
    updateProgressAndControls();
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function buildResults() {
  const out = [];
  for (const sample of state.samples) {
    const file = state.files[sample.fileIndex];
    const text = file.chats[sample.messageIndex];
    const key = `${sample.fileIndex}:${sample.messageIndex}`;
    const selection = Array.from(state.selections.get(key) || []);
    out.push({ file: file.name, text, annotation: selection });
  }
  return out;
}

function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || 'annotations.json';
  document.body.appendChild(a);
  a.click();
  URL.revokeObjectURL(url);
  a.remove();
}

// Build per-file annotations for a given file index (omit filename field)
function buildResultsForFile(fileIndex) {
  const file = state.files[fileIndex];
  const out = [];
  if (!file) return out;
  for (let mi = 0; mi < file.chats.length; mi++) {
    const key = `${fileIndex}:${mi}`;
    const selection = Array.from(state.selections.get(key) || []);
    out.push({ text: file.chats[mi], annotation: selection });
  }
  return out;
}

function filenameWithSuffix(originalName, suffix) {
  const dot = originalName.lastIndexOf('.');
  if (dot <= 0) return `${originalName}${suffix}.json`;
  const base = originalName.slice(0, dot);
  return `${base}${suffix}.json`;
}

function saveFileAnnotations(fileIndex) {
  if (state.savedFiles.has(fileIndex)) return; // avoid duplicate downloads
  const suffix = (els.outputSuffix?.value || '-chat-annotated').trim() || '-chat-annotated';
  const file = state.files[fileIndex];
  if (!file) return;
  const data = buildResultsForFile(fileIndex);
  const outName = filenameWithSuffix(file.name, suffix);
  downloadJson(outName, data);
  state.savedFiles.add(fileIndex);
}

function saveAllPendingAndFinish() {
  if (state.samples.length === 0) return;
  for (let fi = 0; fi < state.files.length; fi++) {
    saveFileAnnotations(fi);
  }
}

els.setupForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    // Folder selection yields multiple File entries
    assert(els.inputFile.files.length >= 1, 'Please choose a folder containing JSON files.');
  assert(els.categoriesFile.files.length === 1, 'Please choose exactly one categories JSONL file.');

    // Collect .json files, then shuffle deterministically using provided seed
    const fileList = Array.from(els.inputFile.files)
      .filter(f => f.name.toLowerCase().endsWith('.json'));
    assert(fileList.length > 0, 'No .json files found in the selected folder.');

    // Seeded RNG (Mulberry32)
    function xmur3(str){let h=1779033703^str.length;for(let i=0;i<str.length;i++){h=Math.imul(h^str.charCodeAt(i),3432918353);h=h<<13|h>>>19;}return function(){h=Math.imul(h^h>>>16,2246822507);h=Math.imul(h^h>>>13,3266489909);return (h^h>>>16)>>>0;}};
    function mulberry32(a){return function(){let t=a+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296;}}
    const seedStr = (els.seedInput?.value ?? '0') + '';
    const seed = xmur3(seedStr)();
    const rand = mulberry32(seed);
    // Fisher-Yates shuffle with rand()
    const shuffled = fileList.slice();
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }

    const nFiles = Math.max(1, Math.min(Number(els.sampleCount.value || '1'), shuffled.length));
    const chosen = shuffled.slice(0, nFiles);

    // Read and parse the chosen files
    const fileTexts = await Promise.all(chosen.map(readTextFile));
    state.files = [];
    for (let i = 0; i < chosen.length; i++) {
      try {
        const raw = JSON.parse(fileTexts[i]);
        const { normalized, originals, roles } = normalizeFileContent(raw);
        if (normalized.length === 0) continue;
        state.files.push({ name: chosen[i].name, content: raw, chats: normalized, rawChats: originals, roles });
      } catch (err) {
        console.warn('Skipping invalid JSON:', chosen[i].name);
      }
    }
    assert(state.files.length > 0, 'No valid chat data found in selected files.');

    const catTxt = await readTextFile(els.categoriesFile.files[0]);
    const cats = await parseCategoriesJSONL(catTxt);
    assert(cats.length > 0, 'No valid categories found in JSONL. Expect lines with { ss_technique, ss_definition?, ss_example? }');

    state.categories = cats;

    // Build per-message samples across all selected files, in file order then message order
    const samples = [];
    for (let fi = 0; fi < state.files.length; fi++) {
      const file = state.files[fi];
      for (let mi = 0; mi < file.chats.length; mi++) {
        samples.push({ fileIndex: fi, messageIndex: mi });
      }
    }
    state.samples = samples;
    state.sampleIndex = 0;
    state.selections = new Map();
    state.sampleLimit = samples.length;
  state.savedFiles = new Set();

    document.getElementById('setup').classList.add('hidden');
    els.annotator.classList.remove('hidden');

    renderTechniques();
    renderContext();
    updateProgressAndControls();
  } catch (err) {
    alert(err.message || String(err));
  }
});

els.noneBtn.addEventListener('click', () => noneSelection());
els.techSearch.addEventListener('input', () => renderTechniques(els.techSearch.value));
els.nextBtn.addEventListener('click', () => nextSample());
els.prevBtn.addEventListener('click', () => prevSample());

els.finishBtn.addEventListener('click', () => {
  saveAllPendingAndFinish();
});
