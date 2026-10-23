// Settings saved by earlier versions used a different key prefix.
["RelaySettings", "RelayAuxSelected", "RelayPttKey", "RelayOutputDevice", "TrainingDataset", "TrainingModelName"].forEach(key => {
  try {
    const old = localStorage.getItem("workingJoe" + key);
    if (old !== null && localStorage.getItem("voiceDashboard" + key) === null) localStorage.setItem("voiceDashboard" + key, old);
  } catch (_) {}
});
// Settings of the removed voice changer page.
try { ["voiceDashboardRvcDataset", "voiceDashboardRvcModelName", "voiceDashboardGuide"].forEach(key => localStorage.removeItem(key)); }
catch (_) {}
let state = null;
let voiceDataset = "";
try { voiceDataset = localStorage.getItem("voiceDashboardVoiceDataset") || ""; }
catch (_) {}
let auxSelected = new Set();
let relayAuxSelected = new Set();
try { relayAuxSelected = new Set(JSON.parse(localStorage.getItem("voiceDashboardRelayAuxSelected") || "[]")); }
catch (_) { relayAuxSelected = new Set(); }
let refsSignature = "";
let modelsSignature = "";
const $ = id => document.getElementById(id);

// The dashboard's own audio player in place of the browser's: a play button, a position bar
// and the time. Hiding an audio element hides its player too.
const PLAY_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
const PAUSE_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>';
function clockTime(seconds) {
  if (!Number.isFinite(seconds)) return "0:00";
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}
function attachPlayer(audio) {
  const label = audio.dataset.label || "audio";
  const bar = document.createElement("div");
  bar.className = "player";
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "player-toggle";
  const seek = document.createElement("input");
  seek.type = "range";
  seek.className = "player-seek";
  seek.min = "0";
  seek.max = "1000";
  seek.value = "0";
  seek.setAttribute("aria-label", `Position in the ${label}`);
  const time = document.createElement("span");
  time.className = "player-time";
  bar.append(toggle, seek, time);
  audio.after(bar);
  let seeking = false;
  const update = () => {
    const loaded = Boolean(audio.currentSrc || audio.getAttribute("src"));
    toggle.disabled = !loaded;
    toggle.innerHTML = audio.paused ? PLAY_ICON : PAUSE_ICON;
    toggle.setAttribute("aria-label", `${audio.paused ? "Play" : "Pause"} the ${label}`);
    seek.disabled = !loaded || !Number.isFinite(audio.duration);
    if (!seeking) seek.value = audio.duration ? String(Math.round(audio.currentTime / audio.duration * 1000)) : "0";
    time.textContent = `${clockTime(audio.currentTime)} / ${clockTime(audio.duration)}`;
  };
  toggle.onclick = () => { if (audio.paused) audio.play().catch(() => {}); else audio.pause(); };
  seek.addEventListener("input", () => {
    seeking = true;
    if (Number.isFinite(audio.duration)) audio.currentTime = seek.value / 1000 * audio.duration;
  });
  seek.addEventListener("change", () => { seeking = false; });
  ["play", "pause", "ended", "timeupdate", "loadedmetadata", "durationchange", "emptied", "loadstart"].forEach(name => audio.addEventListener(name, update));
  update();
  // Mic to Voice's output sets its volume in Settings, so its player has no volume button.
  if (!audio.hasAttribute("data-settings-volume")) addVolume(audio, bar, label);
}

// One volume for every player with a volume button, remembered between visits.
const VOLUME_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M11 5 6 9H2v6h4l5 4z" fill="currentColor" stroke="none"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M18.5 5.5a9 9 0 0 1 0 13"/></svg>';
const MUTED_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M11 5 6 9H2v6h4l5 4z" fill="currentColor" stroke="none"/><path d="m22 9-6 6M16 9l6 6"/></svg>';
const volumeControls = [];
let playerVolume = 1;
try {
  const saved = Number(localStorage.getItem("voiceDashboardPlayerVolume") ?? 1);
  if (Number.isFinite(saved)) playerVolume = Math.max(0, Math.min(1, saved));
} catch (_) {}

function setPlayerVolume(volume) {
  playerVolume = volume;
  volumeControls.forEach(control => {
    control.audio.volume = volume;
    control.range.value = String(Math.round(volume * 100));
    control.value.textContent = `${Math.round(volume * 100)}%`;
    control.button.innerHTML = volume ? VOLUME_ICON : MUTED_ICON;
  });
  try { localStorage.setItem("voiceDashboardPlayerVolume", String(volume)); } catch (_) {}
}

function addVolume(audio, bar, label) {
  bar.classList.add("has-volume");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "player-volume";
  button.setAttribute("aria-label", `Volume of the ${label}`);
  button.setAttribute("aria-expanded", "false");
  const panel = document.createElement("div");
  panel.className = "player-volume-panel";
  panel.hidden = true;
  const range = document.createElement("input");
  range.type = "range";
  range.min = "0";
  range.max = "100";
  range.setAttribute("aria-label", `Volume of the ${label}`);
  const value = document.createElement("span");
  value.className = "player-volume-value";
  panel.append(range, value);
  bar.append(button, panel);
  volumeControls.push({audio, range, value, button});
  const open = show => {
    panel.hidden = !show;
    button.setAttribute("aria-expanded", show ? "true" : "false");
  };
  button.onclick = () => {
    open(panel.hidden);
    if (!panel.hidden) range.focus();
  };
  range.addEventListener("input", () => setPlayerVolume(Number(range.value) / 100));
  // Esc closes the slider, not the window it is in.
  panel.addEventListener("keydown", event => {
    if (event.key !== "Escape") return;
    event.preventDefault();
    event.stopPropagation();
    open(false);
    button.focus();
  });
  document.addEventListener("pointerdown", event => {
    if (!panel.hidden && !bar.contains(event.target)) open(false);
  });
  setPlayerVolume(playerVolume);
}
document.querySelectorAll("audio").forEach(attachPlayer);
let relayListening = false;
let relayQueue = [];
let relayBusy = false;
let relayLastUrl = "";
let relayNextId = 1;
let relayStream = null;
let relayAudioCtx = null;
let relaySource = null;
let relayProcessor = null;
let relayKeepAlive = null;
let relayPushHeld = false;
// Counts the starts of listening, so a start that Stop Everything cut short knows it.
let relayStartCount = 0;
let relayBindingPtt = false;
let relayCapturing = false;
let relayCurrentChunks = [];
let relayCurrentFrames = 0;
let relayLastVoiceAt = 0;
let relayLastLevelDb = -120;
let relayCurrentItem = null;
let relayPlaybackResolve = null;
let armRelayWhenReady = false;
// Mic to Voice's services start only when Start is pressed.
let systemStartBusy = false;
// Stop Everything raises it, so a start that is still waiting, for example for the microphone
// prompt, does not go on.
let startToken = 0;
let lastGpuBusy = false;
// A dataset created on this page, kept selected until a dataset list includes it: a list that
// was requested before the dataset existed can arrive after it.
let newTrainingDataset = null;
let welcomeChecked = false;
let lastSystemMessage = "";
let datasetUploadFiles = [];
// Transcripts of the files waiting to be added, typed or written by speech recognition, so a
// redraw of the list keeps them. A file without one is still being transcribed.
const datasetUploadTexts = new WeakMap();
const datasetUploadInputs = new WeakMap();
const transcriptQueue = [];
let transcriptBusy = false;
// Relay workspace view: "capture" shows the recording control alone; "result" shows one queue item.
let relayView = {mode: "capture", itemId: null};
let relayTranscribing = false;
let relayStatusText = "";
let relayLoadedUrl = "";
let relayLevelTick = 0;
let lastSystemSnapshot = null;
let lastSystemStatus = "";
let pttBindingCancelledAt = 0;
let engineBusy = false;
let logRaw = "";
let generateOutputs = [];
// The generated line shown in the player.
let generateShown = null;
let textLangDataset = "";
let configErrorShown = false;
let relaySavedSettings = {};
try { relaySavedSettings = JSON.parse(localStorage.getItem("voiceDashboardRelaySettings") || "{}"); }
catch (_) { relaySavedSettings = {}; }

const RELAY_SETTING_IDS = [
  "relayGptModel", "relaySovitsModel", "relayReference",
  "relayOutputDevice", "relayPttKey", "relayMonitorLocal", "relayPushToTalk",
  "relayAutoSpeak", "relayVolume", "relayTopK", "relayTopP",
  "relayTemperature", "relaySpeed", "relayPause", "relayRepPenalty",
  "relaySpoken", "relaySpeaks"
];

// The language you speak to Mic to Voice, and the one its voice speaks.
function relayLanguages() {
  return {spoken: $("relaySpoken").value, speaks: $("relaySpeaks").value};
}

// Chinese and Cantonese are both written in Chinese characters, so no translation is needed
// between them; see written in dashboard/asr.py.
function relayTranslates() {
  const written = language => language === "yue" ? "zh" : language;
  const {spoken, speaks} = relayLanguages();
  return written(spoken) !== written(speaks);
}

function renderRelayLanguageNote() {
  const notes = [];
  if ($("relaySpoken").value !== "en") {
    notes.push("The first time, speech recognition downloads a model for languages other than English, about 145 MB with the standard settings.");
  }
  if (relayTranslates()) {
    notes.push("Your words are translated by NLLB-200, which downloads about 650 MB the first time. Its license allows only non-commercial use.");
  }
  $("relayLanguageNote").textContent = notes.join(" ");
  $("relayLanguageNote").hidden = !notes.length;
}

function recentActivity(message) {
  if (message) $("recentActivity").textContent = message;
}

function showError(message) {
  $("globalErrorText").textContent = message || "An unknown error occurred.";
  $("globalError").classList.add("visible");
  recentActivity(`Failed: ${message}`);
}

window.alert = showError;
$("dismissError").onclick = () => $("globalError").classList.remove("visible");
$("openErrorLogs").onclick = () => activateTab("logs");

function saveRelaySettings() {
  const saved = {};
  RELAY_SETTING_IDS.forEach(id => {
    const element = $(id);
    if (!element) return;
    if (element.tagName === "SELECT" && !element.options.length && relaySavedSettings[id] !== undefined) {
      saved[id] = relaySavedSettings[id];
    } else {
      saved[id] = element.type === "checkbox" ? element.checked : element.value;
    }
  });
  relaySavedSettings = saved;
  localStorage.setItem("voiceDashboardRelaySettings", JSON.stringify(saved));
}

function loadRelaySettings() {
  RELAY_SETTING_IDS.forEach(id => {
    const element = $(id);
    if (!element || relaySavedSettings[id] === undefined) return;
    if (element.type === "checkbox") element.checked = Boolean(relaySavedSettings[id]);
    else if (element.tagName !== "SELECT" || Array.from(element.options).some(option => option.value === relaySavedSettings[id])) element.value = relaySavedSettings[id];
  });
}

function componentCard(cardId, statusId, value) {
  const card = $(cardId);
  const label = $(statusId);
  card.classList.remove("ready", "starting", "failed");
  if (value === "ready") {
    card.classList.add("ready");
    label.textContent = "Ready";
  } else if (value === "starting") {
    card.classList.add("starting");
    label.textContent = "Starting";
  } else if (value === "failed") {
    card.classList.add("failed");
    label.textContent = "Failed";
  } else {
    label.textContent = "Off";
  }
}

function modelSlug(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64) || "voice-model";
}

function suggestedTranscript(filename) {
  const stem = String(filename || "").replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
  if (!stem) return "";
  const text = stem[0].toUpperCase() + stem.slice(1);
  return /[.!?]$/.test(text) ? text : text + ".";
}

function datasetLabel(dataset) {
  return `${dataset.name} — ${dataset.wav_count} ${dataset.wav_count === 1 ? "clip" : "clips"}`;
}

function fillDatasetSelects(datasets, activeId, requestedId) {
  // The server answers with the voice it was asked for, or the first one when that is gone. An
  // answer to an older request, from before the voice was changed, does not undo the change.
  const chosen = voiceDataset === requestedId ? activeId : voiceDataset;
  ["voiceDataset", "relayDataset"].forEach(id => {
    const select = $(id);
    select.innerHTML = "";
    datasets.forEach(dataset => option(select, dataset.id, datasetLabel(dataset)));
    if (datasets.some(dataset => dataset.id === chosen)) select.value = chosen;
  });
  if (chosen !== voiceDataset) {
    voiceDataset = chosen;
    try { localStorage.setItem("voiceDashboardVoiceDataset", voiceDataset); } catch (_) {}
  }
}

function fillTrainingDatasets(datasets) {
  const select = $("trainingDataset");
  const old = (newTrainingDataset && newTrainingDataset.id) || select.value || localStorage.getItem("voiceDashboardTrainingDataset") || "";
  select.innerHTML = "";
  (datasets || []).forEach(dataset => option(select, dataset.id, datasetLabel(dataset)));
  if (newTrainingDataset && (datasets || []).some(dataset => dataset.id === newTrainingDataset.id)) newTrainingDataset = null;
  else if (newTrainingDataset) option(select, newTrainingDataset.id, newTrainingDataset.name);
  if (Array.from(select.options).some(item => item.value === old)) select.value = old;
  const selected = (datasets || []).find(dataset => dataset.id === select.value);
  if (selected) {
    $("trainingDatasetStatus").textContent = selected.ready ? "Ready" : (selected.wav_count || selected.list_rows ? "Clips and transcripts do not match" : "No clips yet");
    $("trainingDatasetStatus").className = selected.ready ? "status-line good" : "status-line warn";
    updateUploadButton();
  }
  renderDatasetList(datasets || [], select.value);
  loadClips();
}

// The clips already in the selected dataset: listen, correct a transcript, or delete a clip.
// Rows are kept while the list grows, so typing in one is not interrupted.
let clipsSignature = "";
let clipsDataset = "";
let clipsData = {clips: [], seconds: 0};
const clipRows = new Map();
const clipAudio = new Audio();
let clipPlaying = "";
const STOP_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><rect x="6" y="6" width="12" height="12"/></svg>';

function formatLength(seconds) {
  if (seconds === null || seconds === undefined) return "unreadable";
  if (seconds < 60) return `${Number(seconds).toFixed(1)} s`;
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)} min ${whole % 60} s`;
}

// The server says which clips training skips, and why; see training_skips in dashboard/datasets.py.
function clipProblem(clip) {
  if (clip.duration === null || clip.duration === undefined) return "The file cannot be read.";
  return clip.skipped || "";
}

async function loadClips(force = false) {
  const dataset = ((state && state.datasets) || []).find(row => row.id === $("trainingDataset").value);
  const signature = dataset ? `${dataset.id}|${dataset.wav_count}|${dataset.list_rows}` : "";
  if (!force && signature === clipsSignature) return;
  clipsSignature = signature;
  if (!dataset) {
    clipsData = {clips: [], seconds: 0};
    renderClips("");
    return;
  }
  // Another dataset's clips, and the references made from them, go at once, not when the new list arrives.
  if (dataset.id !== clipsDataset) {
    clipsData = {clips: [], seconds: 0};
    renderClips(dataset.id);
  }
  try {
    const data = await (await fetch(`/api/dataset-clips?dataset=${encodeURIComponent(dataset.id)}`)).json();
    if (data.ok && $("trainingDataset").value === dataset.id) {
      clipsData = data;
      renderClips(dataset.id);
    }
  } catch (_) {
    clipsSignature = "";
  }
}

// The saved epochs of the model named in step 2, each used to say the same test line.
function savedEpochs(kind) {
  const name = modelSlug($("trainingModelName").value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(kind === "gpt" ? `^${name}-e(\\d+)\\.ckpt$` : `^${name}_e(\\d+)_s\\d+\\.pth$`);
  const version = $("sovitsVersion").value;
  const epochs = ((state && state.models && state.models[kind]) || [])
    .filter(row => kind === "gpt" || row.version === version)
    .map(row => pattern.exec(row.name)).filter(Boolean).map(match => match[1]);
  return new Set(epochs).size;
}

function sovitsLabel() {
  return $("sovitsVersion").value === "v2Pro" ? "SoVITS" : `${$("sovitsVersion").value} SoVITS`;
}

let compareRowsSignature = "";
let compareChosen = "";
function renderCompare() {
  const kind = $("compareKind").value;
  const count = savedEpochs(kind);
  const job = state && state.jobs && state.jobs.compare;
  const running = Boolean(job && job.status === "running");
  // Models are listed once WSL runs; until then the comparison itself finds the epochs.
  const listed = Boolean(state && state.models_listed);
  $("compareStop").hidden = !running;
  $("compareRun").disabled = running || (listed && !count) || !$("compareReference").value;
  $("compareInfo").textContent = !$("compareReference").options.length
    ? "Compare needs a clip of 3 to 10 seconds in this dataset, as the reference."
    : !listed
    ? "Generates one line with each saved epoch, so you can choose the best by ear."
    : (count
      ? `Generates one line with ${count === 1 ? "the one" : `each of the ${count}`} saved ${kind === "gpt" ? "GPT" : sovitsLabel()} epoch${count === 1 ? "" : "s"} of ${modelSlug($("trainingModelName").value)}, so you can choose the best by ear.`
      : `${modelSlug($("trainingModelName").value)} has no saved ${kind === "gpt" ? "GPT" : sovitsLabel()} epochs yet. Train it first.`);
  if (!job) return;
  const last = (job.lines || []).slice(-1)[0] || "Starting...";
  $("compareStatus").textContent = running ? last : (job.status === "failed" ? `${last} See Logs for details.` : (job.status === "stopped" ? "Stopped." : (compareChosen || "Done.")));
  $("compareStatus").className = job.status === "failed" ? "status-line bad" : (!running && compareChosen ? "status-line good" : "status-line");
  const results = job.results || [];
  const signature = results.map(row => row.url).join("|");
  if (signature === compareRowsSignature) return;
  compareRowsSignature = signature;
  const list = $("compareResults");
  list.innerHTML = "";
  results.forEach(row => {
    const line = document.createElement("div");
    line.className = "compare-row";
    const label = document.createElement("span");
    label.textContent = `${row.kind === "gpt" ? "GPT" : "SoVITS"} epoch ${row.epoch}`;
    const audio = document.createElement("audio");
    audio.src = row.url;
    audio.dataset.label = label.textContent;
    const use = document.createElement("button");
    use.className = "secondary";
    use.textContent = "Use this";
    use.onclick = () => {
      // Text to Voice and Mic to Voice speak with this epoch from now on.
      [["gptModel", row.gpt], ["sovitsModel", row.sovits], ["relayGptModel", row.gpt], ["relaySovitsModel", row.sovits]].forEach(([id, path]) => {
        if (Array.from($(id).options).some(option => option.value === path)) $(id).value = path;
      });
      saveRelaySettings();
      compareChosen = `Text to Voice and Mic to Voice now use ${label.textContent}.`;
      $("compareStatus").className = "status-line good";
      $("compareStatus").textContent = compareChosen;
    };
    line.append(label, audio, use);
    list.appendChild(line);
    attachPlayer(audio);
  });
}

function fillCompareReferences() {
  // Clips that can be a reference: 3 to 10 seconds long.
  const select = $("compareReference");
  const old = select.value;
  select.innerHTML = "";
  (clipsData.clips || []).filter(clip => clip.duration >= 3 && clip.duration <= 10).forEach(clip => option(select, clip.wav, clip.text));
  if (Array.from(select.options).some(item => item.value === old)) select.value = old;
  renderCompare();
}

// One player for the clip list and the files waiting to be added. A row's button plays its
// audio; pressing it again, or another row's, stops it.
let clipButton = null;
let clipObjectUrl = "";
function stopClip() {
  clipAudio.pause();
  clipAudio.removeAttribute("src");
  clipAudio.load();
  if (clipButton) clipButton.innerHTML = PLAY_ICON;
  if (clipObjectUrl) URL.revokeObjectURL(clipObjectUrl);
  clipButton = null;
  clipObjectUrl = "";
  clipPlaying = "";
}
clipAudio.onended = stopClip;

function playRowAudio(key, button, source) {
  const playing = clipPlaying === key;
  stopClip();
  if (playing) return;
  clipPlaying = key;
  clipButton = button;
  button.innerHTML = STOP_ICON;
  // A file not yet added plays from the computer.
  if (source instanceof Blob) clipObjectUrl = URL.createObjectURL(source);
  clipAudio.src = clipObjectUrl || source;
  clipAudio.volume = playerVolume;
  clipAudio.play().catch(() => { if (clipPlaying === key) stopClip(); });
}

function rowPlayButton(label) {
  const play = document.createElement("button");
  play.type = "button";
  play.className = "secondary icon-button";
  play.innerHTML = PLAY_ICON;
  play.setAttribute("aria-label", `Play ${label}`);
  return play;
}

function clipRow(clip) {
  const row = document.createElement("div");
  row.className = "clip-row";
  const play = rowPlayButton(clip.wav);
  play.onclick = () => playRowAudio(clip.wav, play, clip.url);
  const text = document.createElement("input");
  text.value = clip.text;
  text.setAttribute("aria-label", `Transcript of ${clip.wav}`);
  text.title = clip.wav;
  let saved = clip.text;
  text.onchange = async () => {
    try {
      saved = (await post("/api/dataset-clip-text", {dataset_id: clipsDataset, wav: clip.wav, text: text.value})).text;
      text.value = saved;
      checkCorrected.add(checkKey(clip.wav));
      renderClipNotes();
    } catch (error) {
      text.value = saved;
      showError(error.message);
    }
  };
  const length = document.createElement("span");
  length.className = "clip-length";
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "secondary icon-button clip-delete";
  remove.title = "Delete this clip";
  remove.setAttribute("aria-label", `Delete ${clip.wav}`);
  remove.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';
  remove.onclick = async () => {
    if (clipPlaying === clip.wav) stopClip();
    const next = row.nextElementSibling || row.previousElementSibling;
    try {
      await post("/api/dataset-clip-delete", {dataset_id: clipsDataset, wav: clip.wav});
      // A clip added later under this name is not the one the check heard.
      checkGone.add(checkKey(clip.wav));
      await refresh();
      await loadClips(true);
      if (next && next.isConnected) next.querySelector(".clip-delete").focus();
    } catch (error) {
      showError(error.message);
    }
  };
  const note = document.createElement("div");
  note.className = "clip-note";
  note.hidden = true;
  row.append(play, text, length, remove, note);
  return {row, play, length, note, text, setSaved: value => { saved = value; }};
}

// What the last check of this dataset found, by clip. A transcript corrected since is not
// flagged again.
let checkFound = new Map();
let checkSignature = "";
const checkCorrected = new Set();
const checkGone = new Set();
const checkKey = wav => `${clipsDataset}/${wav}`;
function renderCheck() {
  const job = state && state.jobs && state.jobs["dataset-check"];
  const mine = Boolean(job && job.dataset === clipsDataset);
  const running = Boolean(mine && job.status === "running");
  $("checkClips").disabled = running;
  $("checkClipsStop").hidden = !running;
  const results = mine ? job.results || [] : [];
  const signature = (mine ? job.dataset : "") + "|" + results.map(row => `${row.wav}:${row.problems.length}`).join("|");
  if (signature !== checkSignature) {
    checkSignature = signature;
    checkFound = new Map(results.map(row => [row.wav, row]));
    renderClipNotes();
  }
  if (!mine) {
    $("checkStatus").textContent = "";
    return;
  }
  const last = (job.lines || []).slice(-1)[0] || "Starting...";
  $("checkStatus").textContent = job.status === "stopped" ? `Check stopped. ${checkFound.size} clip${checkFound.size === 1 ? "" : "s"} found so far.`
    : (job.status === "failed" ? `${last} See Logs for details.` : last);
  $("checkStatus").className = job.status === "failed" ? "status-line bad" : "status-line";
}

function clipProblems(wav) {
  const found = checkFound.get(wav);
  if (!found || checkGone.has(checkKey(wav))) return [];
  return found.problems.filter(problem => !(checkCorrected.has(checkKey(wav)) && problem.startsWith("The transcript differs")));
}

function renderClipNotes() {
  // With nothing left to look at, as in another dataset or once all are corrected, the filter
  // turns off rather than hide every clip.
  const count = Array.from(clipRows.keys()).filter(wav => clipProblems(wav).length).length;
  if (!count) $("checkFilter").checked = false;
  $("checkFilterLabel").hidden = !count;
  clipRows.forEach((parts, wav) => {
    const problems = clipProblems(wav);
    const found = checkFound.get(wav);
    parts.note.innerHTML = "";
    parts.note.hidden = !problems.length;
    if (problems.length) {
      const text = document.createElement("span");
      text.textContent = problems.join(" ");
      parts.note.append(text);
      if (found.heard !== null && !checkCorrected.has(checkKey(wav))) {
        const heard = document.createElement("span");
        heard.className = "heard";
        heard.textContent = found.heard ? `Heard: "${found.heard}"` : "Heard no words.";
        parts.note.append(heard);
        if (found.heard) {
          const use = document.createElement("button");
          use.type = "button";
          use.className = "secondary";
          use.textContent = "Use heard text";
          use.onclick = async () => {
            try {
              const saved = (await post("/api/dataset-clip-text", {dataset_id: clipsDataset, wav, text: found.heard})).text;
              parts.text.value = saved;
              parts.setSaved(saved);
              checkCorrected.add(checkKey(wav));
              renderClipNotes();
            } catch (error) { showError(error.message); }
          };
          parts.note.append(use);
        }
      }
    }
    parts.row.hidden = $("checkFilter").checked && !problems.length;
  });
}
$("checkFilter").onchange = renderClipNotes;

$("checkClips").onclick = async () => {
  $("checkClips").disabled = true;
  try {
    await post("/api/dataset-check", {dataset_id: clipsDataset});
    checkCorrected.clear();
    checkGone.clear();
    $("checkStatus").className = "status-line";
    $("checkStatus").textContent = "Starting...";
    refresh();
  } catch (error) {
    showError(error.message);
    $("checkClips").disabled = false;
  }
};

function renderClips(datasetId) {
  if (datasetId !== clipsDataset) {
    stopClip();
    clipRows.clear();
    $("clipList").innerHTML = "";
    clipsDataset = datasetId;
  }
  const clips = clipsData.clips || [];
  const names = new Set(clips.map(clip => clip.wav));
  clipRows.forEach((parts, wav) => {
    if (names.has(wav)) return;
    if (clipPlaying === wav) stopClip();
    parts.row.remove();
    clipRows.delete(wav);
  });
  let problems = 0;
  clips.forEach(clip => {
    if (!clipRows.has(clip.wav)) {
      const parts = clipRow(clip);
      clipRows.set(clip.wav, parts);
      $("clipList").appendChild(parts.row);
    }
    const parts = clipRows.get(clip.wav);
    const problem = clipProblem(clip);
    if (problem) problems++;
    parts.length.textContent = formatLength(clip.duration);
    parts.length.title = problem;
    parts.row.classList.toggle("warn", Boolean(problem));
  });
  const summary = $("clipSummary");
  summary.hidden = !clips.length;
  $("checkRow").hidden = !clips.length;
  const short = clips.filter(clip => clip.duration !== null && clip.duration < 2).length;
  const long = clips.filter(clip => clip.duration > 10).length;
  const middle = clips.filter(clip => clip.duration >= 2 && clip.duration <= 10).length;
  summary.textContent = `${clips.length} clip${clips.length === 1 ? "" : "s"}, ${formatLength(clipsData.seconds)} in total`
    + (short || long ? `: ${short} under 2 s, ${middle} of 2 to 10 s, ${long} over 10 s.` : ".")
    + (problems ? ` Training skips ${problems} of them: see the lengths marked in yellow.` : "")
    + (clips.length && clipsData.seconds < 60 ? " A minute or more of clips trains a better voice." : "");
  checkSignature = "";
  renderCheck();
  summary.classList.toggle("warn", problems > 0);
  fillCompareReferences();
}

// Redrawn only when a dataset or the choice changes, so keyboard focus stays in the list.
let datasetListSignature = null;
function renderDatasetList(datasets, selectedId) {
  const signature = selectedId + "|" + datasets.map(row => `${row.id}:${row.name}:${row.wav_count}:${row.ready}`).join("|");
  if (signature === datasetListSignature) return;
  datasetListSignature = signature;
  const list = $("datasetList");
  list.innerHTML = "";
  datasets.forEach(dataset => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "history-item";
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", dataset.id === selectedId ? "true" : "false");
    const name = document.createElement("span");
    name.className = "history-text";
    name.textContent = dataset.name;
    const meta = document.createElement("span");
    meta.className = "history-meta";
    const clips = document.createElement("span");
    clips.textContent = `${dataset.wav_count} ${dataset.wav_count === 1 ? "clip" : "clips"}`;
    const ready = document.createElement("span");
    ready.textContent = dataset.ready ? "ready" : (dataset.wav_count ? "incomplete" : "no clips yet");
    meta.append(clips, ready);
    row.append(name, meta);
    row.onclick = () => {
      $("trainingDataset").value = dataset.id;
      $("trainingDataset").onchange();
    };
    list.appendChild(row);
  });
  $("datasetListCount").textContent = datasets.length ? `${datasets.length} ${datasets.length === 1 ? "dataset" : "datasets"}` : "No datasets yet";
  $("noDatasets").hidden = datasets.length > 0;
  $("trainingSteps").hidden = !datasets.length;
}

function openCreateDataset() {
  $("createDatasetDialog").showModal();
  $("newDatasetName").focus();
}
$("newDatasetButton").onclick = openCreateDataset;
$("newDatasetEmpty").onclick = openCreateDataset;

function renderDatasetUploads() {
  const list = $("datasetUploadRows");
  list.innerHTML = "";
  datasetUploadFiles.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "upload-row";
    const play = rowPlayButton(file.name);
    play.onclick = () => playRowAudio(file, play, file);
    // A file that plays while the list is redrawn keeps its stop button.
    if (clipPlaying === file) {
      play.innerHTML = STOP_ICON;
      clipButton = play;
    }
    const name = document.createElement("div");
    name.className = "file-name";
    name.title = file.name;
    name.textContent = file.name;
    const transcript = document.createElement("input");
    transcript.value = datasetUploadTexts.get(file) || "";
    transcript.placeholder = datasetUploadTexts.has(file) ? "Exact spoken transcript" : "Writing the transcript...";
    transcript.setAttribute("aria-label", `Transcript of ${file.name}`);
    transcript.oninput = () => {
      datasetUploadTexts.set(file, transcript.value);
      updateUploadButton();
    };
    datasetUploadInputs.set(file, transcript);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "secondary icon-button";
    remove.title = "Remove from the list";
    remove.setAttribute("aria-label", `Remove ${file.name}`);
    remove.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';
    remove.onclick = () => {
      if (clipPlaying === file) stopClip();
      datasetUploadFiles = datasetUploadFiles.filter(other => other !== file);
      renderDatasetUploads();
    };
    row.append(play, name, transcript, remove);
    list.appendChild(row);
  });
  $("datasetUploadHint").hidden = !datasetUploadFiles.length;
  updateUploadButton();
}

let uploadBusy = false;
function updateUploadButton() {
  const selected = state && (state.datasets || []).find(dataset => dataset.id === $("trainingDataset").value);
  const waiting = datasetUploadFiles.some(file => !datasetUploadTexts.has(file));
  $("uploadDatasetAudio").hidden = !datasetUploadFiles.length;
  $("uploadDatasetAudio").disabled = uploadBusy || !datasetUploadFiles.length || !selected || waiting;
}

function setUploadTranscript(file, text) {
  datasetUploadTexts.set(file, text);
  const input = datasetUploadInputs.get(file);
  if (input) {
    input.value = text;
    input.placeholder = "Exact spoken transcript";
  }
  updateUploadButton();
}

// Speech recognition writes the transcripts, one file at a time, in the dataset's language.
function queueTranscripts(files) {
  transcriptQueue.push(...files.filter(file => !datasetUploadTexts.has(file)));
  runTranscripts();
}

async function runTranscripts() {
  if (transcriptBusy) return;
  transcriptBusy = true;
  let failure = "";
  try {
    while (transcriptQueue.length) {
      const file = transcriptQueue.shift();
      if (!datasetUploadFiles.includes(file) || datasetUploadTexts.has(file)) continue;
      if (failure) {
        setUploadTranscript(file, suggestedTranscript(file.name));
        continue;
      }
      const left = 1 + transcriptQueue.filter(other => datasetUploadFiles.includes(other) && !datasetUploadTexts.has(other)).length;
      $("datasetUploadProgress").className = "status-line";
      $("datasetUploadProgress").textContent = `Writing transcripts, ${left} left...`;
      // The first transcript also loads, and the first time downloads, the speech model.
      const slow = setTimeout(() => {
        $("datasetUploadProgress").textContent = `Writing transcripts, ${left} left. The first time also downloads the speech model (about 480 MB)...`;
      }, 8000);
      try {
        const result = await post("/api/dataset-transcribe", {dataset_id: $("trainingDataset").value, filename: file.name, audio_base64: await fileToBase64(file)});
        if (!datasetUploadTexts.has(file)) setUploadTranscript(file, result.text);
      } catch (error) {
        failure = error.message;
        if (!datasetUploadTexts.has(file)) setUploadTranscript(file, suggestedTranscript(file.name));
      } finally {
        clearTimeout(slow);
      }
    }
    $("datasetUploadProgress").className = failure ? "status-line bad" : "status-line";
    $("datasetUploadProgress").textContent = failure
      ? `Automatic transcripts failed, so the file names were used. ${failure}`
      : (datasetUploadFiles.length ? "Transcripts written." : "");
  } finally {
    transcriptBusy = false;
  }
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

loadRelaySettings();
renderRelayLanguageNote();
// Migrate a saved volume from the old 0-4x scale to 0-100%; new-scale values are always > 4.
if (relaySavedSettings.relayVolume !== undefined && Number(relaySavedSettings.relayVolume) <= 4) {
  $("relayVolume").value = Math.round(Number(relaySavedSettings.relayVolume) / 4 * 100);
}
function activateTab(name) {
  if (!$(`tab-${name}`)) name = "relay";
  document.body.dataset.tab = name;
  document.querySelectorAll(".tab-button").forEach(button => {
    const active = button.dataset.tab === name;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".tab-panel").forEach(panel => {
    panel.classList.toggle("active", panel.id === `tab-${name}`);
  });
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  try { localStorage.setItem("voiceDashboardTab", name); } catch (_) {}
  renderStartButton();
  // Before the first status arrives, the page is still loading and has nothing to show.
  if (name === "relay" && lastSystemSnapshot) renderRelayCenter();
}
document.querySelectorAll(".tab-button, [data-go]").forEach(button => {
  button.onclick = () => {
    activateTab(button.dataset.tab || button.dataset.go);
    refresh();
  };
});
// The page opens on the tab used last, and the first time on Mic to Voice. Without a voice,
// that page points to Voice Training.
let savedTab = "";
try { savedTab = localStorage.getItem("voiceDashboardTab") || ""; } catch (_) {}
activateTab(location.hash.slice(1) || savedTab || "relay");
function option(select, value, label) {
  const opt = document.createElement("option");
  opt.value = value;
  opt.textContent = label;
  select.appendChild(opt);
}

function dbFromChunk(chunk) {
  if (!chunk || !chunk.length) return null;
  let sum = 0;
  for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
  const rms = Math.sqrt(sum / chunk.length);
  return 20 * Math.log10(Math.max(rms, 0.000001));
}

function modelLabel(row) {
  const plus = row.version === "v2ProPlus" && !row.path.endsWith(".ckpt");
  return `${row.name} (${plus ? "v2ProPlus, " : ""}${row.size_mb} MB)`;
}

function fillSelect(select, rows, oldValue) {
  select.innerHTML = "";
  rows.forEach(row => option(select, row.path, modelLabel(row)));
  if (oldValue && rows.some(row => row.path === oldValue)) select.value = oldValue;
}

function fillReferences(refs) {
  const old = $("reference").value;
  $("reference").innerHTML = "";
  refs.forEach(ref => {
    option($("reference"), ref.id, ref.label);
  });
  if (old && refs.some(ref => ref.id === old)) $("reference").value = old;
  const selected = refs.find(ref => ref.id === $("reference").value) || refs[0];
  if (selected) setReference(selected, false);
  // Choosing another voice dataset sets the text language to that dataset's language.
  if (selected && state && state.reference_dataset !== textLangDataset) {
    textLangDataset = state.reference_dataset;
    $("textLang").value = selected.lang;
  }
  renderAuxRefs(state ? state.all_references : refs);
}

// One clip picker implementation serves both the Generate page and the Relay settings dialog.
const generateAux = {
  ids: {search: "auxSearch", list: "auxRefsList", selected: "auxSelectedList", preview: "auxPreview", previewStatus: "auxPreviewStatus", count: "auxCount"},
  activeRef: () => $("reference").value,
  selected: auxSelected,
  onChange: null
};
const relayAux = {
  ids: {search: "relayAuxSearch", list: "relayAuxRefsList", selected: "relayAuxSelectedList", preview: "relayAuxPreview", previewStatus: "relayAuxPreviewStatus", count: "relayAuxCount"},
  activeRef: () => $("relayReference").value,
  selected: relayAuxSelected,
  onChange: () => {
    localStorage.setItem("voiceDashboardRelayAuxSelected", JSON.stringify(Array.from(relayAuxSelected)));
    updateRelayPresetStatus();
  }
};

function renderAuxPicker(picker, refs) {
  const activeRef = picker.activeRef();
  const validIds = new Set((state ? state.all_references : refs).filter(ref => ref.valid_aux_reference).map(ref => ref.id));
  Array.from(picker.selected).forEach(id => {
    if (id === activeRef || !validIds.has(id)) picker.selected.delete(id);
  });
  const filter = $(picker.ids.search).value.trim().toLowerCase();
  const list = $(picker.ids.list);
  list.innerHTML = "";
  refs
    .filter(ref => !filter || ref.label.toLowerCase().includes(filter) || ref.text.toLowerCase().includes(filter))
    .forEach(ref => {
      const row = document.createElement("div");
      const selected = picker.selected.has(ref.id);
      const canAdd = ref.valid_aux_reference && ref.id !== activeRef;
      row.className = "check-row" + (!canAdd ? " disabled" : "") + (selected ? " selected" : "") + (!ref.valid_aux_reference ? " invalid" : "");
      const detail = document.createElement("div");
      detail.className = "clip-title";
      detail.textContent = ref.label;
      const duration = document.createElement("span");
      duration.className = "clip-time";
      duration.textContent = ref.duration ? `${ref.duration}s${ref.valid_aux_reference ? "" : " long"}` : "unknown";
      detail.appendChild(duration);
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = selected ? "mini-button" : "secondary mini-button";
      toggle.textContent = selected ? "Remove" : (ref.valid_aux_reference ? "Add" : "Preview only");
      toggle.disabled = !canAdd && !selected;
      toggle.onclick = () => {
        if (picker.selected.has(ref.id)) picker.selected.delete(ref.id);
        else if (canAdd) picker.selected.add(ref.id);
        renderAuxPicker(picker, refs);
      };
      const play = document.createElement("button");
      play.type = "button";
      play.className = "secondary mini-button";
      play.textContent = "Play";
      play.onclick = event => {
        event.preventDefault();
        event.stopPropagation();
        previewClip(picker, ref);
      };
      row.appendChild(detail);
      row.appendChild(toggle);
      row.appendChild(play);
      list.appendChild(row);
    });
  updatePickerCount(picker);
  if (picker.onChange) picker.onChange();
}

function updatePickerCount(picker) {
  $(picker.ids.count).textContent = `${picker.selected.size} selected`;
  const holder = $(picker.ids.selected);
  holder.innerHTML = "";
  if (!state) return;
  Array.from(picker.selected)
    .map(id => state.all_references.find(ref => ref.id === id))
    .filter(Boolean)
    .forEach(ref => {
      const pill = document.createElement("span");
      pill.className = "pill";
      const text = document.createElement("button");
      text.type = "button";
      text.className = "pill-play";
      text.textContent = ref.label;
      text.onclick = () => previewClip(picker, ref);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "x";
      remove.setAttribute("aria-label", `Remove ${ref.label}`);
      remove.onclick = () => {
        picker.selected.delete(ref.id);
        renderAuxPicker(picker, state.all_references);
      };
      pill.appendChild(text);
      pill.appendChild(remove);
      holder.appendChild(pill);
    });
}

function previewClip(picker, ref) {
  const player = $(picker.ids.preview);
  $(picker.ids.previewStatus).textContent = `Previewing: ${ref.label}`;
  player.pause();
  player.src = ref.local_url;
  player.load();
  const started = player.play();
  if (started && typeof started.catch === "function") {
    started.catch(error => {
      $(picker.ids.previewStatus).textContent = `Preview loaded: ${ref.label}. Press play on the audio bar.`;
      console.warn("Preview playback did not auto-start:", error);
    });
  }
}

function renderAuxRefs(refs) { renderAuxPicker(generateAux, refs); }
function renderRelayAuxRefs() { if (state) renderAuxPicker(relayAux, state.all_references); }

// The clip whose words fill the reference text. An edit of them is kept while that clip stays
// chosen; another clip, as after a change of voice, brings its own words.
let promptClip = "";
function setReference(ref, rewriteText = true) {
  $("refAudio").src = ref.local_url;
  const clip = `${state ? state.reference_dataset : ""}/${ref.id}`;
  if (rewriteText || clip !== promptClip || !$("promptText").value) $("promptText").value = ref.text;
  promptClip = clip;
}

function setRelayReference(ref) {
  if (!ref) return;
  $("relayRefAudio").src = ref.local_url;
}

function fillRelayReferences(refs) {
  const old = $("relayReference").value || relaySavedSettings.relayReference || "";
  $("relayReference").innerHTML = "";
  refs.forEach(ref => option($("relayReference"), ref.id, ref.label));
  if (old && refs.some(ref => ref.id === old)) $("relayReference").value = old;
  const selected = refs.find(ref => ref.id === $("relayReference").value) || refs[0];
  if (selected) {
    $("relayReference").value = selected.id;
    setRelayReference(selected);
  }
  updateRelayPresetStatus();
}

// The Mic to Voice models. Their lists fill only once WSL runs, which can be after Mic to Voice
// starts, so the saved choice stands in until then.
function relayModels() {
  return {
    gpt: $("relayGptModel").value || relaySavedSettings.relayGptModel || "",
    sovits: $("relaySovitsModel").value || relaySavedSettings.relaySovitsModel || "",
  };
}

function relayRef() {
  if (!state) return null;
  return state.references.find(ref => ref.id === $("relayReference").value) || state.references[0] || null;
}

function relayAuxRefs() {
  if (!state) return [];
  const ref = relayRef();
  const activeId = ref ? ref.id : "";
  return Array.from(relayAuxSelected)
    .map(id => state.all_references.find(item => item.id === id))
    .filter(Boolean)
    .filter(item => item.valid_aux_reference && item.id !== activeId);
}

function updateRelayPresetStatus() {
  $("relayAuxCount").textContent = `${relayAuxRefs().length} selected`;
}

function updateRelayVolumeLabel() {
  const value = Number($("relayVolume").value || 0);
  $("relayVolumeLabel").textContent = `${Math.round(value)}%`;
  $("relayAudio").volume = Math.max(0, Math.min(1, value / 100));
}

function describePttKey(code) {
  const names = {
    Mouse0: "Mouse Left",
    Mouse1: "Mouse Middle",
    Mouse2: "Mouse Right",
    Mouse3: "Mouse Back",
    Mouse4: "Mouse Forward",
    ShiftLeft: "Left Shift",
    ShiftRight: "Right Shift",
    ControlLeft: "Left Ctrl",
    ControlRight: "Right Ctrl",
    AltLeft: "Left Alt",
    AltRight: "Right Alt",
    Space: "Space",
    Enter: "Enter",
    Tab: "Tab",
    Escape: "Escape"
  };
  if (names[code]) return names[code];
  if (/^Key[A-Z]$/.test(code)) return code.slice(3);
  if (/^Digit[0-9]$/.test(code)) return code.slice(5);
  if (/^Numpad[0-9]$/.test(code)) return "Numpad " + code.slice(6);
  return code || "Unassigned";
}

function pttMouseCode(event) {
  return `Mouse${event.button}`;
}

// Codes the global helper (ptt_helper.py) can observe; anything else only works while this page is focused.
const GLOBAL_KEY_CODES = /^(ShiftLeft|ShiftRight|ControlLeft|ControlRight|AltLeft|AltRight|Space|Enter|Tab|Escape|Key[A-Z]|Digit[0-9]|Numpad[0-9]|Mouse[0-4])$/;

function renderPttKey() {
  const code = $("relayPttKey").value;
  const control = $("relaySetPttKey");
  const caps = $("relayPttKeyLabel");
  caps.innerHTML = "";
  if (relayBindingPtt) {
    caps.textContent = "Press a key or mouse button";
    $("relayShortcutNote").textContent = "Escape keeps the current key.";
    control.setAttribute("aria-label", "Press a key or mouse button to set the push-to-talk shortcut. Escape keeps the current key.");
  } else {
    const cap = document.createElement("kbd");
    cap.className = "keycap";
    cap.textContent = describePttKey(code);
    caps.appendChild(cap);
    const note = document.createElement("span");
    note.textContent = "push-to-talk key";
    caps.appendChild(note);
    $("relayShortcutNote").textContent = GLOBAL_KEY_CODES.test(code) ? "" : "This key only works while this page is focused.";
    control.setAttribute("aria-label", `Push-to-talk shortcut: ${describePttKey(code)}. Activate to change.`);
  }
  control.classList.toggle("capturing", relayBindingPtt);
}

function startPttBinding() {
  if (relayBindingPtt || Date.now() - pttBindingCancelledAt < 400) return;
  relayBindingPtt = true;
  renderPttKey();
}

function cancelPttBinding() {
  relayBindingPtt = false;
  pttBindingCancelledAt = Date.now();
  setRelayPttKey($("relayPttKey").value);
}

function setRelayPttKey(code) {
  const key = code || "ShiftLeft";
  $("relayPttKey").value = key;
  renderPttKey();
  localStorage.setItem("voiceDashboardRelayPttKey", key);
  saveRelaySettings();
  post("/api/ptt-binding", {binding: key}).catch(error => showError(error.message));
}

async function refreshRelayDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    relayStatus("audio device list unavailable");
    return;
  }
  try {
    const permissionStream = await navigator.mediaDevices.getUserMedia({audio: true});
    permissionStream.getTracks().forEach(track => track.stop());
    const oldOutput = $("relayOutputDevice").value || relaySavedSettings.relayOutputDevice || localStorage.getItem("voiceDashboardRelayOutputDevice") || "";
    const devices = await navigator.mediaDevices.enumerateDevices();
    $("relayOutputDevice").innerHTML = "";
    option($("relayOutputDevice"), "", "Default speakers");
    devices.filter(device => device.kind === "audiooutput").forEach(device => {
      option($("relayOutputDevice"), device.deviceId, device.label || `Output ${$("relayOutputDevice").length + 1}`);
    });
    if (oldOutput) $("relayOutputDevice").value = oldOutput;
    await applyRelayOutputDevice();
    relayStatus("audio devices ready");
  } catch (error) {
    relayStatus("audio devices failed");
    console.error(error);
  }
}

async function autoLoadRelayDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  try {
    // Device labels are only exposed once mic permission is granted; when they are
    // present, refreshRelayDevices() will not trigger a permission prompt.
    const devices = await navigator.mediaDevices.enumerateDevices();
    if (devices.some(device => device.kind === "audiooutput" && device.label)) await refreshRelayDevices();
  } catch (error) {
    console.error(error);
  }
}

async function applyRelayOutputDevice() {
  const audio = $("relayAudio");
  const deviceId = $("relayOutputDevice").value;
  localStorage.setItem("voiceDashboardRelayOutputDevice", deviceId);
  saveRelaySettings();
  if (!audio.setSinkId) {
    relayStatus("output selection needs Chrome or Edge");
    return;
  }
  await audio.setSinkId(deviceId).catch(error => {
    relayStatus("output route failed");
    console.error(error);
  });
}

// Internal status strings map to a short hint under the main status label. Empty = nothing extra to say.
const STATUS_HINTS = {
  "starting local speech": "Starting speech recognition. The first recording can take a moment.",
  "starting TTS engine": "Starting the voice engine. This can take a minute or two.",
  "transcription failed": "Transcription failed. Check the Logs page.",
  "translating": "Translating the typed line.",
  "translation failed": "Translation failed. Check the Logs page.",
  "local speech failed": "The microphone or speech recognition failed. Check the Logs page.",
  "audio devices failed": "The audio device list failed. Allow microphone access and try again.",
  "audio device list unavailable": "Audio devices are unavailable in this browser.",
  "output selection needs Chrome or Edge": "Output device selection needs Chrome or Edge.",
  "output route failed": "Audio could not be routed to the selected output device.",
  "press play on output": "The browser blocked automatic playback. Press play on the player.",
  "no speech recognized": "No speech was recognized. Try again, a little closer to the microphone.",
  "recording too short": "That was too short. Hold the key a little longer while speaking.",
  "relay failed": "", "generating": "", "playing generated line": "", "line queued": "", "idle": "",
  "push-to-talk ready": "", "listening": "", "not listening": "", "listening local": "",
  "recording push-to-talk": "", "transcribing locally": "", "current line cancelled": "",
  "audio devices ready": "", "transcribed": ""
};
const ACTIVE_STATES = new Set(["queued", "generating", "ready", "playing", "failed"]);
const HISTORY_STATES = {queued: "Waiting", generating: "Generating", ready: "Ready", playing: "Playing", completed: "", failed: "Failed", cancelled: "Cancelled", transcribed: "Not spoken"};
const ICONS = {
  mic: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/><path d="M8 21h8"/></svg>',
  dot: '<span class="rec-dot"></span>',
  pen: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h4L18 10l-4-4L4 16v4z"/><path d="M13 7l4 4"/><path d="M4 22h16"/></svg>',
  clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/><path class="hand" d="M12 8v4l3 2"/></svg>',
  bars: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="5" width="3" height="14" rx="1"/><rect x="8" y="5" width="3" height="14" rx="1"/><rect x="13" y="5" width="3" height="14" rx="1"/><rect x="18" y="5" width="3" height="14" rx="1"/></svg>',
  speaker: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9v6h4l5 4V5L8 9H4z"/><path class="wave" d="M16 9a4 4 0 0 1 0 6"/><path class="wave" d="M18.5 6.5a7.5 7.5 0 0 1 0 11"/></svg>',
  alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l10 18H2L12 3z"/><path d="M12 10v4"/><path d="M12 17.5h.01"/></svg>'
};

function relayStatus(text) {
  relayStatusText = text || "";
  recentActivity(text);
  renderRelayCenter();
}

function statusHint() {
  if (!relayStatusText) return "";
  if (/^push-to-talk key:/.test(relayStatusText)) return "";
  return STATUS_HINTS[relayStatusText] !== undefined ? STATUS_HINTS[relayStatusText] : relayStatusText;
}

function formatClock(stamp) {
  try { return new Date(stamp).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }
  catch (_) { return ""; }
}

function viewedRelayItem() {
  if (relayView.mode !== "result") return null;
  return relayQueue.find(item => item.id === relayView.itemId) || null;
}

function selectRelayItem(id) {
  relayView = {mode: "result", itemId: id};
  closeHistoryDrawer();
  renderRelayQueue();
}

function showCaptureView() {
  relayView = {mode: "capture", itemId: null};
  closeHistoryDrawer();
  renderRelayQueue();
}

function closeHistoryDrawer() {
  $("relayHistory").classList.remove("open");
  $("historyToggle").setAttribute("aria-expanded", "false");
}

function renderRelayQueue() {
  const list = $("relayQueue");
  list.innerHTML = "";
  const items = relayQueue.slice(-60).reverse();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent = "Recordings from this session appear here.";
    list.appendChild(empty);
  }
  items.forEach(item => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "history-item";
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", relayView.mode === "result" && relayView.itemId === item.id ? "true" : "false");
    const text = document.createElement("span");
    text.className = "history-text";
    // What you said, which for a translated line is not what the voice says.
    text.textContent = item.heard || item.text;
    const meta = document.createElement("span");
    meta.className = "history-meta";
    const stateSpan = document.createElement("span");
    stateSpan.className = `history-state ${item.status}`;
    stateSpan.textContent = HISTORY_STATES[item.status] !== undefined ? HISTORY_STATES[item.status] : item.status;
    const time = document.createElement("span");
    time.textContent = formatClock(item.createdAt);
    meta.append(stateSpan, time);
    row.append(text, meta);
    row.onclick = () => selectRelayItem(item.id);
    list.appendChild(row);
  });
  const waiting = relayQueue.filter(item => item.status === "queued").length;
  $("relayQueueCount").textContent = waiting ? `${waiting} waiting` : (relayQueue.length ? `${relayQueue.length} this session` : "Nothing recorded yet");
  $("relayClearQueue").hidden = !waiting;
  $("historyCount").textContent = relayQueue.length ? `(${relayQueue.length})` : "";
  renderRelayCenter();
}

function relayPrimaryState(item) {
  const ptt = $("relayPushToTalk").checked;
  const system = lastSystemSnapshot || {};
  if (relayPushHeld || (!ptt && relayCapturing)) {
    return {key: "recording", icon: "dot", label: ptt ? "Listening" : "Hearing speech", hint: ptt ? "Release to transcribe." : "Pause to transcribe."};
  }
  if (relayTranscribing) return {key: "transcribing", icon: "pen", label: "Transcribing", hint: ""};
  // The viewed item while it is still being worked on, or, in the capture view, whatever is being worked on now.
  const active = item && ACTIVE_STATES.has(item.status) ? item : (!item && relayCurrentItem ? relayCurrentItem : null);
  if (active) {
    const suffix = active === item ? "" : " for the previous line";
    switch (active.status) {
      case "queued": {
        const ahead = relayQueue.filter(row => (row.status === "queued" && row.id < active.id) || ["generating", "ready", "playing"].includes(row.status)).length;
        return {key: "queued", icon: "clock", label: ahead ? `Waiting, ${ahead} ahead` : "Waiting for the voice engine", hint: statusHint()};
      }
      case "generating": return {key: "generating", icon: "bars", label: "Generating voice" + suffix, hint: statusHint()};
      case "ready": return {key: "generating", icon: "bars", label: "Voice ready, starting playback", hint: ""};
      case "playing": return {key: "playing", icon: "speaker", label: "Playing" + suffix, hint: ""};
      case "failed": return {key: "failed", icon: "alert", label: "Voice generation failed", hint: active.error || statusHint()};
    }
  }
  if (!ptt) {
    if (relayListening && relayAudioCtx) return {key: "ready", icon: "", label: "Listening for speech", hint: "Voice activation is on. Speak, then pause, to transcribe."};
    if (relayListening) return {key: "ready", icon: "", label: "Starting microphone", hint: statusHint()};
    return {key: "ready", icon: "", label: "Press to start listening", hint: "Voice activation is on. Turn Hold to talk on to record by holding instead."};
  }
  let hint = statusHint();
  // Only while the voice is starting or running; when it is off, why it is off matters more.
  if (!hint && needsPageClick() && (systemStartBusy || ["starting", "ready", "degraded"].includes(system.status))) {
    hint = "Click anywhere on this page once. Browsers play the voice only after that.";
  }
  if (!hint && relayCurrentItem && item && item !== relayCurrentItem) hint = "Another line is still being generated or played.";
  if (!hint) {
    if (relayListening && !relayAudioCtx) hint = "Speech recognition is starting.";
    else if (system.status === "stopped" && systemStartBusy) hint = "Starting speech recognition and the voice engine.";
    else if (system.status === "stopped" && !relayListening && state && state.gpu_busy) hint = "Training is using the GPU. Press Start to load the voice anyway.";
    else if (system.status === "stopped" && !relayListening) hint = "Press Start to turn on speech recognition and the voice engine.";
    else if (system.status === "starting") hint = system.message || "Starting the system.";
    else if (system.status === "failed" || system.status === "degraded") hint = system.message || "";
  }
  return {key: "ready", icon: "", label: "Hold to talk", hint};
}

// The system can already be running when the page loads, as after a reload, but browsers refuse
// to play sound on a page that was never clicked.
function needsPageClick() {
  return Boolean(navigator.userActivation) && !navigator.userActivation.hasBeenActive;
}
["pointerdown", "keydown"].forEach(name => document.addEventListener(name, () => setTimeout(renderRelayCenter), {capture: true, once: true}));

function renderRelayCenter() {
  const item = viewedRelayItem();
  $("relayWorkspace").dataset.view = item ? "result" : "capture";
  $("relayResult").hidden = !item;
  const primary = relayPrimaryState(item);
  $("relayStatusBox").className = `capture-status ${primary.key}`;
  $("relayStatusIcon").className = `status-icon ${primary.key}`;
  $("relayStatusIcon").innerHTML = ICONS[primary.icon] || "";
  $("relayStatusText").textContent = primary.label;
  $("relayStatusHint").textContent = primary.hint || "";
  const ptt = $("relayPushToTalk").checked;
  $("relayPttButton").setAttribute("aria-label", ptt ? "Hold to talk" : "Start listening");
  $("relayPttButton").setAttribute("aria-pressed", relayPushHeld ? "true" : "false");
  const active = item && ACTIVE_STATES.has(item.status) ? item : (!item ? relayCurrentItem : null);
  $("relayCancelCurrent").classList.toggle("visible", Boolean(relayCurrentItem && active === relayCurrentItem));
  if (item) renderRelayResult(item);
}

function renderRelayResult(item) {
  $("relayTranscript").textContent = item.heard || item.text;
  $("relayTranslation").textContent = item.heard ? item.text : "";
  $("relayTranslationBlock").hidden = !item.heard;
  const hasAudio = Boolean(item.url);
  $("relayPlayerBlock").hidden = !hasAudio;
  if (hasAudio) {
    const audio = $("relayAudio");
    const mismatch = relayLoadedUrl !== item.url;
    if (mismatch && (audio.paused || audio.ended)) {
      // Nothing is playing, so load this item's audio without starting it.
      relayLoadedUrl = item.url;
      audio.src = item.url;
      $("relayAudio").hidden = false;
      $("relayPlayThis").hidden = true;
    } else {
      // Another line is playing through the shared player; offer to switch instead of showing its controls here.
      $("relayAudio").hidden = mismatch;
      $("relayPlayThis").hidden = !mismatch;
    }
  }
  const actions = $("relayResultActions");
  actions.innerHTML = "";
  if (["transcribed", "failed", "cancelled"].includes(item.status)) {
    const speak = document.createElement("button");
    speak.type = "button";
    speak.className = "mini-button";
    speak.textContent = item.status === "transcribed" ? "Speak this line" : "Try again";
    speak.onclick = () => requeueRelayItem(item);
    actions.appendChild(speak);
  }
  const meta = [];
  if (item.status === "transcribed") meta.push("Transcribed only. Speak automatically is off.");
  else if (item.status === "cancelled") meta.push("Cancelled before playback.");
  else if (item.status === "completed") meta.push("Spoken");
  if (item.name) meta.push(item.name);
  meta.push(formatClock(item.createdAt));
  $("relayNowPlaying").textContent = meta.filter(Boolean).join(" · ");
}

function setRelayLevel(db) {
  if (++relayLevelTick % 2) return;
  const level = Math.max(0, Math.min(1, (db + 60) / 50));
  $("relayPttButton").style.setProperty("--level", level.toFixed(2));
}

// Waits for the voice engine, unless the line it is for is cancelled meanwhile.
async function ensureRelayApiReady(item) {
  if (state && state.api_ready) return;
  relayStatus("starting TTS engine");
  await post("/api/start-api", relayModels());
  for (let i = 0; i < 80; i++) {
    await new Promise(resolve => setTimeout(resolve, 1500));
    if (item && item.status === "cancelled") return;
    await refresh();
    if (state && state.api_ready) return;
  }
  throw new Error("TTS engine did not become ready.");
}

async function generateRelayLine(text, generationId, lang) {
  const ref = relayRef();
  if (!ref) throw new Error("Pick a reference clip in Settings.");
  const aux = relayAuxRefs().map(item => item.wsl_path);
  return await post("/api/generate", {
    ...relayModels(),
    ref_audio_path: ref.wsl_path,
    aux_ref_audio_paths: aux,
    prompt_text: ref.text,
    prompt_lang: ref.lang,
    text,
    text_lang: lang,
    text_split_method: "cut5",
    seed: -1,
    top_k: $("relayTopK").value,
    top_p: $("relayTopP").value,
    temperature: $("relayTemperature").value,
    speed_factor: $("relaySpeed").value,
    fragment_interval: $("relayPause").value,
    repetition_penalty: $("relayRepPenalty").value,
    streaming_mode: 3,
    overlap_length: 2,
    min_chunk_length: 8,
    generation_id: generationId,
    voice: voiceDataset,
    // Relay lines are played once; the server keeps only the newest ones.
    relay: true
  });
}

function playRelayUrl(url, label) {
  relayLastUrl = url;
  relayLoadedUrl = url;
  const audio = $("relayAudio");
  audio.pause();
  audio.src = url + "?t=" + Date.now();
  applyRelayOutputDevice();
  updateRelayVolumeLabel();
  renderRelayCenter();
  const started = audio.play();
  if (started && typeof started.catch === "function") {
    started.catch(() => relayStatus("press play on output"));
  }
  if ($("relayMonitorLocal").checked && $("relayOutputDevice").value) {
    const monitor = document.getElementById("relayLocalMonitor") || document.createElement("audio");
    monitor.id = "relayLocalMonitor";
    monitor.style.display = "none";
    monitor.volume = audio.volume;
    monitor.src = url + "?monitor=" + Date.now();
    if (!monitor.parentElement) document.body.appendChild(monitor);
    const monitorStarted = monitor.play();
    if (monitorStarted && typeof monitorStarted.catch === "function") monitorStarted.catch(() => {});
  }
}

// A line keeps the language its voice speaks, so Speak this line and Try again use it even after
// the setting changed. heard is what was said, when the voice says it in another language.
function newRelayItem(line, status) {
  const clean = (line.text || "").replace(/\s+/g, " ").trim();
  if (!clean) return null;
  const item = {id: relayNextId++, generationId: `relay-${Date.now()}-${relayNextId}`, text: clean, heard: line.heard || "",
    lang: line.lang || "en", status, url: "", name: "", error: "", createdAt: Date.now()};
  relayQueue.push(item);
  relayView = {mode: "result", itemId: item.id};
  return item;
}

function enqueueRelayLine(line) {
  const item = newRelayItem(line, "queued");
  if (!item) return;
  renderRelayQueue();
  relayStatus("line queued");
  processRelayQueue();
}

// Used for "Speak this line" (transcribed while Speak automatically was off) and "Try again" after a failure.
function requeueRelayItem(item) {
  item.status = "queued";
  item.url = "";
  item.name = "";
  item.error = "";
  item.generationId = `relay-${Date.now()}-${item.id}`;
  relayView = {mode: "result", itemId: item.id};
  renderRelayQueue();
  relayStatus("line queued");
  processRelayQueue();
}

async function processRelayQueue() {
  if (relayBusy) return;
  relayBusy = true;
  try {
    while (relayQueue.some(item => item.status === "queued")) {
      const item = relayQueue.find(row => row.status === "queued");
      relayCurrentItem = item;
      item.status = "generating";
      $("relayCancelCurrent").classList.add("visible");
      renderRelayQueue();
      relayStatus("generating");
      try {
        await ensureRelayApiReady(item);
        // A line cancelled while the engine was starting is not generated at all.
        if (item.status === "cancelled") continue;
        const result = await generateRelayLine(item.text, item.generationId, item.lang);
        if (item.status === "cancelled") continue;
        item.status = "ready";
        item.url = result.url;
        item.name = result.name;
        renderRelayQueue();
        item.status = "playing";
        renderRelayQueue();
        relayStatus("playing generated line");
        playRelayUrl(item.url, item.name);
        await new Promise(resolve => {
          relayPlaybackResolve = resolve;
          const audio = $("relayAudio");
          audio.onended = resolve;
          audio.onerror = resolve;
        });
        relayPlaybackResolve = null;
        if (item.status !== "cancelled") item.status = "completed";
        renderRelayQueue();
      } catch (error) {
        if (item.status !== "cancelled") {
          item.status = "failed";
          item.error = error.message;
          relayStatus("relay failed");
          showError(error.message);
        }
        renderRelayQueue();
      } finally {
        relayCurrentItem = null;
        $("relayCancelCurrent").classList.remove("visible");
        renderRelayCenter();
      }
    }
    if (!relayListening) relayStatus("idle");
  } finally {
    relayBusy = false;
  }
}

function mergeFloatChunks(chunks, frameCount) {
  const merged = new Float32Array(frameCount);
  let offset = 0;
  chunks.forEach(chunk => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });
  return merged;
}

function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeString = (offset, value) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return new Blob([buffer], {type: "audio/wav"});
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function ensureRelayAsrReady() {
  if (state && state.relay_asr_ready) return;
  relayStatus("starting local speech");
  await post("/api/start-relay-asr");
  for (let i = 0; i < 80; i++) {
    await new Promise(resolve => setTimeout(resolve, 500));
    if (!relayListening) throw new Error("Listening was stopped.");
    await refresh();
    if (state && state.relay_asr_ready) return;
  }
  throw new Error("Local speech recognizer did not become ready.");
}

// A recording's words as a line for the voice: its text, in the language the voice speaks, and,
// when that is another language, what was heard.
async function transcribeRelayAudio(samples, sampleRate) {
  const wav = encodeWav(samples, sampleRate);
  const audio_base64 = await blobToBase64(wav);
  const languages = relayLanguages();
  const result = await post("/api/transcribe", {audio_base64, ...languages});
  return {text: (result.text || "").replace(/\s+/g, " ").trim(), heard: result.heard || "", lang: languages.speaks};
}

// A typed line counts as said, so it is translated in the same way.
async function speakTypedRelayLine() {
  const text = $("relayManualText").value.replace(/\s+/g, " ").trim();
  if (!text) return;
  const languages = relayLanguages();
  if (!relayTranslates()) {
    enqueueRelayLine({text, lang: languages.speaks});
    return;
  }
  $("relaySpeakManual").disabled = true;
  relayStatus("translating");
  try {
    const result = await post("/api/translate", {text, ...languages});
    if (!result.text.trim()) throw new Error("Translation failed: it came back empty.");
    enqueueRelayLine({text: result.text, heard: result.heard, lang: languages.speaks});
  } catch (error) {
    relayStatus("translation failed");
    showError(error.message);
  } finally {
    $("relaySpeakManual").disabled = false;
  }
}

function resetRelayCapture() {
  relayCapturing = false;
  relayCurrentChunks = [];
  relayCurrentFrames = 0;
  relayLastVoiceAt = 0;
}

function beginRelayPush() {
  if (!$("relayPushToTalk").checked) return;
  if (!relayListening) {
    try { startRelayListening(); }
    catch (e) { alert(e.message); return; }
  }
  if (!relayAudioCtx) {
    relayStatus("starting local speech");
    return;
  }
  if (relayPushHeld) return;
  resetRelayCapture();
  relayPushHeld = true;
  relayCapturing = true;
  $("relayPttButton").classList.add("active");
  relayStatus("recording push-to-talk");
}

function endRelayPush() {
  if (!relayPushHeld) return;
  relayPushHeld = false;
  $("relayPttButton").classList.remove("active");
  finishRelayCapture();
  if (relayListening && !relayBusy) relayStatus("push-to-talk ready");
}

function finishRelayCapture() {
  if (!relayCurrentFrames) {
    resetRelayCapture();
    return;
  }
  const frames = relayCurrentFrames;
  const chunks = relayCurrentChunks.slice();
  resetRelayCapture();
  const minFrames = Math.floor((relayAudioCtx ? relayAudioCtx.sampleRate : 48000) * 0.45);
  if (frames < minFrames) {
    relayStatus("recording too short");
    return;
  }
  const samples = mergeFloatChunks(chunks, frames);
  const sampleRate = relayAudioCtx ? relayAudioCtx.sampleRate : 48000;
  relayTranscribing = true;
  relayStatus("transcribing locally");
  transcribeRelayAudio(samples, sampleRate)
    .then(line => {
      relayTranscribing = false;
      if (!line.text) {
        relayStatus("no speech recognized");
        return;
      }
      if ($("relayAutoSpeak").checked) {
        enqueueRelayLine(line);
      } else {
        newRelayItem(line, "transcribed");
        renderRelayQueue();
        relayStatus("transcribed");
      }
    })
    .catch(error => {
      relayTranscribing = false;
      // The cause is in Logs; see translate_relay in dashboard/asr.py for the translation's.
      relayStatus(error.message.startsWith("Translation failed") ? "translation failed" : "transcription failed");
      console.error(error);
    });
}

function startRelayListening() {
  if (!navigator.mediaDevices) throw new Error("Browser microphone capture is unavailable.");
  if (relayListening && relayAudioCtx) {
    relayStatus("listening local");
    return;
  }
  relayListening = true;
  // Stop Everything can come while speech recognition or the microphone is still starting. The
  // start then goes no further, and opens no microphone that nothing would close.
  const start = ++relayStartCount;
  const stopped = () => start !== relayStartCount || !relayListening;
  relayStatus("starting local speech");
  ensureRelayAsrReady().then(async () => {
    if (stopped()) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: {ideal: 1},
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      }
    });
    if (stopped()) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    relayStream = stream;
    relayAudioCtx = new AudioContext({latencyHint: "interactive"});
    relaySource = relayAudioCtx.createMediaStreamSource(relayStream);
    relayProcessor = relayAudioCtx.createScriptProcessor(4096, 1, 1);
    relayKeepAlive = relayAudioCtx.createGain();
    relayKeepAlive.gain.value = 0;
    relayProcessor.onaudioprocess = event => {
      if (!relayListening) return;
      const input = event.inputBuffer.getChannelData(0);
      const db = dbFromChunk(input);
      relayLastLevelDb = db === null ? -120 : db;
      const now = performance.now();
      if ($("relayPushToTalk").checked) {
        if (relayPushHeld) {
          if (!relayCapturing) {
            relayCapturing = true;
            relayCurrentChunks = [];
            relayCurrentFrames = 0;
          }
          relayCurrentChunks.push(new Float32Array(input));
          relayCurrentFrames += input.length;
          setRelayLevel(relayLastLevelDb);
        }
        return;
      }
      const isVoice = relayLastLevelDb > -48;
      if (isVoice && !relayCapturing) {
        relayCapturing = true;
        relayLastVoiceAt = now;
        $("relayPttButton").classList.add("active");
        renderRelayCenter();
      }
      if (relayCapturing) {
        relayCurrentChunks.push(new Float32Array(input));
        relayCurrentFrames += input.length;
        if (isVoice) relayLastVoiceAt = now;
        const capturedMs = relayCurrentFrames / relayAudioCtx.sampleRate * 1000;
        setRelayLevel(relayLastLevelDb);
        if ((now - relayLastVoiceAt > 850 && capturedMs > 700) || capturedMs > 9000) {
          $("relayPttButton").classList.remove("active");
          finishRelayCapture();
          renderRelayCenter();
        }
      }
    };
    relaySource.connect(relayProcessor);
    relayProcessor.connect(relayKeepAlive);
    relayKeepAlive.connect(relayAudioCtx.destination);
    if (relayAudioCtx.state === "suspended") await relayAudioCtx.resume();
    if (stopped()) return;
    relayStatus($("relayPushToTalk").checked ? "push-to-talk ready" : "listening local");
  }).catch(error => {
    // A start that was stopped meanwhile has nothing to report.
    if (stopped()) return;
    relayListening = false;
    relayStatus("local speech failed");
    alert(error.message);
  });
}

function stopRelayListening() {
  relayListening = false;
  relayStartCount++;
  relayPushHeld = false;
  $("relayPttButton").classList.remove("active");
  if (relayProcessor) relayProcessor.disconnect();
  relayProcessor = null;
  if (relaySource) relaySource.disconnect();
  relaySource = null;
  if (relayKeepAlive) relayKeepAlive.disconnect();
  relayKeepAlive = null;
  if (relayStream) relayStream.getTracks().forEach(track => track.stop());
  relayStream = null;
  if (relayAudioCtx) relayAudioCtx.close();
  relayAudioCtx = null;
  // Stop Everything drops a recording it cuts off: transcribing it would start speech
  // recognition again, and with Speak automatically on, the voice engine too.
  resetRelayCapture();
  relayStatus(relayBusy ? "generating" : "not listening");
}

function updateSystemView(system) {
  if (!system) return;
  const status = system.status || "stopped";
  lastSystemSnapshot = system;
  $("systemChip").textContent = status === "ready" ? "System ready" : (status === "stopped" ? "System stopped" : (system.message || status));
  $("systemChip").title = system.error ? `${system.message} ${system.error}` : (system.message || status);
  $("systemChip").className = status === "ready" ? "island-system-state good" : (status === "failed" || status === "degraded" ? "island-system-state bad" : "island-system-state");
  if (status !== lastSystemStatus) {
    lastSystemStatus = status;
    renderRelayCenter();
  }
  renderStartButton();
  // Stop Everything stays clickable: services can be running even while the
  // system status reads "stopped", and stop-system is safe to call at any time.
  const components = system.components || {};
  componentCard("relayAsrCard", "relayAsrStatus", status === "starting" && system.phase === "asr" ? "starting" : components.asr);
  componentCard("relayTtsCard", "relayTtsStatus", status === "starting" && system.phase === "tts" ? "starting" : components.tts);
  componentCard("relayPttCard", "relayPttStatus", status === "starting" && system.phase === "ptt" ? "starting" : components.ptt);
  if (system.message && system.message !== lastSystemMessage) {
    lastSystemMessage = system.message;
    recentActivity(system.error ? `${system.message} ${system.error}` : system.message);
    if (system.error) showError(`${system.message} ${system.error}`);
  }
  if (status === "ready" && armRelayWhenReady && !relayListening) {
    armRelayWhenReady = false;
    startRelayListening();
  }
}

function hasVoice() {
  return Boolean(state && (state.datasets || []).some(dataset => dataset.wav_count > 0));
}

// Start shows only on Mic to Voice: as Start while the system is stopped, and as Retry after a
// failed start. During training it still loads the voice, for example to test a checkpoint.
function renderStartButton() {
  const status = (lastSystemSnapshot || {}).status || "";
  const failed = status === "failed" || status === "degraded";
  $("startSystem").hidden = document.body.dataset.tab !== "relay" || systemStartBusy
    || !hasVoice() || !(failed || status === "stopped");
  $("startSystem").textContent = failed ? "Retry" : "Start";
}

async function startSystem() {
  const token = ++startToken;
  systemStartBusy = true;
  renderStartButton();
  renderRelayCenter();
  $("globalError").classList.remove("visible");
  armRelayWhenReady = true;
  recentActivity("Requesting microphone access...");
  try {
    saveRelaySettings();
    await refreshRelayDevices();
    if (token !== startToken) return;
    // The languages, so their models load while the system starts.
    await post("/api/start-system", {...relayModels(), binding: $("relayPttKey").value, ...relayLanguages()});
    recentActivity("System startup requested.");
    await refreshSystem();
  } catch (error) {
    armRelayWhenReady = false;
    showError(error.message);
  } finally {
    systemStartBusy = false;
    renderStartButton();
    renderRelayCenter();
  }
}

async function refreshSystem() {
  try {
    const response = await fetch("/api/system-state");
    const data = await response.json();
    updateSystemView(data.system);
  } catch (_) {
    $("systemChip").textContent = "dashboard connection lost";
    $("systemChip").title = "dashboard connection lost";
    $("systemChip").className = "chip bad";
  }
}

async function runSetupCheck() {
  $("setupSummary").textContent = "Checking local components...";
  const list = $("setupList");
  list.innerHTML = "";
  try {
    const response = await fetch("/api/setup-status");
    const checks = await response.json();
    try {
      const permissionStream = await navigator.mediaDevices.getUserMedia({audio: true});
      permissionStream.getTracks().forEach(track => track.stop());
      const devices = await navigator.mediaDevices.enumerateDevices();
      const microphones = devices.filter(device => device.kind === "audioinput");
      const outputs = devices.filter(device => device.kind === "audiooutput");
      const cableDevices = devices.filter(device => /VB-Audio|CABLE Input|CABLE Output/i.test(device.label || ""));
      checks.microphone = {status: microphones.length ? "ready" : "missing", detail: `${microphones.length} microphone input(s) detected`};
      checks.audio_output = {status: outputs.length ? "ready" : "missing", detail: `${outputs.length} audio output(s) detected`};
      checks.vb_cable = {status: cableDevices.length ? "ready" : "missing", detail: cableDevices.length ? cableDevices.map(device => device.label).join(", ") : "VB-CABLE was not found in browser audio devices."};
    } catch (error) {
      checks.microphone = {status: "needs_configuration", detail: "Allow microphone access to complete this check."};
      checks.audio_output = {status: "needs_configuration", detail: "Allow audio-device access to complete this check."};
    }
    const names = {
      config: "config.json",
      dashboard_python: "Dashboard Python",
      wsl_files: "GPT-SoVITS / WSL",
      gpt_models: "GPT models",
      sovits_models: "SoVITS models",
      asr_environment: "Speech recognition",
      asr_model_cache: "ASR model cache",
      ptt_helper: "Global PTT helper",
      gpu: "Voice engine device",
      microphone: "Microphone",
      audio_output: "Audio outputs",
      vb_cable: "VB-CABLE / audio"
    };
    let ready = 0;
    Object.entries(checks).forEach(([key, check]) => {
      if (check.status === "ready") ready += 1;
      const row = document.createElement("div");
      row.className = "setup-row";
      const name = document.createElement("strong");
      name.textContent = names[key] || key;
      const status = document.createElement("span");
      status.className = `setup-state ${check.status}`;
      status.textContent = check.status.replaceAll("_", " ");
      const detail = document.createElement("span");
      detail.className = "small";
      detail.textContent = check.detail || "";
      row.append(name, status, detail);
      list.appendChild(row);
    });
    $("setupSummary").textContent = `${ready} of ${Object.keys(checks).length} checks ready.`;
  } catch (error) {
    $("setupSummary").textContent = "System check failed.";
    showError(error.message);
  }
}

// A refresh asked for while another runs waits for it and runs once more, as the one running
// asked with the choices from before.
let refreshRunning = null;
let refreshQueued = null;
function refresh() {
  if (refreshRunning) {
    // Also after a refresh that failed, as when the dashboard restarts.
    if (!refreshQueued) refreshQueued = refreshRunning.catch(() => {}).then(() => { refreshQueued = null; return refresh(); });
    return refreshQueued;
  }
  refreshRunning = refreshOnce().finally(() => { refreshRunning = null; });
  return refreshRunning;
}

async function refreshOnce() {
  const requested = voiceDataset;
  const res = await fetch(`/api/state?dataset=${encodeURIComponent(requested)}`);
  state = await res.json();
  if (state.config_error && !configErrorShown) {
    configErrorShown = true;
    showError(state.config_error);
  }
  fillTrainingDatasets(state.datasets || []);
  fillDatasetSelects(state.datasets || [], state.reference_dataset || "", requested);
  document.body.dataset.voice = hasVoice() ? "yes" : "no";
  if (Boolean(state.gpu_busy) !== lastGpuBusy) {
    lastGpuBusy = Boolean(state.gpu_busy);
    renderRelayCenter();
  }
  const gptOld = $("gptModel").value;
  const sovitsOld = $("sovitsModel").value;
  const relayGptOld = $("relayGptModel").value || relaySavedSettings.relayGptModel || "";
  const relaySovitsOld = $("relaySovitsModel").value || relaySavedSettings.relaySovitsModel || "";
  const nextModelsSignature = [...state.models.gpt, ...state.models.sovits].map(row => `${row.path}:${row.mtime}`).join("|");
  if (nextModelsSignature !== modelsSignature) {
    modelsSignature = nextModelsSignature;
    fillSelect($("gptModel"), state.models.gpt, gptOld);
    fillSelect($("sovitsModel"), state.models.sovits, sovitsOld);
    fillSelect($("relayGptModel"), state.models.gpt, relayGptOld);
    fillSelect($("relaySovitsModel"), state.models.sovits, relaySovitsOld);
  }
  const nextRefsSignature = (state.reference_dataset || "") + "|" + state.all_references.map(ref => `${ref.id}:${ref.duration || ""}:${ref.valid_reference}:${ref.valid_aux_reference}`).join("|");
  if (nextRefsSignature !== refsSignature) {
    refsSignature = nextRefsSignature;
    fillReferences(state.references);
    fillRelayReferences(state.references);
    renderRelayAuxRefs();
  }
  if (!engineBusy) {
    $("engineStatus").textContent = state.api_ready ? "Engine ready" : "Engine off";
    $("engineStatus").className = state.api_ready ? "status-line good" : "status-line";
  }
  updateSystemView(state.system);
  const preparation = state.jobs && state.jobs["dataset-prepare"];
  if (preparation) {
    $("datasetPrepareStatus").textContent = preparation.status === "running" ? `${preparation.name} is running...` : `${preparation.name}: ${preparation.status}`;
    $("datasetPrepareStatus").className = preparation.status === "failed" ? "status-line bad" : (preparation.status === "done" ? "status-line good" : "status-line");
    $("prepareDataset").disabled = preparation.status === "running";
  }
  const jobLine = (key, element, bar) => {
    const job = state.jobs && state.jobs[key];
    const progress = job && job.progress;
    $(bar).hidden = !(job && job.status === "running" && progress);
    if (!job) { $(element).textContent = ""; return; }
    let text = job.status === "running" ? "Running..." : `${job.status}${job.returncode !== null && job.returncode !== undefined ? ` (exit ${job.returncode})` : ""}`;
    if (job.status === "running" && progress) {
      text = progress.epoch ? `Epoch ${progress.epoch} of ${progress.epochs} done` : "Starting...";
      if (progress.seconds_left) text += `, about ${Math.max(1, Math.round(progress.seconds_left / 60))} min left`;
      $(bar).firstElementChild.style.width = `${Math.min(100, 100 * progress.epoch / progress.epochs)}%`;
    }
    $(element).textContent = text;
    $(element).className = job.status === "failed" ? "status-line bad" : (job.status === "done" ? "status-line good" : "status-line");
  };
  jobLine("train-sovits", "trainSovitsStatus", "trainSovitsBar");
  jobLine("train-gpt", "trainGptStatus", "trainGptBar");
  renderCompare();
  const trainingJob = ["dataset-sync", "dataset-prepare", "train-sovits", "train-gpt"].map(key => state.jobs[key]).find(job => job && job.status === "running");
  $("trainingJobsStatus").textContent = trainingJob ? `${trainingJob.name} is running` : "";
  const jobLogs = Object.values(state.jobs || {})
    .filter(job => job.lines && job.lines.length)
    .map(job => [`--- ${job.name}: ${job.status} ---`, ...job.lines].join("\n"))
    .join("\n\n");
  logRaw = [state.log || "", jobLogs].filter(Boolean).join("\n\n");
  renderLog();
  $("modelList").innerHTML = "";
  const savedModels = [...state.models.gpt, ...state.models.sovits];
  $("modelCount").textContent = savedModels.length ? `${savedModels.length} files` : (state.models_listed ? "none yet" : "listed once WSL is running");
  savedModels.forEach(row => $("modelList").appendChild(modelLine(row.path, row.size_mb)));
  renderRestore();
  renderCutStatus();
  renderCheck();
  showWelcome();
  // Loaded data can move the highlighted element.
  placeTour();
}

function modelLine(label, sizeMb) {
  const div = document.createElement("div");
  div.className = "model-line";
  const name = document.createElement("span");
  name.textContent = label;
  const size = document.createElement("span");
  size.textContent = `${sizeMb} MB`;
  div.append(name, size);
  return div;
}

// Guided tour. The page darkens except for one highlighted element, and a pop-up beside it
// explains it. The tour moves through the tabs.
// Steps marked "voice" show controls that exist only once a voice does.
const TOUR_STEPS = [
  {title: "Welcome", text: "This tour shows what each part of the dashboard does and how to use it. Use Next and Back, or the arrow keys. Esc ends the tour."},
  {targets: [".tablist"], title: "Tabs", text: "Mic to Voice and Text to Voice speak in a voice. Voice Training creates voices. Logs and System Check help when something does not work."},
  {tab: "training", targets: ["#newDatasetButton"], title: "Create a voice", text: "Every voice starts as a dataset. Press + New dataset, then give it a name and a language."},
  {tab: "training", targets: ["#clipsStep", "#noDatasets"], title: "Add clips", text: "Add clips, or long recordings, which are cut into clips at pauses. Speech recognition writes each transcript. Check them in the list, where every clip can be played, corrected or deleted. Check clips finds clips with a wrong transcript, noise or distortion."},
  {tab: "training", targets: ["#prepareStep", "#noDatasets"], title: "Sync the clips", text: "Sync Only copies the clips to WSL, which is all Text to Voice and Mic to Voice need. Sync again after adding clips."},
  {tab: "training", targets: ["#trainPanel", "#noDatasets"], title: "Train the voice", text: "Training makes the voice closer to its clips, after Sync & Prepare. It takes a while, and it stops the voice first to free the GPU."},
  {tab: "training", targets: ["#compareStep", "#noDatasets"], title: "Compare epochs", text: "Training saves the voice every few epochs. Compare says one test line with each saved epoch, so you can keep the one that sounds best."},
  {tab: "training", targets: ["#trainingModeRestore"], title: "Restore audio", text: "Clean up recordings before training: remove music, echo or noise, and restore missing high frequencies. Each result is added to the list, to restore again or add to a dataset."},
  {tab: "generate", targets: ["#targetText", "#tab-generate .no-voice-note"], title: "Text to Voice", text: "Type what the voice should say.", noVoiceText: "Here you type what a voice should say. It needs a voice first, made on Voice Training."},
  {tab: "generate", voice: true, targets: [".voice-line"], title: "Choose the voice", text: "Pick the voice and a reference clip. The clip's tone shapes how the line sounds, and you can play it here."},
  {tab: "generate", voice: true, targets: [".compose-actions"], title: "Generate", text: "Generate makes the line. The voice engine starts by itself the first time, which takes a moment. The gear opens the models, the text language and finer controls."},
  {tab: "generate", targets: ["#tab-generate .history"], title: "Your lines", text: "Every line you generate is listed here. Click one to play it again."},
  {tab: "relay", targets: ["#relayPttButton", "#relayCapture .no-voice-note"], title: "Mic to Voice", text: "Press Start to turn on speech recognition and the voice engine. Then hold this button or your push-to-talk key, speak, and release. Your words are spoken in the voice.", noVoiceText: "Here you speak, and your words are spoken in a voice. It needs a voice first, made on Voice Training."},
  {tab: "relay", voice: true, targets: ["#relaySetPttKey"], title: "Push-to-talk key", text: "Click to choose another key. It also works while another app, such as a game, is in front."},
  {tab: "relay", voice: true, targets: [".relay-options"], title: "Options", text: "With Hold to talk off, the microphone listens all the time instead. With Speak automatically off, your words are only written down. The gear sets the output device, the volume, the voice, and the languages: the one you speak and the one the voice speaks, into which your words are translated."},
  {tab: "relay", targets: ["#relayHistory"], title: "Recordings", text: "Everything you said in this session is listed here, to read or play again."},
  {targets: ["#systemIsland"], title: "Status bar", text: "Shows whether speech recognition, the voice engine and the global key are running. Start on Mic to Voice starts them, and Stop Everything stops them and frees the GPU."},
  {targets: ["#openGuide"], title: "Take the tour again", text: "Press Guide to see this tour again at any time."},
];
let tour = null;

function startTour() {
  document.querySelectorAll("dialog[open]").forEach(dialog => dialog.close());
  endRelayPush();
  tour = {
    steps: TOUR_STEPS.filter(step => !step.voice || hasVoice()),
    index: 0,
    returnTab: document.body.dataset.tab,
    returnFocus: document.activeElement,
  };
  $("tour").hidden = false;
  showTourStep();
}

function endTour() {
  if (!tour) return;
  const {returnTab, returnFocus} = tour;
  tour = null;
  $("tour").hidden = true;
  activateTab(returnTab);
  refresh();
  if (returnFocus && returnFocus.focus) returnFocus.focus();
}

function moveTour(change) {
  if (!tour) return;
  const index = tour.index + change;
  if (index < 0) return;
  if (index >= tour.steps.length) return endTour();
  tour.index = index;
  showTourStep();
}

function tourTarget(step) {
  for (const selector of step.targets || []) {
    const element = document.querySelector(selector);
    if (element && element.getClientRects().length) return element;
  }
  return null;
}

function showTourStep() {
  const step = tour.steps[tour.index];
  if (step.tab && document.body.dataset.tab !== step.tab) activateTab(step.tab);
  if (step.tab === "training") setTrainingMode("train");
  const target = tourTarget(step);
  if (target) target.scrollIntoView({block: "center", inline: "nearest"});
  $("tourCount").textContent = `${tour.index + 1} of ${tour.steps.length}`;
  $("tourTitle").textContent = step.title;
  $("tourText").textContent = !hasVoice() && step.noVoiceText ? step.noVoiceText : step.text;
  $("tourBack").disabled = tour.index === 0;
  $("tourNext").textContent = tour.index === tour.steps.length - 1 ? "Finish" : "Next";
  placeTour();
  $("tourNext").focus();
}

// The pop-up goes below the element, else above, right or left, else at the bottom of the window.
function placeTour() {
  if (!tour) return;
  const target = tourTarget(tour.steps[tour.index]);
  const spot = $("tourSpot");
  const pop = $("tourPop");
  const gap = 12;
  const pad = 6;
  const width = window.innerWidth;
  const height = window.innerHeight;
  const popWidth = pop.offsetWidth;
  const popHeight = pop.offsetHeight;
  $("tour").classList.toggle("no-spot", !target);
  spot.hidden = !target;
  let top = (height - popHeight) / 2;
  let left = (width - popWidth) / 2;
  if (target) {
    const rect = target.getBoundingClientRect();
    const box = {top: rect.top - pad, left: rect.left - pad, right: rect.right + pad, bottom: rect.bottom + pad};
    Object.assign(spot.style, {top: `${box.top}px`, left: `${box.left}px`, width: `${box.right - box.left}px`, height: `${box.bottom - box.top}px`});
    // The highlight follows the element's corners, so a round button gets a round highlight.
    const radius = getComputedStyle(target).borderTopLeftRadius;
    spot.style.borderRadius = radius.endsWith("%") ? radius : `${parseFloat(radius) + pad}px`;
    const middleX = rect.left + rect.width / 2 - popWidth / 2;
    const middleY = rect.top + rect.height / 2 - popHeight / 2;
    if (box.bottom + gap + popHeight <= height - gap) [top, left] = [box.bottom + gap, middleX];
    else if (box.top - gap - popHeight >= gap) [top, left] = [box.top - gap - popHeight, middleX];
    else if (box.right + gap + popWidth <= width - gap) [top, left] = [middleY, box.right + gap];
    else if (box.left - gap - popWidth >= gap) [top, left] = [middleY, box.left - gap - popWidth];
    else [top, left] = [height - popHeight - gap, middleX];
  }
  pop.style.top = `${Math.round(Math.max(gap, Math.min(top, height - popHeight - gap)))}px`;
  pop.style.left = `${Math.round(Math.max(gap, Math.min(left, width - popWidth - gap)))}px`;
}

$("tourNext").onclick = () => moveTour(1);
$("tourBack").onclick = () => moveTour(-1);
$("tourEnd").onclick = endTour;
// While the tour runs, keys belong to it, so the push-to-talk key does not start a recording.
document.addEventListener("keydown", event => {
  if (!tour) return;
  event.stopPropagation();
  if (event.key === "Escape") { event.preventDefault(); endTour(); }
  else if (event.key === "ArrowRight") { event.preventDefault(); moveTour(1); }
  else if (event.key === "ArrowLeft") { event.preventDefault(); moveTour(-1); }
  else if (event.key === "Tab") {
    // Focus stays in the pop-up.
    event.preventDefault();
    const buttons = [...$("tourPop").querySelectorAll("button:not(:disabled)")];
    const index = buttons.indexOf(document.activeElement);
    buttons[(index + (event.shiftKey ? buttons.length - 1 : 1)) % buttons.length].focus();
  }
}, true);
["mousedown", "pointerdown", "auxclick", "contextmenu"].forEach(name => $("tour").addEventListener(name, event => event.stopPropagation()));
window.addEventListener("resize", placeTour);
window.addEventListener("scroll", placeTour, true);

// Offered once, to a dashboard without datasets. Closing it in any way counts as seen.
function showWelcome() {
  if (welcomeChecked || !state) return;
  welcomeChecked = true;
  let seen = false;
  try { seen = localStorage.getItem("voiceDashboardWelcomeSeen") === "1"; } catch (_) {}
  if (!seen && !(state.datasets || []).length) $("welcomeDialog").showModal();
}

async function post(path, data = {}) {
  const res = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});
  const json = await res.json();
  if (!res.ok || json.ok === false) throw new Error(json.error || "Request failed");
  return json;
}

$("reference").addEventListener("change", () => {
  const ref = state.references.find(r => r.id === $("reference").value);
  if (ref) setReference(ref, true);
  renderAuxRefs(state.all_references);
});
$("auxSearch").addEventListener("input", () => {
  if (state) renderAuxRefs(state.all_references);
});
$("clearAuxRefs").onclick = () => {
  auxSelected.clear();
  if (state) renderAuxRefs(state.all_references);
};

// Starts the voice engine with the Generate page's model selection and waits until it accepts connections.
async function ensureGenerateEngine() {
  if (state && state.api_ready) return true;
  engineBusy = true;
  $("engineStatus").className = "status-line warn";
  $("engineStatus").textContent = "Starting the voice engine, this can take a minute or two...";
  try {
    await post("/api/start-api", {gpt: $("gptModel").value, sovits: $("sovitsModel").value});
    // The server replies immediately after spawning the engine; poll until it accepts connections.
    for (let i = 0; i < 160; i++) {
      await new Promise(resolve => setTimeout(resolve, 1500));
      await refresh();
      if (state && state.api_ready) break;
    }
    const ready = Boolean(state && state.api_ready);
    $("engineStatus").className = ready ? "status-line good" : "status-line bad";
    $("engineStatus").textContent = ready ? "Engine ready" : "Engine did not become ready. Check the Logs page.";
    return ready;
  } catch (e) {
    $("engineStatus").className = "status-line bad";
    $("engineStatus").textContent = "Engine failed to start.";
    throw e;
  } finally {
    engineBusy = false;
  }
}
$("startApi").onclick = async () => {
  $("startApi").disabled = true;
  try { await ensureGenerateEngine(); }
  catch (e) { alert(e.message); }
  finally { $("startApi").disabled = false; refresh(); }
};
$("stopApi").onclick = async () => { await post("/api/stop-api"); $("engineStatus").className = "status-line"; $("engineStatus").textContent = "Engine stopped"; refresh(); };

function renderLog() {
  const filter = $("logFilter").value.trim().toLowerCase();
  const lines = logRaw ? logRaw.split("\n") : [];
  const shown = filter ? lines.filter(line => line.toLowerCase().includes(filter)) : lines;
  const log = $("log");
  log.textContent = shown.join("\n");
  $("logStatus").textContent = filter ? `${shown.length} of ${lines.length} lines` : "";
  if ($("logFollow").checked) log.scrollTop = log.scrollHeight;
}
$("logFilter").addEventListener("input", renderLog);
$("logFollow").addEventListener("change", renderLog);
$("copyLog").onclick = async () => {
  try {
    await navigator.clipboard.writeText(logRaw);
    $("logStatus").textContent = "Copied.";
  } catch (error) {
    $("logStatus").textContent = "Copy failed. Select the text and copy it manually.";
  }
};

function renderGenerateHistory() {
  const list = $("generateHistory");
  list.innerHTML = "";
  const items = generateOutputs.slice().reverse();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent = "Lines you generate appear here.";
    list.appendChild(empty);
  }
  items.forEach(item => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "history-item";
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", item === generateShown ? "true" : "false");
    row.title = item.name;
    const text = document.createElement("span");
    text.className = "history-text";
    text.textContent = item.text;
    const meta = document.createElement("span");
    meta.className = "history-meta";
    const time = document.createElement("span");
    time.textContent = formatClock(item.createdAt);
    meta.append(time);
    row.append(text, meta);
    row.onclick = () => showGeneratedLine(item, true);
    list.appendChild(row);
  });
  $("generateHistoryCount").textContent = items.length ? `${items.length} this session` : "Nothing generated yet";
}

function showGeneratedLine(item, play) {
  generateShown = item;
  $("outputAudio").src = item.url + "?t=" + Date.now();
  $("outputName").textContent = item.name;
  $("generateResult").hidden = false;
  if (play) $("outputAudio").play().catch(() => {});
  renderGenerateHistory();
}

$("generateNewLine").onclick = () => {
  generateShown = null;
  $("targetText").value = "";
  $("generateResult").hidden = true;
  $("generateStatus").textContent = "";
  renderGenerateHistory();
  $("targetText").focus();
};
$("openGenerateSettings").onclick = () => $("generateSettingsDialog").showModal();
renderGenerateHistory();
$("relayReference").addEventListener("change", () => {
  const ref = relayRef();
  if (ref) setRelayReference(ref);
  renderRelayAuxRefs();
});
$("relayAuxSearch").addEventListener("input", renderRelayAuxRefs);
$("relayClearAuxRefs").onclick = () => {
  relayAuxSelected.clear();
  renderRelayAuxRefs();
};
$("relayVolume").addEventListener("input", updateRelayVolumeLabel);
$("refreshRelayDevices").onclick = refreshRelayDevices;
$("relayOutputDevice").addEventListener("change", applyRelayOutputDevice);
$("relaySetPttKey").onclick = () => startPttBinding();
$("relayPushToTalk").addEventListener("change", () => {
  if (!$("relayPushToTalk").checked) endRelayPush();
  relayStatus($("relayPushToTalk").checked ? "push-to-talk ready" : "listening local");
});
$("relayAutoSpeak").addEventListener("change", renderRelayCenter);
$("relayPttButton").addEventListener("pointerdown", event => {
  event.preventDefault();
  // Keep pointer events on the button while held, even if the pointer drifts off it.
  try { event.currentTarget.setPointerCapture(event.pointerId); } catch (_) {}
  beginRelayPush();
});
// Voice-activated mode: the button starts the microphone, since nothing is held.
$("relayPttButton").addEventListener("click", () => {
  if ($("relayPushToTalk").checked || relayListening) return;
  try { startRelayListening(); }
  catch (error) { showError(error.message); }
});
// Keyboard users can hold Space or Enter on the focused button, mirroring pointer hold and release.
$("relayPttButton").addEventListener("keydown", event => {
  if (event.code !== "Space" && event.code !== "Enter") return;
  if (event.code === $("relayPttKey").value) return;
  event.preventDefault();
  if (event.repeat) return;
  if (!$("relayPushToTalk").checked) {
    if (!relayListening) $("relayPttButton").click();
    return;
  }
  beginRelayPush();
});
$("relayPttButton").addEventListener("keyup", event => {
  if (event.code !== "Space" && event.code !== "Enter") return;
  if (event.code === $("relayPttKey").value) return;
  event.preventDefault();
  endRelayPush();
});
$("relayNewRecording").onclick = () => {
  showCaptureView();
  $("relayPttButton").focus();
};
$("historyToggle").onclick = () => {
  const open = $("relayHistory").classList.toggle("open");
  $("historyToggle").setAttribute("aria-expanded", open ? "true" : "false");
};
$("openRelaySettings").onclick = () => $("relaySettingsDialog").showModal();
$("openGuide").onclick = startTour;
$("welcomeDialog").addEventListener("close", () => {
  try { localStorage.setItem("voiceDashboardWelcomeSeen", "1"); } catch (_) {}
});
$("welcomeNo").onclick = () => $("welcomeDialog").close();
$("welcomeYes").onclick = () => {
  $("welcomeDialog").close();
  startTour();
};
$("relayPlayThis").onclick = () => {
  const item = viewedRelayItem();
  if (item && item.url) playRelayUrl(item.url, item.name);
};
["pointerup", "pointercancel", "pointerleave"].forEach(name => {
  $("relayPttButton").addEventListener(name, event => {
    event.preventDefault();
    endRelayPush();
  });
});
document.addEventListener("keydown", event => {
  if (relayBindingPtt) {
    event.preventDefault();
    if (event.code === "Escape") {
      cancelPttBinding();
      return;
    }
    relayBindingPtt = false;
    setRelayPttKey(event.code);
    relayStatus(`push-to-talk key: ${describePttKey(event.code)}`);
    return;
  }
  const tag = (event.target && event.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "select" || tag === "textarea") return;
  if (event.code === $("relayPttKey").value && !event.repeat) {
    event.preventDefault();
    beginRelayPush();
  }
});

$("startSystem").onclick = () => startSystem();

$("stopSystem").onclick = async () => {
  $("stopSystem").disabled = true;
  armRelayWhenReady = false;
  startToken += 1;
  stopRelayListening();
  try {
    await post("/api/stop-system");
  } catch (error) {
    showError(error.message);
  } finally {
    $("stopSystem").disabled = false;
    await refreshSystem();
  }
};

$("runSetupCheck").onclick = runSetupCheck;

RELAY_SETTING_IDS.forEach(id => {
  const element = $(id);
  if (!element) return;
  element.addEventListener(element.type === "range" ? "input" : "change", saveRelaySettings);
});

const relayProfiles = {
  fast: {relayTopK: "10", relayTopP: "0.5", relayTemperature: "0.45", relaySpeed: "1", relayPause: "0.15", relayRepPenalty: "1.4"},
  balanced: {relayTopK: "15", relayTopP: "0.6", relayTemperature: "0.55", relaySpeed: "0.95", relayPause: "0.25", relayRepPenalty: "1.35"},
  quality: {relayTopK: "20", relayTopP: "0.7", relayTemperature: "0.6", relaySpeed: "0.95", relayPause: "0.3", relayRepPenalty: "1.3"}
};
document.querySelectorAll("[data-relay-profile]").forEach(button => {
  button.onclick = () => {
    Object.entries(relayProfiles[button.dataset.relayProfile] || {}).forEach(([id, value]) => { $(id).value = value; });
    saveRelaySettings();
    recentActivity(`${button.textContent.trim()} preset applied.`);
  };
});

const pttEvents = new EventSource("/api/events");
pttEvents.addEventListener("ptt", event => {
  try {
    const ptt = JSON.parse(event.data);
    if (!ptt.held) endRelayPush();
    else if (!tour) beginRelayPush();
  } catch (error) {
    console.error(error);
  }
});
document.addEventListener("keyup", event => {
  if (event.code === $("relayPttKey").value) {
    event.preventDefault();
    endRelayPush();
  }
});
document.addEventListener("mousedown", event => {
  if (relayBindingPtt) {
    event.preventDefault();
    if (event.target && event.target.closest && event.target.closest("#relaySetPttKey")) {
      // Clicking the shortcut control again cancels capture instead of binding that mouse button.
      cancelPttBinding();
      return;
    }
    relayBindingPtt = false;
    const code = pttMouseCode(event);
    setRelayPttKey(code);
    relayStatus(`push-to-talk key: ${describePttKey(code)}`);
    return;
  }
  const tag = (event.target && event.target.tagName || "").toLowerCase();
  if (event.button < 3 && (tag === "input" || tag === "select" || tag === "textarea" || tag === "button")) return;
  if (pttMouseCode(event) === $("relayPttKey").value) {
    event.preventDefault();
    beginRelayPush();
  }
});
document.addEventListener("mouseup", event => {
  if (pttMouseCode(event) === $("relayPttKey").value) {
    event.preventDefault();
    endRelayPush();
  }
});
document.addEventListener("auxclick", event => {
  if (relayBindingPtt || pttMouseCode(event) === $("relayPttKey").value) event.preventDefault();
});
document.addEventListener("contextmenu", event => {
  if (relayBindingPtt || $("relayPttKey").value === "Mouse2") event.preventDefault();
});
setRelayPttKey(localStorage.getItem("voiceDashboardRelayPttKey") || $("relayPttKey").value);
updateRelayVolumeLabel();
autoLoadRelayDevices();
$("relaySpeakManual").onclick = speakTypedRelayLine;
["relaySpoken", "relaySpeaks"].forEach(id => $(id).addEventListener("change", renderRelayLanguageNote));
$("relayClearQueue").onclick = () => {
  relayQueue = relayQueue.filter(item => item.status !== "queued");
  if (!viewedRelayItem()) relayView = {mode: "capture", itemId: null};
  renderRelayQueue();
  relayStatus(relayListening ? "push-to-talk ready" : "idle");
};
$("relayCancelCurrent").onclick = async () => {
  const item = relayCurrentItem;
  if (!item) return;
  item.status = "cancelled";
  renderRelayQueue();
  $("relayAudio").pause();
  if (relayPlaybackResolve) {
    relayPlaybackResolve();
    relayPlaybackResolve = null;
  }
  $("relayCancelCurrent").classList.remove("visible");
  relayStatus("current line cancelled");
  try { await post("/api/cancel-generation", {generation_id: item.generationId}); }
  catch (error) { showError(error.message); }
};
$("relayReplayLast").onclick = () => {
  const item = viewedRelayItem();
  if (item && item.url) playRelayUrl(item.url, item.name);
  else if (relayLastUrl) playRelayUrl(relayLastUrl, "replay");
};
renderRelayQueue();
$("trainingDataset").onchange = () => {
  localStorage.setItem("voiceDashboardTrainingDataset", $("trainingDataset").value);
  datasetUploadFiles = [];
  $("datasetAudioFiles").value = "";
  renderDatasetUploads();
  fillTrainingDatasets(state ? state.datasets : []);
  loadClips(true);
};
$("trainingModelName").value = localStorage.getItem("voiceDashboardTrainingModelName") || $("trainingModelName").value;
const updateTrainingModelName = () => {
  const slug = modelSlug($("trainingModelName").value);
  $("trainingModelSlug").textContent = slug;
  localStorage.setItem("voiceDashboardTrainingModelName", $("trainingModelName").value);
};
$("trainingModelName").oninput = updateTrainingModelName;
updateTrainingModelName();

$("createDataset").onclick = async () => {
  $("createDataset").disabled = true;
  try {
    const result = await post("/api/create-dataset", {
      name: $("newDatasetName").value,
      speaker: $("newDatasetSpeaker").value,
      language: $("newDatasetLanguage").value
    });
    localStorage.setItem("voiceDashboardTrainingDataset", result.dataset.id);
    newTrainingDataset = {id: result.dataset.id, name: result.dataset.name};
    fillTrainingDatasets(state ? state.datasets : []);
    datasetUploadFiles = [];
    renderDatasetUploads();
    $("newDatasetName").value = "";
    $("createDatasetDialog").close();
    recentActivity(`Dataset ${result.dataset.name} created.`);
    await refresh();
  } catch (error) {
    showError(error.message);
  } finally {
    $("createDataset").disabled = false;
  }
};

// Recordings longer than this are cut into clips at pauses on the server; shorter ones are clips.
const CUT_LONGER_THAN = 20;

function audioSeconds(file) {
  // The length of an audio file, or 0 when the browser cannot read it.
  return new Promise(resolve => {
    const url = URL.createObjectURL(file);
    const audio = new Audio();
    const timer = setTimeout(() => done(0), 10000);
    function done(seconds) {
      clearTimeout(timer);
      audio.removeAttribute("src");
      URL.revokeObjectURL(url);
      resolve(Number.isFinite(seconds) ? seconds : 0);
    }
    audio.preload = "metadata";
    audio.onloadedmetadata = () => done(audio.duration);
    audio.onerror = () => done(0);
    audio.src = url;
  });
}

let cutSending = "";
const UPLOAD_LIMIT = 4 * 1024 ** 3;
async function sendToCut(file, datasetId) {
  if (file.size > UPLOAD_LIMIT) throw new Error("Files over 4 GB cannot be sent. Cut the recording into shorter files first.");
  const query = `dataset=${encodeURIComponent(datasetId)}&filename=${encodeURIComponent(file.name)}`;
  const response = await fetch(`/api/dataset-cut?${query}`, {method: "POST", headers: {"Content-Type": "application/octet-stream"}, body: file});
  const result = await response.json().catch(() => ({}));
  if (!response.ok || result.ok === false) throw new Error(result.error || "The upload failed.");
}

// A voice as one file: both models and the reference clips.
const importVoiceFile = document.createElement("input");
importVoiceFile.type = "file";
importVoiceFile.accept = ".zip";
importVoiceFile.hidden = true;
document.body.appendChild(importVoiceFile);
function shareStatus(text, kind = "") {
  $("shareStatus").className = `status-line ${kind}`;
  $("shareStatus").textContent = text;
}
document.querySelectorAll("[data-import-voice]").forEach(button => { button.onclick = () => importVoiceFile.click(); });
importVoiceFile.onchange = async () => {
  const file = importVoiceFile.files[0];
  importVoiceFile.value = "";
  if (!file) return;
  if (file.size > UPLOAD_LIMIT) return showError("This file is over 4 GB, far larger than a voice file.");
  const buttons = Array.from(document.querySelectorAll("[data-import-voice]"));
  buttons.forEach(button => { button.disabled = true; button.textContent = "Importing..."; });
  shareStatus(`Importing ${file.name}...`);
  recentActivity(`Importing the voice ${file.name}...`);
  try {
    const response = await fetch("/api/voice-import", {method: "POST", headers: {"Content-Type": "application/zip"}, body: file});
    const result = await response.json().catch(() => ({}));
    if (!response.ok || result.ok === false) throw new Error(result.error || "The import failed.");
    // The imported voice becomes the voice of Text to Voice and Mic to Voice.
    voiceDataset = result.dataset;
    try { localStorage.setItem("voiceDashboardVoiceDataset", voiceDataset); } catch (_) {}
    refsSignature = "";
    modelsSignature = "";
    await refresh();
    const pairs = [["gptModel", result.gpt], ["sovitsModel", result.sovits], ["relayGptModel", result.gpt], ["relaySovitsModel", result.sovits]];
    const found = pairs.every(([id, path]) => Array.from($(id).options).some(item => item.value === path));
    pairs.forEach(([id, path]) => {
      if (Array.from($(id).options).some(item => item.value === path)) $(id).value = path;
    });
    saveRelaySettings();
    shareStatus(found ? `Imported ${result.name}. Text to Voice and Mic to Voice now use it.` : `Imported ${result.name}. Choose its models in Settings.`, "good");
    recentActivity(`Imported the voice ${result.name}.`);
  } catch (error) {
    shareStatus("");
    showError(error.message);
  } finally {
    buttons.forEach(button => { button.disabled = false; button.textContent = "Import a voice"; });
  }
};
$("exportVoice").onclick = async () => {
  $("exportVoice").disabled = true;
  shareStatus("Packing the voice...");
  try {
    const result = await post("/api/voice-export", {gpt: $("gptModel").value, sovits: $("sovitsModel").value, dataset_id: voiceDataset,
      reference: $("reference").value, prompt_text: $("promptText").value, extra_references: Array.from(auxSelected)});
    const link = document.createElement("a");
    link.href = result.url;
    link.download = result.name;
    document.body.append(link);
    link.click();
    link.remove();
    shareStatus(`Saved as ${result.name}.`, "good");
  } catch (error) {
    shareStatus("");
    showError(error.message);
  } finally {
    $("exportVoice").disabled = false;
  }
};

$("speechApiAddress").textContent = `${location.origin}/v1`;
$("compareKind").onchange = renderCompare;
// The version chosen last is kept, as Compare epochs looks for SoVITS epochs of that version.
try { $("sovitsVersion").value = localStorage.getItem("voiceDashboardSovitsVersion") || "v2Pro"; } catch (_) {}
if (!$("sovitsVersion").value) $("sovitsVersion").value = "v2Pro";
$("sovitsVersion").addEventListener("change", () => {
  try { localStorage.setItem("voiceDashboardSovitsVersion", $("sovitsVersion").value); } catch (_) {}
  renderCompare();
});
$("trainingModelName").addEventListener("input", renderCompare);
$("compareRun").onclick = async () => {
  $("compareRun").disabled = true;
  try {
    await post("/api/compare-epochs", {model: $("trainingModelName").value, kind: $("compareKind").value, dataset_id: $("trainingDataset").value,
      reference: $("compareReference").value, text: $("compareText").value, version: $("sovitsVersion").value});
    compareRowsSignature = null;
    compareChosen = "";
    $("compareResults").innerHTML = "";
    $("compareStatus").className = "status-line";
    $("compareStatus").textContent = "Starting...";
    refresh();
  } catch (error) {
    showError(error.message);
    renderCompare();
  }
};

$("datasetAddFiles").onclick = () => $("datasetAudioFiles").click();
$("datasetAudioFiles").onchange = async () => {
  // Clips join the list, which can also hold recordings from Restore audio; long recordings
  // are cut into clips on the server.
  const picked = Array.from($("datasetAudioFiles").files || []);
  $("datasetAudioFiles").value = "";
  const datasetId = $("trainingDataset").value;
  const clips = [];
  const long = [];
  // A length the browser cannot read is 0; ffmpeg on the server reads more formats.
  for (const file of picked) {
    const seconds = await audioSeconds(file);
    (seconds === 0 || seconds > CUT_LONGER_THAN ? long : clips).push(file);
  }
  if ($("trainingDataset").value !== datasetId) return;
  if (clips.length) {
    datasetUploadFiles.push(...clips);
    renderDatasetUploads();
    queueTranscripts(clips);
  }
  for (const file of long) {
    cutSending = file.name;
    $("datasetCutStatus").className = "status-line";
    $("datasetCutStatus").textContent = `Sending ${file.name}...`;
    try {
      await sendToCut(file, datasetId);
    } catch (error) {
      showError(`${file.name}: ${error.message}`);
    } finally {
      cutSending = "";
    }
  }
  if (long.length) refresh();
};

// Clips the server has cut join the list of files to check, with their transcripts, once each.
const cutTaken = new Set();
async function takeCutClips(job) {
  for (const clip of job.results || []) {
    if (cutTaken.has(clip.url) || job.dataset !== $("trainingDataset").value) continue;
    cutTaken.add(clip.url);
    try {
      const response = await fetch(clip.url);
      if (!response.ok) continue;
      const file = new File([await response.blob()], clip.name, {type: "audio/wav"});
      if (job.dataset !== $("trainingDataset").value) continue;
      datasetUploadTexts.set(file, clip.text);
      datasetUploadFiles.push(file);
      renderDatasetUploads();
    } catch (_) {
      cutTaken.delete(clip.url);
    }
  }
}

function renderCutStatus() {
  const job = state && state.jobs && state.jobs["dataset-cut"];
  if (job) takeCutClips(job);
  const mine = Boolean(job && job.dataset === $("trainingDataset").value);
  const running = Boolean(mine && job.status === "running");
  $("datasetCutStop").hidden = !running;
  // A recording on its way to the server says so until it has arrived.
  if (cutSending) return;
  if (!mine) {
    $("datasetCutStatus").textContent = "";
    return;
  }
  const last = (job.lines || []).slice(-1)[0] || "Starting...";
  $("datasetCutStatus").textContent = running ? last : (job.status === "stopped" ? "Cutting stopped." : (job.status === "failed" ? `${last} See Logs for details.` : last));
  $("datasetCutStatus").className = job.status === "failed" ? "status-line bad" : "status-line";
}

$("uploadDatasetAudio").onclick = async () => {
  if (uploadBusy || !datasetUploadFiles.length) return;
  if (datasetUploadFiles.some(file => !(datasetUploadTexts.get(file) || "").trim())) return showError("Every selected audio file needs a transcript.");
  const datasetId = $("trainingDataset").value;
  const total = datasetUploadFiles.length;
  uploadBusy = true;
  updateUploadButton();
  let added = 0;
  try {
    // The list as it is now: a file removed meanwhile is not sent, and a transcript edited
    // meanwhile is sent as it is. Switching datasets empties the list, which ends the upload.
    while (datasetUploadFiles.length && $("trainingDataset").value === datasetId) {
      const file = datasetUploadFiles[0];
      $("datasetUploadProgress").className = "status-line";
      $("datasetUploadProgress").textContent = `Adding ${added + 1} of ${total}: ${file.name}`;
      const audio = await fileToBase64(file);
      if (!datasetUploadFiles.includes(file)) continue;
      await post("/api/upload-dataset-audio", {
        dataset_id: datasetId,
        filename: file.name,
        audio_base64: audio,
        text: (datasetUploadTexts.get(file) || "").trim(),
        speaker: "",
        language: ""
      });
      added++;
      // An added file leaves the list, so a failure later does not add it twice. Only its row
      // goes, so typing in another row goes on.
      datasetUploadFiles = datasetUploadFiles.filter(other => other !== file);
      if (clipPlaying === file) stopClip();
      const input = datasetUploadInputs.get(file);
      if (input) input.parentElement.remove();
      $("datasetUploadHint").hidden = !datasetUploadFiles.length;
    }
    $("datasetUploadProgress").textContent = `${added} file${added === 1 ? "" : "s"} added.`;
  } catch (error) {
    $("datasetUploadProgress").className = "status-line bad";
    $("datasetUploadProgress").textContent = `${added} of ${total} added. The rest are still in the list.`;
    showError(error.message);
  } finally {
    uploadBusy = false;
    updateUploadButton();
    if (added) await refresh();
  }
};

$("syncDataset").onclick = async () => {
  try {
    await post("/api/sync-dataset", {dataset_id: $("trainingDataset").value});
    $("datasetPrepareStatus").textContent = "Dataset sync started.";
    refresh();
  } catch (error) { showError(error.message); }
};
$("prepareDataset").onclick = async () => {
  $("prepareDataset").disabled = true;
  try {
    await post("/api/prepare-dataset", {dataset_id: $("trainingDataset").value, model_name: $("trainingModelName").value});
    $("datasetPrepareStatus").textContent = `Preparing features for ${modelSlug($("trainingModelName").value)}...`;
    refresh();
  } catch (error) { showError(error.message); }
  finally { $("prepareDataset").disabled = false; }
};
["voiceDataset", "relayDataset"].forEach(id => $(id).addEventListener("change", () => {
  voiceDataset = $(id).value;
  try { localStorage.setItem("voiceDashboardVoiceDataset", voiceDataset); } catch (_) {}
  refsSignature = "";
  refresh();
}));
$("trainSovits").onclick = async () => {
  try {
    await post("/api/start-train", {kind: "sovits", dataset_id: $("trainingDataset").value, model_name: $("trainingModelName").value, version: $("sovitsVersion").value, epochs: $("sovitsEpochs").value, batch_size: $("sovitsBatch").value, save_every: $("sovitsSave").value});
    refresh();
  } catch (error) { showError(error.message); }
};
$("trainGpt").onclick = async () => {
  try {
    await post("/api/start-train", {kind: "gpt", dataset_id: $("trainingDataset").value, model_name: $("trainingModelName").value, epochs: $("gptEpochs").value, batch_size: $("gptBatch").value, save_every: $("gptSave").value});
    refresh();
  } catch (error) { showError(error.message); }
};
document.querySelectorAll("[data-stop]").forEach(btn => {
  btn.onclick = async () => {
    btn.disabled = true;
    try {
      await post("/api/stop-job", {key: btn.dataset.stop});
    } catch (error) {
      showError(error.message);
    } finally {
      btn.disabled = false;
      refresh();
    }
  };
});
$("generate").onclick = async () => {
  const ref = state.references.find(r => r.id === $("reference").value);
  if (!ref) return alert("Pick a reference clip.");
  $("generate").disabled = true;
  $("generateStatus").className = "status-line";
  if (!(state && state.api_ready)) {
    $("generateStatus").textContent = state && state.gpu_busy
      ? "Starting the voice engine first. Training is using the same GPU, so both can run slower or run out of memory."
      : "Starting the voice engine first...";
    try {
      if (!await ensureGenerateEngine()) throw new Error("The voice engine did not become ready.");
    } catch (e) {
      $("generateStatus").className = "status-line bad";
      $("generateStatus").textContent = "Engine failed to start.";
      $("generate").disabled = false;
      return alert(e.message);
    }
  }
  $("generateStatus").textContent = "Generating...";
  const aux = Array.from(auxSelected)
    .map(id => state.all_references.find(r => r.id === id))
    .filter(Boolean)
    .filter(r => r.valid_aux_reference)
    .filter(r => r.id !== ref.id)
    .map(r => r.wsl_path);
  try {
    const result = await post("/api/generate", {
      gpt: $("gptModel").value,
      sovits: $("sovitsModel").value,
      ref_audio_path: ref.wsl_path,
      aux_ref_audio_paths: aux,
      prompt_text: $("promptText").value,
      prompt_lang: ref.lang,
      text: $("targetText").value,
      text_lang: $("textLang").value,
      text_split_method: $("splitMethod").value,
      seed: $("seed").value,
      top_k: $("topK").value,
      top_p: $("topP").value,
      temperature: $("temperature").value,
      speed_factor: $("speed").value,
      fragment_interval: $("pause").value,
      repetition_penalty: $("repPenalty").value,
      voice: voiceDataset
    });
    const line = {text: $("targetText").value.trim(), url: result.url, name: result.name, createdAt: Date.now()};
    generateOutputs.push(line);
    showGeneratedLine(line, false);
    $("generateStatus").className = "status-line good";
    $("generateStatus").textContent = "Done.";
  } catch (e) {
    $("generateStatus").className = "status-line bad";
    $("generateStatus").textContent = "Failed.";
    alert(e.message);
  } finally {
    $("generate").disabled = false;
    refresh();
  }
};

// Restore audio: LavaSR's noise remover and high-frequency restoration, for recordings that
// can then be added to a dataset. Every result is a new recording in the list, so the steps can
// be combined: for example noise removal first, then high frequencies.
let restoreFocus = "";
const restoreChecked = new Set();
let restoreListSignature = "";
// The result to show when it appears, if the view is still on the recording it is made from.
let restoreAwait = null;
const RESTORE_SETTING_IDS = ["restoreMusic", "restoreMusicModel", "restoreEcho", "restoreDenoise", "restoreEnhance", "restoreQuality", "restoreKeep", "restoreModel", "restorePieces"];
// The options that need restore_env, which Setup-Restore-Env.bat creates.
const LAVASR_IDS = ["restoreDenoise", "restoreEnhance", "restoreQuality", "restoreKeep", "restoreModel", "restorePieces"];

function setTrainingMode(mode) {
  $("tab-training").dataset.mode = mode;
  document.querySelectorAll(".mode-tabs [data-mode]").forEach(button => {
    button.setAttribute("aria-selected", button.dataset.mode === mode ? "true" : "false");
  });
  try { localStorage.setItem("voiceDashboardTrainingMode", mode); } catch (_) {}
}

function restoreItems() {
  return (state && state.restore && state.restore.items) || [];
}

function restoreItem(id) {
  return restoreItems().find(item => item.id === id) || null;
}

function restoreTargets() {
  return restoreChecked.size ? [...restoreChecked] : (restoreFocus ? [restoreFocus] : []);
}

function describeRestore(item) {
  return item.steps && item.steps.length ? item.steps[item.steps.length - 1] : `original, ${Number(item.sample_rate) / 1000} kHz`;
}

function renderRestore() {
  const restore = (state && state.restore) || {installed: false, items: []};
  const items = restore.items || [];
  const job = state && state.jobs && state.jobs.restore;
  const status = job ? job.status : "";
  if (restoreAwait && restoreFocus !== restoreAwait.from) restoreAwait = null;
  if (restoreAwait && items.some(item => item.id === restoreAwait.to)) {
    restoreFocus = restoreAwait.to;
    restoreAwait = null;
  }
  if (!items.some(item => item.id === restoreFocus)) restoreFocus = items.length ? items[0].id : "";
  [...restoreChecked].forEach(id => { if (!items.some(item => item.id === id)) restoreChecked.delete(id); });
  // Music and echo removal run in WSL, so the tab works before restore_env is installed.
  $("restoreMissing").hidden = Boolean(restore.installed);
  LAVASR_IDS.forEach(id => { $(id).disabled = !restore.installed; });
  $("restoreEmpty").hidden = items.length > 0;
  $("restoreSteps").hidden = !items.length;
  // Redrawn only when the list or the recording in view changes, so focus stays put.
  const signature = restoreFocus + "|" + items.map(item => item.id).join(",");
  if (signature !== restoreListSignature) {
    restoreListSignature = signature;
    renderRestoreList(items);
  }
  renderRestoreFocus();
  const running = status === "running";
  $("restoreRun").disabled = running || !restoreFocus;
  if (job) {
    const progress = (job.lines || []).filter(line => /^(Loading|Downloading|Separated|Restored|Failed|Done|\d+ of \d+ file)/.test(line)).slice(-1)[0] || "Starting...";
    const failed = `${/ failed\.$/.test(progress) ? progress : "Failed."} See Logs for details.`;
    $("restoreStatus").textContent = running ? progress : (status === "done" ? "Done." : (status === "failed" ? failed : "Stopped."));
    $("restoreStatus").className = status === "failed" ? "status-line bad" : (status === "done" ? "status-line good" : "status-line");
  }
}

function renderRestoreList(items) {
  const list = $("restoreList");
  list.innerHTML = "";
  items.forEach(item => {
    const row = document.createElement("div");
    row.className = "restore-row";
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = restoreChecked.has(item.id);
    check.setAttribute("aria-label", `Select ${item.name}`);
    check.onchange = () => {
      if (check.checked) restoreChecked.add(item.id);
      else restoreChecked.delete(item.id);
      renderRestoreFocus();
    };
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    button.setAttribute("aria-current", item.id === restoreFocus ? "true" : "false");
    button.setAttribute("aria-selected", item.id === restoreFocus ? "true" : "false");
    const name = document.createElement("span");
    name.className = "history-text";
    name.textContent = item.name;
    const meta = document.createElement("span");
    meta.className = "history-meta";
    const what = document.createElement("span");
    what.textContent = describeRestore(item);
    const time = document.createElement("span");
    time.textContent = formatClock((item.created || 0) * 1000);
    meta.append(what, time);
    button.append(name, meta);
    button.onclick = () => {
      restoreFocus = item.id;
      renderRestore();
    };
    row.append(check, button);
    list.appendChild(row);
  });
}

function renderRestoreFocus() {
  const items = restoreItems();
  const selected = restoreChecked.size;
  $("restoreCount").textContent = items.length
    ? `${items.length} recording${items.length === 1 ? "" : "s"}${selected ? `, ${selected} selected` : ""}`
    : "No audio yet";
  const item = restoreItem(restoreFocus);
  if (!item) return;
  const parent = item.made_from ? restoreItem(item.made_from) : null;
  $("restoreTitle").textContent = item.name;
  $("restoreInfo").textContent = `${Number(item.sample_rate) / 1000} kHz · ${item.duration} s`;
  $("restoreParentBlock").hidden = !parent;
  if (!parent) $("restoreParent").pause();
  $("restoreItemLabel").textContent = item.made_from ? "After" : "Recording";
  // Only a different recording is loaded, so a refresh does not stop playback.
  if (parent && $("restoreParent").dataset.item !== parent.id) {
    $("restoreParent").src = parent.url;
    $("restoreParent").dataset.item = parent.id;
  }
  if ($("restoreItem").dataset.item !== item.id) {
    $("restoreItem").src = item.url;
    $("restoreItem").dataset.item = item.id;
  }
  let history = item.steps && item.steps.length ? `Made by: ${item.steps.join(", then ")}.` : "Not restored yet.";
  if (item.made_from && !parent) history += " The recording it was made from was deleted.";
  $("restoreHistory").textContent = history;
  $("restoreRun").textContent = selected ? `Restore ${selected} selected` : "Restore";
  $("restoreToDataset").textContent = selected ? `Add ${selected} to dataset` : "Add to dataset";
  $("restoreDelete").textContent = selected ? `Delete ${selected}` : "Delete";
}

function restoreSettings() {
  return {
    music: $("restoreMusic").checked ? $("restoreMusicModel").value : "",
    echo: $("restoreEcho").checked,
    denoise: $("restoreDenoise").checked && !$("restoreDenoise").disabled,
    enhance: $("restoreEnhance").checked && !$("restoreEnhance").disabled,
    quality: $("restoreQuality").value,
    keep_below: $("restoreKeep").value ? Number($("restoreKeep").value) * 1000 : null,
    model: $("restoreModel").value,
    pieces: $("restorePieces").checked,
  };
}

document.querySelectorAll(".mode-tabs [data-mode]").forEach(button => {
  button.addEventListener("click", () => setTrainingMode(button.dataset.mode));
});
let savedTrainingMode = "train";
try { savedTrainingMode = localStorage.getItem("voiceDashboardTrainingMode") || "train"; } catch (_) {}
setTrainingMode(savedTrainingMode === "restore" ? "restore" : "train");

try {
  const saved = JSON.parse(localStorage.getItem("voiceDashboardRestoreSettings") || "{}") || {};
  RESTORE_SETTING_IDS.forEach(id => {
    if (saved[id] === undefined) return;
    if ($(id).type === "checkbox") $(id).checked = Boolean(saved[id]);
    else $(id).value = saved[id];
  });
} catch (_) {}
RESTORE_SETTING_IDS.forEach(id => $(id).addEventListener("change", () => {
  const saved = {};
  RESTORE_SETTING_IDS.forEach(key => { saved[key] = $(key).type === "checkbox" ? $(key).checked : $(key).value; });
  try { localStorage.setItem("voiceDashboardRestoreSettings", JSON.stringify(saved)); } catch (_) {}
}));

$("restoreAddButton").onclick = () => $("restoreFiles").click();
$("restoreAddEmpty").onclick = () => $("restoreFiles").click();
$("restoreFiles").onchange = async () => {
  const files = Array.from($("restoreFiles").files || []);
  $("restoreFiles").value = "";
  for (let index = 0; index < files.length; index++) {
    $("restoreCount").textContent = `Adding ${index + 1} of ${files.length}...`;
    try {
      const result = await post("/api/restore-upload", {filename: files[index].name, audio_base64: await fileToBase64(files[index])});
      restoreFocus = result.item.id;
    } catch (error) {
      showError(`${files[index].name}: ${error.message}`);
    }
  }
  await refresh();
};
$("restoreRun").onclick = async () => {
  $("restoreRun").disabled = true;
  try {
    const result = await post("/api/restore-run", {item_ids: restoreTargets(), settings: restoreSettings()});
    const made = (result.results || {})[restoreFocus];
    restoreAwait = made ? {from: restoreFocus, to: made} : null;
    $("restoreStatus").className = "status-line";
    $("restoreStatus").textContent = "Starting...";
    refresh();
  } catch (error) {
    $("restoreRun").disabled = false;
    showError(error.message);
  }
};
$("restoreToDataset").onclick = async () => {
  if (!state || !(state.datasets || []).length) return showError("Create a dataset first, on the Train tab.");
  const chosen = restoreTargets().map(restoreItem).filter(Boolean);
  const long = chosen.filter(item => Number(item.duration) > CUT_LONGER_THAN);
  const items = chosen.filter(item => !long.includes(item));
  const files = [];
  $("restoreToDataset").disabled = true;
  try {
    if (long.length) await post("/api/dataset-cut-restored", {dataset_id: $("trainingDataset").value, item_ids: long.map(item => item.id)});
    for (const item of items) {
      const response = await fetch(item.url);
      if (!response.ok) throw new Error(`${item.name} could not be loaded. It may have been deleted.`);
      files.push(new File([await response.blob()], `${item.name}.wav`, {type: "audio/wav"}));
    }
  } catch (error) {
    return showError(error.message);
  } finally {
    $("restoreToDataset").disabled = false;
  }
  datasetUploadFiles.push(...files);
  renderDatasetUploads();
  queueTranscripts(files);
  setTrainingMode("train");
  if (long.length) refresh();
  $("clipsStep").scrollIntoView({block: "start"});

};
$("restoreDownload").onclick = () => {
  const item = restoreItem(restoreFocus);
  if (!item) return;
  const link = document.createElement("a");
  link.href = item.url;
  link.download = `${item.name}${item.made_from ? "-restored" : ""}.wav`;
  document.body.append(link);
  link.click();
  link.remove();
};
$("restoreDelete").onclick = async () => {
  const ids = restoreTargets();
  if (!ids.length || !confirm(`Delete ${ids.length === 1 ? "this recording" : `${ids.length} recordings`}? This cannot be undone.`)) return;
  ["restoreItem", "restoreParent"].forEach(id => {
    $(id).pause();
    $(id).removeAttribute("src");
    $(id).load();
    delete $(id).dataset.item;
  });
  for (const id of ids) {
    try { await post("/api/restore-delete", {item_id: id}); }
    catch (error) { showError(error.message); }
  }
  restoreChecked.clear();
  restoreFocus = "";
  refresh();
};

refresh();
refreshSystem();
setInterval(refresh, 5000);
setInterval(refreshSystem, 750);
