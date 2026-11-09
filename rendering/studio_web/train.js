"use strict";

const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const draftKey = "inksy-worksheet-v1";
const worksheet = document.getElementById("worksheet");
const form = document.getElementById("worksheet-form");
const name = document.getElementById("style-name");
const train = document.getElementById("train");
const status = document.getElementById("training-status");
const progress = document.getElementById("training-progress");
const error = document.getElementById("training-error");
const openStyle = document.getElementById("open-style");
const samples = Array(alphabet.length * 2).fill(null);
const canvases = [];
const photoInput = document.getElementById("worksheet-photo");
const photoPreview = document.getElementById("photo-preview");
const photoCanvas = document.getElementById("photo-canvas");
const photoStatus = document.getElementById("photo-status");
const importPhoto = document.getElementById("import-photo");
const resetMarkers = document.getElementById("reset-markers");
const sourcePhoto = document.createElement("canvas");
const markerNames = ["top left", "top right", "bottom right", "bottom left"];
const markers = [];
let photoImported = false;
let importing = false;
let busy = false;

function refresh() {
  const filled = samples.filter(Boolean).length;
  document.getElementById("worksheet-count").textContent = filled + " / " + samples.length + " samples";
  train.disabled = busy || importing || filled !== samples.length || !name.value.trim();
  name.disabled = busy;
  photoInput.disabled = busy || importing;
  resetMarkers.disabled = busy || importing;
  importPhoto.disabled = busy || importing || markers.length !== 4;
}

function saveDraft() {
  try {
    localStorage.setItem(draftKey, JSON.stringify({name: name.value, samples, photoImported}));
  } catch {
    status.textContent = "Draft storage is unavailable. Keep this tab open.";
  }
  refresh();
}

function drawSample(index) {
  const ctx = canvases[index].getContext("2d");
  const saved = samples[index];
  const image = new Image();
  image.onload = () => {
    if (samples[index] === saved) ctx.drawImage(image, 0, 0);
  };
  image.src = "data:image/png;base64," + saved;
}

function addBox(group, letter, slot) {
  const index = alphabet.indexOf(letter) * 2 + slot;
  const box = document.createElement("div");
  const canvas = document.createElement("canvas");
  canvases[index] = canvas;
  canvas.width = canvas.height = 96;
  canvas.setAttribute("aria-label", letter + ", sample " + (slot + 1));
  const ctx = canvas.getContext("2d");
  ctx.strokeStyle = ctx.fillStyle = "#24211c";
  ctx.lineWidth = 3;
  ctx.lineCap = ctx.lineJoin = "round";
  let pointer = null;
  let last = null;
  function point(event) {
    const rect = canvas.getBoundingClientRect();
    return [(event.clientX - rect.left) * 96 / rect.width,
      (event.clientY - rect.top) * 96 / rect.height].map(value => Math.max(3, Math.min(93, value)));
  }
  canvas.addEventListener("pointerdown", event => {
    if (busy || pointer !== null || event.button !== 0) return;
    pointer = event.pointerId;
    canvas.setPointerCapture(pointer);
    last = point(event);
    ctx.beginPath();
    ctx.arc(...last, 1.5, 0, 2 * Math.PI);
    ctx.fill();
  });
  canvas.addEventListener("pointermove", event => {
    if (pointer !== event.pointerId) return;
    const current = point(event);
    ctx.beginPath();
    ctx.moveTo(...last);
    ctx.lineTo(...current);
    ctx.stroke();
    last = current;
  });
  function finish(event) {
    if (pointer !== event.pointerId) return;
    pointer = null;
    samples[index] = canvas.toDataURL("image/png").split(",")[1];
    saveDraft();
  }
  canvas.addEventListener("pointerup", finish);
  canvas.addEventListener("pointercancel", finish);
  canvas.addEventListener("lostpointercapture", finish);
  const clear = document.createElement("button");
  clear.type = "button";
  clear.className = "text-button";
  clear.textContent = "Clear";
  clear.setAttribute("aria-label", "Clear " + letter + ", sample " + (slot + 1));
  clear.addEventListener("click", () => {
    if (busy) return;
    ctx.clearRect(0, 0, 96, 96);
    samples[index] = null;
    saveDraft();
  });
  if (samples[index]) drawSample(index);
  box.append(canvas, clear);
  group.append(box);
}

function showMarkers() {
  const ctx = photoCanvas.getContext("2d");
  ctx.drawImage(sourcePhoto, 0, 0);
  const size = Math.max(9, photoCanvas.width / 90);
  markers.forEach(([x, y], index) => {
    ctx.fillStyle = "#932c19";
    ctx.beginPath();
    ctx.arc(x * photoCanvas.width, y * photoCanvas.height, size, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "white";
    ctx.font = "bold " + size * 1.4 + "px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(index + 1, x * photoCanvas.width, y * photoCanvas.height);
  });
  photoStatus.textContent = markers.length < 4
    ? "Select the center of the " + markerNames[markers.length] + " black square (" + (markers.length + 1) + " of 4)."
    : "All four markers selected. Fill the empty boxes to review the crops.";
  refresh();
}

photoInput.addEventListener("change", async () => {
  const file = photoInput.files[0];
  photoPreview.hidden = true;
  photoStatus.textContent = "";
  markers.length = 0;
  refresh();
  if (!file) return;
  if (file.size > 20_000_000) {
    photoStatus.textContent = "Choose a JPEG or PNG smaller than 20 MB.";
    return;
  }
  const url = URL.createObjectURL(file);
  const image = new Image();
  try {
    image.src = url;
    await image.decode();
    if (photoInput.files[0] !== file) return;
    const scale = Math.min(1, 2400 / Math.max(image.naturalWidth, image.naturalHeight));
    sourcePhoto.width = photoCanvas.width = Math.round(image.naturalWidth * scale);
    sourcePhoto.height = photoCanvas.height = Math.round(image.naturalHeight * scale);
    if (Math.min(sourcePhoto.width, sourcePhoto.height) < 500) {
      throw new Error("Take a larger photo with the worksheet filling the frame.");
    }
    const ctx = sourcePhoto.getContext("2d", {alpha: false});
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, sourcePhoto.width, sourcePhoto.height);
    ctx.drawImage(image, 0, 0, sourcePhoto.width, sourcePhoto.height);
    photoPreview.hidden = false;
    showMarkers();
  } catch (problem) {
    if (photoInput.files[0] === file) {
      photoStatus.textContent = problem instanceof DOMException ? "Could not open this photo. Try a JPEG or PNG." : problem.message;
    }
  } finally {
    URL.revokeObjectURL(url);
  }
});

photoCanvas.addEventListener("click", event => {
  if (busy || importing || markers.length === 4) return;
  const rect = photoCanvas.getBoundingClientRect();
  markers.push([(event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height]);
  showMarkers();
});
resetMarkers.addEventListener("click", () => {
  markers.length = 0;
  showMarkers();
});
importPhoto.addEventListener("click", async () => {
  importing = true;
  photoStatus.textContent = "Straightening the photo and finding the samples…";
  refresh();
  try {
    const photo = sourcePhoto.toDataURL("image/jpeg", 0.88).split(",")[1];
    const response = await fetch("/api/worksheet-photo", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({photo, markers}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not read the worksheet.");
    if (result.samples.length !== samples.length) throw new Error("The worksheet version does not match.");
    let added = 0;
    result.samples.forEach((sample, index) => {
      if (samples[index] || !sample) return;
      samples[index] = sample;
      drawSample(index);
      added += 1;
    });
    if (added) photoImported = true;
    const remaining = samples.filter(sample => !sample).length;
    photoStatus.textContent = "Added " + added + " samples. " + (remaining
      ? remaining + " boxes still need a sample. Draw in them or try another photo."
      : "Review the labeled crops below before training.");
    saveDraft();
  } catch (problem) {
    photoStatus.textContent = problem.message;
  } finally {
    importing = false;
    refresh();
  }
});

function showState(state) {
  busy = state.phase === "training";
  progress.hidden = !busy;
  if (busy) {
    progress.max = state.total;
    progress.value = state.step;
    status.textContent = "Training · " + state.step + " / " + state.total + " updates";
    setTimeout(poll, 1000);
  } else if (state.phase === "done") {
    status.textContent = "Your style is ready.";
    openStyle.href = "/?style=" + encodeURIComponent(state.style);
    openStyle.hidden = false;
  } else if (state.phase === "error") {
    error.textContent = state.error;
    error.hidden = false;
    status.textContent = "Ready to retry.";
  }
  refresh();
}

async function poll() {
  try {
    const response = await fetch("/api/training");
    if (!response.ok) throw new Error("Training status is unavailable.");
    showState(await response.json());
  } catch {
    status.textContent = "Reconnecting to the studio…";
    setTimeout(poll, 3000);
  }
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (train.disabled) return;
  busy = true;
  error.hidden = true;
  openStyle.hidden = true;
  status.textContent = "Checking your samples…";
  refresh();
  try {
    const response = await fetch("/api/train", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name: name.value, samples, alphabet: "alphanumeric", photoImported}),
    });
    const state = await response.json();
    if (!response.ok) throw new Error(state.error || "Could not start training.");
    showState(state);
  } catch (problem) {
    busy = false;
    error.textContent = problem.message;
    error.hidden = false;
    status.textContent = "Correct the sample and try again.";
    refresh();
  }
});

try {
  const draft = JSON.parse(localStorage.getItem(draftKey) || "null");
  if (draft && Array.isArray(draft.samples) && [52, samples.length].includes(draft.samples.length)) {
    name.value = typeof draft.name === "string" ? draft.name.slice(0, 40) : "";
    photoImported = draft.photoImported === true;
    draft.samples.forEach((sample, index) => {
      if (typeof sample === "string" && /^[A-Za-z0-9+/=]{1,50000}$/.test(sample)) {
        samples[index] = sample;
      }
    });
  }
} catch {
  status.textContent = "Draft storage is unavailable. Keep this tab open.";
}
for (const letter of alphabet) {
  const group = document.createElement("section");
  group.className = "letter-group";
  const title = document.createElement("h2");
  title.textContent = letter;
  group.append(title);
  addBox(group, letter, 0);
  addBox(group, letter, 1);
  worksheet.append(group);
}
name.addEventListener("input", saveDraft);
refresh();
poll();
