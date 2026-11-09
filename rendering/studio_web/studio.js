"use strict";

const sampleText = "A note for Friday\n\nMeet me at 4:30 in Room B-12.\nBring 2 pens, 5 sheets, and your ideas.\n\nSee you there!";
const form = document.getElementById("editor-form");
const input = document.getElementById("text");
const generate = document.getElementById("generate");
const download = document.getElementById("download");
const previous = document.getElementById("previous");
const next = document.getElementById("next");
const status = document.getElementById("render-status");
const error = document.getElementById("error");
const image = document.getElementById("page-image");
const preview = document.getElementById("preview-window");
const style = document.getElementById("style");
let pages = [];
let pageIndex = 0;
let renderedText = null;
let renderedStyle = null;
let busy = false;

function fullAlphabet() {
  return style.value === "standard" || style.selectedOptions[0]?.dataset.full === "true";
}

function refresh() {
  const fresh = input.value === renderedText && style.value === renderedStyle;
  document.getElementById("input-note").textContent = fullAlphabet()
    ? "English letters, numbers, and common punctuation."
    : "This style uses lowercase letters and common punctuation.";
  document.getElementById("character-count").textContent = input.value.length.toLocaleString("en-US") + " / 6,000";
  generate.disabled = busy || !input.value.trim();
  download.disabled = busy || !pages.length || !fresh;
  previous.disabled = pageIndex === 0;
  next.disabled = pageIndex + 1 >= pages.length;
  image.hidden = !pages.length;
  document.getElementById("empty-preview").hidden = pages.length > 0;
  document.getElementById("page-count").textContent =
    "Page " + (pages.length ? pageIndex + 1 : 0) + " of " + pages.length;
  if (pages.length) {
    image.src = pages[pageIndex];
    image.alt = "Generated handwritten page " + (pageIndex + 1);
  }
  preview.setAttribute("aria-busy", String(busy));
  if (busy) status.textContent = "Generating handwriting…";
  else if (pages.length && !fresh) status.textContent = "Generate again to update the preview.";
  else if (pages.length) status.textContent = pages.length === 1 ? "1 page ready" : pages.length + " pages ready";
  else status.textContent = "Ready to generate";
}

async function renderText(event) {
  event.preventDefault();
  if (busy || !input.value.trim()) return;
  const text = input.value;
  const selectedStyle = style.value;
  busy = true;
  error.hidden = true;
  refresh();
  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text, style: selectedStyle}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not generate a page.");
    const newPages = result.pages.map(encoded => {
      const bytes = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
      return URL.createObjectURL(new Blob([bytes], {type: "image/png"}));
    });
    pages.forEach(url => URL.revokeObjectURL(url));
    pages = newPages;
    pageIndex = 0;
    renderedText = text;
    renderedStyle = selectedStyle;
  } catch (problem) {
    error.textContent = problem instanceof Error ? problem.message : "Could not generate a page.";
    error.hidden = false;
  } finally {
    busy = false;
    refresh();
  }
}

form.addEventListener("submit", renderText);
style.addEventListener("change", refresh);
input.addEventListener("input", () => {
  error.hidden = true;
  refresh();
});
input.addEventListener("keydown", event => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    form.requestSubmit();
  }
});
document.getElementById("sample").addEventListener("click", () => {
  input.value = fullAlphabet() ? sampleText : sampleText.toLowerCase().replace(/[0-9]/g, "");
  refresh();
  form.requestSubmit();
});
previous.addEventListener("click", () => { pageIndex -= 1; refresh(); });
next.addEventListener("click", () => { pageIndex += 1; refresh(); });
download.addEventListener("click", () => {
  if (download.disabled) return;
  const link = document.createElement("a");
  link.href = pages[pageIndex];
  link.download = "inksy-page-" + (pageIndex + 1) + ".png";
  document.body.appendChild(link);
  link.click();
  link.remove();
});

async function initialize() {
  try {
    const response = await fetch("/api/styles");
    if (response.ok) {
      const result = await response.json();
      document.getElementById("train-link").hidden = result.training_available === false;
      if (result.hosted) document.getElementById("studio-location").textContent = "Online demo";
      for (const saved of result.styles) {
        const option = new Option(saved.name, saved.id);
        option.dataset.full = String(saved.alphabet?.length === 62);
        style.add(option);
      }
      const chosen = new URLSearchParams(location.search).get("style");
      if (Array.from(style.options).some(option => option.value === chosen)) style.value = chosen;
    }
  } finally {
    document.getElementById("sample").click();
  }
}

initialize().catch(() => {});
