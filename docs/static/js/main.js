// SPDX-License-Identifier: Apache-2.0
// RAWATTA project page: small progressive enhancements. The page works without JavaScript.
import { parseSceneIndex } from "./formats.js";

// ---------------------------------------------------------------------------------------------
// Set this to the arXiv identifier (for example "2611.01234") once the paper is announced.
// While it is empty, the Paper and PDF buttons stay disabled and read "coming soon".
const ARXIV_ID = "";
// ---------------------------------------------------------------------------------------------

const EXAMPLES_INDEX = "static/examples/index.json";

function applyArxiv() {
  if (!ARXIV_ID) return;
  const urls = { abs: `https://arxiv.org/abs/${ARXIV_ID}`, pdf: `https://arxiv.org/pdf/${ARXIV_ID}` };
  document.querySelectorAll("[data-arxiv]").forEach((a) => {
    const url = urls[a.dataset.arxiv];
    if (!url) return;
    a.href = url;
    a.removeAttribute("aria-disabled");
    a.removeAttribute("title");
    a.querySelectorAll(".soon").forEach((s) => s.remove());
  });
  const bib = document.getElementById("bibtex-code");
  if (bib) bib.textContent = bib.textContent.replace("XXXX.XXXXX", ARXIV_ID);
}

function addCopyButtons() {
  document.querySelectorAll(".codeblock").forEach((block) => {
    const code = block.querySelector("code");
    if (!code) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy";
    btn.textContent = "Copy";
    btn.setAttribute("aria-label", "Copy to clipboard");
    btn.addEventListener("click", async () => {
      const text = code.textContent;
      let ok = false;
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(text);
          ok = true;
        }
      } catch (_) { ok = false; }
      if (!ok) {
        // Fallback: select the text so the reader can copy it by hand.
        const range = document.createRange();
        range.selectNodeContents(code);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      }
      btn.textContent = ok ? "Copied" : "Selected";
      setTimeout(() => { btn.textContent = "Copy"; }, 1600);
    });
    block.appendChild(btn);
  });
}

// index.json is written by examples/python/gallery_index.py; see formats.js for its schema.
async function loadSceneList(indexUrl) {
  try {
    const res = await fetch(indexUrl, { cache: "no-cache" });
    if (!res.ok) return [];
    return parseSceneIndex(await res.json());
  } catch (_) {
    return [];
  }
}

// The interactive section stays hidden unless static/examples/index.json lists at least one scene.
// three.js and the point clouds are fetched only when the section approaches the viewport.
async function initExamples() {
  const section = document.getElementById("interactive");
  if (!section) return;
  const indexUrl = new URL(EXAMPLES_INDEX, document.baseURI);
  const scenes = await loadSceneList(indexUrl.href);
  if (!scenes.length) return;
  section.hidden = false;
  const status = document.getElementById("viewer-status");
  const base = new URL(".", indexUrl).href; // scene folders are relative to index.json
  let started = false;
  const start = () => {
    if (started) return;
    started = true;
    if (status) status.textContent = "Loading the 3D viewer…";
    import("./viewer.js")
      .then((m) => m.mountViewer(section, scenes, base))
      .catch(() => { if (status) status.textContent = "The 3D viewer could not be loaded (it needs WebGL and access to cdn.jsdelivr.net)."; });
  };
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { io.disconnect(); start(); }
    }, { rootMargin: "400px 0px" });
    io.observe(section);
  } else {
    start();
  }
}

applyArxiv();
addCopyButtons();
initExamples();
