// SPDX-License-Identifier: Apache-2.0
// Browser checks for the project page (run by check.sh inside mcr.microsoft.com/playwright:v1.63.0-noble).
// For each target x viewport x colour scheme: console errors, page errors, failed requests, broken images,
// broken internal links and anchors, horizontal overflow, axe-core WCAG 2 A/AA violations; plus screenshots.
// Also: the quick start shows the canonical commands verbatim, no "Ours" label appears, and (dev target) the viewer
// loads its scene with all cameras (shipped: buddha, 34; dev: the synthetic scene, 8) and links the photo licence
// (CC BY 4.0 for both) in its caption.
// Usage: node site-src/dev/check.cjs [-h|--help] [SCREENSHOT_DIR]   (needs the two servers that check.sh starts)
const fs = require("fs");
const path = require("path");
if (process.argv.includes("-h") || process.argv.includes("--help")) {
  console.log(fs.readFileSync(__filename, "utf8").split("\n").slice(1, 7).map((l) => l.replace(/^\/\/ ?/, "")).join("\n"));
  process.exit(0);
}
const MODS = process.env.PW_MODULES || "/w/.venv/pw/node_modules";
const { chromium } = require(path.join(MODS, "playwright"));
const AXE = fs.readFileSync(path.join(MODS, "axe-core", "axe.min.js"), "utf8");
const OUT = process.argv[2] || "/w/site-src/dev/out/shots";
fs.mkdirSync(OUT, { recursive: true });

const targets = [
  // docs/ exactly as committed: the real gallery, whose first scene (buddha) has 34 cameras
  { name: "ship", url: "http://127.0.0.1:8811/rawatta/", viewer: true, cams: 34 },
  // docs/ + synthetic example scene (8 cameras)
  { name: "dev", url: "http://127.0.0.1:8812/rawatta/", viewer: true, cams: 8 },
];
const viewports = [
  { name: "1280", width: 1280, height: 800, dsf: 1, mobile: false },
  { name: "390", width: 390, height: 844, dsf: 2, mobile: true },
  { name: "360", width: 360, height: 740, dsf: 2, mobile: true },
];
const schemes = ["light", "dark"];
// The release's canonical user commands; the page must show them exactly (inside its code blocks).
const CANONICAL = [
  'docker run --rm --user "$(id -u):$(id -g)" -v ~/.cache/huggingface:/hf:ro -v ~/.cache/rawatta:/weights \\\n' +
  "  --entrypoint download-weights ghcr.io/sezer-muhammed/rawatta:latest --token-file /hf/token --dir /weights",
  'docker run --rm --gpus all --user "$(id -u):$(id -g)" -v ~/.cache/rawatta:/weights:ro -v "$PWD":/work \\\n' +
  "  ghcr.io/sezer-muhammed/rawatta:latest run --images photos --config rawatta --out out",
];
const SYNTH_LICENCE = "https://creativecommons.org/licenses/by/4.0/";

(async () => {
  const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const report = [];
  let failures = 0;
  for (const t of targets) {
    for (const vp of viewports) {
      for (const scheme of schemes) {
        const tag = `${t.name}-${vp.name}-${scheme}`;
        const ctx = await browser.newContext({ viewport: { width: vp.width, height: vp.height }, deviceScaleFactor: vp.dsf,
          isMobile: vp.mobile, hasTouch: vp.mobile, colorScheme: scheme });
        const page = await ctx.newPage();
        const r = { tag, consoleErrors: [], pageErrors: [], failedRequests: [], badResponses: [], hosts: new Set() };
        page.on("console", (m) => { if (m.type() === "error") r.consoleErrors.push(m.text()); });
        page.on("pageerror", (e) => r.pageErrors.push(String(e)));
        page.on("requestfailed", (q) => r.failedRequests.push(`${q.url()} ${q.failure() && q.failure().errorText}`));
        page.on("response", (s) => { if (s.status() >= 400) r.badResponses.push(`${s.status()} ${s.url()}`); });
        page.on("request", (q) => { try { r.hosts.add(new URL(q.url()).host); } catch (_) {} });
        await page.goto(t.url, { waitUntil: "load" });
        // scroll through the page so lazy images and the lazy viewer load
        await page.evaluate(async () => {
          const step = Math.round(window.innerHeight * 0.7);
          for (let y = 0; y < document.documentElement.scrollHeight; y += step) { window.scrollTo(0, y); await new Promise((ok) => setTimeout(ok, 120)); }
          window.scrollTo(0, 0);
        });
        await page.waitForLoadState("networkidle");
        // lazy images: bring each into view, then give them up to 15 s to finish (a real failure stays incomplete)
        for (const img of await page.locator("img").all()) { if (await img.isVisible()) await img.scrollIntoViewIfNeeded(); }
        try {
          await page.waitForFunction(() => [...document.images].filter((i) => !i.closest("[hidden]")).every((i) => i.complete && i.naturalWidth > 0), null, { timeout: 15000 });
        } catch (_) { /* reported below */ }
        await page.evaluate(() => window.scrollTo(0, 0));
        if (t.viewer) {
          await page.locator("#interactive").scrollIntoViewIfNeeded();
          try {
            await page.waitForFunction(() => /points/.test(document.getElementById("viewer-status").textContent), null, { timeout: 45000 });
          } catch (_) { /* reported below */ }
          await page.waitForTimeout(1500);
          r.viewerStatus = await page.locator("#viewer-status").textContent();
          r.viewerOk = new RegExp(`points · ${t.cams} cameras$`).test(r.viewerStatus || "");
          r.captionLinks = await page.evaluate(() => [...document.querySelectorAll("#viewer-caption a")].map((a) => ({ href: a.href, text: a.textContent })));
          r.captionOk = r.captionLinks.some((l) => l.href === "https://creativecommons.org/licenses/by/4.0/" && l.text === "CC BY 4.0");
          await page.locator("#interactive").screenshot({ path: path.join(OUT, `${tag}-viewer.png`) });
          // orbit a little and take a second shot, to see the controls work
          const box = await page.locator("#viewer-canvas canvas").boundingBox();
          if (box && !vp.mobile) {
            await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
            await page.mouse.down();
            await page.mouse.move(box.x + box.width / 2 + 160, box.y + box.height / 2 + 40, { steps: 8 });
            await page.mouse.up();
            await page.waitForTimeout(800);
            await page.locator("#interactive").screenshot({ path: path.join(OUT, `${tag}-viewer-orbit.png`) });
          }
          await page.evaluate(() => window.scrollTo(0, 0));
        }
        r.interactiveHidden = await page.evaluate(() => document.getElementById("interactive").hidden);
        r.arxiv = await page.evaluate(() => ({
          links: [...document.querySelectorAll("[data-arxiv]")].map((x) => ({ href: x.getAttribute("href"), disabled: x.getAttribute("aria-disabled") })),
          bibId: (document.getElementById("bibtex-code").textContent.match(/arXiv:(\S+)\}/) || [])[1],
        }));
        // only the dev copy flips ARXIV_ID to a fake id (check.sh); the shipped page keeps the placeholder
        const arxivOk = t.name === "dev"
          ? r.arxiv.links.every((l) => l.href && l.href.includes("2611.00001") && !l.disabled) && r.arxiv.bibId === "2611.00001"
          : r.arxiv.links.every((l) => !l.href && l.disabled === "true") && r.arxiv.bibId === "XXXX.XXXXX";
        r.codeText = await page.evaluate(() => [...document.querySelectorAll("#use pre code")].map((c) => c.textContent).join("\n\n"));
        r.missingCommands = CANONICAL.filter((c) => !r.codeText.includes(c));
        r.oursLabel = await page.evaluate(() => /\bOurs\b/.test(document.body.innerText));
        r.overflow = await page.evaluate(() => {
          const de = document.documentElement, vw = de.clientWidth;
          const offenders = [];
          for (const el of document.querySelectorAll("body *")) {
            if (el.closest(".table-wrap, pre, [hidden]")) continue;
            const b = el.getBoundingClientRect();
            if (b.width && (b.right > vw + 1 || b.left < -1)) offenders.push(`${el.tagName.toLowerCase()}${el.id ? "#" + el.id : ""}.${el.className} right=${Math.round(b.right)}`);
          }
          return { scrollWidth: de.scrollWidth, clientWidth: vw, offenders: offenders.slice(0, 10) };
        });
        r.images = await page.evaluate(() => [...document.images].filter((i) => !i.closest("[hidden]"))
          .map((i) => ({ src: i.currentSrc || i.src, ok: i.complete && i.naturalWidth > 0, alt: (i.getAttribute("alt") || "").length })));
        const links = await page.evaluate(() => [...document.querySelectorAll("a[href]")].map((a) => a.href));
        r.brokenLinks = []; r.externalLinks = [];
        const origin = new URL(t.url).origin;
        for (const href of [...new Set(links)]) {
          const u = new URL(href);
          if (u.origin !== origin) { r.externalLinks.push(href); continue; }
          if (u.hash && u.pathname === new URL(t.url).pathname) {
            const id = decodeURIComponent(u.hash.slice(1));
            const exists = await page.evaluate((x) => !!document.getElementById(x), id);
            if (!exists) r.brokenLinks.push(`missing anchor ${u.hash}`);
            continue;
          }
          const res = await ctx.request.get(href);
          if (res.status() >= 400) r.brokenLinks.push(`${res.status()} ${href}`);
        }
        await page.addScriptTag({ content: AXE });
        r.axe = await page.evaluate(async () => {
          const res = await window.axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] } });
          return res.violations.map((v) => ({ id: v.id, impact: v.impact, n: v.nodes.length, nodes: v.nodes.slice(0, 4).map((x) => x.target.join(" ") + " :: " + (x.failureSummary || "").split("\n").slice(0, 2).join(" ")) }));
        });
        await page.screenshot({ path: path.join(OUT, `${tag}-full.png`), fullPage: true });
        if (t.name === "ship") {
          for (const id of ["teaser", "regimes", "method", "results", "scaling", "use", "limitations", "bibtex"]) {
            await page.locator(`#${id}`).screenshot({ path: path.join(OUT, `${tag}-sec-${id}.png`) });
          }
          await page.evaluate(() => window.scrollTo(0, 0));
          await page.screenshot({ path: path.join(OUT, `${tag}-top.png`) });
        }
        r.hosts = [...r.hosts];
        const imgBad = r.images.filter((i) => !i.ok || !i.alt);
        const expectedHosts = [new URL(t.url).host, ...(t.viewer ? ["cdn.jsdelivr.net"] : [])];
        r.unexpectedHosts = r.hosts.filter((h) => !expectedHosts.includes(h));
        r.problems = [
          ...r.consoleErrors.map((x) => "console: " + x), ...r.pageErrors.map((x) => "pageerror: " + x),
          ...r.failedRequests.map((x) => "requestfailed: " + x), ...r.badResponses.map((x) => "http: " + x),
          ...imgBad.map((i) => "image: " + JSON.stringify(i)), ...r.brokenLinks.map((x) => "link: " + x),
          ...(r.overflow.scrollWidth > r.overflow.clientWidth ? ["horizontal overflow: " + JSON.stringify(r.overflow)] : []),
          ...r.axe.map((v) => `axe ${v.id} (${v.impact}) x${v.n}: ${v.nodes.join(" | ")}`),
          ...r.unexpectedHosts.map((h) => "unexpected host: " + h),
          ...(t.viewer && !r.viewerOk ? [`viewer did not load the scene with ${t.cams} cameras: ` + r.viewerStatus] : []),
          ...(t.viewer && !r.captionOk ? ["viewer caption has no licence link to " + SYNTH_LICENCE + ": " + JSON.stringify(r.captionLinks)] : []),
          ...r.missingCommands.map((c) => "canonical command missing from the quick start: " + c.split("\n")[0]),
          ...(r.oursLabel ? ['the page text contains "Ours"'] : []),
          ...(t.viewer === r.interactiveHidden ? [`interactive section hidden=${r.interactiveHidden}`] : []),
          ...(arxivOk ? [] : ["arXiv buttons/BibTeX not in the expected state: " + JSON.stringify(r.arxiv)]),
        ];
        failures += r.problems.length;
        console.log(`${tag}: ${r.problems.length ? "FAIL" : "ok"} | console errors ${r.consoleErrors.length}, images ${r.images.length} (bad ${imgBad.length}), links checked ${links.length} (broken ${r.brokenLinks.length}), scrollWidth ${r.overflow.scrollWidth}/${r.overflow.clientWidth}, axe ${r.axe.length}, hosts ${r.hosts.join(",")}${t.viewer ? ", viewer: " + r.viewerStatus : ""}`);
        for (const p of r.problems) console.log("   - " + p);
        report.push(r);
        await ctx.close();
      }
    }
  }
  fs.writeFileSync(path.join(OUT, "report.json"), JSON.stringify(report, null, 1));
  await browser.close();
  console.log(failures ? `FAILED: ${failures} problem(s)` : "ALL CHECKS PASSED");
  process.exit(failures ? 1 : 0);
})();
