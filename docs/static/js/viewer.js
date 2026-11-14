// SPDX-License-Identifier: Apache-2.0
// Point-cloud viewer for `rawatta run` outputs: points.ply and cameras.json (formats in ./formats.js).
// three.js is pinned in the import map of index.html (three@0.170.0 from cdn.jsdelivr.net).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { parsePLY, parseCameras, cameraCenter } from "./formats.js";

const FRAME0 = [1.0, 0.6, 0.27];   // orange
const FRAMEN = [0.55, 0.77, 1.0];  // light blue

/** Caption under the viewer: the scene's caption text (run, measured latency, photo attribution), then links to the
 *  photo licence and source. Built from text nodes, so nothing in index.json is parsed as HTML. */
function renderCaption(el, s) {
  el.textContent = "";
  if (s.caption) el.append(s.caption);
  const link = (href, text, rel) => {
    const a = document.createElement("a");
    a.href = href;
    a.textContent = text;
    a.rel = rel;
    return a;
  };
  const bits = [];
  if (s.licence || s.licenceUrl) {
    bits.push(["Photo licence: ", s.licenceUrl ? link(s.licenceUrl, s.licence || s.licenceUrl, "license noopener") : s.licence]);
  }
  if (s.source) bits.push(["", link(s.source, "photo source", "noopener")]);
  bits.forEach(([label, node], i) => {
    el.append(i === 0 ? (s.caption ? " " : "") : " · ", label, node);
  });
  if (bits.length) el.append(".");
}

// OpenCV world (x right, y down, z forward) -> three.js (y up, camera looks along -z): rotate 180 degrees about x.
function cv2three(x, y, z) { return [x, -y, -z]; }

function frustumSegments(cams, depth) {
  const pos = [], col = [];
  for (const cam of cams) {
    const { E, K, width, height } = cam;
    const C = cameraCenter(E);
    const toWorld = (p) => [0, 1, 2].map((k) => C[k] + E[k] * p[0] + E[4 + k] * p[1] + E[8 + k] * p[2]);
    const corner = (u, v) => toWorld([((u - K[2]) / K[0]) * depth, ((v - K[5]) / K[4]) * depth, depth]);
    const c = [corner(0, 0), corner(width, 0), corner(width, height), corner(0, height)];
    const color = cam.index === 0 ? FRAME0 : FRAMEN;
    const seg = (a, b) => { pos.push(...cv2three(...a), ...cv2three(...b)); col.push(...color, ...color); };
    for (let i = 0; i < 4; i++) { seg(C, c[i]); seg(c[i], c[(i + 1) % 4]); }
    // small tick above the image's top edge marks "up"
    const mid = toWorld([0, ((0 - K[5]) / K[4]) * depth * 1.25, depth]);
    seg(c[0], mid); seg(mid, c[1]);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  return g;
}

/** Robust centre (per-axis median) and radius (90th percentile distance) of a point sample, in three.js coordinates. */
function robustBounds(positions, count) {
  const step = Math.max(1, Math.floor(count / 20000));
  const xs = [], ys = [], zs = [];
  for (let i = 0; i < count; i += step) {
    const p = cv2three(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
    xs.push(p[0]); ys.push(p[1]); zs.push(p[2]);
  }
  const med = (a) => { const s = Float64Array.from(a).sort(); return s.length ? s[s.length >> 1] : 0; };
  const c = [med(xs), med(ys), med(zs)];
  const d = xs.map((x, i) => Math.hypot(x - c[0], ys[i] - c[1], zs[i] - c[2])).sort((a, b) => a - b);
  const r = d.length ? d[Math.min(d.length - 1, Math.floor(d.length * 0.9))] : 1;
  return { center: new THREE.Vector3(...c), radius: Math.max(r, 1e-3) };
}

export function mountViewer(section, scenes, baseUrl) {
  const host = section.querySelector("#viewer-canvas");
  const status = section.querySelector("#viewer-status");
  const chips = section.querySelector("#viewer-scenes");
  const camsBox = section.querySelector("#viewer-cams");
  const sizeInput = section.querySelector("#viewer-size");
  const resetBtn = section.querySelector("#viewer-reset");
  const caption = section.querySelector("#viewer-caption");
  const say = (t) => { status.textContent = t; };

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  } catch (_) {
    say("WebGL is not available in this browser, so the 3D example cannot be shown.");
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace; // PLY colours are display sRGB bytes: pass them through unchanged
  renderer.setClearColor(new THREE.Color().setRGB(21 / 255, 23 / 255, 27 / 255, THREE.LinearSRGBColorSpace), 1); // = CSS #15171b
  host.insertBefore(renderer.domElement, status);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 16 / 10, 0.01, 1000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;

  const material = new THREE.PointsMaterial({ size: 1.5, vertexColors: true, sizeAttenuation: false });
  const lineMaterial = new THREE.LineBasicMaterial({ vertexColors: true });
  let points = null, frusta = null, home = null, token = 0;

  const resize = () => {
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(host);
  resize();

  const setPointSize = () => { material.size = Number(sizeInput.value) * renderer.getPixelRatio(); };
  sizeInput.addEventListener("input", setPointSize);
  setPointSize();
  camsBox.addEventListener("change", () => { if (frusta) frusta.visible = camsBox.checked; });
  const goHome = () => {
    if (!home) return;
    camera.position.copy(home.position);
    controls.target.copy(home.target);
    camera.near = home.radius * 0.002; camera.far = home.radius * 200;
    camera.updateProjectionMatrix();
    controls.update();
  };
  resetBtn.addEventListener("click", goHome);

  let visible = true, frame = 0;
  const loop = () => {
    frame = 0;
    if (!visible || document.hidden) return;
    controls.update();
    renderer.render(scene, camera);
    frame = requestAnimationFrame(loop);
  };
  const kick = () => { if (!frame) frame = requestAnimationFrame(loop); };
  new IntersectionObserver((es) => { visible = es.some((e) => e.isIntersecting); if (visible) kick(); }).observe(host);
  document.addEventListener("visibilitychange", kick);

  const clear = () => {
    for (const o of [points, frusta]) { if (o) { scene.remove(o); o.geometry.dispose(); } }
    points = frusta = null;
  };

  async function load(s) {
    const my = ++token;
    const url = (f) => new URL(`${s.dir}/${f}`, baseUrl).href;
    say(`Loading ${s.title}…`);
    renderCaption(caption, s);
    try {
      const [plyRes, camRes] = await Promise.all([fetch(url(s.points)), fetch(url(s.cameras))]);
      if (!plyRes.ok) throw new Error(`${s.points}: HTTP ${plyRes.status}`);
      const ply = parsePLY(await plyRes.arrayBuffer());
      // The point cloud is shown even when the cameras cannot be read; the status line says why they are missing.
      let cams = [], camNote = "";
      try {
        if (!camRes.ok) throw new Error(`${s.cameras}: HTTP ${camRes.status}`);
        const parsed = parseCameras(await camRes.json());
        cams = parsed.cams;
        if (parsed.skipped) camNote = ` (${parsed.skipped} unreadable)`;
      } catch (e) {
        camNote = ` (cameras not shown: ${e.message})`;
      }
      if (my !== token) return;
      clear();
      const n = ply.count;
      const pos3 = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const p = cv2three(ply.positions[3 * i], ply.positions[3 * i + 1], ply.positions[3 * i + 2]);
        pos3[3 * i] = p[0]; pos3[3 * i + 1] = p[1]; pos3[3 * i + 2] = p[2];
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos3, 3));
      const col = ply.colors || new Uint8Array(n * 3).fill(200);
      g.setAttribute("color", new THREE.Uint8BufferAttribute(col, 3, true));
      points = new THREE.Points(g, material);
      scene.add(points);
      const { center, radius } = robustBounds(ply.positions, n);
      if (cams.length) {
        const centers = cams.map((c) => cameraCenter(c.E));
        const gaps = [];
        for (let i = 1; i < centers.length; i++) gaps.push(Math.hypot(...centers[i].map((v, k) => v - centers[i - 1][k])));
        gaps.sort((a, b) => a - b);
        const spacing = gaps.length ? gaps[gaps.length >> 1] : 0;
        const depth = Math.min(Math.max(spacing * 0.6, radius * 0.03), radius * 0.15);
        frusta = new THREE.LineSegments(frustumSegments(cams, depth), lineMaterial);
        frusta.visible = camsBox.checked;
        scene.add(frusta);
      }
      // Start behind and slightly above the first camera, looking at the scene centre.
      const c0 = cams.length ? new THREE.Vector3(...cv2three(...cameraCenter(cams[0].E))) : new THREE.Vector3(0, 0, 0);
      const dir3 = c0.clone().sub(center);
      if (dir3.lengthSq() < 1e-12) dir3.set(0, 0, 1);
      dir3.normalize();
      const position = center.clone().addScaledVector(dir3, radius * 2.1).add(new THREE.Vector3(0, radius * 0.5, 0));
      home = { position, target: center.clone(), radius };
      goHome();
      say(`${n.toLocaleString("en-US")} points · ${cams.length} cameras${camNote}`);
      kick();
    } catch (err) {
      if (my === token) say(`Could not load this example (${err.message}).`);
    }
  }

  chips.textContent = "";
  scenes.forEach((s, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = s.title;
    b.setAttribute("aria-pressed", i === 0 ? "true" : "false");
    b.addEventListener("click", () => {
      chips.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", c === b ? "true" : "false"));
      load(s);
    });
    chips.appendChild(b);
  });
  if (scenes.length < 2) chips.hidden = true;
  load(scenes[0]);
}
