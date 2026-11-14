// SPDX-License-Identifier: Apache-2.0
// Unit tests for docs/static/js/formats.js (the viewer's file readers). Plain Node, no packages.
// Usage: node site-src/dev/test_formats.mjs [SYNTHETIC_SCENE_DIR]
//   SYNTHETIC_SCENE_DIR: optional output folder of make_synthetic_scene.py, checked as well.
// Fixtures (site-src/dev/fixtures/): cameras.rawatta-run.json is what `rawatta run`'s serializer writes,
// gallery-index.json what examples/python/gallery_index.py writes; see make_fixtures.sh for how both were made.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

if (process.argv.includes("-h") || process.argv.includes("--help")) {
  console.log(readFileSync(fileURLToPath(import.meta.url), "utf8").split("\n").slice(1, 6).map((l) => l.replace(/^\/\/ ?/, "")).join("\n"));
  process.exit(0);
}
const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");
// formats.js is an ES module without a package.json around it: import it from a data: URL so any Node version takes it.
const src = readFileSync(join(root, "docs/static/js/formats.js"), "utf8");
const F = await import("data:text/javascript;base64," + Buffer.from(src).toString("base64"));
const json = (p) => JSON.parse(readFileSync(p, "utf8"));
let n = 0;
const test = (name, fn) => { fn(); n++; console.log("ok - " + name); };

test("cameras.json from rawatta run's serializer", () => {
  const { cams, skipped } = F.parseCameras(json(join(here, "fixtures/cameras.rawatta-run.json")));
  assert.equal(skipped, 0);
  assert.equal(cams.length, 2);
  assert.deepEqual(cams.map((c) => c.name), ["a.png", "frame 2.png"]);
  assert.deepEqual(cams.map((c) => c.index), [0, 1]);
  assert.deepEqual([cams[1].width, cams[1].height], [512, 336]);
  assert.deepEqual(cams[0].E, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
  assert.deepEqual(cams[1].E, [0, -1, 0, 0.5, 1, 0, 0, -1, 0, 0, 1, 2]);
  assert.deepEqual(cams[1].K, [400, 0, 256, 0, 410, 168, 0, 0, 1]);
  // C = -R^T t for R = [[0,-1,0],[1,0,0],[0,0,1]], t = (0.5, -1, 2)
  assert.deepEqual(F.cameraCenter(cams[1].E).map((v) => v + 0), [1, 0.5, -2]);
  assert.deepEqual(F.cameraCenter(cams[0].E).map((v) => v + 0), [0, 0, 0]);
});

test("cameras.json: wrong shapes are refused or skipped", () => {
  assert.throws(() => F.parseCameras([]), /expected/);
  assert.throws(() => F.parseCameras({ cameras: [] }), /expected/);
  assert.throws(() => F.parseCameras({ width: 4, height: 4, frames: [{ w2c: [[1, 0, 0]], intrinsics: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] }] }), /no frame/);
  const good = { image: "x", w2c: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], intrinsics: [[2, 0, 1], [0, 2, 1], [0, 0, 1]] };
  const r = F.parseCameras({ width: 2, height: 2, frames: [good, { ...good, w2c: [[1, 0, 0, "a"], [0, 1, 0, 0], [0, 0, 1, 0]] }, good] });
  assert.equal(r.cams.length, 2);
  assert.equal(r.skipped, 1);
  assert.deepEqual(r.cams.map((c) => c.index), [0, 2]);
  assert.deepEqual(F.parseCameras({ width: 2, height: 2, frames: [] }), { cams: [], skipped: 0 });
});

test("index.json from examples/python/gallery_index.py", () => {
  const scenes = F.parseSceneIndex(json(join(here, "fixtures/gallery-index.json")));
  assert.equal(scenes.length, 1);
  const s = scenes[0];
  assert.equal(s.id, "buddha");
  assert.equal(s.title, "Buddha head");
  assert.deepEqual([s.dir, s.points, s.cameras], ["buddha", "points.ply", "cameras.json"]);
  assert.equal(s.licence, "CC BY 4.0");
  assert.equal(s.licenceUrl, "https://creativecommons.org/licenses/by/4.0/");
  assert.equal(s.source, "https://github.com/alicevision/dataset_buddha");
  assert.equal(s.synthetic, true);
  assert.match(s.caption, /^4 photos, --config rawatta\. .*Photos: Buddha Dataset/);
});

test("index.json: the shipped empty index, and unsafe entries", () => {
  assert.deepEqual(F.parseSceneIndex(json(join(root, "docs/static/examples/index.json"))), []);
  assert.deepEqual(F.parseSceneIndex(null), []);
  assert.deepEqual(F.parseSceneIndex([{ id: "a" }]), []);
  const scenes = F.parseSceneIndex({ scenes: [
    { id: "a" },
    { id: "b", dir: "../outside" },
    { id: "c", dir: "https://example.org/x" },
    { id: "d", points: "/etc/points.ply" },
    { id: "e", licence: "CC0 1.0", licence_url: "javascript:alert(1)", source: "ftp://x" },
    { title: "no id" },
  ] });
  assert.deepEqual(scenes.map((s) => s.id), ["a", "e"]);
  assert.deepEqual([scenes[0].dir, scenes[0].points, scenes[0].cameras, scenes[0].title], ["a", "points.ply", "cameras.json", "a"]);
  assert.equal(scenes[1].licenceUrl, null);
  assert.equal(scenes[1].source, null);
  assert.equal(scenes[1].licence, "CC0 1.0");
});

test("points.ply as rawatta run writes it", () => {
  // header of write_points_ply in crates/vggt-omega/src/output.rs; 15 bytes per vertex
  const header = "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n" +
    "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
  const h = Buffer.from(header, "ascii");
  const body = Buffer.alloc(3 * 15);
  [[1, -2, 3.5, 255, 0, 7], [NaN, 0, 0, 1, 1, 1], [0.25, 0.5, -0.125, 1, 2, 3]].forEach((v, i) => {
    body.writeFloatLE(v[0], 15 * i); body.writeFloatLE(v[1], 15 * i + 4); body.writeFloatLE(v[2], 15 * i + 8);
    body[15 * i + 12] = v[3]; body[15 * i + 13] = v[4]; body[15 * i + 14] = v[5];
  });
  const buf = Buffer.concat([h, body]);
  const ply = F.parsePLY(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
  assert.equal(ply.count, 2); // the NaN vertex is dropped
  assert.deepEqual(Array.from(ply.positions), [1, -2, 3.5, 0.25, 0.5, -0.125]);
  assert.deepEqual(Array.from(ply.colors), [255, 0, 7, 1, 2, 3]);
  assert.throws(() => F.parsePLY(new TextEncoder().encode("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n").buffer), /binary_little_endian/);
});

const synth = process.argv[2];
if (synth) {
  test("synthetic test scene uses the same formats", () => {
    const { cams, skipped } = F.parseCameras(json(join(synth, "synthetic/cameras.json")));
    assert.equal(skipped, 0);
    assert.equal(cams.length, 8);
    assert.deepEqual(F.cameraCenter(cams[0].E).map((v) => Math.abs(v)), [0, 0, 0]);
    const scenes = F.parseSceneIndex(json(join(synth, "index.json")));
    assert.equal(scenes.length, 1);
    assert.equal(scenes[0].dir, "synthetic");
    assert.ok(scenes[0].licenceUrl && scenes[0].synthetic);
    const buf = readFileSync(join(synth, "synthetic/points.ply"));
    assert.ok(F.parsePLY(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length)).count > 1000);
  });
}
console.log(`all ${n} format tests passed`);
