// SPDX-License-Identifier: Apache-2.0
// Readers for the files the interactive example loads. No three.js here, so the parsers can be tested in Node
// (site-src/dev/test_formats.mjs) and main.js can read the scene list without loading the viewer.
//
//   static/examples/index.json      written by examples/python/gallery_index.py (examples/make_gallery.sh)
//   <scene dir>/cameras.json        written by `rawatta run` (crates/vggt-omega/src/output.rs, CameraDoc)
//   <scene dir>/points.ply          written by `rawatta run`, subsampled by examples/python/subsample_ply.py

const TYPE_SIZE = { char: 1, int8: 1, uchar: 1, uint8: 1, short: 2, int16: 2, ushort: 2, uint16: 2,
  int: 4, int32: 4, uint: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8 };
const GETTER = { char: "getInt8", int8: "getInt8", uchar: "getUint8", uint8: "getUint8", short: "getInt16", int16: "getInt16",
  ushort: "getUint16", uint16: "getUint16", int: "getInt32", int32: "getInt32", uint: "getUint32", uint32: "getUint32",
  float: "getFloat32", float32: "getFloat32", double: "getFloat64", float64: "getFloat64" };

/**
 * Vertex element of a binary little-endian PLY file. `rawatta run` and subsample_ply.py write
 * `x y z` (float) and `red green blue` (uchar). Returns {positions: Float32Array, colors: Uint8Array|null, count};
 * vertices with a non-finite coordinate are dropped.
 */
export function parsePLY(buffer) {
  const bytes = new Uint8Array(buffer);
  const head = new TextDecoder("latin1").decode(bytes.subarray(0, Math.min(bytes.length, 65536)));
  const end = head.indexOf("end_header");
  if (!head.startsWith("ply") || end < 0) throw new Error("not a PLY file");
  let start = end + "end_header".length;
  if (head[start] === "\r") start++;
  if (head[start] === "\n") start++;
  let format = "", count = 0, inVertex = false, seenVertex = false;
  const props = [];
  for (const line of head.slice(0, end).split(/\r?\n/)) {
    const t = line.trim().split(/\s+/);
    if (t[0] === "format") format = t[1];
    else if (t[0] === "element") {
      inVertex = t[1] === "vertex";
      if (inVertex) { count = parseInt(t[2], 10) || 0; seenVertex = true; }
      else if (!seenVertex) throw new Error("the vertex element must come first");
    } else if (t[0] === "property" && inVertex) {
      if (t[1] === "list") throw new Error("list properties on vertices are not supported");
      props.push({ type: t[1], name: t[2] });
    }
  }
  if (format !== "binary_little_endian") throw new Error(`unsupported PLY format "${format}" (need binary_little_endian)`);
  let stride = 0;
  const field = {};
  for (const p of props) {
    if (!TYPE_SIZE[p.type]) throw new Error(`unknown PLY type ${p.type}`);
    field[p.name] = { offset: stride, get: GETTER[p.type], type: p.type };
    stride += TYPE_SIZE[p.type];
  }
  if (!field.x || !field.y || !field.z) throw new Error("PLY vertices need x, y, z");
  if (start + count * stride > bytes.length) throw new Error("PLY file is truncated");
  const rgb = field.red && field.green && field.blue ? [field.red, field.green, field.blue] : null;
  const scale = rgb ? (rgb[0].type.startsWith("float") || rgb[0].type.startsWith("double") ? 255 : (TYPE_SIZE[rgb[0].type] === 2 ? 1 / 257 : 1)) : 1;
  const dv = new DataView(bytes.buffer, bytes.byteOffset + start, count * stride);
  const positions = new Float32Array(count * 3);
  const colors = rgb ? new Uint8Array(count * 3) : null;
  const fx = field.x, fy = field.y, fz = field.z;
  let n = 0;
  for (let i = 0, o = 0; i < count; i++, o += stride) {
    const x = dv[fx.get](o + fx.offset, true), y = dv[fy.get](o + fy.offset, true), z = dv[fz.get](o + fz.offset, true);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    positions[3 * n] = x; positions[3 * n + 1] = y; positions[3 * n + 2] = z;
    if (colors) {
      for (let c = 0; c < 3; c++) {
        const v = dv[rgb[c].get](o + rgb[c].offset, true) * scale;
        colors[3 * n + c] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
    n++;
  }
  return { positions: positions.subarray(0, 3 * n), colors: colors ? colors.subarray(0, 3 * n) : null, count: n };
}

/** A rows x cols matrix given as nested arrays (the serde layout of `[[f32; cols]; rows]`), flattened row-major. */
function matrix(v, rows, cols) {
  if (!Array.isArray(v) || v.length !== rows) return null;
  const out = [];
  for (const row of v) {
    if (!Array.isArray(row) || row.length !== cols) return null;
    for (const x of row) {
      if (typeof x !== "number" || !Number.isFinite(x)) return null;
      out.push(x);
    }
  }
  return out;
}

/**
 * cameras.json exactly as `rawatta run` writes it (crates/vggt-omega/src/output.rs, CameraDoc / FrameDoc):
 *   {"convention": "opencv w2c (x right, y down, z forward); world = frame 0; model-scale units",
 *    "width": W, "height": H,
 *    "frames": [{"image": "name.jpg", "w2c": [[r00, r01, r02, t0], [..], [..]], "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
 *                "fov_y": rad, "fov_x": rad, "pose_enc": [9 numbers]}, ...]}
 * The intrinsics are in the preprocessed image's pixels (W x H). Returns
 * {cams: [{E: 12 numbers (w2c row-major), K: 9 numbers, width, height, name, index}], skipped}; `index` is the
 * position in `frames`, so index 0 is the world frame. Throws if the document does not have this shape.
 */
export function parseCameras(doc) {
  if (!doc || typeof doc !== "object" || Array.isArray(doc) || !Array.isArray(doc.frames)) {
    throw new Error('cameras.json: expected {"width", "height", "frames": [...]} as written by rawatta run');
  }
  const W = Number(doc.width), H = Number(doc.height);
  const cams = [];
  let skipped = 0;
  doc.frames.forEach((f, i) => {
    const E = f && typeof f === "object" ? matrix(f.w2c, 3, 4) : null;
    const K = f && typeof f === "object" ? matrix(f.intrinsics, 3, 3) : null;
    if (!E || !K || !(K[0] > 0) || !(K[4] > 0)) { skipped++; return; }
    cams.push({
      E, K,
      width: W > 0 ? W : 2 * K[2],
      height: H > 0 ? H : 2 * K[5],
      name: typeof f.image === "string" ? f.image : `frame ${i}`,
      index: i,
    });
  });
  if (doc.frames.length && !cams.length) throw new Error("cameras.json: no frame has a 3x4 w2c and 3x3 intrinsics");
  return { cams, skipped };
}

/** Camera centre in world coordinates of a w2c [R | t] (12 numbers, row-major): C = -R^T t. */
export function cameraCenter(E) {
  const t = [E[3], E[7], E[11]];
  return [0, 1, 2].map((k) => -(E[k] * t[0] + E[4 + k] * t[1] + E[8 + k] * t[2]));
}

/** The string if it is an absolute http(s) URL, else null (keeps javascript: and similar out of links). */
export function httpUrl(u) {
  if (typeof u !== "string" || !u) return null;
  try {
    const url = new URL(u);
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : null;
  } catch (_) {
    return null;
  }
}

/** A path relative to the folder of index.json: no scheme, no leading slash, no "..". */
function relPath(p, fallback) {
  if (p === undefined || p === null || p === "") return fallback;
  if (typeof p !== "string") return null;
  const s = p.replace(/^\.\/+/, "").replace(/\/+$/, "");
  if (!s || /^[a-z][a-z0-9+.-]*:/i.test(s) || s.startsWith("/") || s.split("/").includes("..")) return null;
  return s;
}

/**
 * static/examples/index.json as examples/python/gallery_index.py writes it:
 *   {"scenes": [{"id", "title", "dir", "points", "cameras", "thumb", "caption",
 *                "frames", "config", "latency_ms", "latency_iqr_ms", "reps", "warmup", "gpu", "point_count",
 *                "licence", "licence_url", "source", "author", "attribution", "changes", "kind", ["synthetic"]}]}
 * `dir` is relative to the folder of index.json; `points` and `cameras` are file names in it. Returns the usable
 * scenes with the fields the page shows; entries without an id or with unsafe paths are left out.
 */
export function parseSceneIndex(doc) {
  const list = doc && typeof doc === "object" && Array.isArray(doc.scenes) ? doc.scenes : [];
  const out = [];
  for (const s of list) {
    if (!s || typeof s !== "object" || typeof s.id !== "string" || !s.id) continue;
    const dir = relPath(s.dir, s.id), points = relPath(s.points, "points.ply"), cameras = relPath(s.cameras, "cameras.json");
    if (!dir || !points || !cameras) continue;
    const str = (v) => (typeof v === "string" && v.trim() ? v.trim() : null);
    out.push({
      id: s.id,
      title: str(s.title) || s.id,
      dir, points, cameras,
      caption: str(s.caption),
      licence: str(s.licence),
      licenceUrl: httpUrl(s.licence_url),
      source: httpUrl(s.source),
      synthetic: s.synthetic === true,
    });
  }
  return out;
}
