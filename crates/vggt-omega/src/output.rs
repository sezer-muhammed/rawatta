// SPDX-License-Identifier: LicenseRef-FAIR-Noncommercial-Research
// A port of Meta's VGGT-Ω architecture; see LICENSE-VGGT-OMEGA.
//! Engine outputs on disk: frame sampling, coloured point clouds (a plain
//! `x y z red green blue` PLY, and an INRIA-layout PLY of small isotropic
//! Gaussians for splat viewers), a COLMAP text model, `.npy` arrays and
//! `cameras.json`.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::{DenseHost, Prepared, VggtCameras};

/// Evenly spaced indices: `n` of `0..len`, first and last included.
pub fn sample_indices(len: usize, n: usize) -> Vec<usize> {
    if len <= n {
        return (0..len).collect();
    }
    if n == 1 {
        return vec![0];
    }
    (0..n)
        .map(|i| (i * (len - 1) + (n - 1) / 2) / (n - 1))
        .collect()
}

/// One coloured point; `size` is the footprint of one pixel at its depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub xyz: [f32; 3],
    pub rgb: [u8; 3],
    pub size: f32,
}

/// World points of every pixel whose confidence is above the
/// `percentile`-th percentile, thinned by a regular stride to at most
/// `max_points`.
pub fn point_cloud(p: &Prepared, d: &DenseHost, percentile: f32, max_points: usize) -> Vec<Point> {
    let g = p.geom;
    let n = g.frames * g.pixels();
    if n == 0 || d.conf.len() != n || d.world.len() != n * 3 {
        return Vec::new();
    }
    let mut sorted: Vec<f32> = d.conf.iter().copied().filter(|c| c.is_finite()).collect();
    sorted.sort_by(f32::total_cmp);
    let q = (percentile.clamp(0.0, 100.0) / 100.0 * (sorted.len().max(1) - 1) as f32) as usize;
    let thr = sorted.get(q).copied().unwrap_or(f32::INFINITY);
    let kept = d.conf.iter().filter(|&&c| c >= thr).count();
    let stride = kept.div_ceil(max_points.max(1)).max(1);
    let mut out = Vec::with_capacity(kept / stride + 1);
    let mut seen = 0usize;
    for i in 0..n {
        if d.conf[i] < thr {
            continue;
        }
        seen += 1;
        if !(seen - 1).is_multiple_of(stride) {
            continue;
        }
        let xyz = [d.world[i * 3], d.world[i * 3 + 1], d.world[i * 3 + 2]];
        if !xyz.iter().all(|v| v.is_finite()) {
            continue;
        }
        let depth = d.depth.get(i).copied().unwrap_or(1.0);
        out.push(Point {
            xyz,
            rgb: [p.pixels[i * 3], p.pixels[i * 3 + 1], p.pixels[i * 3 + 2]],
            // One pixel at the image centre of a ~60° view: depth · 2·tan(30°) / W.
            size: depth * 1.1547 / g.width as f32 * (stride as f32).sqrt(),
        });
    }
    out
}

/// Coloured point cloud as a binary little-endian PLY: per vertex
/// `float x, y, z` and `uchar red, green, blue` (15 bytes), the layout every
/// point viewer (MeshLab, CloudCompare, Open3D, Blender) reads.
pub fn write_points_ply(pts: &[Point], out: &mut dyn Write) -> Result<()> {
    let h = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n\
         property float x\nproperty float y\nproperty float z\n\
         property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",
        pts.len()
    );
    out.write_all(h.as_bytes())?;
    let mut buf = Vec::with_capacity(pts.len() * 15);
    for p in pts {
        for v in p.xyz {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&p.rgb);
    }
    out.write_all(&buf)?;
    Ok(())
}

/// INRIA 3DGS PLY with degree-0 colour: one small isotropic Gaussian per
/// point (for splat viewers).
pub fn write_splat_ply(pts: &[Point], out: &mut dyn Write) -> Result<()> {
    const SH_C0: f32 = 0.282_094_8;
    let props = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0",
        "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ];
    let mut h = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n",
        pts.len()
    );
    for p in props {
        h.push_str(&format!("property float {p}\n"));
    }
    h.push_str("end_header\n");
    out.write_all(h.as_bytes())?;
    let mut buf = Vec::with_capacity(pts.len() * props.len() * 4);
    for p in pts {
        let s = p.size.max(1e-6).ln();
        let dc = p.rgb.map(|v| (v as f32 / 255.0 - 0.5) / SH_C0);
        let vals = [
            p.xyz[0], p.xyz[1], p.xyz[2], 0.0, 0.0, 0.0, dc[0], dc[1], dc[2], 4.0, s, s, s, 1.0,
            0.0, 0.0, 0.0,
        ];
        for v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    out.write_all(&buf)?;
    Ok(())
}

/// Rotation matrix (row-major) → quaternion `[w, x, y, z]`.
pub fn mat_to_quat(r: [[f32; 3]; 3]) -> [f64; 4] {
    let m = r.map(|row| row.map(|v| v as f64));
    let t = m[0][0] + m[1][1] + m[2][2];
    let q = if t > 0.0 {
        let s = (t + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        ]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        ]
    };
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let q = q.map(|v| v / n);
    if q[0] < 0.0 { q.map(|v| -v) } else { q }
}

/// COLMAP text model `(cameras.txt, images.txt, points3D.txt)`: one PINHOLE
/// camera per frame, poses, points without tracks.
pub fn colmap_text(names: &[String], c: &VggtCameras, pts: &[Point]) -> (String, String, String) {
    let mut cameras = String::from(
        "# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n",
    );
    let mut images = String::from(
        "# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
    );
    for (i, ((e, k), name)) in c
        .extrinsics
        .iter()
        .zip(&c.intrinsics)
        .zip(names)
        .enumerate()
    {
        let id = i + 1;
        cameras.push_str(&format!(
            "{id} PINHOLE {} {} {} {} {} {}\n",
            c.width, c.height, k[0], k[4], k[2], k[5]
        ));
        let q = mat_to_quat([[e[0], e[1], e[2]], [e[4], e[5], e[6]], [e[8], e[9], e[10]]]);
        let name = name.replace(char::is_whitespace, "_");
        images.push_str(&format!(
            "{id} {} {} {} {} {} {} {} {id} {name}\n\n",
            q[0], q[1], q[2], q[3], e[3], e[7], e[11]
        ));
    }
    let mut points = String::with_capacity(pts.len() * 48);
    points.push_str("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n");
    for (i, p) in pts.iter().enumerate() {
        points.push_str(&format!(
            "{} {} {} {} {} {} {} 0\n",
            i + 1,
            p.xyz[0],
            p.xyz[1],
            p.xyz[2],
            p.rgb[0],
            p.rgb[1],
            p.rgb[2]
        ));
    }
    (cameras, images, points)
}

/// `cameras.json`: per frame the image name, the world-to-camera 3×4
/// (OpenCV axes, world = frame 0), the 3×3 intrinsics in preprocessed
/// pixels (`width` × `height`), the fields of view (radians) and the raw
/// pose encoding.
#[derive(Debug, Serialize)]
pub struct CameraDoc<'a> {
    pub convention: &'static str,
    pub width: usize,
    pub height: usize,
    pub frames: Vec<FrameDoc<'a>>,
}

#[derive(Debug, Serialize)]
pub struct FrameDoc<'a> {
    /// Input file name (for a video: the extracted frame's name).
    pub image: &'a str,
    /// World-to-camera 3×4, row-major.
    pub w2c: [[f32; 4]; 3],
    /// 3×3 intrinsics, preprocessed pixels.
    pub intrinsics: [[f32; 3]; 3],
    pub fov_y: f32,
    pub fov_x: f32,
    pub pose_enc: [f32; 9],
}

pub fn cameras_doc<'a>(names: &'a [String], c: &VggtCameras) -> CameraDoc<'a> {
    CameraDoc {
        convention: "opencv w2c (x right, y down, z forward); world = frame 0; model-scale units",
        width: c.width,
        height: c.height,
        frames: names
            .iter()
            .zip(c.extrinsics.iter().zip(&c.intrinsics).zip(&c.pose_enc))
            .map(|(image, ((e, k), pe))| FrameDoc {
                image,
                w2c: [
                    [e[0], e[1], e[2], e[3]],
                    [e[4], e[5], e[6], e[7]],
                    [e[8], e[9], e[10], e[11]],
                ],
                intrinsics: [[k[0], k[1], k[2]], [k[3], k[4], k[5]], [k[6], k[7], k[8]]],
                fov_y: pe[7],
                fov_x: pe[8],
                pose_enc: *pe,
            })
            .collect(),
    }
}

/// A little-endian f32 `.npy` (format 1.0) of `shape`.
pub fn npy_bytes(shape: &[usize], data: &[f32]) -> Result<Vec<u8>> {
    ensure!(
        shape.iter().product::<usize>() == data.len(),
        "npy: shape {shape:?} does not hold {} values",
        data.len()
    );
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let tuple = if dims.len() == 1 {
        format!("{},", dims[0])
    } else {
        dims.join(", ")
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({tuple}), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend((header.len() as u16).to_le_bytes());
    b.extend(header.as_bytes());
    b.reserve(data.len() * 4);
    for v in data {
        b.extend(v.to_le_bytes());
    }
    Ok(b)
}

/// Writes [`npy_bytes`] to `path`, creating its directory.
pub fn write_npy(path: &Path, shape: &[usize], data: &[f32]) -> Result<()> {
    let b = npy_bytes(shape, data)?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    std::fs::write(path, b).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cams() -> VggtCameras {
        VggtCameras {
            pose_enc: vec![
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 1.1],
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 1.1],
            ],
            extrinsics: vec![
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                // 90° about z, then t = (0.5, -1, 2).
                [0.0, -1.0, 0.0, 0.5, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 2.0],
            ],
            intrinsics: vec![[400.0, 0.0, 256.0, 0.0, 410.0, 168.0, 0.0, 0.0, 1.0]; 2],
            width: 512,
            height: 336,
        }
    }

    fn pts() -> Vec<Point> {
        vec![
            Point {
                xyz: [1.0, -2.0, 3.5],
                rgb: [255, 0, 7],
                size: 0.01,
            },
            Point {
                xyz: [0.25, 0.5, -0.125],
                rgb: [1, 2, 3],
                size: 0.01,
            },
        ]
    }

    #[test]
    fn points_ply_header_and_records() {
        let mut b = Vec::new();
        write_points_ply(&pts(), &mut b).unwrap();
        let end = b.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
        let header = std::str::from_utf8(&b[..end]).unwrap();
        assert_eq!(
            header,
            "ply\nformat binary_little_endian 1.0\nelement vertex 2\n\
             property float x\nproperty float y\nproperty float z\n\
             property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        );
        let body = &b[end..];
        assert_eq!(body.len(), 2 * 15);
        let f = |o: usize| f32::from_le_bytes(body[o..o + 4].try_into().unwrap());
        assert_eq!([f(0), f(4), f(8)], [1.0, -2.0, 3.5]);
        assert_eq!(&body[12..15], &[255, 0, 7]);
        assert_eq!([f(15), f(19), f(23)], [0.25, 0.5, -0.125]);
        assert_eq!(&body[27..30], &[1, 2, 3]);
        let mut e = Vec::new();
        write_points_ply(&[], &mut e).unwrap();
        assert!(
            std::str::from_utf8(&e)
                .unwrap()
                .contains("element vertex 0\n")
        );
    }

    #[test]
    fn colmap_text_format() {
        let names = vec!["a.png".to_string(), "frame 2.png".to_string()];
        let (cams, imgs, points) = colmap_text(&names, &two_cams(), &pts());
        let data = |s: &str| {
            s.lines()
                .filter(|l| !l.starts_with('#'))
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        assert!(cams.starts_with("# Camera list"));
        assert_eq!(
            data(&cams),
            [
                "1 PINHOLE 512 336 400 410 256 168",
                "2 PINHOLE 512 336 400 410 256 168"
            ]
        );
        // Two lines per image: the pose line and an empty POINTS2D line.
        let lines = data(&imgs);
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], "1 1 0 0 0 0 0 0 1 a.png");
        assert_eq!(lines[1], "");
        let f: Vec<&str> = lines[2].split(' ').collect();
        assert_eq!(f.len(), 10);
        assert_eq!((f[0], f[8], f[9]), ("2", "2", "frame_2.png"));
        let q: Vec<f64> = f[1..5].iter().map(|v| v.parse().unwrap()).collect();
        let h = std::f64::consts::FRAC_1_SQRT_2;
        for (a, b) in q.iter().zip([h, 0.0, 0.0, h]) {
            assert!((a - b).abs() < 1e-6, "{q:?}");
        }
        assert_eq!(&f[5..8], &["0.5", "-1", "2"]);
        assert_eq!(lines[3], "");
        assert_eq!(
            data(&points),
            ["1 1 -2 3.5 255 0 7 0", "2 0.25 0.5 -0.125 1 2 3 0"]
        );
    }

    #[test]
    fn cameras_json_schema() {
        let names = vec!["a.png".to_string(), "b.png".to_string()];
        let v = serde_json::to_value(cameras_doc(&names, &two_cams())).unwrap();
        let top: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(top, ["convention", "frames", "height", "width"]);
        assert_eq!(
            (v["width"].as_u64(), v["height"].as_u64()),
            (Some(512), Some(336))
        );
        let frames = v["frames"].as_array().unwrap();
        assert_eq!(frames.len(), 2);
        let keys: Vec<&str> = frames[1]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            ["fov_x", "fov_y", "image", "intrinsics", "pose_enc", "w2c"]
        );
        assert_eq!(frames[1]["image"], "b.png");
        let w2c = frames[1]["w2c"].as_array().unwrap();
        assert_eq!(w2c.len(), 3);
        assert!(w2c.iter().all(|r| r.as_array().unwrap().len() == 4));
        assert_eq!(w2c[0][3].as_f64(), Some(0.5));
        assert_eq!(w2c[2][3].as_f64(), Some(2.0));
        let k = frames[1]["intrinsics"].as_array().unwrap();
        assert_eq!(k.len(), 3);
        assert_eq!(k[0][0].as_f64(), Some(400.0));
        assert_eq!(k[1][2].as_f64(), Some(168.0));
        assert_eq!(k[2][2].as_f64(), Some(1.0));
        assert_eq!(frames[1]["pose_enc"].as_array().unwrap().len(), 9);
        assert!((frames[1]["fov_y"].as_f64().unwrap() - 0.9).abs() < 1e-6);
        assert!((frames[1]["fov_x"].as_f64().unwrap() - 1.1).abs() < 1e-6);
    }

    #[test]
    fn npy_shapes_round_trip() {
        let dir = std::env::temp_dir().join(format!("rawatta-npy-{}", std::process::id()));
        let p = dir.join("sub/extrinsics.npy");
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
        write_npy(&p, &[2, 3, 4], &data).unwrap();
        let b = std::fs::read(&p).unwrap();
        let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
        let header = std::str::from_utf8(&b[10..10 + hl]).unwrap();
        assert!(
            header.starts_with("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3, 4), }")
        );
        assert_eq!(b.len(), 10 + hl + 24 * 4);
        assert_eq!(
            f32::from_le_bytes(b[10 + hl + 4..10 + hl + 8].try_into().unwrap()),
            0.5
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn npy_header_is_aligned_and_parsable() {
        let b = npy_bytes(&[2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(&b[..8], b"\x93NUMPY\x01\x00");
        let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
        assert_eq!((10 + hl) % 64, 0);
        let header = std::str::from_utf8(&b[10..10 + hl]).unwrap();
        assert!(header.contains("'shape': (2, 3)"), "{header}");
        assert!(header.ends_with('\n'));
        let vals: Vec<f32> = b[10 + hl..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(vals, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(npy_bytes(&[4], &[1.0]).is_err());
        let one = npy_bytes(&[1], &[1.0]).unwrap();
        assert!(
            std::str::from_utf8(&one[10..one.len() - 4])
                .unwrap()
                .contains("(1,)")
        );
    }
}
