// SPDX-License-Identifier: LicenseRef-FAIR-Noncommercial-Research
// A port of Meta's VGGT-Ω architecture; see LICENSE-VGGT-OMEGA.
//! Multi-view geometry metrics in the VGGT / VGGT-Ω conventions.
//!
//! - Pose: relative pose over every frame pair `(i < j)`, rotation error
//!   from quaternions (`acos(1 - 2·(1 - <q, q'>²))`) and translation-direction
//!   error folded to [0, 90] degrees; AUC@k is the mean of the cumulative
//!   histogram of `max(rot, trans)` over integer bins `0..k` (×100).
//!   RRA@k / RTA@k are the fractions of pairs under k degrees.
//! - Depth: per-frame median scale + shift alignment, mask
//!   `0.1 < gt < 100`, δ<1.25 and AbsRel.
//! - Points: accuracy (pred → GT nearest distance), completeness
//!   (GT → pred) and their mean (Chamfer), after Umeyama Sim(3) alignment.
//!
//! Re-implemented from the definitions in VGGT-Ω's `eval/metrics.py`
//! (Apache-2.0, read for parity); `parity_with_official` checks every number
//! against outputs of that file on real sequences.

// Small fixed-size linear algebra reads clearer with index loops.
#![allow(clippy::needless_range_loop)]

/// Row-major `[R | t]`, world-to-camera.
pub type Extr = [[f64; 4]; 3];

type M3 = [[f64; 3]; 3];
type V3 = [f64; 3];

fn inv(e: &Extr) -> Extr {
    let mut o = [[0.0; 4]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = e[j][i];
        }
    }
    for i in 0..3 {
        o[i][3] = -(0..3).map(|k| o[i][k] * e[k][3]).sum::<f64>();
    }
    o
}

fn compose(a: &Extr, b: &Extr) -> Extr {
    let mut o = [[0.0; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            o[i][j] =
                (0..3).map(|k| a[i][k] * b[k][j]).sum::<f64>() + if j == 3 { a[i][3] } else { 0.0 };
        }
    }
    o
}

/// The official `_matrix_to_quaternion` (xyzw, w ≥ 0).
fn quat(m: &Extr) -> [f64; 4] {
    let (m00, m01, m02) = (m[0][0], m[0][1], m[0][2]);
    let (m10, m11, m12) = (m[1][0], m[1][1], m[1][2]);
    let (m20, m21, m22) = (m[2][0], m[2][1], m[2][2]);
    let sp = |v: f64| if v > 0.0 { v.sqrt() } else { 0.0 };
    let mag = [
        sp(1.0 + m00 + m11 + m22),
        sp(1.0 + m00 - m11 - m22),
        sp(1.0 - m00 + m11 - m22),
        sp(1.0 - m00 - m11 + m22),
    ];
    let cand = [
        [mag[0] * mag[0], m21 - m12, m02 - m20, m10 - m01],
        [m21 - m12, mag[1] * mag[1], m10 + m01, m02 + m20],
        [m02 - m20, m10 + m01, mag[2] * mag[2], m12 + m21],
        [m10 - m01, m20 + m02, m21 + m12, mag[3] * mag[3]],
    ];
    // argmax: first maximum, as torch.argmax.
    let mut k = 0;
    for i in 1..4 {
        if mag[i] > mag[k] {
            k = i;
        }
    }
    let d = 2.0 * mag[k].max(0.1);
    let wxyz = cand[k].map(|v| v / d);
    let q = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]];
    if q[3] < 0.0 { q.map(|v| -v) } else { q }
}

/// Rotation and translation-direction errors in degrees for every pair
/// `(i, j)`, `i < j`, in `torch.combinations` order.
pub fn pose_errors(pred: &[Extr], gt: &[Extr]) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(pred.len(), gt.len());
    let (mut rot, mut tr) = (Vec::new(), Vec::new());
    for i in 0..gt.len() {
        for j in i + 1..gt.len() {
            let rg = compose(&gt[j], &inv(&gt[i]));
            let rp = compose(&pred[j], &inv(&pred[i]));
            let (qg, qp) = (quat(&rg), quat(&rp));
            let dq: f64 = (0..4).map(|k| qg[k] * qp[k]).sum();
            let loss = (1.0 - dq * dq).max(1e-15);
            rot.push((1.0 - 2.0 * loss).acos().to_degrees());
            let unit = |e: &Extr| {
                let n = (e[0][3].powi(2) + e[1][3].powi(2) + e[2][3].powi(2)).sqrt() + 1e-15;
                [e[0][3] / n, e[1][3] / n, e[2][3] / n]
            };
            let (tg, tp) = (unit(&rg), unit(&rp));
            let d: f64 = (0..3).map(|k| tg[k] * tp[k]).sum();
            let tl = (1.0 - d * d).max(1e-15);
            let t = (1.0 - tl).sqrt().acos().to_degrees();
            tr.push(t.min((180.0 - t).abs()));
        }
    }
    (rot, tr)
}

/// AUC@k in percent (official `_auc`).
pub fn auc(rot: &[f64], tr: &[f64], k: usize) -> f64 {
    assert!(!rot.is_empty() && rot.len() == tr.len());
    let mut hist = vec![0usize; k];
    for (r, t) in rot.iter().zip(tr) {
        let e = r.max(*t);
        // np.histogram bins [0,1), ..., [k-1, k] (last bin closed).
        if (0.0..=k as f64).contains(&e) {
            hist[(e.floor() as usize).min(k - 1)] += 1;
        }
    }
    let n = rot.len() as f64;
    let mut c = 0.0;
    let mut s = 0.0;
    for h in hist {
        c += h as f64 / n;
        s += c;
    }
    s / k as f64 * 100.0
}

/// (RRA@k, RTA@k) in percent.
pub fn rra_rta(rot: &[f64], tr: &[f64], k: f64) -> (f64, f64) {
    let n = rot.len() as f64;
    let f = |v: &[f64]| v.iter().filter(|e| **e < k).count() as f64 / n * 100.0;
    (f(rot), f(tr))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthMetrics {
    pub delta125: f64,
    pub abs_rel: f64,
    pub frames: usize,
}

fn median(v: &mut [f64]) -> f64 {
    // np.median: mean of the two middle values for even lengths.
    let n = v.len();
    v.sort_by(f64::total_cmp);
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Official `depth_metrics` on frames of `h·w` values (pred already resized
/// to the GT); frames with < 10 valid pixels are skipped.
pub fn depth_metrics(pred: &[f32], gt: &[f32], hw: usize) -> Option<DepthMetrics> {
    assert_eq!(pred.len(), gt.len());
    let (mut d, mut a) = (Vec::new(), Vec::new());
    for (p, g) in pred.chunks(hw).zip(gt.chunks(hw)) {
        let idx: Vec<usize> = (0..hw)
            .filter(|&i| g[i].is_finite() && g[i] > 0.1 && g[i] < 100.0)
            .collect();
        if idx.len() < 10 {
            continue;
        }
        let gm: Vec<f64> = idx
            .iter()
            .map(|&i| (g[i] as f64).clamp(0.1, 100.0))
            .collect();
        let pm: Vec<f64> = idx
            .iter()
            .map(|&i| (p[i] as f64).clamp(0.1, 100.0))
            .collect();
        let mg = median(&mut gm.clone());
        let mp = median(&mut pm.clone());
        let mut ratio: Vec<f64> = gm
            .iter()
            .zip(&pm)
            .map(|(g, p)| (g - mg + 1e-8) / (p - mp + 1e-8))
            .collect();
        let scale = median(&mut ratio);
        let mut res: Vec<f64> = gm.iter().zip(&pm).map(|(g, p)| g - scale * p).collect();
        let shift = median(&mut res);
        let (mut ar, mut dl) = (0.0, 0usize);
        for (g, p) in gm.iter().zip(&pm) {
            let al = (p * scale + shift).max(1e-6);
            ar += (al - g).abs() / g;
            if (al / g).max(g / al) < 1.25 {
                dl += 1;
            }
        }
        a.push(ar / gm.len() as f64);
        d.push(dl as f64 / gm.len() as f64);
    }
    if d.is_empty() {
        return None;
    }
    Some(DepthMetrics {
        delta125: d.iter().sum::<f64>() / d.len() as f64,
        abs_rel: a.iter().sum::<f64>() / a.len() as f64,
        frames: d.len(),
    })
}

// ------------------------------------------------------------ point clouds

/// Umeyama similarity `b ≈ s·R·a + t` (least squares).
pub fn umeyama(a: &[V3], b: &[V3]) -> (f64, M3, V3) {
    let n = a.len() as f64;
    let mean = |p: &[V3]| {
        let mut m = [0.0; 3];
        for v in p {
            for k in 0..3 {
                m[k] += v[k] / n;
            }
        }
        m
    };
    let (ma, mb) = (mean(a), mean(b));
    let mut cov = [[0.0; 3]; 3];
    let mut va = 0.0;
    for (x, y) in a.iter().zip(b) {
        for i in 0..3 {
            va += (x[i] - ma[i]).powi(2) / n;
            for j in 0..3 {
                cov[i][j] += (y[i] - mb[i]) * (x[j] - ma[j]) / n;
            }
        }
    }
    let (u, sig, v) = svd3(&cov);
    let mut d = [1.0, 1.0, 1.0];
    if det(&u) * det(&v) < 0.0 {
        d[2] = -1.0;
    }
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (0..3).map(|k| u[i][k] * d[k] * v[j][k]).sum();
        }
    }
    let s = (0..3).map(|k| sig[k] * d[k]).sum::<f64>() / va.max(1e-300);
    let mut t = [0.0; 3];
    for i in 0..3 {
        t[i] = mb[i] - s * (0..3).map(|k| r[i][k] * ma[k]).sum::<f64>();
    }
    (s, r, t)
}

fn det(m: &M3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// SVD of a 3×3 matrix via Jacobi on `AᵀA`: `A = U·diag(σ)·Vᵀ`, σ descending.
fn svd3(a: &M3) -> (M3, V3, M3) {
    let mut s = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            s[i][j] = (0..3).map(|k| a[k][i] * a[k][j]).sum();
        }
    }
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..64 {
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if s[p][q].abs() < 1e-300 {
                continue;
            }
            let th = 0.5 * (s[q][q] - s[p][p]) / s[p][q];
            let t = th.signum() / (th.abs() + (th * th + 1.0).sqrt());
            let t = if th == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let sn = t * c;
            for k in 0..3 {
                let (skp, skq) = (s[k][p], s[k][q]);
                s[k][p] = c * skp - sn * skq;
                s[k][q] = sn * skp + c * skq;
            }
            for k in 0..3 {
                let (spk, sqk) = (s[p][k], s[q][k]);
                s[p][k] = c * spk - sn * sqk;
                s[q][k] = sn * spk + c * sqk;
            }
            for row in v.iter_mut() {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - sn * vq;
                row[q] = sn * vp + c * vq;
            }
        }
    }
    let mut ord = [0usize, 1, 2];
    ord.sort_by(|&x, &y| s[y][y].total_cmp(&s[x][x]));
    let mut vs = [[0.0; 3]; 3];
    let mut sig = [0.0; 3];
    for (c, &o) in ord.iter().enumerate() {
        sig[c] = s[o][o].max(0.0).sqrt();
        for r in 0..3 {
            vs[r][c] = v[r][o];
        }
    }
    let mut u = [[0.0; 3]; 3];
    for c in 0..3 {
        let av: V3 = std::array::from_fn(|r| (0..3).map(|k| a[r][k] * vs[k][c]).sum());
        let n = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 1e-12 {
            for r in 0..3 {
                u[r][c] = av[r] / n;
            }
        } else {
            // Rank-deficient column: complete to an orthonormal basis.
            let (a0, a1) = (c.wrapping_sub(1) % 3, (c + 1) % 3);
            let x = [u[0][a0], u[1][a0], u[2][a0]];
            let y = [u[0][a1], u[1][a1], u[2][a1]];
            let z = [
                x[1] * y[2] - x[2] * y[1],
                x[2] * y[0] - x[0] * y[2],
                x[0] * y[1] - x[1] * y[0],
            ];
            for r in 0..3 {
                u[r][c] = z[r];
            }
        }
    }
    (u, sig, vs)
}

/// A static 3-d tree for exact nearest-neighbour distances.
pub struct KdTree {
    pts: Vec<V3>,
}

impl KdTree {
    pub fn new(mut pts: Vec<V3>) -> Self {
        fn build(p: &mut [V3], depth: usize) {
            if p.len() <= 1 {
                return;
            }
            let ax = depth % 3;
            let m = p.len() / 2;
            p.select_nth_unstable_by(m, |a, b| a[ax].total_cmp(&b[ax]));
            let (l, r) = p.split_at_mut(m);
            build(l, depth + 1);
            build(&mut r[1..], depth + 1);
        }
        build(&mut pts, 0);
        Self { pts }
    }

    pub fn nearest_sq(&self, q: &V3) -> f64 {
        fn go(p: &[V3], depth: usize, q: &V3, best: &mut f64) {
            if p.is_empty() {
                return;
            }
            let m = p.len() / 2;
            let c = &p[m];
            let d = (0..3).map(|k| (c[k] - q[k]).powi(2)).sum::<f64>();
            if d < *best {
                *best = d;
            }
            let ax = depth % 3;
            let diff = q[ax] - c[ax];
            let (near, far) = if diff < 0.0 {
                (&p[..m], &p[m + 1..])
            } else {
                (&p[m + 1..], &p[..m])
            };
            go(near, depth + 1, q, best);
            if diff * diff < *best {
                go(far, depth + 1, q, best);
            }
        }
        let mut best = f64::INFINITY;
        go(&self.pts, 0, q, &mut best);
        best
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointMetrics {
    /// Mean ℓ2 between corresponding points after alignment (Ω's point error).
    pub point_error: f64,
    pub accuracy: f64,
    pub completeness: f64,
    pub chamfer: f64,
}

/// `pred[i]` corresponds to `gt[i]` (same pixel); the prediction is Sim(3)
/// aligned to the GT on these correspondences first.
pub fn point_metrics(pred: &[V3], gt: &[V3]) -> PointMetrics {
    use rayon::prelude::*;
    let (s, r, t) = umeyama(pred, gt);
    let al: Vec<V3> = pred
        .iter()
        .map(|p| std::array::from_fn(|i| s * (0..3).map(|k| r[i][k] * p[k]).sum::<f64>() + t[i]))
        .collect();
    let n = gt.len() as f64;
    let pe = al
        .iter()
        .zip(gt)
        .map(|(a, g)| (0..3).map(|k| (a[k] - g[k]).powi(2)).sum::<f64>().sqrt())
        .sum::<f64>()
        / n;
    let tg = KdTree::new(gt.to_vec());
    let tp = KdTree::new(al.clone());
    let acc = al.par_iter().map(|p| tg.nearest_sq(p).sqrt()).sum::<f64>() / al.len() as f64;
    let comp = gt.par_iter().map(|g| tp.nearest_sq(g).sqrt()).sum::<f64>() / n;
    PointMetrics {
        point_error: pe,
        accuracy: acc,
        completeness: comp,
        chamfer: 0.5 * (acc + comp),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rz(deg: f64, t: V3) -> Extr {
        let (s, c) = deg.to_radians().sin_cos();
        [[c, -s, 0.0, t[0]], [s, c, 0.0, t[1]], [0.0, 0.0, 1.0, t[2]]]
    }

    #[test]
    fn identical_poses_score_full_auc() {
        let p: Vec<Extr> = (0..5)
            .map(|i| rz(i as f64 * 7.0, [i as f64, 0.5, 1.0]))
            .collect();
        let (r, t) = pose_errors(&p, &p);
        assert_eq!(r.len(), 10);
        assert!(r.iter().chain(&t).all(|e| *e < 1e-3));
        assert!(auc(&r, &t, 30) > 99.9);
        assert_eq!(rra_rta(&r, &t, 5.0), (100.0, 100.0));
    }

    #[test]
    fn rotation_error_is_the_relative_angle() {
        let gt = [rz(0.0, [1.0, 0.0, 0.0]), rz(10.0, [1.0, 0.0, 0.0])];
        let pr = [rz(0.0, [1.0, 0.0, 0.0]), rz(14.0, [1.0, 0.0, 0.0])];
        let (r, _) = pose_errors(&pr, &gt);
        assert!((r[0] - 4.0).abs() < 1e-6, "{r:?}");
    }

    #[test]
    fn depth_is_invariant_to_scale_and_shift() {
        let gt: Vec<f32> = (0..400).map(|i| 1.0 + (i % 37) as f32 * 0.1).collect();
        let pr: Vec<f32> = gt.iter().map(|g| 2.0 * g + 0.3).collect();
        let m = depth_metrics(&pr, &gt, 200).unwrap();
        assert_eq!(m.frames, 2);
        assert!(m.abs_rel < 1e-6 && (m.delta125 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn umeyama_recovers_a_similarity_and_points_align() {
        let a: Vec<V3> = (0..200)
            .map(|i| {
                let f = i as f64;
                [(f * 0.37).sin(), (f * 0.11).cos(), (f * 0.05).sin() * 2.0]
            })
            .collect();
        let e = rz(33.0, [0.2, -1.0, 3.0]);
        let b: Vec<V3> = a
            .iter()
            .map(|p| {
                std::array::from_fn(|i| 2.5 * (0..3).map(|k| e[i][k] * p[k]).sum::<f64>() + e[i][3])
            })
            .collect();
        let (s, _, _) = umeyama(&a, &b);
        assert!((s - 2.5).abs() < 1e-9);
        let m = point_metrics(&a, &b);
        assert!(m.point_error < 1e-9 && m.chamfer < 1e-9, "{m:?}");
    }

    #[test]
    fn kdtree_matches_brute_force() {
        let pts: Vec<V3> = (0..500)
            .map(|i| {
                let f = i as f64;
                [(f * 1.3).sin(), (f * 0.7).cos(), (f * 0.3).sin()]
            })
            .collect();
        let t = KdTree::new(pts.clone());
        for q in [[0.1, 0.2, 0.3], [2.0, -1.0, 0.0], [0.0, 0.0, 0.0]] {
            let bf = pts
                .iter()
                .map(|p| (0..3).map(|k| (p[k] - q[k]).powi(2)).sum::<f64>())
                .fold(f64::INFINITY, f64::min);
            assert_eq!(t.nearest_sq(&q), bf);
        }
    }

    /// Outputs of the official `eval/metrics.py` on real sequences. The
    /// fixture holds ground-truth poses and depth of dataset frames, so it is
    /// not committed: `$RAWATTA_METRICS_FIXTURE`, default
    /// `$RAWATTA_DATA/fixtures/omega_metrics_parity.json` (`/data` when unset).
    #[test]
    #[ignore]
    fn parity_with_official() {
        let p = std::env::var_os("RAWATTA_METRICS_FIXTURE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var_os("RAWATTA_DATA")
                    .map_or_else(|| "/data".into(), std::path::PathBuf::from)
                    .join("fixtures/omega_metrics_parity.json")
            });
        let Ok(bytes) = std::fs::read(&p) else {
            panic!("missing {}", p.display());
        };
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let ex = |v: &serde_json::Value| -> Vec<Extr> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    std::array::from_fn(|i| std::array::from_fn(|j| m[i][j].as_f64().unwrap()))
                })
                .collect()
        };
        let f32s = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let mut all_r = Vec::new();
        let mut all_t = Vec::new();
        for s in v["sequences"].as_array().unwrap() {
            let (r, t) = pose_errors(&ex(&s["pred"]), &ex(&s["gt"]));
            for (k, e) in r.iter().enumerate() {
                assert!(
                    (e - s["rotation"][k].as_f64().unwrap()).abs() < 1e-6,
                    "{} rot {k}",
                    s["name"]
                );
                assert!(
                    (t[k] - s["translation"][k].as_f64().unwrap()).abs() < 1e-6,
                    "{} trans {k}",
                    s["name"]
                );
            }
            all_r.extend(r);
            all_t.extend(t);
            if let Some(d) = s.get("depth") {
                let hw = d["h"].as_u64().unwrap() as usize * d["w"].as_u64().unwrap() as usize;
                let m = depth_metrics(&f32s(&d["pred"]), &f32s(&d["gt"]), hw).unwrap();
                assert!(
                    (m.delta125 - d["delta125"].as_f64().unwrap()).abs() < 1e-9,
                    "{}",
                    s["name"]
                );
                assert!(
                    (m.abs_rel - d["AbsRel"].as_f64().unwrap()).abs() < 1e-7,
                    "{}",
                    s["name"]
                );
            }
        }
        for k in [3usize, 30] {
            let o = v["pooled"][format!("AUC@{k}")].as_f64().unwrap();
            assert!((auc(&all_r, &all_t, k) - o).abs() < 1e-9, "AUC@{k}");
        }
    }
}
