// SPDX-License-Identifier: Apache-2.0
//! Uniform frame sampling from a video with the `ffprobe` / `ffmpeg`
//! command-line tools (run as subprocesses; not linked).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};
use vggt_omega::output::sample_indices;

/// A directory under the system temp dir, removed on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Result<Self> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let p = std::env::temp_dir().join(format!("rawatta-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&p).with_context(|| format!("create {}", p.display()))?;
        Ok(Self(p))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tool(name: &str) -> Command {
    let mut c = Command::new(name);
    c.stdin(std::process::Stdio::null());
    c
}

fn ffprobe_count(path: &Path, entries: &str, count_flag: &str) -> Result<usize> {
    let out = tool("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", count_flag])
        .args(["-show_entries", entries, "-of", "csv=p=0"])
        .arg(path)
        .output()
        .context("run ffprobe (install ffmpeg, which provides ffprobe, and put it on PATH)")?;
    ensure!(
        out.status.success(),
        "ffprobe failed on {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr).trim()
    );
    let text = String::from_utf8_lossy(&out.stdout);
    text.trim()
        .trim_end_matches(',')
        .parse()
        .with_context(|| format!("ffprobe frame count {:?}", text.trim()))
}

/// Frames in the first video stream: the packet count (fast, no decode),
/// else the decoded frame count.
pub fn frame_count(path: &Path) -> Result<usize> {
    ffprobe_count(path, "stream=nb_read_packets", "-count_packets")
        .or_else(|_| ffprobe_count(path, "stream=nb_read_frames", "-count_frames"))
}

/// Extracts `n` uniformly spaced frames (first and last included) of the
/// video at `path` into `dir` as PNGs named `frame_<index>.png` (index in
/// the video, 6 digits) and returns `(name, path)` in time order. Rotation
/// metadata is applied, as players show the video.
pub fn extract_frames(path: &Path, n: usize, dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    ensure!(path.is_file(), "video {} not found", path.display());
    ensure!(n >= 1, "--max-frames must be at least 1");
    let total = frame_count(path)?;
    ensure!(total > 0, "{} has no video frames", path.display());
    let keep = sample_indices(total, n);
    let expr = keep
        .iter()
        .map(|i| format!("eq(n\\,{i})"))
        .collect::<Vec<_>>()
        .join("+");
    // One image per selected frame needs variable-frame-rate output: the
    // option is `-fps_mode vfr` from FFmpeg 5.1 on (the only spelling in 9.0
    // and later) and `-vsync vfr` before 5.1, so retry with the old one when
    // ffmpeg does not know the new one.
    let extract = |sync: &str| {
        tool("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(path)
            .args(["-vf", &format!("select={expr}"), sync, "vfr"])
            .arg(dir.join("f%06d.png"))
            .output()
            .context("run ffmpeg (install ffmpeg and put it on PATH)")
    };
    let mut out = extract("-fps_mode")?;
    if !out.status.success() && String::from_utf8_lossy(&out.stderr).contains("fps_mode") {
        out = extract("-vsync")?;
    }
    if !out.status.success() {
        bail!(
            "ffmpeg failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    files.sort();
    ensure!(
        !files.is_empty(),
        "ffmpeg extracted no frames from {}",
        path.display()
    );
    // The packet count can differ from the decoded frames (rarely); then
    // the frames keep their extraction order as names.
    let exact = files.len() == keep.len();
    if !exact {
        eprintln!(
            "rawatta: asked ffmpeg for {} frames of {}, got {}",
            keep.len(),
            path.display(),
            files.len()
        );
    }
    files
        .into_iter()
        .enumerate()
        .map(|(k, f)| {
            let name = if exact {
                format!("frame_{:06}.png", keep[k])
            } else {
                format!("frame_seq{:06}.png", k)
            };
            let dst = dir.join(&name);
            std::fs::rename(&f, &dst).with_context(|| format!("rename {}", f.display()))?;
            Ok((name, dst))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn have_ffmpeg() -> bool {
        ["ffmpeg", "ffprobe"].iter().all(|t| {
            tool(t)
                .arg("-version")
                .output()
                .is_ok_and(|o| o.status.success())
        })
    }

    /// A 50-frame synthetic clip, 10 frames sampled, and a lossless clip
    /// whose pixels identify each frame. Where ffmpeg is not
    /// installed it skips with a note written straight to stderr (the test
    /// harness does not capture that); `RAWATTA_REQUIRE_FFMPEG=1` makes it
    /// fail instead. The dev container has ffmpeg.
    #[test]
    fn samples_frames_from_a_video() {
        if !have_ffmpeg() {
            let msg = "samples_frames_from_a_video: SKIPPED, ffmpeg/ffprobe not on PATH";
            if std::env::var_os("RAWATTA_REQUIRE_FFMPEG").is_some_and(|v| v == "1") {
                panic!("{msg}");
            }
            use std::io::Write as _;
            let _ = writeln!(std::io::stderr(), "{msg}");
            return;
        }
        let tmp = TempDir::new("video-test").unwrap();
        let clip = tmp.0.join("clip.mp4");
        let st = tool("ffmpeg")
            .args(["-v", "error", "-f", "lavfi", "-i"])
            .arg("testsrc=size=96x64:rate=25:duration=2")
            .args(["-pix_fmt", "yuv420p"])
            .arg(&clip)
            .status()
            .unwrap();
        assert!(st.success());
        assert_eq!(frame_count(&clip).unwrap(), 50);
        let dir = tmp.0.join("frames");
        std::fs::create_dir_all(&dir).unwrap();
        let got = extract_frames(&clip, 10, &dir).unwrap();
        let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
        let want: Vec<String> = sample_indices(50, 10)
            .iter()
            .map(|i| format!("frame_{i:06}.png"))
            .collect();
        assert_eq!(names, want);
        assert_eq!(names[0], "frame_000000.png");
        assert_eq!(names[9], "frame_000049.png");
        let img = vggt_omega::preprocess::decode(&std::fs::read(&got[3].1).unwrap()).unwrap();
        assert_eq!((img.width, img.height), (96, 64));
        // More frames asked than the clip has: every frame, once.
        let dir2 = tmp.0.join("all");
        std::fs::create_dir_all(&dir2).unwrap();
        assert_eq!(extract_frames(&clip, 500, &dir2).unwrap().len(), 50);

        // The pixels are the selected frames: a lossless clip (FFV1, RGB)
        // whose frame n is grey level 5n, so every extracted image names
        // its source frame.
        let grey = tmp.0.join("grey.mkv");
        let st = tool("ffmpeg")
            .args(["-v", "error", "-f", "lavfi", "-i"])
            .arg("color=c=black:size=96x64:rate=25:duration=2,format=rgb24,geq=r=N*5:g=N*5:b=N*5")
            .args(["-c:v", "ffv1", "-pix_fmt", "bgr0"])
            .arg(&grey)
            .status()
            .unwrap();
        assert!(st.success());
        assert_eq!(frame_count(&grey).unwrap(), 50);
        let dir3 = tmp.0.join("grey");
        std::fs::create_dir_all(&dir3).unwrap();
        let got = extract_frames(&grey, 7, &dir3).unwrap();
        let keep = sample_indices(50, 7);
        assert_eq!(got.len(), keep.len());
        for ((name, path), &i) in got.iter().zip(&keep) {
            assert_eq!(name, &format!("frame_{i:06}.png"));
            let img = vggt_omega::preprocess::decode(&std::fs::read(path).unwrap()).unwrap();
            for (x, y) in [(0, 0), (95, 63), (48, 32)] {
                let px = img.pixel(x, y);
                assert_eq!(px, [5 * i as u8; 3], "{name} at ({x}, {y})");
            }
        }
    }
}
