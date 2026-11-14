// SPDX-License-Identifier: Apache-2.0
//! Config names, hardware gates and the weights location, shared by `run`
//! and `bench`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use vggt_omega::bench::BenchConfig;

/// The named settings users pick from (the full list of aliases, including
/// the paper's controls, is in `configs/GRAMMAR.md`).
pub const NAMED: [&str; 3] = ["rawatta", "rawatta-fast", "exact"];

/// The default `--config`.
pub const DEFAULT_CONFIG: &str = "rawatta";

/// The keys a config file (`configs/*.toml` format) may have. Only
/// `grammar` is required.
pub const CONFIG_FILE_KEYS: [&str; 6] = [
    "name",
    "display_name",
    "grammar",
    "description",
    "min_compute_capability",
    "paper_reference",
];

/// A config name, grammar string or config file, resolved.
#[derive(Debug, Clone)]
pub struct Resolved {
    /// What the user passed (`rawatta`, or a grammar string), or the
    /// `name` of a config file (its file stem when it has none).
    pub name: String,
    /// The grammar string it stands for.
    pub grammar: String,
    pub config: BenchConfig,
    /// The config file it came from, if any.
    pub file: Option<PathBuf>,
    /// The compute capability it needs: the engine's gate for the grammar,
    /// or a config file's higher `min_compute_capability`.
    min_cc: (i32, i32),
}

impl Resolved {
    pub fn min_compute_capability(&self) -> (i32, i32) {
        self.min_cc
    }

    /// Whether the config runs GEMMs in FP8 and so needs the FP8 side file.
    pub fn needs_fp8_weights(&self) -> bool {
        self.config.fp8_blocks != vggt_omega::trunk::Fp8Blocks::NONE
    }
}

/// Whether `--config` names a config file rather than a name or grammar
/// string (grammar strings never end in `.toml`).
pub fn is_config_file(arg: &str) -> bool {
    arg.trim().to_ascii_lowercase().ends_with(".toml")
}

/// Resolves `rawatta`, `rawatta-fast`, `exact`, any other alias, a raw
/// grammar string such as `dense@sagepp~L4=sage.pvh16/L15=sage.pvh16`, or
/// the path of a `.toml` config file in the `configs/` format.
pub fn resolve_config(arg: &str) -> Result<Resolved> {
    let arg = arg.trim();
    if is_config_file(arg) {
        let path = Path::new(arg);
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read config file {}", path.display()))?;
        return parse_config_file(&text, path);
    }
    resolve_grammar(arg, arg)
}

fn resolve_grammar(grammar: &str, name: &str) -> Result<Resolved> {
    let config = BenchConfig::named(grammar).map_err(|e| {
        anyhow!(
            "unknown or malformed config {grammar:?}: {e:#}\n\
             Named settings: {}. Any string of the config grammar works too \
             (<schedule>[@<precision>][~<overrides>][+fp8[:<blocks>]], see configs/GRAMMAR.md), \
             and so does the path of a .toml config file in the configs/ format.",
            NAMED.join(", ")
        )
    })?;
    Ok(Resolved {
        name: name.to_string(),
        grammar: BenchConfig::grammar_of(grammar).to_string(),
        min_cc: config.min_compute_capability(),
        config,
        file: None,
    })
}

/// Parses a config file (`configs/rawatta.toml` format): `grammar` is a
/// name or grammar string; `name` names the config (default: the file
/// stem); `min_compute_capability` may raise the hardware gate but not
/// lower it below what the grammar needs. The other keys are descriptive.
pub fn parse_config_file(text: &str, path: &Path) -> Result<Resolved> {
    let shown = path.display();
    let t: toml::Table =
        toml::from_str(text).with_context(|| format!("parse config file {shown}"))?;
    for k in t.keys() {
        ensure!(
            CONFIG_FILE_KEYS.contains(&k.as_str()),
            "config file {shown}: unknown key {k:?} (known keys: {})",
            CONFIG_FILE_KEYS.join(", ")
        );
    }
    let field = |k: &str| -> Result<Option<&str>> {
        t.get(k)
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| anyhow!("config file {shown}: {k} must be a string"))
            })
            .transpose()
    };
    let grammar = field("grammar")?
        .map(str::trim)
        .ok_or_else(|| anyhow!("config file {shown}: missing key grammar (a config string)"))?;
    let name = match field("name")? {
        Some(n) => n.trim().to_string(),
        None => path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| grammar.to_string()),
    };
    ensure!(!name.is_empty(), "config file {shown}: name is empty");
    for k in ["display_name", "description", "paper_reference"] {
        field(k)?;
    }
    let mut r = resolve_grammar(grammar, &name).with_context(|| format!("config file {shown}"))?;
    if let Some(cc) = field("min_compute_capability")? {
        let cc = parse_cc(cc.trim()).ok_or_else(|| {
            anyhow!(
                "config file {shown}: min_compute_capability must look like \"8.9\", got {cc:?}"
            )
        })?;
        let engine = r.min_cc;
        ensure!(
            cc >= engine,
            "config file {shown}: min_compute_capability {}.{} is below the {}.{} that its grammar \
             {:?} needs (see configs/GRAMMAR.md, Hardware)",
            cc.0,
            cc.1,
            engine.0,
            engine.1,
            r.grammar
        );
        r.min_cc = cc;
    }
    r.file = Some(path.to_path_buf());
    Ok(r)
}

/// Name and compute capability of GPU 0, the device the engine uses.
pub fn gpu0() -> Result<(String, (i32, i32))> {
    let r = rawatta_nn::gpu::probe();
    let d = r.devices.first().ok_or_else(|| {
        anyhow!(
            "no CUDA GPU found{} (run `rawatta info` for details)",
            r.error.map(|e| format!(": {e}")).unwrap_or_default()
        )
    })?;
    Ok((
        d.name.clone(),
        parse_cc(&d.compute_capability).unwrap_or((0, 0)),
    ))
}

pub fn sm(cc: (i32, i32)) -> String {
    format!("sm_{}{}", cc.0, cc.1)
}

/// Parses `"8.9"` into `(8, 9)`.
pub fn parse_cc(s: &str) -> Option<(i32, i32)> {
    let (a, b) = s.split_once('.')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

/// Fails with a clear message when `gpu` (compute capability `have`) cannot
/// run `cfg`.
pub fn check_compute(cfg: &Resolved, have: (i32, i32), gpu: &str) -> Result<()> {
    if have < (8, 0) {
        bail!(
            "{gpu} has compute capability {}.{} ({}); the rawatta engine needs 8.0 or newer \
             (bf16 tensor cores: Ampere, Ada, Hopper, Blackwell). No config runs on this GPU.",
            have.0,
            have.1,
            sm(have)
        );
    }
    let need = cfg.min_compute_capability();
    if have < need && cfg.config.min_compute_capability() <= have {
        // Only the config file's own min_compute_capability rules it out.
        bail!(
            "config {:?} ({}) sets min_compute_capability {}.{} ({}+), but {gpu} is {}.{} ({}).",
            cfg.name,
            cfg.file
                .as_deref()
                .map_or_else(|| "a config file".into(), |f| f.display().to_string()),
            need.0,
            need.1,
            sm(need),
            have.0,
            have.1,
            sm(have)
        );
    }
    if have < need {
        bail!(
            "config {:?} needs compute capability {}.{}+ ({}+: FP8 tensor cores, i.e. Ada, Hopper or \
             Blackwell), but {gpu} is {}.{} ({}).\n\
             Use --config exact to run the exact engine (bf16 attention, needs sm_80+) on this GPU.",
            cfg.name,
            need.0,
            need.1,
            sm(need),
            have.0,
            have.1,
            sm(have)
        );
    }
    Ok(())
}

/// `--weights`, else `$RAWATTA_WEIGHTS`, else the cache default.
pub fn weights_path(arg: Option<&Path>) -> PathBuf {
    arg.map(Path::to_path_buf)
        .unwrap_or_else(vggt_omega::default_weights_path)
}

/// Where the weights are looked for when neither `--weights` nor
/// `RAWATTA_WEIGHTS` is given.
fn cache_default() -> String {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"));
    base.join("rawatta")
        .join(vggt_omega::WEIGHTS_FILE)
        .display()
        .to_string()
}

/// The canonical weight download, Docker form (the host needs only Docker
/// and the NVIDIA driver; the token file comes from `hf auth login`).
pub const DOWNLOAD_DOCKER: &str = "docker run --rm --user \"$(id -u):$(id -g)\" \
-v ~/.cache/huggingface:/hf:ro -v ~/.cache/rawatta:/weights \\\n          \
--entrypoint download-weights ghcr.io/sezer-muhammed/rawatta:latest --token-file /hf/token --dir /weights";

/// The canonical weight download, native form (needs curl, python3 >= 3.10
/// and numpy).
pub const DOWNLOAD_NATIVE: &str = "tools/download_weights.sh";

/// Where `rawatta` is running, for the missing-weights message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightsSource {
    /// Inside the Docker image (it sets `RAWATTA_CACHE`): the weights come
    /// from the host folder mounted at `/weights`.
    Image,
    /// Native, at the cache default (no `--weights`, no `RAWATTA_WEIGHTS`).
    NativeDefault,
    /// Native, at a path from `--weights` or `RAWATTA_WEIGHTS`.
    NativeChosen,
}

/// The message for weights missing at `path`.
fn missing_weights_message(path: &Path, source: WeightsSource) -> String {
    let docker = format!("mkdir -p ~/.cache/huggingface ~/.cache/rawatta\n    {DOWNLOAD_DOCKER}");
    let licence = "The weights are gated on Hugging Face under the FAIR Noncommercial Research License \
                   (research use only): request access on the model page first.";
    match source {
        WeightsSource::Image => format!(
            "VGGT-Omega weights not found at {}.\n\
             Mount the host folder that holds {} at /weights (-v ~/.cache/rawatta:/weights:ro), \
             or download them into it once on the host:\n    {docker}\n{licence}",
            path.display(),
            vggt_omega::WEIGHTS_FILE
        ),
        WeightsSource::NativeDefault | WeightsSource::NativeChosen => {
            let native = if source == WeightsSource::NativeDefault {
                format!(
                    "{DOWNLOAD_NATIVE}   (writes {}, the default location)",
                    path.display()
                )
            } else {
                let parent = path
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                format!("{DOWNLOAD_NATIVE} --dir {}", parent.display())
            };
            format!(
                "VGGT-Omega weights not found at {}.\n\
                 Set RAWATTA_WEIGHTS=<file> (or pass --weights <file>), or download them once:\n  \
                 Native: {native}\n  \
                 Docker:\n    {docker}\n{licence}",
                path.display()
            )
        }
    }
}

/// Fails with a message that names `RAWATTA_WEIGHTS` and the download
/// commands when the weights (or the FP8 side file a `+fp8` config needs)
/// are missing.
pub fn check_weights(path: &Path, fp8: bool) -> Result<()> {
    if !path.is_file() {
        let source = if std::env::var_os("RAWATTA_CACHE").is_some_and(|v| !v.is_empty()) {
            WeightsSource::Image
        } else if path.display().to_string() == cache_default() {
            WeightsSource::NativeDefault
        } else {
            WeightsSource::NativeChosen
        };
        bail!("{}", missing_weights_message(path, source));
    }
    if fp8 {
        let side = vggt_omega::weights::fp8_companion(path);
        if !side.is_file() {
            bail!(
                "this config runs FP8 GEMMs (+fp8) and needs the FP8 side file {}, which is missing.\n\
                 Create it with tools/convert_weights.py --fp8 (or download with {DOWNLOAD_NATIVE} --fp8), \
                 or drop +fp8 from the config.",
                side.display()
            );
        }
    }
    Ok(())
}

/// Parses `--crop-aspect`: a height/width ratio as `0.65625` or `336/512`.
pub fn parse_aspect(s: &str) -> Result<f64, String> {
    let v = match s.split_once('/') {
        Some((h, w)) => {
            let h: f64 = h
                .trim()
                .parse()
                .map_err(|_| format!("bad height in {s:?}"))?;
            let w: f64 = w
                .trim()
                .parse()
                .map_err(|_| format!("bad width in {s:?}"))?;
            h / w
        }
        None => s
            .trim()
            .parse()
            .map_err(|_| format!("expected h/w such as 0.75 or 3/4, got {s:?}"))?,
    };
    if !(0.5..=2.0).contains(&v) {
        return Err(format!(
            "crop aspect (height / width) must be within [0.5, 2], got {v}"
        ));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_settings_resolve_to_the_paper_strings() {
        let want = [
            ("rawatta", "dense@sagepp~L4=sage.pvh16/L15=sage.pvh16"),
            (
                "rawatta-fast",
                "L0:frame/L3:merge4/L4:frame/L5:frame/L19:merge4@sagepp~L15=sage.pvh16",
            ),
            ("exact", "dense@bf16"),
        ];
        for (name, grammar) in want {
            let r = resolve_config(name).unwrap();
            assert_eq!(r.grammar, grammar, "{name}");
            // The alias and its grammar string build the same engine config.
            let direct = resolve_config(grammar).unwrap();
            assert_eq!(r.config.schedule, direct.config.schedule, "{name}");
            assert_eq!(r.config.sage_prec, direct.config.sage_prec, "{name}");
            assert_eq!(r.config.fp8_blocks, direct.config.fp8_blocks, "{name}");
        }
        assert_eq!(resolve_config(" exact ").unwrap().name, "exact");
        assert_eq!(
            resolve_config(DEFAULT_CONFIG)
                .unwrap()
                .min_compute_capability(),
            (8, 9)
        );
        // Raw grammar strings pass through unchanged.
        let raw = resolve_config("frame3@sagepp+fp8").unwrap();
        assert_eq!(raw.grammar, "frame3@sagepp+fp8");
        assert!(raw.needs_fp8_weights());
        assert!(!resolve_config("rawatta").unwrap().needs_fp8_weights());
        let err = format!("{:#}", resolve_config("rawata").unwrap_err());
        assert!(err.contains("rawatta, rawatta-fast, exact"), "{err}");
        assert!(err.contains("GRAMMAR.md"), "{err}");
    }

    #[test]
    fn compute_capability_gate() {
        let ra = resolve_config("rawatta").unwrap();
        let fast = resolve_config("rawatta-fast").unwrap();
        let exact = resolve_config("exact").unwrap();
        for cc in [(8, 0), (8, 6), (8, 7), (8, 8)] {
            for c in [&ra, &fast] {
                let e = format!("{:#}", check_compute(c, cc, "Test GPU").unwrap_err());
                assert!(e.contains("--config exact"), "{e}");
                assert!(e.contains("sm_89"), "{e}");
                assert!(e.contains(&sm(cc)), "{e}");
            }
            check_compute(&exact, cc, "Test GPU").unwrap();
        }
        for cc in [(8, 9), (9, 0), (10, 0), (12, 0)] {
            check_compute(&ra, cc, "Test GPU").unwrap();
            check_compute(&fast, cc, "Test GPU").unwrap();
            check_compute(&exact, cc, "Test GPU").unwrap();
        }
        let e = format!("{:#}", check_compute(&exact, (7, 5), "T4").unwrap_err());
        assert!(e.contains("8.0 or newer"), "{e}");
        assert_eq!(parse_cc("8.9"), Some((8, 9)));
        assert_eq!(parse_cc("12.0"), Some((12, 0)));
        assert_eq!(parse_cc("x"), None);
    }

    #[test]
    fn missing_weights_message_names_the_right_fix() {
        let p = Path::new("/nonexistent/vggt_omega_1b_512.safetensors");
        for source in [
            WeightsSource::Image,
            WeightsSource::NativeDefault,
            WeightsSource::NativeChosen,
        ] {
            let e = super::missing_weights_message(p, source);
            assert!(e.contains("--entrypoint download-weights"), "{e}");
            assert!(e.contains("ghcr.io/sezer-muhammed/rawatta:latest"), "{e}");
            assert!(e.contains("--token-file /hf/token --dir /weights"), "{e}");
            // The host folders are created before Docker would create them as root.
            let mkdir = e
                .find("mkdir -p ~/.cache/huggingface ~/.cache/rawatta")
                .expect(&e);
            assert!(mkdir < e.find("docker run").expect(&e), "{e}");
            assert!(e.contains("Noncommercial"), "{e}");
            assert!(
                e.contains("/nonexistent/vggt_omega_1b_512.safetensors"),
                "{e}"
            );
        }
        // Native: RAWATTA_WEIGHTS comes first, then the two download commands.
        let e = super::missing_weights_message(p, WeightsSource::NativeDefault);
        let env = e.find("RAWATTA_WEIGHTS").expect(&e);
        assert!(env < e.find("--weights").expect(&e), "{e}");
        assert!(
            e.contains("Native: tools/download_weights.sh   (writes /nonexistent/"),
            "{e}"
        );
        assert!(e.contains("the default location"), "{e}");
        // A path from --weights / RAWATTA_WEIGHTS: download into its folder.
        let e = super::missing_weights_message(p, WeightsSource::NativeChosen);
        assert!(
            e.contains("Native: tools/download_weights.sh --dir /nonexistent\n"),
            "{e}"
        );
        assert!(!e.contains("default location"), "{e}");
        // In the image: mount the host folder; no native script, no host paths.
        let e = super::missing_weights_message(
            Path::new("/weights/vggt_omega_1b_512.safetensors"),
            WeightsSource::Image,
        );
        assert!(
            e.contains("at /weights (-v ~/.cache/rawatta:/weights:ro)"),
            "{e}"
        );
        assert!(!e.contains("Native"), "{e}");
        assert!(!e.contains("--weights <file>"), "{e}");
        assert!(!e.contains("RAWATTA_WEIGHTS"), "{e}");
        // The FP8 side file is checked when the config needs it.
        let dir = std::env::temp_dir().join(format!("rawatta-w-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let w = dir.join("w.safetensors");
        std::fs::write(&w, b"x").unwrap();
        check_weights(&w, false).unwrap();
        let e = format!("{:#}", check_weights(&w, true).unwrap_err());
        assert!(
            e.contains("w_fp8.safetensors") && e.contains("--fp8"),
            "{e}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A file in a fresh temporary folder, removed with the folder.
    struct TmpFile(PathBuf);
    impl TmpFile {
        fn new(name: &str, text: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "rawatta-cfg-{}-{}",
                std::process::id(),
                name.replace(['.', '/'], "_")
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let p = dir.join(name);
            std::fs::write(&p, text).unwrap();
            Self(p)
        }
        fn arg(&self) -> String {
            self.0.display().to_string()
        }
    }
    impl Drop for TmpFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(self.0.parent().unwrap());
        }
    }

    /// `--config <file.toml>`: the shipped files resolve exactly like their
    /// names, and a minimal user file works.
    #[test]
    fn config_files_resolve() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
        for name in NAMED {
            let path = root.join(format!("{name}.toml"));
            let f = resolve_config(&path.display().to_string()).unwrap();
            let n = resolve_config(name).unwrap();
            assert_eq!(f.name, name);
            assert_eq!(f.grammar, n.grammar, "{name}");
            assert_eq!(f.config.schedule, n.config.schedule, "{name}");
            assert_eq!(f.config.sage_prec, n.config.sage_prec, "{name}");
            assert_eq!(f.config.fp8_blocks, n.config.fp8_blocks, "{name}");
            assert_eq!(f.min_compute_capability(), n.min_compute_capability());
            assert_eq!(f.file.as_deref(), Some(path.as_path()));
            assert!(n.file.is_none());
        }
        // Only grammar is required; the name defaults to the file stem.
        let mine = TmpFile::new("my-schedule.toml", "grammar = \"L0:frame@sagepp\"\n");
        let r = resolve_config(&mine.arg()).unwrap();
        assert_eq!(
            (r.name.as_str(), r.grammar.as_str()),
            ("my-schedule", "L0:frame@sagepp")
        );
        assert_eq!(r.min_compute_capability(), (8, 9));
        // A higher gate is kept; a lower one than the grammar needs fails.
        let hi = TmpFile::new(
            "hi.TOML",
            "name = \"exact-hopper\"\ngrammar = \"exact\"\nmin_compute_capability = \"9.0\"\n",
        );
        let r = resolve_config(&hi.arg()).unwrap();
        assert_eq!(
            (r.name.as_str(), r.grammar.as_str()),
            ("exact-hopper", "dense@bf16")
        );
        assert_eq!(r.min_compute_capability(), (9, 0));
        let e = format!("{:#}", check_compute(&r, (8, 9), "Test GPU").unwrap_err());
        assert!(e.contains("sets min_compute_capability 9.0"), "{e}");
        check_compute(&r, (9, 0), "Test GPU").unwrap();
        let lo = TmpFile::new(
            "lo.toml",
            "grammar = \"dense@sagepp\"\nmin_compute_capability = \"8.0\"\n",
        );
        let e = format!("{:#}", resolve_config(&lo.arg()).unwrap_err());
        assert!(e.contains("below the 8.9"), "{e}");
        // Mistakes are named.
        for (file, text, want) in [
            (
                "typo.toml",
                "grammer = \"dense@bf16\"\n",
                "unknown key \"grammer\"",
            ),
            ("nogrammar.toml", "name = \"x\"\n", "missing key grammar"),
            (
                "bad.toml",
                "grammar = \"dense@nope\"\n",
                "unknown or malformed config",
            ),
            ("num.toml", "grammar = 3\n", "grammar must be a string"),
            (
                "cc.toml",
                "grammar = \"exact\"\nmin_compute_capability = 8.9\n",
                "must be a string",
            ),
            ("syntax.toml", "grammar = \n", "parse config file"),
        ] {
            let f = TmpFile::new(file, text);
            let e = format!("{:#}", resolve_config(&f.arg()).unwrap_err());
            assert!(e.contains(want), "{file}: {e}");
            assert!(e.contains(file), "{file}: {e}");
        }
        let e = format!("{:#}", resolve_config("/nonexistent/x.toml").unwrap_err());
        assert!(e.contains("read config file /nonexistent/x.toml"), "{e}");
        assert!(!is_config_file("dense@bf16") && is_config_file(" a/b.Toml "));
    }

    #[test]
    fn crop_aspect_parsing() {
        assert_eq!(parse_aspect("0.65625"), Ok(0.65625));
        assert_eq!(parse_aspect("336/512"), Ok(0.65625));
        assert_eq!(parse_aspect("3 / 4"), Ok(0.75));
        assert!(parse_aspect("0.2").is_err());
        assert!(parse_aspect("a/b").is_err());
    }
}
