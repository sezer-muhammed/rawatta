// SPDX-License-Identifier: LicenseRef-FAIR-Noncommercial-Research
//! Test helpers: `.npy` / `.npz` reading for the oracle dumps, and where
//! tests write their outputs.

// A general reader and writer; the tests use part of it.
#[allow(dead_code)]
pub mod npz;

/// The workspace `target/` (or `$CARGO_TARGET_DIR`), whatever the test's
/// working directory.
pub fn target_dir() -> std::path::PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .filter(|d| !d.is_empty())
        .map_or_else(
            || std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"),
            std::path::PathBuf::from,
        )
}
