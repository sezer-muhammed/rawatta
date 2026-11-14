// SPDX-License-Identifier: Apache-2.0
//! The files in `configs/` agree with the engine's aliases and hardware
//! gates.

use crate::support::{parse_cc, resolve_config};

const FILES: [(&str, &str); 3] = [
    ("rawatta", include_str!("../../../configs/rawatta.toml")),
    (
        "rawatta-fast",
        include_str!("../../../configs/rawatta-fast.toml"),
    ),
    ("exact", include_str!("../../../configs/exact.toml")),
];

const GRAMMAR_MD: &str = include_str!("../../../configs/GRAMMAR.md");

#[test]
fn config_files_match_the_aliases() {
    for (name, text) in FILES {
        let t: toml::Table = toml::from_str(text).unwrap_or_else(|e| panic!("{name}.toml: {e}"));
        let field = |k: &str| {
            t.get(k)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("{name}.toml: missing string field {k}"))
        };
        assert_eq!(field("name"), name);
        let r = resolve_config(name).unwrap();
        assert_eq!(field("grammar"), r.grammar, "{name}.toml grammar");
        let cc = parse_cc(field("min_compute_capability")).unwrap();
        assert_eq!(
            cc,
            r.min_compute_capability(),
            "{name}.toml compute capability"
        );
        assert!(field("description").len() > 80, "{name}.toml description");
        assert!(
            field("paper_reference").contains("Configuration names"),
            "{name}.toml paper_reference"
        );
        assert!(!field("display_name").is_empty());
        // GRAMMAR.md lists the same string for the name.
        assert!(
            GRAMMAR_MD.contains(&format!("| `{}`", r.grammar)),
            "GRAMMAR.md lacks {name}'s string"
        );
    }
}

/// Every alias written out in GRAMMAR.md's tables resolves to that string.
#[test]
fn grammar_md_tables_match_the_aliases() {
    let mut checked = 0;
    for line in GRAMMAR_MD.lines().filter(|l| l.starts_with("| `")) {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        let name = cells[1].trim_matches('`').trim_end_matches("` (default)");
        let Some(&(_, grammar)) = vggt_omega::bench::ALIASES.iter().find(|(a, _)| *a == name)
        else {
            continue;
        };
        // Rows whose last string cell is a grammar string must match.
        if let Some(s) = cells
            .iter()
            .rev()
            .find(|c| c.starts_with('`') && c.ends_with('`') && c.len() > 2)
        {
            let s = s.trim_matches('`');
            if s.contains('@') || s.starts_with("L") || s == "dense@bf16" {
                assert_eq!(s, grammar, "GRAMMAR.md row for {name}");
                checked += 1;
            }
        }
    }
    assert!(checked >= 10, "only {checked} GRAMMAR.md rows checked");
}

/// The hardware GRAMMAR.md states for each config matches the engine's
/// gate: the "Needs" column of the named settings, and "all need sm_89+"
/// for the other aliases.
#[test]
fn grammar_md_hardware_matches_the_gate() {
    let mut named = 0;
    let mut others = 0;
    let mut in_others = false;
    for line in GRAMMAR_MD.lines() {
        if line.starts_with("## ") {
            in_others = line == "## Other named configs";
        }
        if !line.starts_with("| `") {
            continue;
        }
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        let name = cells[1].trim_matches('`').trim_end_matches("` (default)");
        if crate::support::NAMED.contains(&name) {
            let need = resolve_config(name).unwrap().min_compute_capability();
            let want = format!("sm_{}{}+", need.0, need.1);
            assert_eq!(cells[3], want, "GRAMMAR.md Needs column for {name}");
            named += 1;
        } else if in_others {
            let need = resolve_config(name).unwrap().min_compute_capability();
            assert_eq!(need, (8, 9), "GRAMMAR.md says {name} needs sm_89+");
            others += 1;
        }
    }
    assert_eq!(named, 3, "named settings rows");
    assert!(others >= 13, "only {others} other aliases checked");
    assert!(GRAMMAR_MD.contains("All of them need sm_89+"));
}
