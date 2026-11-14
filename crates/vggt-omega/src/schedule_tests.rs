// SPDX-License-Identifier: LicenseRef-FAIR-Noncommercial-Research
//! Attention-schedule checks that need no GPU: the `v1` control schedule's
//! layer map and the scratch accounting of the quantised workspaces.

use super::*;

#[test]
fn v1_schedule_layers() {
    let s = AttnSchedule::v1(true);
    for i in 0..24 {
        let want = if REGISTER_LAYERS.contains(&i) {
            InterAttn::Register
        } else if FRAME_LAYERS.contains(&i) {
            InterAttn::Frame
        } else if MERGE4_LAYERS.contains(&i) {
            InterAttn::Merge4
        } else {
            InterAttn::Sage
        };
        assert_eq!(s.0[i], want, "layer {i}");
    }
    assert_eq!(
        AttnSchedule::v1(false)
            .0
            .iter()
            .filter(|&&a| a == InterAttn::Global)
            .count(),
        19 - 3 - 4
    );
    let w = AttnSchedule::v1_wide(false);
    assert_eq!(w.0[8], InterAttn::Merge4);
    assert_eq!(w.0[22], InterAttn::Merge4);
    assert_eq!(AttnSchedule::named("dense").unwrap(), AttnSchedule::dense());
    assert!(AttnSchedule::named("nope").is_err());
    for (name, _) in [("v1+sage", 0), ("v1-wide+sage", 0), ("sage", 0), ("v1", 0)] {
        let s = AttnSchedule::named(name).unwrap();
        assert!(super::trunk_schedule(&s).is_ok(), "{name}");
    }
}

#[test]
fn sage_workspaces_are_counted() {
    let g = trunk::TrunkGeom {
        frames: 200,
        grid_h: 21,
        grid_w: 32,
    };
    let dense = trunk::Trunk::scratch_bytes(&g);
    let s = super::trunk_schedule(&AttnSchedule::v1(true)).unwrap();
    let approx = trunk::Trunk::scratch_bytes_for(&g, &s);
    // Q8, K8 and V8 of both workspaces (dense sage, merge4) borrow the
    // trunk's `big` buffer; only the (per-token) scales and partials
    // remain, well under one INT8 Q (16 heads x 64 x tokens).
    let extra = approx - dense;
    let tokens = 200 * 689;
    assert!(extra > 0 && extra < 16 * 64 * tokens, "{extra}");
    let d = super::trunk_schedule(&AttnSchedule::dense()).unwrap();
    assert_eq!(trunk::Trunk::scratch_bytes_for(&g, &d), dense);
}
