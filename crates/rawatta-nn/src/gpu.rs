// SPDX-License-Identifier: Apache-2.0
//! CUDA device discovery and contexts, built on `cudarc`.
//!
//! The CUDA driver and NVRTC are loaded when first used, not at link time, so
//! this crate compiles everywhere and only fails at runtime on machines
//! without an NVIDIA driver. [`probe`] turns that failure into a value.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use cudarc::driver::{CudaContext, sys};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub ordinal: u32,
    pub name: String,
    /// `"8.9"` style; pick nvcc/NVRTC targets from this.
    pub compute_capability: String,
    pub total_memory_bytes: u64,
    pub multiprocessors: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuReport {
    /// `"12.8"` style, or `None` when no driver could be loaded.
    pub driver_version: Option<String>,
    pub devices: Vec<GpuInfo>,
    /// Why no devices were found, when that is the case.
    pub error: Option<String>,
}

/// Lists CUDA devices without ever panicking, even with no driver installed.
pub fn probe() -> GpuReport {
    // Checking first keeps cudarc from printing a panic on driverless hosts;
    // catch_unwind stays as the backstop.
    if !unsafe { sys::is_culib_present() } {
        return GpuReport {
            driver_version: None,
            devices: Vec::new(),
            error: Some("CUDA driver library not found on this machine".into()),
        };
    }
    match catch_unwind(AssertUnwindSafe(probe_inner)) {
        Ok(Ok(report)) => report,
        Ok(Err(e)) => GpuReport {
            driver_version: None,
            devices: Vec::new(),
            error: Some(format!("{e:#}")),
        },
        Err(_) => GpuReport {
            driver_version: None,
            devices: Vec::new(),
            error: Some("CUDA driver library not found on this machine".into()),
        },
    }
}

/// Why the driver lists no device, for a `cuInit` / `cuDeviceGetCount`
/// error code. The driver itself is loaded at that point.
pub fn device_error_message(code: sys::CUresult) -> String {
    match code {
        sys::CUresult::CUDA_ERROR_NO_DEVICE => {
            "the NVIDIA driver is loaded but reports no CUDA device \
             (CUDA_ERROR_NO_DEVICE): check CUDA_VISIBLE_DEVICES, and in Docker run with --gpus all"
                .into()
        }
        c => format!("the NVIDIA driver is loaded but listing devices failed ({c:?})"),
    }
}

fn probe_inner() -> Result<GpuReport> {
    let mut raw = 0;
    unsafe { sys::cuDriverGetVersion(&mut raw) }
        .result()
        .map_err(|e| anyhow!("cuDriverGetVersion: {e:?}"))?;
    let driver_version = Some(format!("{}.{}", raw / 1000, (raw % 1000) / 10));

    // The driver loaded: a failure from here on (no visible device, say) is
    // reported with the driver version, not as a missing driver.
    let count = match CudaContext::device_count() {
        Ok(n) => n,
        Err(e) => {
            return Ok(GpuReport {
                driver_version,
                devices: Vec::new(),
                error: Some(device_error_message(e.0)),
            });
        }
    };
    let mut devices = Vec::with_capacity(count.max(0) as usize);
    for ordinal in 0..count.max(0) as usize {
        let ctx = context(ordinal)?;
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| anyhow!("compute capability: {e:?}"))?;
        devices.push(GpuInfo {
            ordinal: ordinal as u32,
            name: ctx.name().map_err(|e| anyhow!("device name: {e:?}"))?,
            compute_capability: format!("{major}.{minor}"),
            total_memory_bytes: ctx.total_mem().map_err(|e| anyhow!("total mem: {e:?}"))? as u64,
            multiprocessors: ctx
                .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                .map_err(|e| anyhow!("SM count: {e:?}"))? as u32,
        });
    }
    Ok(GpuReport {
        driver_version,
        devices,
        error: None,
    })
}

pub fn context(ordinal: usize) -> Result<Arc<CudaContext>> {
    if !unsafe { sys::is_culib_present() } {
        return Err(anyhow!(
            "CUDA driver library not found; run on a host with an NVIDIA driver"
        ));
    }
    CudaContext::new(ordinal).map_err(|e| anyhow!("open CUDA device {ordinal}: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_device_is_not_a_missing_driver() {
        let m = device_error_message(sys::CUresult::CUDA_ERROR_NO_DEVICE);
        assert!(m.contains("CUDA_ERROR_NO_DEVICE"), "{m}");
        assert!(m.contains("driver is loaded"), "{m}");
        assert!(m.contains("CUDA_VISIBLE_DEVICES"), "{m}");
        let m = device_error_message(sys::CUresult::CUDA_ERROR_UNKNOWN);
        assert!(m.contains("CUDA_ERROR_UNKNOWN"), "{m}");
    }
}
