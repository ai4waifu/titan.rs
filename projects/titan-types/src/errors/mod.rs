use std::{
    error::Error,
    fmt::{Debug, Display, Formatter},
};

mod convert;
mod display;

/// The result type of this crate.
pub type Result<T> = std::result::Result<T, TitanError>;

/// A boxed error kind with optional debug detail (not a localization contract).
#[derive(Clone)]
pub struct TitanError {
    kind: Box<TitanErrorKind>,
    /// Backend/debug detail for developers; not a stable user-facing string.
    detail: Option<String>,
}

impl TitanError {
    /// Construct from a kind with no detail.
    pub fn from_kind(kind: TitanErrorKind) -> Self {
        Self {
            kind: Box::new(kind),
            detail: None,
        }
    }

    /// Construct from a kind plus opaque debug detail.
    pub fn with_detail(kind: TitanErrorKind, detail: impl Into<String>) -> Self {
        Self {
            kind: Box::new(kind),
            detail: Some(detail.into()),
        }
    }

    /// Stable machine-readable kind.
    pub fn kind(&self) -> TitanErrorKind {
        *self.kind
    }

    /// Optional debug detail (not for localization).
    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }
}

/// Stable Titan error categories for DXO / host mapping (`DXO_TITAN_*`).
///
/// Variants must not embed natural-language user copy.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TitanErrorKind {
    /// Device id / adapter could not be resolved.
    DeviceNotFound,
    /// Device was lost or reset during execution.
    DeviceLost,
    /// Operation mixed tensors or buffers from different devices.
    CrossDevice,
    /// Cross-stream dependency was invalid or incomplete.
    CrossStream,
    /// Waiting on an event / stream failed.
    EventWaitFailed,
    /// Device allocation failed (OOM or allocator error).
    AllocationFailed,
    /// Kernel ABI / binding contract mismatch.
    InvalidAbi,
    /// Requested kernel or op is unsupported on this backend.
    KernelUnsupported,
    /// Kernel launch / dispatch failed.
    KernelLaunchFailed,
    /// Device → host readback failed.
    ReadbackFailed,
    /// Host → device upload failed.
    UploadFailed,
    /// Requested backend is not available on this machine.
    BackendUnavailable,
    /// Fallback when no more specific kind applies.
    UnknownError,
}

impl TitanErrorKind {
    /// Stable snake-ish label for diagnostics / DXO mapping (not localized).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DeviceNotFound => "device_not_found",
            Self::DeviceLost => "device_lost",
            Self::CrossDevice => "cross_device",
            Self::CrossStream => "cross_stream",
            Self::EventWaitFailed => "event_wait_failed",
            Self::AllocationFailed => "allocation_failed",
            Self::InvalidAbi => "invalid_abi",
            Self::KernelUnsupported => "kernel_unsupported",
            Self::KernelLaunchFailed => "kernel_launch_failed",
            Self::ReadbackFailed => "readback_failed",
            Self::UploadFailed => "upload_failed",
            Self::BackendUnavailable => "backend_unavailable",
            Self::UnknownError => "unknown_error",
        }
    }

    /// Heuristic map from HAL `operation` names to a kind.
    pub fn from_hal_operation(operation: &str) -> Self {
        let op = operation.trim().to_ascii_lowercase();
        match op.as_str() {
            "allocate" | "alloc" => Self::AllocationFailed,
            "upload" | "copy_h2d" | "write" => Self::UploadFailed,
            "download" | "readback" | "copy_d2h" | "read" => Self::ReadbackFailed,
            "wait_event" | "wait" | "synchronize" => Self::EventWaitFailed,
            "launch" | "dispatch" | "execute" => Self::KernelLaunchFailed,
            "load_kernel" | "compile" => Self::KernelUnsupported,
            "create_stream" | "create_event" | "open" | "enumerate" => Self::BackendUnavailable,
            "device_lost" => Self::DeviceLost,
            other if other.contains("cross") && other.contains("device") => Self::CrossDevice,
            other if other.contains("cross") && other.contains("stream") => Self::CrossStream,
            other if other.contains("abi") => Self::InvalidAbi,
            _ => Self::UnknownError,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hal_operation_maps() {
        assert_eq!(TitanErrorKind::from_hal_operation("upload"), TitanErrorKind::UploadFailed);
        assert_eq!(TitanErrorKind::from_hal_operation("allocate"), TitanErrorKind::AllocationFailed);
        assert_eq!(TitanErrorKind::from_hal_operation("wait_event"), TitanErrorKind::EventWaitFailed);
        assert_eq!(TitanErrorKind::from_hal_operation("download"), TitanErrorKind::ReadbackFailed);
    }

    #[test]
    fn with_detail_preserves_kind() {
        let err = TitanError::with_detail(TitanErrorKind::UploadFailed, "oom");
        assert_eq!(err.kind(), TitanErrorKind::UploadFailed);
        assert_eq!(err.detail(), Some("oom"));
    }
}
