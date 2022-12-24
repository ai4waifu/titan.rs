//! Living-15 shaped diagnostics for Graph IR validation and pass failures.
//!
//! Titan emits stable machine codes and English debug messages only.
//! Localization belongs to DXO catalogs.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Severity matching the DXO / Living `15` wire.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Hard failure.
    Error,
    /// Recoverable warning.
    Warning,
    /// Informational note.
    Info,
}

/// Structured diagnostic payload (JSON-serializable, not localized).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IrDiagnostic {
    /// Stable code (`DXO_IR_*`).
    pub code: String,
    /// Severity.
    pub severity: Severity,
    /// English debug message (not a translation key).
    pub message: String,
    /// Structured args for host catalogs.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub args: BTreeMap<String, String>,
    /// Extra debug details.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub details: BTreeMap<String, String>,
    /// Optional operator / pass / op name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
}

impl IrDiagnostic {
    /// Construct an error diagnostic.
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            severity: Severity::Error,
            message: message.into(),
            args: BTreeMap::new(),
            details: BTreeMap::new(),
            operation: None,
        }
    }

    /// Attach a string arg.
    pub fn with_arg(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.args.insert(key.into(), value.into());
        self
    }

    /// Attach a detail field.
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }

    /// Attach operation label.
    pub fn with_operation(mut self, operation: impl Into<String>) -> Self {
        self.operation = Some(operation.into());
        self
    }

    /// Serialize to Living `15` JSON object text.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("IrDiagnostic is always serializable")
    }
}

/// Stable IR diagnostic codes (wire shared with DXO).
pub mod codes {
    /// Graph schema version mismatch or missing schema field.
    pub const SCHEMA_INVALID: &str = "DXO_IR_SCHEMA_INVALID";
    /// Value id referenced but not registered.
    pub const UNKNOWN_VALUE: &str = "DXO_IR_UNKNOWN_VALUE";
    /// Duplicate value id registration.
    pub const DUPLICATE_VALUE: &str = "DXO_IR_DUPLICATE_VALUE";
    /// Duplicate node id.
    pub const DUPLICATE_NODE: &str = "DXO_IR_DUPLICATE_NODE";
    /// Graph structure invariant broken (empty outputs, dangling refs, …).
    pub const GRAPH_INVALID: &str = "DXO_IR_GRAPH_INVALID";
    /// Shape / dtype constraint unsatisfiable.
    pub const SHAPE_CONSTRAINT_UNSAT: &str = "DXO_IR_SHAPE_CONSTRAINT_UNSAT";
    /// Declared pass failed its contract.
    pub const PASS_FAILED: &str = "DXO_IR_PASS_FAILED";
    /// Serialization roundtrip / document version unsupported.
    pub const SERIALIZE_UNSUPPORTED: &str = "DXO_IR_SERIALIZE_UNSUPPORTED";
}
