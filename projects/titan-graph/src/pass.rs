//! Pass declaration contract and registry skeleton (Living `16` / phase-03).

use crate::diagnostic::{codes, IrDiagnostic};
use serde::{Deserialize, Serialize};

/// Pipeline stage a pass is allowed to run in.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PassStage {
    /// Term capture / normalize.
    Term,
    /// Core analysis and rewrite.
    Core,
    /// Partition / stream / buffer schedule.
    Scheduled,
    /// Final executable plan materialization.
    Executable,
}

/// How a pass reports contract failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PassFailureBehavior {
    /// Abort the compile with a structured diagnostic.
    Abort,
    /// Skip the pass and continue (must still emit a warning diagnostic).
    SkipWithWarning,
}

/// Declared pass contract — name, stage, invariants, failure behavior.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PassDecl {
    /// Stable pass name (`canonicalize`, `dce`, `fusion.elementwise`, …).
    pub name: String,
    /// Allowed pipeline stage.
    pub stage: PassStage,
    /// Human-readable invariant list (English debug; not localized).
    pub invariants: Vec<String>,
    /// Failure behavior when invariants break.
    pub failure: PassFailureBehavior,
}

impl PassDecl {
    /// Build a declaration.
    pub fn new(
        name: impl Into<String>,
        stage: PassStage,
        invariants: impl IntoIterator<Item = impl Into<String>>,
        failure: PassFailureBehavior,
    ) -> Self {
        Self {
            name: name.into(),
            stage,
            invariants: invariants.into_iter().map(Into::into).collect(),
            failure,
        }
    }

    /// Emit a Living `15` diagnostic for a declared pass failure.
    pub fn failure_diagnostic(&self, detail: impl Into<String>) -> IrDiagnostic {
        IrDiagnostic::error(codes::PASS_FAILED, format!("pass '{}' failed its contract", self.name))
            .with_arg("pass", self.name.clone())
            .with_arg("stage", format!("{:?}", self.stage).to_ascii_lowercase())
            .with_detail("debug", detail)
            .with_operation(self.name.clone())
    }
}

/// Ordered registry of pass declarations (skeleton — no execution yet).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PassRegistry {
    decls: Vec<PassDecl>,
}

impl PassRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a pass; rejects duplicate names.
    pub fn register(&mut self, decl: PassDecl) -> Result<(), IrDiagnostic> {
        if self.decls.iter().any(|d| d.name == decl.name) {
            return Err(IrDiagnostic::error(codes::PASS_FAILED, format!("duplicate pass '{}'", decl.name))
                .with_arg("pass", decl.name));
        }
        self.decls.push(decl);
        Ok(())
    }

    /// Lookup by name.
    pub fn get(&self, name: &str) -> Option<&PassDecl> {
        self.decls.iter().find(|d| d.name == name)
    }

    /// All declarations in registration order.
    pub fn decls(&self) -> &[PassDecl] {
        &self.decls
    }

    /// Declarations filtered by stage.
    pub fn by_stage(&self, stage: PassStage) -> Vec<&PassDecl> {
        self.decls.iter().filter(|d| d.stage == stage).collect()
    }
}

/// Minimal built-in declaration set for contract tests (not a full pipeline).
pub fn builtin_pass_registry() -> PassRegistry {
    let mut registry = PassRegistry::new();
    let entries = [
        PassDecl::new(
            "validate",
            PassStage::Core,
            ["graph references are closed", "schema version matches"],
            PassFailureBehavior::Abort,
        ),
        PassDecl::new(
            "canonicalize",
            PassStage::Core,
            ["attrs are sorted", "effects unchanged"],
            PassFailureBehavior::Abort,
        ),
        PassDecl::new(
            "dce",
            PassStage::Core,
            ["dead pure nodes removed", "outputs preserved"],
            PassFailureBehavior::Abort,
        ),
        PassDecl::new(
            "fusion.elementwise",
            PassStage::Core,
            ["fused region is pure", "alias sets disjoint"],
            PassFailureBehavior::Abort,
        ),
        PassDecl::new(
            "schedule.streams",
            PassStage::Scheduled,
            ["effect order preserved", "cross-stream waits explicit"],
            PassFailureBehavior::Abort,
        ),
        PassDecl::new(
            "plan.executable",
            PassStage::Executable,
            ["all kernels bound", "buffers planned"],
            PassFailureBehavior::Abort,
        ),
    ];
    for decl in entries {
        registry.register(decl).expect("builtin pass names are unique");
    }
    registry
}
