#![warn(missing_docs)]
//! Concrete model family descriptors. Weights and heavy backends stay external.

use titan_model::{
    DeploymentTarget, ManifestState, ModelCapabilities, ModelCatalogEntry, ModelDescriptor, ModelFamilyId,
    ModelManifestSummary, ModelRegistry, ModelSchema, ModelVariantId,
};

/// Registers the built-in family descriptors without global constructor side effects.
pub fn registry() -> ModelRegistry {
    let mut registry = ModelRegistry::default();
    for descriptor in built_in_descriptors() {
        registry.register(descriptor).expect("built-in model ids are unique");
    }
    registry
}

/// Returns the stable descriptors shipped by this crate.
pub fn built_in_descriptors() -> Vec<ModelDescriptor> {
    [
        ("vision", "classifier", true, false),
        ("language", "transformer", true, true),
        ("recommendation", "two-tower", true, false),
        ("reinforcement", "actor-critic", true, false),
        ("forecasting", "transformer", true, false),
        ("audio", "streaming-encoder", true, true),
        ("graph", "message-passing", true, false),
    ]
    .into_iter()
    .map(|(family, variant, training, streaming)| ModelDescriptor {
        family: ModelFamilyId(family.into()),
        variant: ModelVariantId(variant.into()),
        capabilities: ModelCapabilities { training, generation: family == "language", streaming, native: true, wasm: false },
        schema: ModelSchema { input: "versioned tensor schema".into(), output: "versioned tensor schema".into(), version: 1 },
    })
    .collect()
}

/// Projects the built-in registry into the safe, read-only model directory API.
pub fn catalog() -> Vec<ModelCatalogEntry> {
    built_in_descriptors()
        .into_iter()
        .map(|descriptor| ModelCatalogEntry {
            family: descriptor.family,
            variant: descriptor.variant,
            schema: descriptor.schema,
            capabilities: descriptor.capabilities,
            manifest: ModelManifestSummary {
                schema_version: 1,
                state: ManifestState::Ready,
                deployment_targets: vec![DeploymentTarget::Native],
            },
            diagnostic_summary: "descriptor registered; no runtime health included".into(),
        })
        .collect()
}
