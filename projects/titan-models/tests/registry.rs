use titan_model::{ModelFamilyId, ModelVariantId};

#[test]
fn built_in_models_are_explicitly_discoverable() {
    let registry = titan_models::registry();
    let transformer = registry.find(&ModelFamilyId("language".into()), &ModelVariantId("transformer".into()));
    assert!(transformer.is_some());
    assert!(transformer.unwrap().capabilities.generation);
}

#[test]
fn catalog_exposes_manifest_versions_and_deployment_targets() {
    let catalog = titan_models::catalog();
    assert_eq!(catalog.len(), 7);
    assert_eq!(catalog[0].manifest.schema_version, 1);
    assert!(!catalog[0].manifest.deployment_targets.is_empty());
}
