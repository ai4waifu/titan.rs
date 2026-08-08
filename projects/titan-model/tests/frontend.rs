use titan_model::{ApiError, ApiErrorCode, ApiJsonSchema, ApiResponse, DeploymentManifest, DeploymentTarget, OnnxModel};
#[test]
fn deployment_and_interchange_are_portable() {
    let manifest = DeploymentManifest { model: "demo".into(), target: DeploymentTarget::Native, backend: "cpu".into() };
    assert!(manifest.encode().contains("Native"));
    assert!(OnnxModel::linear("demo").encode().contains("MatMul"));
}

#[test]
fn public_api_contract_is_serializable_and_versioned() {
    let error = ApiError {
        schema_version: 1,
        request_id: "request-1".into(),
        code: ApiErrorCode::RunNotFound,
        message: "run is not visible".into(),
        retryable: false,
    };
    assert_eq!(serde_json::to_value(error).unwrap()["code"], "run_not_found");
    let response = ApiResponse {
        schema_version: 1,
        request_id: "request-2".into(),
        generated_at: "2026-08-08T00:00:00Z".into(),
        data: Vec::<String>::new(),
    };
    assert_eq!(serde_json::to_value(response).unwrap()["schema_version"], 1);
    assert_eq!(ApiJsonSchema::documents()["$schema"], "https://json-schema.org/draft/2020-12/schema");
}
