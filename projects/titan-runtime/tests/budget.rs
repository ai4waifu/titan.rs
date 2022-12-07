use titan_runtime::{ResourceBudget, ResourceRequest};

#[test]
fn budget_rejects_before_allocation() {
    let report = ResourceBudget { device_bytes: 100, host_bytes: 200, concurrency: 2 }.assess(ResourceRequest {
        device_bytes: 101,
        host_bytes: 50,
        concurrency: 1,
    });
    assert!(!report.feasible);
    assert_eq!(report.device_available, 0);
    assert_eq!(report.host_available, 150);
    assert_eq!(report.concurrency_available, 1);
}
