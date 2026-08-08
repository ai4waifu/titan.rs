#[test]
fn invalid_declarations_report_actionable_diagnostics() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/kernel_zero_block_size.rs");
    cases.compile_fail("tests/ui/kernel_invalid_backend.rs");
    cases.compile_fail("tests/ui/distributed_zero_world.rs");
}
