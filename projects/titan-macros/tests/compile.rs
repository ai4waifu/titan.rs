use titan_macros::{distributed, kernel, neural, parameters};

#[neural]
struct DemoModel;

#[parameters]
struct DemoParameters;

#[kernel(block_size = 64, vector_width = 4, pipeline_depth = 2, shared_memory_padding = 0, backend = Auto)]
fn demo_kernel() {}

#[distributed(world = 1, strategy = "data_parallel")]
fn demo_distributed() {}

#[test]
fn metadata_is_stable_and_items_remain_callable() {
    assert_eq!(__TITAN_DemoModel_META, "neural:DemoModel");
    assert_eq!(__TITAN_DemoParameters_META, "parameters:DemoParameters");
    assert_eq!(__TITAN_demo_kernel_META, "kernel:demo_kernel");
    assert_eq!(__TITAN_demo_distributed_META, "distributed:demo_distributed");
    demo_kernel();
    demo_distributed();
}
