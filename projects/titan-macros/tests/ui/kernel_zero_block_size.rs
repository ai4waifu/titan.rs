use titan_macros::kernel;

#[kernel(block_size = 0)]
fn invalid_kernel() {}

fn main() {}
