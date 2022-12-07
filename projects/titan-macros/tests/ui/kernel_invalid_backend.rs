use titan_macros::kernel;

#[kernel(backend = Cuda)]
fn invalid_kernel() {}

fn main() {}
