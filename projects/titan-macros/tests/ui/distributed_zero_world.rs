use titan_macros::distributed;

#[distributed(world = 0)]
fn invalid_distributed() {}

fn main() {}
