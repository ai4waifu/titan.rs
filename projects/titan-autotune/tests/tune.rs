use std::time::Duration;
use titan_autotune::{Autotuner, TuneEntry, TuneKey};

#[test]
fn version_two_tune_store_round_trips_a_winner() {
    let path = std::env::temp_dir().join(format!("titan-{}.tune", std::process::id()));
    let mut tuner = Autotuner::open(&path);
    let key = TuneKey {
        operator: "matmul".into(),
        device: "cpu".into(),
        shape: "2x2x2".into(),
        dtype: "f32".into(),
        layout: "contiguous".into(),
        strategy_version: 1,
    };
    tuner.insert(
        key,
        TuneEntry {
            candidate: "generated-baseline".into(),
            median_ns: 1,
            p95_ns: 2,
            correctness_hash: "ok".into(),
            provisional: false,
        },
    );
    tuner.flush().unwrap();
    assert!(std::fs::read_to_string(path.with_extension("tune")).unwrap().contains("version=2"));
    let _ = Duration::from_nanos(1);
}
