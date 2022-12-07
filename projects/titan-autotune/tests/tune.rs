use std::{fs, time::Duration};
use titan_autotune::{Autotuner, MatmulKey};

#[test]
fn normalizes_legacy_extension_and_writes_versioned_tune() {
    let path = std::env::temp_dir().join(format!("titan-autotune-{}.cache", std::process::id()));
    let tune = path.with_extension("tune");
    let _ = fs::remove_file(&path);
    let _ = fs::remove_file(&tune);
    let mut tuner = Autotuner::open(&path);
    let key = MatmulKey { backend: "cpu".into(), m: 2, n: 3, k: 4 };
    assert_eq!(tuner.choose(key, |_| Duration::from_micros(1)), 8);
    let text = fs::read_to_string(&tune).unwrap();
    assert!(text.starts_with("# titan.tune version=1\n"));
    assert!(!path.exists());
    let _ = fs::remove_file(tune);
}
