#![warn(missing_docs)]
//! Deterministic candidate selection with a portable line-based cache.

use std::{
    collections::HashMap,
    fs, io,
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MatmulKey {
    pub backend: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}
impl MatmulKey {
    pub fn encode(&self) -> String {
        format!("{},{},{},{}", self.backend, self.m, self.n, self.k)
    }
}

#[derive(Debug)]
pub struct Autotuner {
    cache_path: PathBuf,
    choices: HashMap<MatmulKey, usize>,
}
impl Autotuner {
    pub fn open(cache_path: impl Into<PathBuf>) -> Self {
        let mut cache_path = cache_path.into();
        // Tune files are the only persisted autotune format.  Normalize legacy
        // callers so an old `.cache` path can never be written again.
        if cache_path.extension().and_then(|ext| ext.to_str()) != Some("tune") {
            cache_path.set_extension("tune");
        }
        let mut choices = HashMap::new();
        if let Ok(text) = fs::read_to_string(&cache_path) {
            for line in text.lines().filter(|line| !line.is_empty() && !line.starts_with("#")) {
                let p: Vec<_> = line.split(',').collect();
                if p.len() == 5 {
                    if let (Ok(m), Ok(n), Ok(k), Ok(t)) = (p[1].parse(), p[2].parse(), p[3].parse(), p[4].parse()) {
                        choices.insert(MatmulKey { backend: p[0].into(), m, n, k }, t);
                    }
                }
            }
        }
        Self { cache_path, choices }
    }
    pub fn choose<F>(&mut self, key: MatmulKey, mut benchmark: F) -> usize
    where
        F: FnMut(usize) -> Duration,
    {
        if let Some(&tile) = self.choices.get(&key) {
            return tile;
        }
        let mut best = (8, Duration::MAX);
        for tile in [8, 16, 32, 64] {
            let elapsed = benchmark(tile);
            if elapsed < best.1 {
                best = (tile, elapsed);
            }
        }
        self.choices.insert(key, best.0);
        let _ = self.flush();
        best.0
    }
    pub fn flush(&self) -> io::Result<()> {
        if let Some(parent) = self.cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut rows: Vec<_> = self.choices.iter().map(|(k, v)| format!("{},{}", k.encode(), v)).collect();
        rows.sort();
        let mut output = String::from("# titan.tune version=1\n");
        output.push_str(&rows.join("\n"));
        if !rows.is_empty() {
            output.push('\n');
        }
        fs::write(&self.cache_path, output)
    }

    /// Records production telemetry as a candidate result and persists an improvement.
    pub fn record_feedback(&mut self, key: MatmulKey, tile: usize, observed: Duration, incumbent: Duration) -> bool {
        if observed < incumbent {
            self.choices.insert(key, tile);
            let _ = self.flush();
            true
        }
        else {
            false
        }
    }
}

pub fn measure(mut operation: impl FnMut()) -> Duration {
    let start = Instant::now();
    operation();
    start.elapsed()
}
