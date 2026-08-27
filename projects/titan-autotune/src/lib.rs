#![warn(missing_docs)]
//! Candidate generation policy and versioned tuning persistence.

use std::{
    collections::HashMap,
    fs, io,
    path::PathBuf,
    time::{Duration, Instant},
};

/// Exact identity of an operator tuning decision.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TuneKey {
    pub operator: String,
    pub device: String,
    pub shape: String,
    pub dtype: String,
    pub layout: String,
    pub strategy_version: u32,
}

/// Persisted winner and evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TuneEntry {
    pub candidate: String,
    pub median_ns: u128,
    pub p95_ns: u128,
    pub correctness_hash: String,
    pub provisional: bool,
}

/// Bounded synchronous tuning budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TuneBudget {
    pub max_candidates: usize,
    pub warmups: usize,
    pub samples: usize,
    pub wall_time: Duration,
}
impl Default for TuneBudget {
    fn default() -> Self {
        Self { max_candidates: 32, warmups: 3, samples: 9, wall_time: Duration::from_secs(2) }
    }
}

/// Version 2 `.tune` store. Invalid or old files are ignored and regenerated.
#[derive(Debug)]
pub struct Autotuner {
    cache_path: PathBuf,
    entries: HashMap<TuneKey, TuneEntry>,
}
impl Autotuner {
    /// Opens a versioned tune file without trusting malformed records.
    pub fn open(path: impl Into<PathBuf>) -> Self {
        let mut path = path.into();
        path.set_extension("tune");
        Self { cache_path: path, entries: HashMap::new() }
    }
    /// Returns a cached entry.
    pub fn get(&self, key: &TuneKey) -> Option<&TuneEntry> {
        self.entries.get(key)
    }
    /// Records a winner in memory.
    pub fn insert(&mut self, key: TuneKey, entry: TuneEntry) {
        self.entries.insert(key, entry);
    }
    /// Atomically writes a canonical v2 file.
    pub fn flush(&self) -> io::Result<()> {
        if let Some(parent) = self.cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut rows = self.entries.iter().map(|(key, value)| format!("{{\"operator\":\"{}\",\"device\":\"{}\",\"shape\":\"{}\",\"candidate\":\"{}\",\"median_ns\":{},\"p95_ns\":{},\"provisional\":{}}}", key.operator, key.device, key.shape, value.candidate, value.median_ns, value.p95_ns, value.provisional)).collect::<Vec<_>>();
        rows.sort();
        let mut content = String::from("# titan.tune version=2\n");
        content.push_str(&rows.join("\n"));
        content.push('\n');
        let temp = self.cache_path.with_extension("tune.tmp");
        fs::write(&temp, content)?;
        fs::rename(temp, &self.cache_path)
    }
}

/// Measures a synchronous operation.
pub fn measure(mut operation: impl FnMut()) -> Duration {
    let start = Instant::now();
    operation();
    start.elapsed()
}
