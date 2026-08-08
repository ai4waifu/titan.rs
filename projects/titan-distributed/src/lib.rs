#![warn(missing_docs)]
//! Local deterministic collective contracts for in-process testing.

#[derive(Debug, Clone, PartialEq)]
pub enum CollectiveError {
    EmptyWorld,
    InconsistentLengths,
    /// The checkpoint text cannot be decoded.
    InvalidCheckpoint,
    InvalidManifest,
    RunMismatch,
    EpochMismatch,
    SequenceMismatch,
    Timeout,
    ChecksumMismatch,
}
impl std::fmt::Display for CollectiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "collective failed: {self:?}")
    }
}
impl std::error::Error for CollectiveError {}

/// A deterministic checksum for local protocol records (FNV-1a, rendered as hex).
pub fn checksum(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// A local-only frame carrying run, epoch and monotonic sequence metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalFrame {
    pub run_id: String,
    pub epoch: u64,
    pub sequence: u64,
    pub payload: Vec<u8>,
    pub checksum: String,
}

/// Deterministic in-process transport contract. It performs no network I/O.
#[derive(Clone, Debug)]
pub struct LocalTransport {
    run_id: String,
    epoch: u64,
    next_sequence: u64,
    timeout: std::time::Duration,
}

impl LocalTransport {
    pub fn new(run_id: impl Into<String>, epoch: u64, timeout: std::time::Duration) -> Self {
        Self { run_id: run_id.into(), epoch, next_sequence: 0, timeout }
    }

    pub fn send(&mut self, payload: impl AsRef<[u8]>, timeout: std::time::Duration) -> Result<LocalFrame, CollectiveError> {
        if timeout.is_zero() || self.timeout.is_zero() {
            return Err(CollectiveError::Timeout);
        }
        let payload = payload.as_ref().to_vec();
        let frame = LocalFrame {
            run_id: self.run_id.clone(),
            epoch: self.epoch,
            sequence: self.next_sequence,
            checksum: checksum(&payload),
            payload,
        };
        self.next_sequence = self.next_sequence.checked_add(1).ok_or(CollectiveError::SequenceMismatch)?;
        Ok(frame)
    }

    pub fn receive(&mut self, frame: &LocalFrame, timeout: std::time::Duration) -> Result<Vec<u8>, CollectiveError> {
        if timeout.is_zero() || self.timeout.is_zero() {
            return Err(CollectiveError::Timeout);
        }
        if frame.run_id != self.run_id { return Err(CollectiveError::RunMismatch); }
        if frame.epoch != self.epoch { return Err(CollectiveError::EpochMismatch); }
        if frame.sequence != self.next_sequence { return Err(CollectiveError::SequenceMismatch); }
        if checksum(&frame.payload) != frame.checksum { return Err(CollectiveError::ChecksumMismatch); }
        self.next_sequence = self.next_sequence.checked_add(1).ok_or(CollectiveError::SequenceMismatch)?;
        Ok(frame.payload.clone())
    }
}

pub trait Collective {
    fn all_reduce_sum(&self, shards: &[Vec<f32>]) -> Result<Vec<f32>, CollectiveError>;
    fn reduce_scatter_sum(&self, shards: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, CollectiveError>;
    fn all_gather(&self, shards: &[Vec<f32>]) -> Result<Vec<f32>, CollectiveError>;
}
#[derive(Debug, Default)]
pub struct LocalRing;
impl Collective for LocalRing {
    fn all_reduce_sum(&self, shards: &[Vec<f32>]) -> Result<Vec<f32>, CollectiveError> {
        let Some(first) = shards.first()
        else {
            return Err(CollectiveError::EmptyWorld);
        };
        if shards.iter().any(|s| s.len() != first.len()) {
            return Err(CollectiveError::InconsistentLengths);
        }
        let mut total = vec![0.; first.len()];
        for shard in shards {
            for (out, value) in total.iter_mut().zip(shard) {
                *out += value;
            }
        }
        Ok(total)
    }

    fn reduce_scatter_sum(&self, shards: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, CollectiveError> {
        let total = self.all_reduce_sum(shards)?;
        if total.len() % shards.len() != 0 {
            return Err(CollectiveError::InconsistentLengths);
        }
        let width = total.len() / shards.len();
        Ok(total.chunks(width).map(|c| c.to_vec()).collect())
    }

    fn all_gather(&self, shards: &[Vec<f32>]) -> Result<Vec<f32>, CollectiveError> {
        if shards.is_empty() {
            return Err(CollectiveError::EmptyWorld);
        }
        Ok(shards.iter().flat_map(|s| s.iter().copied()).collect())
    }
}

/// Atomically committed checkpoint manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointManifest {
    pub run_id: String,
    pub step: usize,
    pub shards: u32,
    pub committed: bool,
    pub checksum: String,
}
impl CheckpointManifest {
    pub fn commit(run_id: impl Into<String>, step: usize, shards: u32, checksum: impl Into<String>) -> Self {
        Self { run_id: run_id.into(), step, shards, committed: true, checksum: checksum.into() }
    }
    pub fn validate(&self) -> Result<(), CollectiveError> {
        if self.run_id.is_empty() || self.shards == 0 || !self.committed || self.checksum.is_empty() {
            Err(CollectiveError::InvalidManifest)
        }
        else {
            Ok(())
        }
    }

    /// Validates all identity and integrity fields before restoring state.
    pub fn validate_recovery(&self, run_id: &str, step: usize, payload: &[u8]) -> Result<(), CollectiveError> {
        self.validate()?;
        if self.run_id != run_id || self.step != step { return Err(CollectiveError::InvalidManifest); }
        if self.checksum != checksum(payload) { return Err(CollectiveError::ChecksumMismatch); }
        Ok(())
    }
}

/// Data/model parallel execution strategies accepted by the distributed API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Strategy {
    DataParallel,
    TensorParallel,
    Pipeline1F1B,
    Fsdp,
    Zero { stage: u8 },
}

/// Recoverable training state owned by the framework rather than a Python process.
#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint {
    pub step: usize,
    pub weights: Vec<f32>,
    pub strategy: Strategy,
}
impl Checkpoint {
    /// Encodes state for durable storage; callers can write the returned data to file, Redis, or S3.
    pub fn encode(&self) -> String {
        format!(
            "step={}\nstrategy={:?}\nweights={}",
            self.step,
            self.strategy,
            self.weights.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
        )
    }
    /// Decodes portable checkpoint text for the supported strategy set.
    pub fn decode(source: &str) -> Result<Self, CollectiveError> {
        let mut step = None;
        let mut weights = None;
        let mut strategy = None;
        for line in source.lines() {
            if let Some(v) = line.strip_prefix("step=") {
                step = v.parse().ok();
            }
            else if let Some(v) = line.strip_prefix("weights=") {
                weights = Some(if v.is_empty() {
                    Vec::new()
                }
                else {
                    v.split(',')
                        .map(str::parse)
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| CollectiveError::InvalidCheckpoint)?
                });
            }
            else if let Some(v) = line.strip_prefix("strategy=") {
                strategy = match v {
                    "DataParallel" => Some(Strategy::DataParallel),
                    "TensorParallel" => Some(Strategy::TensorParallel),
                    "Pipeline1F1B" => Some(Strategy::Pipeline1F1B),
                    "Fsdp" => Some(Strategy::Fsdp),
                    "Zero { stage: 1 }" => Some(Strategy::Zero { stage: 1 }),
                    "Zero { stage: 2 }" => Some(Strategy::Zero { stage: 2 }),
                    "Zero { stage: 3 }" => Some(Strategy::Zero { stage: 3 }),
                    _ => None,
                };
            }
        }
        match (step, weights, strategy) {
            (Some(step), Some(weights), Some(strategy)) => Ok(Self { step, weights, strategy }),
            _ => Err(CollectiveError::InvalidCheckpoint),
        }
    }
}
