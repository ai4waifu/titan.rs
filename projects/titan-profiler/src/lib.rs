#![warn(missing_docs)]
//! In-process operator timing collection.

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};
#[derive(Clone, Debug)]
pub struct Span {
    pub name: String,
    pub elapsed: Duration,
}

/// Stable identifiers attached to every telemetry record.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TelemetryContext {
    pub run_id: String,
    pub model_id: String,
    pub graph_id: String,
    pub rank: u32,
    pub step: u64,
}
/// A normalized event emitted by runtime components.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryEvent {
    pub sequence: u64,
    pub context: TelemetryContext,
    pub kind: String,
    pub name: String,
    pub value_ns: u64,
}
/// Bounded in-process collector. Events beyond capacity are dropped and counted.
#[derive(Debug)]
pub struct Collector {
    capacity: usize,
    events: Vec<TelemetryEvent>,
    dropped: u64,
    next_sequence: u64,
}
impl Collector {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, events: Vec::new(), dropped: 0, next_sequence: 0 }
    }
    pub fn push(&mut self, mut event: TelemetryEvent) {
        event.sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        if self.events.len() < self.capacity {
            self.events.push(event)
        }
        else {
            self.dropped += 1;
        }
    }
    pub fn drain(&mut self) -> Vec<TelemetryEvent> {
        std::mem::take(&mut self.events)
    }
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}
#[derive(Debug, Default)]
pub struct Profiler {
    spans: Vec<Span>,
}
impl Profiler {
    pub fn measure<T>(&mut self, name: impl Into<String>, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = f();
        self.spans.push(Span { name: name.into(), elapsed: start.elapsed() });
        result
    }
    pub fn summary(&self) -> BTreeMap<String, Duration> {
        let mut totals = BTreeMap::new();
        for span in &self.spans {
            *totals.entry(span.name.clone()).or_default() += span.elapsed;
        }
        totals
    }

    /// Produces dependency-free telemetry suitable for a local collector.
    pub fn export(&self) -> String {
        self.summary()
            .into_iter()
            .map(|(name, elapsed)| format!("{name},{}", elapsed.as_nanos()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}
