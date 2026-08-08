use titan_profiler::{Collector, TelemetryContext, TelemetryEvent};

fn event(name: &str) -> TelemetryEvent {
    TelemetryEvent {
        sequence: u64::MAX,
        context: TelemetryContext::default(),
        kind: "test".into(),
        name: name.into(),
        value_ns: 0,
    }
}

#[test]
fn collector_assigns_monotonic_sequences_even_when_dropping() {
    let mut collector = Collector::new(2);
    collector.push(event("first"));
    collector.push(event("second"));
    collector.push(event("dropped"));
    let events = collector.drain();
    assert_eq!(events.iter().map(|event| event.sequence).collect::<Vec<_>>(), vec![0, 1]);
    assert_eq!(collector.dropped(), 1);
    collector.push(event("after-drain"));
    assert_eq!(collector.drain()[0].sequence, 3);
}
