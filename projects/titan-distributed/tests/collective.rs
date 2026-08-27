use std::time::Duration;

use titan_distributed::{CheckpointManifest, Collective, CollectiveError, LocalRing, LocalTransport, checksum};

#[test]
fn sum() {
    assert_eq!(LocalRing.all_reduce_sum(&[vec![1., 2.], vec![3., 4.]]).unwrap(), vec![4., 6.]);
}

#[test]
fn scatter_gather_and_manifest() {
    assert_eq!(
        LocalRing.reduce_scatter_sum(&[vec![1., 2., 3., 4.], vec![1., 1., 1., 1.]]).unwrap(),
        vec![vec![2., 3.], vec![4., 5.]]
    );
    assert_eq!(LocalRing.all_gather(&[vec![1., 2.], vec![3.]]).unwrap(), vec![1., 2., 3.]);
    assert!(CheckpointManifest::commit("run", 1, 2, "abc").validate().is_ok());
}

#[test]
fn local_transport_checks_run_epoch_sequence_timeout_and_checksum() {
    let timeout = Duration::from_millis(1);
    let mut sender = LocalTransport::new("run-a", 7, timeout);
    let mut receiver = LocalTransport::new("run-a", 7, timeout);
    let frame = sender.send(b"payload", timeout).unwrap();
    assert_eq!(receiver.receive(&frame, timeout).unwrap(), b"payload");

    let mut wrong_run = LocalTransport::new("run-b", 7, timeout);
    assert_eq!(wrong_run.receive(&frame, timeout), Err(CollectiveError::RunMismatch));
    let mut wrong_epoch = LocalTransport::new("run-a", 8, timeout);
    assert_eq!(wrong_epoch.receive(&frame, timeout), Err(CollectiveError::EpochMismatch));
    assert_eq!(receiver.receive(&frame, timeout), Err(CollectiveError::SequenceMismatch));
    assert_eq!(sender.send(b"x", Duration::ZERO), Err(CollectiveError::Timeout));

    let mut corrupt = sender.send(b"payload", timeout).unwrap();
    corrupt.checksum = "broken".to_string();
    let mut verifier = LocalTransport::new("run-a", 7, timeout);
    verifier.receive(&frame, timeout).unwrap();
    assert_eq!(verifier.receive(&corrupt, timeout), Err(CollectiveError::ChecksumMismatch));
}

#[test]
fn recovery_validates_manifest_identity_and_payload_checksum() {
    let payload = b"checkpoint bytes";
    let manifest = CheckpointManifest::commit("run-a", 4, 1, checksum(payload));
    assert!(manifest.validate_recovery("run-a", 4, payload).is_ok());
    assert_eq!(manifest.validate_recovery("run-a", 5, payload), Err(CollectiveError::InvalidManifest));
    assert_eq!(manifest.validate_recovery("run-a", 4, b"changed"), Err(CollectiveError::ChecksumMismatch));
}
