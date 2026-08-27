use std::sync::Arc;
use titan_hal::{BackendDriver, Buffer, BufferBinding, EncodedLaunchArgs};
use titan_types::{BackendId, DeviceId};

#[derive(Debug)]
struct FakeBuffer(DeviceId, u64);
impl Buffer for FakeBuffer {
    fn device(&self) -> DeviceId {
        self.0
    }
    fn byte_len(&self) -> usize {
        16
    }
    fn identity(&self) -> u64 {
        self.1
    }
}

struct EmptyDriver;
impl BackendDriver for EmptyDriver {
    fn id(&self) -> BackendId {
        BackendId::Cpu
    }
    fn enumerate(&self) -> Result<Vec<titan_types::DeviceFingerprint>, titan_hal::HalError> {
        Ok(Vec::new())
    }
    fn open(&self, device: titan_types::DeviceId) -> Result<std::sync::Arc<dyn titan_hal::DeviceSession>, titan_hal::HalError> {
        Err(titan_hal::HalError { operation: "open", detail: format!("no fake device: {device:?}") })
    }
}

#[test]
fn backend_contract_does_not_expose_operator_methods() {
    assert_eq!(EmptyDriver.id(), BackendId::Cpu);
    assert!(EmptyDriver.enumerate().unwrap().is_empty());
}

fn device(ordinal: u32) -> DeviceId {
    DeviceId { backend: BackendId::Cpu, ordinal }
}

fn binding(slot: u32, device: DeviceId) -> BufferBinding {
    let buffer = Arc::new(FakeBuffer(device, slot as u64)) as Arc<dyn Buffer>;
    BufferBinding { slot, buffer, device_id: device }
}

#[test]
fn duplicate_slots_are_rejected_at_construction() {
    let error = EncodedLaunchArgs::try_new(vec![1], vec![2], [binding(3, device(0)), binding(3, device(0))]).unwrap_err();
    assert_eq!(error.operation, "encode");
    assert!(error.detail.contains("duplicate"));
}

#[test]
fn missing_slots_are_rejected_by_validation() {
    let encoded = EncodedLaunchArgs::try_new(vec![], vec![], [binding(1, device(0))]).unwrap();
    let error = encoded.validate_for(device(0), &[1, 2]).unwrap_err();
    assert!(error.detail.contains("missing"));
    assert!(error.detail.contains("2"));
}

#[test]
fn cross_session_slots_are_rejected_by_validation() {
    let encoded = EncodedLaunchArgs::try_new(vec![], vec![], [binding(1, device(1))]).unwrap();
    let error = encoded.validate_for(device(0), &[1]).unwrap_err();
    assert!(error.detail.contains("cross-session"));
}

#[test]
fn bindings_round_trip_with_canonical_bytes() {
    let expected = vec![9, 8, 7];
    let encoded = EncodedLaunchArgs::try_new(vec![1, 2], expected.clone(), [binding(4, device(0))]).unwrap();
    assert_eq!(encoded.payload(), &[1, 2]);
    assert_eq!(encoded.canonical_abi(), expected.as_slice());
    assert_eq!(encoded.bindings().get(&4).unwrap().device_id, device(0));
    encoded.validate_for(device(0), &[4]).unwrap();
}
