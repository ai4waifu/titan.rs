use std::sync::Arc;
use titan_hal::Buffer;
use titan_kernel::{AbiArg, BasicBlock, BlockId, KernelAbi, KernelArg, KernelModule, LaunchConfig};
use titan_types::{BackendId, DType, DeviceId, KernelId};

#[derive(Debug)]
struct FakeBuffer(DeviceId);
impl Buffer for FakeBuffer {
    fn device(&self) -> DeviceId {
        self.0
    }
    fn byte_len(&self) -> usize {
        4
    }
    fn identity(&self) -> u64 {
        1
    }
}

#[test]
fn kernel_ir_and_abi_have_stable_non_empty_identity() {
    let abi = KernelAbi { version: 1, args: vec![], launch: LaunchConfig::default(), workspace_bytes: 0 };
    let ir = KernelModule {
        kernel_id: KernelId("matmul".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi: abi.clone(),
    };
    ir.verify().unwrap();
    assert!(!ir.kernel_id.0.is_empty());
    assert!(!abi.hash().is_empty());
}

#[test]
fn encode_returns_hal_arguments_with_canonical_bytes_and_binding() {
    let abi = KernelAbi {
        version: 1,
        args: vec![AbiArg::Buffer { dtype: DType::F32, writable: false, alignment: 4 }],
        launch: LaunchConfig::default(),
        workspace_bytes: 0,
    };
    let device = DeviceId { backend: BackendId::Cpu, ordinal: 0 };
    let buffer = Arc::new(FakeBuffer(device)) as Arc<dyn Buffer>;
    let encoded =
        abi.encode(&[KernelArg::Buffer { slot: 5, dtype: DType::F32, writable: false, alignment: 4, buffer }]).unwrap();
    assert_eq!(encoded.canonical_abi(), abi.canonical_bytes().as_slice());
    assert_eq!(encoded.bindings().get(&5).unwrap().device_id, device);
    assert_eq!(abi.decode(encoded.payload()).unwrap(), vec![titan_kernel::DecodedArg::Buffer { slot: 5 }]);
}
