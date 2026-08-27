#![warn(missing_docs)]
//! Portable CPU DeviceSession implementation.
use std::{
    collections::HashMap,
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};
use titan_hal::{
    BackendDriver, Buffer, DeviceSession, EncodedLaunchArgs, Event, HalError, LaunchGeometry, LoadedKernel, Stream,
};
use titan_kernel::{AbiArg, KernelAbi, KernelError, KernelModule, TargetCompiler};
use titan_types::{AbiHash, BackendId, DeviceFingerprint, DeviceId, KernelId, KernelLaunchMetadata};

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);
#[derive(Debug)]
struct CpuBuffer {
    device: DeviceId,
    id: u64,
    bytes: Mutex<Vec<u8>>,
}
impl Buffer for CpuBuffer {
    fn device(&self) -> DeviceId {
        self.device
    }
    fn byte_len(&self) -> usize {
        self.bytes.lock().map(|b| b.len()).unwrap_or(0)
    }
    fn identity(&self) -> u64 {
        self.id
    }
}
#[derive(Debug)]
struct CpuStream(DeviceId);
impl Stream for CpuStream {
    fn device(&self) -> DeviceId {
        self.0
    }
}
#[derive(Debug)]
struct CpuEvent(DeviceId);
impl Event for CpuEvent {
    fn device(&self) -> DeviceId {
        self.0
    }
}
#[derive(Debug)]
struct CpuKernel(DeviceId, AbiHash, KernelId, KernelLaunchMetadata);
impl LoadedKernel for CpuKernel {
    fn device(&self) -> DeviceId {
        self.0
    }
    fn abi_hash(&self) -> &AbiHash {
        &self.1
    }
    fn kernel_id(&self) -> &KernelId {
        &self.2
    }
    fn launch_metadata(&self) -> &KernelLaunchMetadata {
        &self.3
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Clone)]
struct Session {
    fingerprint: DeviceFingerprint,
    buffers: Arc<Mutex<HashMap<u64, Arc<CpuBuffer>>>>,
}
impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuSession").field("device", &self.fingerprint.device).finish()
    }
}
impl DeviceSession for Session {
    fn device(&self) -> DeviceId {
        self.fingerprint.device
    }
    fn fingerprint(&self) -> &DeviceFingerprint {
        &self.fingerprint
    }
    fn allocate(&self, bytes: usize, _alignment: usize) -> Result<Arc<dyn Buffer>, HalError> {
        let buffer = Arc::new(CpuBuffer {
            device: self.device(),
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            bytes: Mutex::new(vec![0; bytes]),
        });
        self.buffers
            .lock()
            .map_err(|_| HalError { operation: "allocate", detail: "registry poisoned".into() })?
            .insert(buffer.id, buffer.clone());
        Ok(buffer)
    }
    fn upload(&self, _stream: &dyn Stream, dst: &dyn Buffer, src: &[u8]) -> Result<Arc<dyn Event>, HalError> {
        if dst.device() != self.device() || src.len() > dst.byte_len() {
            return Err(HalError { operation: "upload", detail: "invalid destination or size".into() });
        }
        let b = self
            .buffers
            .lock()
            .map_err(|_| HalError { operation: "upload", detail: "registry poisoned".into() })?
            .get(&dst.identity())
            .cloned()
            .ok_or(HalError { operation: "upload", detail: "foreign buffer".into() })?;
        let mut guard = b.bytes.lock().map_err(|_| HalError { operation: "upload", detail: "buffer poisoned".into() })?;
        guard[..src.len()].copy_from_slice(src);
        Ok(Arc::new(CpuEvent(self.device())))
    }
    fn download(&self, _stream: &dyn Stream, src: &dyn Buffer, dst: &mut [u8]) -> Result<Arc<dyn Event>, HalError> {
        if src.device() != self.device() || dst.len() > src.byte_len() {
            return Err(HalError { operation: "download", detail: "invalid source or size".into() });
        }
        let b = self
            .buffers
            .lock()
            .map_err(|_| HalError { operation: "download", detail: "registry poisoned".into() })?
            .get(&src.identity())
            .cloned()
            .ok_or(HalError { operation: "download", detail: "foreign buffer".into() })?;
        let guard = b.bytes.lock().map_err(|_| HalError { operation: "download", detail: "buffer poisoned".into() })?;
        dst.copy_from_slice(&guard[..dst.len()]);
        Ok(Arc::new(CpuEvent(self.device())))
    }
    fn copy(&self, _stream: &dyn Stream, dst: &dyn Buffer, src: &dyn Buffer, bytes: usize) -> Result<Arc<dyn Event>, HalError> {
        if dst.device() != self.device() || src.device() != self.device() || bytes > dst.byte_len() || bytes > src.byte_len() {
            return Err(HalError { operation: "copy", detail: "invalid buffer or size".into() });
        }
        let registry = self.buffers.lock().map_err(|_| HalError { operation: "copy", detail: "registry poisoned".into() })?;
        let d = registry
            .get(&dst.identity())
            .cloned()
            .ok_or(HalError { operation: "copy", detail: "foreign destination".into() })?;
        let s =
            registry.get(&src.identity()).cloned().ok_or(HalError { operation: "copy", detail: "foreign source".into() })?;
        let data =
            s.bytes.lock().map_err(|_| HalError { operation: "copy", detail: "buffer poisoned".into() })?[..bytes].to_vec();
        let mut guard = d.bytes.lock().map_err(|_| HalError { operation: "copy", detail: "buffer poisoned".into() })?;
        guard[..bytes].copy_from_slice(&data);
        Ok(Arc::new(CpuEvent(self.device())))
    }
    fn create_stream(&self) -> Result<Arc<dyn Stream>, HalError> {
        Ok(Arc::new(CpuStream(self.device())))
    }
    fn create_event(&self) -> Result<Arc<dyn Event>, HalError> {
        Ok(Arc::new(CpuEvent(self.device())))
    }
    fn load(
        &self,
        artifact: &[u8],
        abi_hash: &AbiHash,
        metadata: KernelLaunchMetadata,
    ) -> Result<Arc<dyn LoadedKernel>, HalError> {
        let expected = cpu_add_artifact(abi_hash);
        if artifact != expected.as_slice() {
            return Err(error("load", "invalid CPU artifact or ABI identity"));
        }
        let expected_metadata = elementwise_add_f32_abi()
            .launch_metadata(&KernelId("elementwise.add.f32".into()))
            .map_err(|_| error("load", "invalid CPU add metadata"))?;
        if metadata != expected_metadata {
            return Err(error("load", "launch metadata does not match CPU add ABI"));
        }
        Ok(Arc::new(CpuKernel(self.device(), abi_hash.clone(), KernelId(metadata.entry.clone()), metadata)))
    }
    fn launch(
        &self,
        stream: &dyn Stream,
        kernel: &dyn LoadedKernel,
        args: &EncodedLaunchArgs,
        geometry: &LaunchGeometry,
    ) -> Result<Arc<dyn Event>, HalError> {
        if stream.device() != self.device() || kernel.device() != self.device() {
            return Err(HalError { operation: "launch", detail: "cross-device handle".into() });
        }
        let kernel = kernel.as_any().downcast_ref::<CpuKernel>().ok_or_else(|| error("launch", "foreign loaded kernel"))?;
        if kernel.abi_hash().0.as_bytes() != args.canonical_abi() {
            return Err(error("launch", "launch ABI does not match loaded artifact"));
        }
        if kernel.kernel_id().0 != "elementwise.add.f32" {
            return Err(error("launch", "unsupported CPU kernel entry"));
        }
        if geometry.block[0] == 0 {
            return Err(error("launch", "zero block size"));
        }
        let mut cursor = 0usize;
        let mut slots = [0u32; 3];
        for slot in &mut slots {
            let end = cursor.checked_add(4).ok_or_else(|| error("launch", "payload overflow"))?;
            if end > args.payload().len() {
                return Err(error("launch", "truncated buffer slot"));
            }
            *slot = u32::from_le_bytes(args.payload()[cursor..end].try_into().unwrap());
            cursor = end;
        }
        if cursor != args.payload().len() {
            return Err(error("launch", "trailing ABI bytes"));
        }
        args.validate_for(self.device(), slots)?;
        let lhs_binding = args.bindings().get(&slots[0]).ok_or_else(|| error("launch", "missing lhs binding"))?;
        let rhs_binding = args.bindings().get(&slots[1]).ok_or_else(|| error("launch", "missing rhs binding"))?;
        let out_binding = args.bindings().get(&slots[2]).ok_or_else(|| error("launch", "missing output binding"))?;
        let registry = self.buffers.lock().map_err(|_| error("launch", "registry poisoned"))?;
        let lhs = registry.get(&lhs_binding.buffer.identity()).cloned();
        let rhs = registry.get(&rhs_binding.buffer.identity()).cloned();
        let out = registry.get(&out_binding.buffer.identity()).cloned();
        let (lhs, rhs, out) = match (lhs, rhs, out) {
            (Some(lhs), Some(rhs), Some(out)) => (lhs, rhs, out),
            _ => return Err(error("launch", "foreign buffer binding")),
        };
        let left = lhs.bytes.lock().map_err(|_| error("launch", "buffer poisoned"))?.clone();
        let right = rhs.bytes.lock().map_err(|_| error("launch", "buffer poisoned"))?.clone();
        if left.len() != right.len() || out.byte_len() != left.len() || left.len() % 4 != 0 {
            return Err(error("launch", "elementwise add buffer lengths disagree"));
        }
        let mut output = out.bytes.lock().map_err(|_| error("launch", "buffer poisoned"))?;
        for (index, chunk) in output.chunks_exact_mut(4).enumerate() {
            let offset = index * 4;
            let value = f32::from_le_bytes(left[offset..offset + 4].try_into().unwrap())
                + f32::from_le_bytes(right[offset..offset + 4].try_into().unwrap());
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        Ok(Arc::new(CpuEvent(self.device())))
    }
    fn poll(&self, event: &dyn Event) -> Result<bool, HalError> {
        Ok(event.device() == self.device())
    }
    fn wait(&self, event: &dyn Event) -> Result<(), HalError> {
        if event.device() == self.device() {
            Ok(())
        }
        else {
            Err(HalError { operation: "wait", detail: "cross-device event".into() })
        }
    }
}

fn error(operation: &'static str, detail: impl Into<String>) -> HalError {
    HalError { operation, detail: detail.into() }
}

/// Foundation ABI used by the CPU f32 elementwise add vertical slice.
pub fn elementwise_add_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
        ],
        launch: titan_kernel::LaunchConfig::default(),
        workspace_bytes: 0,
    }
}

fn cpu_add_artifact(abi_hash: &AbiHash) -> Vec<u8> {
    let mut artifact = b"TITAN_CPU_ELEMENTWISE_ADD_F32\0".to_vec();
    artifact.extend_from_slice(abi_hash.0.as_bytes());
    artifact
}

/// Deterministic native CPU artifact compiler for f32 elementwise add.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuCompiler;
impl TargetCompiler for CpuCompiler {
    fn target(&self) -> titan_kernel::KernelTarget {
        titan_kernel::KernelTarget::CpuAvx2
    }
    fn compile(&self, ir: &KernelModule, abi: &KernelAbi, fingerprint: &DeviceFingerprint) -> Result<Vec<u8>, KernelError> {
        if fingerprint.device.backend != BackendId::Cpu {
            return Err(KernelError::Unsupported("CPU compiler requires CPU device".into()));
        }
        ir.verify()?;
        if ir.kernel_id.0 != "elementwise.add.f32" {
            return Err(KernelError::Unsupported("CPU compiler only emits f32 elementwise add".into()));
        }
        if abi != &elementwise_add_f32_abi() {
            return Err(KernelError::InvalidAbi("unexpected elementwise add ABI".into()));
        }
        Ok(cpu_add_artifact(&abi.abi_hash()))
    }
}

/// Compiles the canonical f32 add artifact through the Foundation compiler contract.
pub fn compile_elementwise_add_f32(fingerprint: &DeviceFingerprint) -> Result<(Vec<u8>, KernelAbi), KernelError> {
    let abi = elementwise_add_f32_abi();
    let module = KernelModule {
        kernel_id: KernelId("elementwise.add.f32".into()),
        entry: titan_kernel::BlockId(0),
        blocks: vec![titan_kernel::BasicBlock { id: titan_kernel::BlockId(0), params: vec![], instructions: vec![] }],
        abi: abi.clone(),
    };
    let artifact = CpuCompiler.compile(&module, &abi, fingerprint)?;
    Ok((artifact, abi))
}

/// CPU backend driver for device ordinal zero.
#[derive(Debug, Default)]
pub struct CpuDriver;
impl BackendDriver for CpuDriver {
    fn id(&self) -> BackendId {
        BackendId::Cpu
    }
    fn enumerate(&self) -> Result<Vec<DeviceFingerprint>, HalError> {
        Ok(vec![DeviceFingerprint {
            device: DeviceId { backend: BackendId::Cpu, ordinal: 0 },
            model: "x86_64".into(),
            driver: "native".into(),
            capability_revision: "avx2-fma".into(),
        }])
    }
    fn open(&self, device: DeviceId) -> Result<Arc<dyn DeviceSession>, HalError> {
        self.enumerate()?
            .into_iter()
            .find(|x| x.device == device)
            .map(|fingerprint| {
                Arc::new(Session { fingerprint, buffers: Arc::new(Mutex::new(HashMap::new())) }) as Arc<dyn DeviceSession>
            })
            .ok_or(HalError { operation: "open", detail: format!("unknown device {device:?}") })
    }
}
