#![warn(missing_docs)]
//! WebGPU backend for Titan's backend-neutral launch contract.

mod wgsl;

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
};
use titan_hal::{
    BackendDriver, Buffer, DeviceSession, EncodedLaunchArgs, Event, HalError, LaunchGeometry, LoadedKernel, Stream,
};
use titan_kernel::{AbiArg, KernelAbi, KernelError};
use titan_types::{AbiHash, BackendId, DeviceFingerprint, DeviceId, KernelId, KernelLaunchMetadata, LaunchArgKind};
use wgpu::util::DeviceExt;
use wgsl::{WgpuKernelKind, WgslArtifact};

static NEXT_BUFFER_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub use wgsl::{WgpuArtifact, WgpuCompiler};

/// Fixed ABI for f32 elementwise add with an explicit element count scalar.
pub fn elementwise_add_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
        ],
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Fixed ABI for a row-major f32 GEMM: `A[M,K] * B[K,N] -> C[M,N]`.
pub fn gemm_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
        ],
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Static contract required by WebGPU F32 contiguous GEMM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GemmF32Descriptor {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lhs_shape: [u32; 2],
    pub rhs_shape: [u32; 2],
    pub output_shape: [u32; 2],
    pub lhs_dtype: titan_types::DType,
    pub rhs_dtype: titan_types::DType,
    pub output_dtype: titan_types::DType,
    pub lhs_contiguous: bool,
    pub rhs_contiguous: bool,
    pub output_contiguous: bool,
    pub transpose_lhs: bool,
    pub transpose_rhs: bool,
}

impl GemmF32Descriptor {
    /// Validates the fixed F32 contiguous GEMM contract before lowering or launch.
    pub fn validate(&self) -> Result<(), KernelError> {
        if self.transpose_lhs || self.transpose_rhs {
            return Err(KernelError::InvalidAbi("WebGPU GEMM does not implement transpose".into()));
        }
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(KernelError::InvalidAbi("WebGPU GEMM M, N, and K must be non-zero".into()));
        }
        if self.lhs_dtype != titan_types::DType::F32
            || self.rhs_dtype != titan_types::DType::F32
            || self.output_dtype != titan_types::DType::F32
        {
            return Err(KernelError::InvalidAbi("WebGPU GEMM requires F32 inputs and output".into()));
        }
        if !self.lhs_contiguous || !self.rhs_contiguous || !self.output_contiguous {
            return Err(KernelError::InvalidAbi(
                "WebGPU GEMM requires contiguous row-major A, B, and C buffers".into(),
            ));
        }
        if self.lhs_shape != [self.m, self.k] || self.rhs_shape != [self.k, self.n] || self.output_shape != [self.m, self.n]
        {
            return Err(KernelError::InvalidAbi(
                "WebGPU GEMM shapes must be A[M,K], B[K,N], and C[M,N]".into(),
            ));
        }
        Ok(())
    }
}

struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    device_id: DeviceId,
}

impl std::fmt::Debug for WgpuContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("WgpuContext").field("device", &self.device_id).finish()
    }
}

struct WgpuDiscovery {
    instance: wgpu::Instance,
    devices: Vec<DeviceFingerprint>,
    adapters: Vec<wgpu::Adapter>,
}

impl std::fmt::Debug for WgpuDiscovery {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("WgpuDiscovery").field("devices", &self.devices).finish()
    }
}

impl WgpuDiscovery {
    fn open() -> Result<Self, HalError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        let mut devices = Vec::with_capacity(adapters.len());
        for (ordinal, adapter) in adapters.iter().enumerate() {
            let info = adapter.get_info();
            devices.push(DeviceFingerprint {
                device: DeviceId { backend: BackendId::Wgpu, ordinal: ordinal as u32 },
                model: info.name,
                driver: format!("{:?}", info.backend),
                capability_revision: format!("webgpu-{}", info.device_type as u8),
            });
        }
        Ok(Self { instance, devices, adapters })
    }

    fn open_session(&self, ordinal: u32) -> Result<Arc<WgpuContext>, HalError> {
        let adapter = self
            .adapters
            .get(ordinal as usize)
            .ok_or_else(|| error("open", "WebGPU device ordinal is unavailable"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("titan-wgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|failure| error("request_device", failure.to_string()))?;
        Ok(Arc::new(WgpuContext {
            device,
            queue,
            device_id: DeviceId { backend: BackendId::Wgpu, ordinal },
        }))
    }
}

/// WebGPU backend registered through the common Titan HAL contract.
#[derive(Clone, Debug)]
pub struct WgpuDriver {
    discovery: Arc<WgpuDiscovery>,
}

impl WgpuDriver {
    /// Opens the WebGPU backend and enumerates adapters.
    pub fn open() -> Result<Self, HalError> {
        Ok(Self { discovery: Arc::new(WgpuDiscovery::open()?) })
    }
}

#[derive(Debug)]
struct WgpuBuffer {
    id: u64,
    context: Arc<WgpuContext>,
    raw: wgpu::Buffer,
    bytes: usize,
}

impl Buffer for WgpuBuffer {
    fn device(&self) -> DeviceId {
        self.context.device_id
    }

    fn byte_len(&self) -> usize {
        self.bytes
    }

    fn identity(&self) -> u64 {
        self.id
    }
}

#[derive(Debug)]
struct WgpuStream {
    context: Arc<WgpuContext>,
}

impl Stream for WgpuStream {
    fn device(&self) -> DeviceId {
        self.context.device_id
    }
}

#[derive(Debug)]
struct WgpuEvent {
    context: Arc<WgpuContext>,
    completed: Arc<AtomicBool>,
}

impl Event for WgpuEvent {
    fn device(&self) -> DeviceId {
        self.context.device_id
    }
}

#[derive(Debug)]
struct WgpuKernel {
    context: Arc<WgpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    abi_hash: AbiHash,
    kernel_id: KernelId,
    metadata: KernelLaunchMetadata,
    kind: WgpuKernelKind,
}

impl LoadedKernel for WgpuKernel {
    fn device(&self) -> DeviceId {
        self.context.device_id
    }

    fn abi_hash(&self) -> &AbiHash {
        &self.abi_hash
    }

    fn kernel_id(&self) -> &KernelId {
        &self.kernel_id
    }

    fn launch_metadata(&self) -> &KernelLaunchMetadata {
        &self.metadata
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
struct WgpuSession {
    context: Arc<WgpuContext>,
    fingerprint: DeviceFingerprint,
    buffers: Mutex<HashMap<u64, Arc<WgpuBuffer>>>,
    streams: Mutex<HashMap<usize, Arc<WgpuStream>>>,
    events: Mutex<HashMap<usize, Arc<WgpuEvent>>>,
}

impl WgpuSession {
    fn buffer(&self, buffer: &dyn Buffer) -> Result<Arc<WgpuBuffer>, HalError> {
        if buffer.device() != self.device() {
            return Err(error("buffer", "cross-device buffer"));
        }
        self.buffers
            .lock()
            .map_err(|_| error("buffer", "registry poisoned"))?
            .get(&buffer.identity())
            .cloned()
            .ok_or_else(|| error("buffer", "foreign buffer"))
    }

    fn stream(&self, stream: &dyn Stream) -> Result<Arc<WgpuStream>, HalError> {
        if stream.device() != self.device() {
            return Err(error("stream", "cross-device stream"));
        }
        self.streams
            .lock()
            .map_err(|_| error("stream", "registry poisoned"))?
            .get(&stream_key(stream))
            .cloned()
            .ok_or_else(|| error("stream", "foreign stream"))
    }

    fn event(&self, event: &dyn Event) -> Result<Arc<WgpuEvent>, HalError> {
        if event.device() != self.device() {
            return Err(error("event", "cross-device event"));
        }
        self.events
            .lock()
            .map_err(|_| error("event", "registry poisoned"))?
            .get(&event_key(event))
            .cloned()
            .ok_or_else(|| error("event", "foreign event"))
    }

    fn register_event(&self, event: Arc<WgpuEvent>) -> Result<Arc<dyn Event>, HalError> {
        self.events
            .lock()
            .map_err(|_| error("event", "registry poisoned"))?
            .insert(Arc::as_ptr(&event) as usize, event.clone());
        Ok(event)
    }

    fn completed_event(&self) -> Result<Arc<dyn Event>, HalError> {
        let event = Arc::new(WgpuEvent {
            context: self.context.clone(),
            completed: Arc::new(AtomicBool::new(true)),
        });
        self.register_event(event)
    }

    fn submitted_event(&self) -> Result<Arc<dyn Event>, HalError> {
        let completed = Arc::new(AtomicBool::new(false));
        let event = Arc::new(WgpuEvent { context: self.context.clone(), completed: completed.clone() });
        self.context.queue.on_submitted_work_done(move || {
            completed.store(true, Ordering::Release);
        });
        self.register_event(event)
    }

    fn bind_group_layout(device: &wgpu::Device, kind: WgpuKernelKind) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("titan-wgpu-kernel"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(wgpu::BufferSize::new(match kind {
                            WgpuKernelKind::GemmF32 => 12,
                            WgpuKernelKind::ElementwiseAddF32 => 4,
                        })
                        .expect("uniform size")),
                    },
                    count: None,
                },
            ],
        })
    }

    fn uniform_bytes(kind: WgpuKernelKind, scalars: &[Vec<u8>]) -> Result<Vec<u8>, HalError> {
        match kind {
            WgpuKernelKind::GemmF32 => {
                if scalars.len() != 3 {
                    return Err(error("launch", "GEMM requires three scalar arguments"));
                }
                let m = i32::from_le_bytes(scalars[0].as_slice().try_into().map_err(|_| error("launch", "invalid m"))?);
                let n = i32::from_le_bytes(scalars[1].as_slice().try_into().map_err(|_| error("launch", "invalid n"))?);
                let k = i32::from_le_bytes(scalars[2].as_slice().try_into().map_err(|_| error("launch", "invalid k"))?);
                Ok([m, n, k].map(i32::to_le_bytes).into_iter().flatten().collect())
            }
            WgpuKernelKind::ElementwiseAddF32 => {
                if scalars.len() != 1 {
                    return Err(error("launch", "elementwise add requires one scalar argument"));
                }
                Ok(scalars[0].clone())
            }
        }
    }
}

impl DeviceSession for WgpuSession {
    fn device(&self) -> DeviceId {
        self.context.device_id
    }

    fn fingerprint(&self) -> &DeviceFingerprint {
        &self.fingerprint
    }

    fn allocate(&self, bytes: usize, _alignment: usize) -> Result<Arc<dyn Buffer>, HalError> {
        if bytes == 0 {
            return Err(error("allocate", "zero-byte WebGPU allocations are unsupported"));
        }
        let raw = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("titan-wgpu-buffer"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffer = Arc::new(WgpuBuffer {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            context: self.context.clone(),
            raw,
            bytes,
        });
        self.buffers
            .lock()
            .map_err(|_| error("allocate", "registry poisoned"))?
            .insert(buffer.id, buffer.clone());
        Ok(buffer)
    }

    fn upload(&self, _stream: &dyn Stream, dst: &dyn Buffer, src: &[u8]) -> Result<Arc<dyn Event>, HalError> {
        let destination = self.buffer(dst)?;
        if src.len() > destination.byte_len() {
            return Err(error("upload", "source exceeds allocation"));
        }
        self.context.queue.write_buffer(&destination.raw, 0, src);
        self.completed_event()
    }

    fn download(&self, _stream: &dyn Stream, src: &dyn Buffer, dst: &mut [u8]) -> Result<Arc<dyn Event>, HalError> {
        let source = self.buffer(src)?;
        if dst.len() > source.byte_len() {
            return Err(error("download", "destination exceeds allocation"));
        }
        let staging = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("titan-wgpu-staging-read"),
            size: dst.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("titan-wgpu-download") });
        encoder.copy_buffer_to_buffer(&source.raw, 0, &staging, 0, dst.len() as u64);
        self.context.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.context.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| error("download", "map callback dropped"))?
            .map_err(|failure| error("download", failure.to_string()))?;
        dst.copy_from_slice(&slice.get_mapped_range()[..dst.len()]);
        drop(slice.get_mapped_range());
        staging.unmap();
        self.completed_event()
    }

    fn copy(&self, _stream: &dyn Stream, dst: &dyn Buffer, src: &dyn Buffer, bytes: usize) -> Result<Arc<dyn Event>, HalError> {
        let destination = self.buffer(dst)?;
        let source = self.buffer(src)?;
        if bytes > source.byte_len() || bytes > destination.byte_len() {
            return Err(error("copy", "transfer exceeds allocation"));
        }
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("titan-wgpu-copy") });
        encoder.copy_buffer_to_buffer(&source.raw, 0, &destination.raw, 0, bytes as u64);
        self.context.queue.submit(Some(encoder.finish()));
        self.submitted_event()
    }

    fn create_stream(&self) -> Result<Arc<dyn Stream>, HalError> {
        let stream = Arc::new(WgpuStream { context: self.context.clone() });
        self.streams
            .lock()
            .map_err(|_| error("stream", "registry poisoned"))?
            .insert(Arc::as_ptr(&stream) as usize, stream.clone());
        Ok(stream)
    }

    fn create_event(&self) -> Result<Arc<dyn Event>, HalError> {
        let event = Arc::new(WgpuEvent {
            context: self.context.clone(),
            completed: Arc::new(AtomicBool::new(false)),
        });
        self.register_event(event)
    }

    fn load(
        &self,
        artifact: &[u8],
        abi_hash: &AbiHash,
        metadata: KernelLaunchMetadata,
    ) -> Result<Arc<dyn LoadedKernel>, HalError> {
        let wgsl = WgslArtifact::from_driver_bytes(artifact).map_err(|detail| error("load", detail))?;
        let kind = if metadata.entry.contains("gemm_f32") {
            WgpuKernelKind::GemmF32
        } else if metadata.entry.contains("elementwise_add_f32") {
            WgpuKernelKind::ElementwiseAddF32
        } else {
            return Err(error("load", format!("unsupported WebGPU entry `{}`", metadata.entry)));
        };
        let bind_group_layout = Self::bind_group_layout(&self.context.device, kind);
        let module = self
            .context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("titan-wgpu-shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl.source().into()),
            });
        let pipeline = self
            .context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("titan-wgpu-pipeline"),
                layout: Some(
                    &self
                        .context
                        .device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("titan-wgpu-pipeline-layout"),
                            bind_group_layouts: &[&bind_group_layout],
                            push_constant_ranges: &[],
                        }),
                ),
                module: &module,
                entry_point: Some(metadata.entry.as_str()),
                compilation_options: Default::default(),
                cache: None,
            });
        Ok(Arc::new(WgpuKernel {
            context: self.context.clone(),
            pipeline,
            bind_group_layout,
            abi_hash: abi_hash.clone(),
            kernel_id: KernelId(metadata.entry.clone()),
            metadata,
            kind,
        }))
    }

    fn launch(
        &self,
        _stream: &dyn Stream,
        kernel: &dyn LoadedKernel,
        args: &EncodedLaunchArgs,
        geometry: &LaunchGeometry,
    ) -> Result<Arc<dyn Event>, HalError> {
        if kernel.device() != self.device() {
            return Err(error("launch", "cross-device kernel"));
        }
        let kernel = kernel
            .as_any()
            .downcast_ref::<WgpuKernel>()
            .ok_or_else(|| error("launch", "foreign loaded kernel"))?;
        if !Arc::ptr_eq(&kernel.context, &self.context) {
            return Err(error("launch", "kernel belongs to another WebGPU session"));
        }
        if args.canonical_abi() != kernel.abi_hash.0.as_bytes() {
            return Err(error("launch", "encoded arguments do not match kernel ABI"));
        }

        let mut cursor = 0usize;
        let mut buffer_slots = Vec::new();
        let mut scalars = Vec::new();
        for kind in &kernel.metadata.arguments {
            match kind {
                LaunchArgKind::Buffer => {
                    let end = cursor.checked_add(4).ok_or_else(|| error("launch", "buffer slot overflow"))?;
                    let slot = args
                        .payload()
                        .get(cursor..end)
                        .ok_or_else(|| error("launch", "truncated buffer slot"))
                        .and_then(|bytes| {
                            bytes.try_into().map(u32::from_le_bytes).map_err(|_| error("launch", "invalid buffer slot"))
                        })?;
                    cursor = end;
                    buffer_slots.push(slot);
                }
                LaunchArgKind::Scalar { byte_len } => {
                    let length_end = cursor.checked_add(4).ok_or_else(|| error("launch", "scalar length overflow"))?;
                    let length = args
                        .payload()
                        .get(cursor..length_end)
                        .ok_or_else(|| error("launch", "truncated scalar length"))
                        .and_then(|bytes| {
                            bytes.try_into().map(u32::from_le_bytes).map_err(|_| error("launch", "invalid scalar length"))
                        })? as usize;
                    cursor = length_end;
                    if length != *byte_len as usize {
                        return Err(error("launch", "scalar width disagrees with ABI"));
                    }
                    let end = cursor.checked_add(length).ok_or_else(|| error("launch", "scalar overflow"))?;
                    let value = args.payload().get(cursor..end).ok_or_else(|| error("launch", "truncated scalar"))?.to_vec();
                    cursor = end;
                    scalars.push(value);
                }
                LaunchArgKind::Shape { rank } => {
                    let length = *rank as usize * 8;
                    let end = cursor.checked_add(length).ok_or_else(|| error("launch", "shape overflow"))?;
                    cursor = end;
                }
            }
        }
        if cursor != args.payload().len() {
            return Err(error("launch", "trailing ABI bytes"));
        }
        if buffer_slots.len() != 3 {
            return Err(error("launch", "expected three buffer bindings"));
        }
        args.validate_for(self.device(), buffer_slots.iter())?;
        let buffers = buffer_slots
            .iter()
            .map(|slot| {
                let binding = args
                    .bindings()
                    .get(slot)
                    .ok_or_else(|| error("launch", format!("missing buffer binding for slot {slot}")))?;
                self.buffer(binding.buffer.as_ref())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let uniform_bytes = Self::uniform_bytes(kernel.kind, &scalars)?;
        let uniform_size = match kernel.kind {
            WgpuKernelKind::GemmF32 => 12,
            WgpuKernelKind::ElementwiseAddF32 => 4,
        };
        let mut uniform_padded = uniform_bytes;
        uniform_padded.resize(uniform_size.max(16), 0);
        let uniform = self
            .context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("titan-wgpu-uniform"),
                contents: &uniform_padded,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("titan-wgpu-bind-group"),
            layout: &kernel.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buffers[0].raw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buffers[1].raw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buffers[2].raw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: uniform.as_entire_binding() },
            ],
        });
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("titan-wgpu-launch") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("titan-wgpu-compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(geometry.grid[0], geometry.grid[1], geometry.grid[2]);
        }
        self.context.queue.submit(Some(encoder.finish()));
        self.submitted_event()
    }

    fn poll(&self, event: &dyn Event) -> Result<bool, HalError> {
        let event = self.event(event)?;
        self.context.device.poll(wgpu::Maintain::Poll);
        Ok(event.completed.load(Ordering::Acquire))
    }

    fn wait(&self, event: &dyn Event) -> Result<(), HalError> {
        let event = self.event(event)?;
        while !event.completed.load(Ordering::Acquire) {
            self.context.device.poll(wgpu::Maintain::Wait);
        }
        Ok(())
    }

    fn wait_event(&self, stream: &dyn Stream, event: &dyn Event) -> Result<(), HalError> {
        let _ = self.stream(stream)?;
        // Prototype queue is host-ordered; validate stream then wait for completion.
        self.wait(event)
    }
}

impl BackendDriver for WgpuDriver {
    fn id(&self) -> BackendId {
        BackendId::Wgpu
    }

    fn enumerate(&self) -> Result<Vec<DeviceFingerprint>, HalError> {
        Ok(self.discovery.devices.clone())
    }

    fn open(&self, device: DeviceId) -> Result<Arc<dyn DeviceSession>, HalError> {
        if device.backend != BackendId::Wgpu {
            return Err(error("open", "non-WebGPU device passed to WebGPU driver"));
        }
        let fingerprint = self
            .discovery
            .devices
            .get(device.ordinal as usize)
            .cloned()
            .ok_or_else(|| error("open", "WebGPU device ordinal is unavailable"))?;
        Ok(Arc::new(WgpuSession {
            context: self.discovery.open_session(device.ordinal)?,
            fingerprint,
            buffers: Mutex::new(HashMap::new()),
            streams: Mutex::new(HashMap::new()),
            events: Mutex::new(HashMap::new()),
        }))
    }
}

fn stream_key(stream: &dyn Stream) -> usize {
    stream as *const dyn Stream as *const () as usize
}

fn event_key(event: &dyn Event) -> usize {
    event as *const dyn Event as *const () as usize
}

fn error(operation: &'static str, detail: impl Into<String>) -> HalError {
    HalError { operation, detail: detail.into() }
}
