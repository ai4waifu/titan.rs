#![warn(missing_docs)]
//! NVIDIA CUDA Driver API backend for Titan's backend-neutral launch contract.
//!
//! The display-driver supplied `nvcuda.dll` is loaded dynamically.  This crate
//! does not link CUDART or NVRTC. PTX is lowered privately from Titan Kernel
//! IR and reaches the Driver API only as a NUL-terminated artifact.

mod ptx;

use libloading::{Library, Symbol};
use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};
use titan_hal::{
    BackendDriver, Buffer, DeviceSession, EncodedLaunchArgs, Event, HalError, LaunchGeometry, LoadedKernel, Stream,
};
use titan_kernel::{AbiArg, KernelAbi, KernelError, KernelModule, KernelTarget, TargetCompiler};
use titan_types::{AbiHash, BackendId, DeviceFingerprint, DeviceId, KernelId, KernelLaunchMetadata, LaunchArgKind};

type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuDevicePtr = u64;
type CuStreamHandle = *mut c_void;
type CuEventHandle = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const CUDA_ERROR_NOT_READY: CuResult = 600;
const CU_JIT_INFO_LOG_BUFFER: i32 = 3;
const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: i32 = 4;
const CU_JIT_ERROR_LOG_BUFFER: i32 = 5;
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: i32 = 6;

type CuInit = unsafe extern "C" fn(u32) -> CuResult;
type CuDeviceGetCount = unsafe extern "C" fn(*mut i32) -> CuResult;
type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, i32) -> CuResult;
type CuDeviceGetName = unsafe extern "C" fn(*mut i8, i32, CuDevice) -> CuResult;
type CuDriverGetVersion = unsafe extern "C" fn(*mut i32) -> CuResult;
type CuDeviceComputeCapability = unsafe extern "C" fn(*mut i32, *mut i32, CuDevice) -> CuResult;
type CuDevicePrimaryCtxRetain = unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult;
type CuDevicePrimaryCtxRelease = unsafe extern "C" fn(CuDevice) -> CuResult;
type CuCtxSetCurrent = unsafe extern "C" fn(CuContext) -> CuResult;
type CuMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult;
type CuMemFree = unsafe extern "C" fn(CuDevicePtr) -> CuResult;
type CuMemcpyHtoDAsync = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize, CuStreamHandle) -> CuResult;
type CuMemcpyDtoHAsync = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize, CuStreamHandle) -> CuResult;
type CuMemcpyDtoDAsync = unsafe extern "C" fn(CuDevicePtr, CuDevicePtr, usize, CuStreamHandle) -> CuResult;
type CuStreamCreate = unsafe extern "C" fn(*mut CuStreamHandle, u32) -> CuResult;
type CuStreamDestroy = unsafe extern "C" fn(CuStreamHandle) -> CuResult;
type CuStreamSynchronize = unsafe extern "C" fn(CuStreamHandle) -> CuResult;
type CuEventCreate = unsafe extern "C" fn(*mut CuEventHandle, u32) -> CuResult;
type CuEventDestroy = unsafe extern "C" fn(CuEventHandle) -> CuResult;
type CuEventRecord = unsafe extern "C" fn(CuEventHandle, CuStreamHandle) -> CuResult;
type CuEventQuery = unsafe extern "C" fn(CuEventHandle) -> CuResult;
type CuEventSynchronize = unsafe extern "C" fn(CuEventHandle) -> CuResult;
type CuModuleLoadDataEx = unsafe extern "C" fn(*mut CuModule, *const c_void, u32, *mut i32, *mut *mut c_void) -> CuResult;
type CuModuleGetFunction = unsafe extern "C" fn(*mut CuFunction, CuModule, *const i8) -> CuResult;
type CuModuleUnload = unsafe extern "C" fn(CuModule) -> CuResult;
type CuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CuStreamHandle,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

/// A discovered NVIDIA Driver API installation.
pub struct CudaDiscovery {
    library: Arc<Library>,
    devices: Vec<DeviceFingerprint>,
}

impl std::fmt::Debug for CudaDiscovery {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("CudaDiscovery").field("devices", &self.devices).finish()
    }
}

/// A retained CUDA primary context.
pub struct CudaContext {
    library: Arc<Library>,
    raw: CuContext,
    device: CuDevice,
    id: DeviceId,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl std::fmt::Debug for CudaContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("CudaContext").field("device", &self.id).finish()
    }
}

impl CudaContext {
    /// Returns the backend-neutral device identity.
    pub fn device(&self) -> DeviceId {
        self.id
    }

    fn activate(&self) -> Result<(), HalError> {
        activate(&self.library, self.raw)
    }

    fn allocate(self: &Arc<Self>, bytes: usize) -> Result<CudaAllocation, HalError> {
        if bytes == 0 {
            return Err(error("allocate", "zero-byte CUDA allocations are unsupported"));
        }
        self.activate()?;
        let allocate: Symbol<CuMemAlloc> = symbol(&self.library, b"cuMemAlloc_v2\0", "resolve_cuMemAlloc")?;
        let mut pointer = 0;
        unsafe { check("cuMemAlloc", allocate(&mut pointer, bytes))? };
        Ok(CudaAllocation { context: self.clone(), pointer, bytes })
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if let Ok(release) = symbol::<CuDevicePrimaryCtxRelease>(
            &self.library,
            b"cuDevicePrimaryCtxRelease\0",
            "resolve_cuDevicePrimaryCtxRelease",
        ) {
            unsafe {
                let _ = release(self.device);
            }
        }
    }
}

/// A driver-owned device allocation.
#[derive(Debug)]
pub struct CudaAllocation {
    context: Arc<CudaContext>,
    pointer: CuDevicePtr,
    bytes: usize,
}

unsafe impl Send for CudaAllocation {}
unsafe impl Sync for CudaAllocation {}

impl CudaAllocation {
    fn raw(&self) -> CuDevicePtr {
        self.pointer
    }

    /// Returns the allocation size in bytes.
    pub fn byte_len(&self) -> usize {
        self.bytes
    }

    /// Returns the owning device identity.
    pub fn device(&self) -> DeviceId {
        self.context.device()
    }
}

impl Drop for CudaAllocation {
    fn drop(&mut self) {
        if self.context.activate().is_ok() {
            if let Ok(free) = symbol::<CuMemFree>(&self.context.library, b"cuMemFree_v2\0", "resolve_cuMemFree") {
                unsafe {
                    let _ = free(self.pointer);
                }
            }
        }
    }
}

impl CudaDiscovery {
    /// Dynamically loads the Driver API and enumerates CUDA devices.
    pub fn open() -> Result<Self, HalError> {
        #[cfg(windows)]
        let library = unsafe { Library::new("nvcuda.dll") };
        #[cfg(not(windows))]
        let library = unsafe { Library::new("libcuda.so.1") };
        let library = Arc::new(library.map_err(|load_error| error("load_driver", load_error.to_string()))?);

        unsafe {
            let init: Symbol<CuInit> = symbol(&library, b"cuInit\0", "resolve_cuInit")?;
            check("cuInit", init(0))?;
            let count_fn: Symbol<CuDeviceGetCount> = symbol(&library, b"cuDeviceGetCount\0", "resolve_cuDeviceGetCount")?;
            let mut count = 0;
            check("cuDeviceGetCount", count_fn(&mut count))?;
            let version_fn: Symbol<CuDriverGetVersion> =
                symbol(&library, b"cuDriverGetVersion\0", "resolve_cuDriverGetVersion")?;
            let mut version = 0;
            check("cuDriverGetVersion", version_fn(&mut version))?;
            let get_device: Symbol<CuDeviceGet> = symbol(&library, b"cuDeviceGet\0", "resolve_cuDeviceGet")?;
            let get_name: Symbol<CuDeviceGetName> = symbol(&library, b"cuDeviceGetName\0", "resolve_cuDeviceGetName")?;
            let capability: Symbol<CuDeviceComputeCapability> =
                symbol(&library, b"cuDeviceComputeCapability\0", "resolve_cuDeviceComputeCapability")?;
            let mut devices = Vec::with_capacity(count.max(0) as usize);
            for ordinal in 0..count {
                let mut device = 0;
                check("cuDeviceGet", get_device(&mut device, ordinal))?;
                let mut name = [0_i8; 256];
                check("cuDeviceGetName", get_name(name.as_mut_ptr(), name.len() as i32, device))?;
                let name = std::ffi::CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
                let mut major = 0;
                let mut minor = 0;
                check("cuDeviceComputeCapability", capability(&mut major, &mut minor, device))?;
                devices.push(DeviceFingerprint {
                    device: DeviceId { backend: BackendId::Cuda, ordinal: ordinal as u32 },
                    model: name,
                    driver: format!("{}.{}", version / 1000, (version % 1000) / 10),
                    capability_revision: format!("sm_{major}{minor}"),
                });
            }
            Ok(Self { library, devices })
        }
    }

    /// Returns driver-discovered CUDA devices.
    pub fn devices(&self) -> &[DeviceFingerprint] {
        &self.devices
    }

    /// Retains a primary context for a discovered CUDA device.
    pub fn open_primary_context(&self, ordinal: u32) -> Result<Arc<CudaContext>, HalError> {
        let fingerprint =
            self.devices.get(ordinal as usize).ok_or_else(|| error("open_context", "CUDA device ordinal is unavailable"))?;
        let get_device: Symbol<CuDeviceGet> = symbol(&self.library, b"cuDeviceGet\0", "resolve_cuDeviceGet")?;
        let retain: Symbol<CuDevicePrimaryCtxRetain> =
            symbol(&self.library, b"cuDevicePrimaryCtxRetain\0", "resolve_cuDevicePrimaryCtxRetain")?;
        let mut device = 0;
        let mut context = std::ptr::null_mut();
        unsafe {
            check("cuDeviceGet", get_device(&mut device, ordinal as i32))?;
            check("cuDevicePrimaryCtxRetain", retain(&mut context, device))?;
        }
        Ok(Arc::new(CudaContext { library: self.library.clone(), raw: context, device, id: fingerprint.device }))
    }
}

/// CUDA backend registered through the common Titan HAL contract.
#[derive(Clone, Debug)]
pub struct CudaDriver {
    discovery: Arc<CudaDiscovery>,
}

impl CudaDriver {
    /// Opens the driver-only CUDA backend.
    pub fn open() -> Result<Self, HalError> {
        Ok(Self { discovery: Arc::new(CudaDiscovery::open()?) })
    }
}

/// The fixed ABI supported by the CUDA f32 elementwise-add lowering.
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

/// Fixed ABI for a contiguous F32 broadcast add over up to four aligned dimensions.
pub fn broadcast_add_f32_abi() -> KernelAbi {
    let mut args = vec![
        AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
        AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
        AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
        AbiArg::Scalar { dtype: titan_types::DType::I32 },
    ];
    args.extend((0..12).map(|_| AbiArg::Scalar { dtype: titan_types::DType::I32 }));
    KernelAbi {
        version: 1,
        args,
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Fixed ABI for a contiguous F32 unary activation with one input, one output, and an element count.
pub fn silu_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
        ],
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Fixed ABI for a contiguous F32 GELU activation with one input, one output, and an element count.
pub fn gelu_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
        ],
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Fixed ABI for contiguous F32 QuickGELU with a runtime-provided slope.
pub fn quick_gelu_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::F32 },
        ],
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Fixed ABI for contiguous F32 softmax over the final logical axis.
pub fn softmax_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
        ],
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Fixed ABI for contiguous F32 sum reduction over the final logical axis.
pub fn reduction_sum_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
        ],
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Fixed ABI for concatenating two contiguous F32 rank-2 tensors on axis zero.
pub fn concat_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
        ],
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Fixed ABI for a contiguous rank-2 F32 matrix transpose: input, output, rows, cols.
pub fn transpose_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
        ],
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Fixed ABI for a rank-1 contiguous F32 slice.
pub fn slice_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
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

/// Fixed ABI for contiguous NCHW F32 nearest-neighbor resize.
pub fn resize_nearest2d_f32_abi() -> KernelAbi {
    let mut args = vec![
        AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
        AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
    ];
    args.extend((0..6).map(|_| AbiArg::Scalar { dtype: titan_types::DType::I32 }));
    KernelAbi {
        version: 1,
        args,
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Static contract required by the CUDA F32 last-axis softmax lowering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SoftmaxF32Descriptor {
    /// Input shape in logical row-major order.
    pub input_shape: [u32; 4],
    /// Output shape in logical row-major order.
    pub output_shape: [u32; 4],
    /// Number of logical dimensions in the input and output.
    pub rank: u8,
    /// Axis reduced by softmax; only the final logical axis is supported.
    pub axis: i8,
    /// Input element type.
    pub input_dtype: titan_types::DType,
    /// Output element type.
    pub output_dtype: titan_types::DType,
    /// Whether the input uses row-major contiguous storage.
    pub input_contiguous: bool,
    /// Whether the output uses row-major contiguous storage.
    pub output_contiguous: bool,
}

/// Fixed ABI for contiguous F32 last-axis LayerNorm with optional affine vectors.
pub fn layer_norm_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::F32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
        ],
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Fixed ABI for contiguous NCHW F32 GroupNorm with optional channel affine vectors.
pub fn group_norm_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::F32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
        ],
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Static contract required by the CUDA F32 contiguous NCHW GroupNorm lowering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupNormF32Descriptor {
    /// Logical NCHW input dimensions.
    pub input_shape: [u32; 4],
    /// Logical NCHW output dimensions.
    pub output_shape: [u32; 4],
    /// Number of channel groups.
    pub groups: u32,
    /// Input element type.
    pub input_dtype: titan_types::DType,
    /// Output element type.
    pub output_dtype: titan_types::DType,
    /// Optional gamma element type.
    pub gamma_dtype: Option<titan_types::DType>,
    /// Optional beta element type.
    pub beta_dtype: Option<titan_types::DType>,
    /// Whether input storage is contiguous NCHW.
    pub input_contiguous: bool,
    /// Whether output storage is contiguous NCHW.
    pub output_contiguous: bool,
    /// Epsilon added to the variance.
    pub epsilon: f32,
}

impl GroupNormF32Descriptor {
    /// Validates the contiguous NCHW GroupNorm contract before lowering or launch.
    pub fn validate(&self) -> Result<(), KernelError> {
        let [n, channels, height, width] = self.input_shape;
        if n == 0 || channels == 0 || height == 0 || width == 0 || self.output_shape != self.input_shape {
            return Err(KernelError::InvalidAbi(
                "CUDA GroupNorm requires matching non-zero NCHW input and output shapes".into(),
            ));
        }
        if self.groups == 0 || channels % self.groups != 0 {
            return Err(KernelError::InvalidAbi("CUDA GroupNorm groups must be non-zero and divide channels".into()));
        }
        if self.input_dtype != titan_types::DType::F32
            || self.output_dtype != titan_types::DType::F32
            || self.gamma_dtype.is_some_and(|dtype| dtype != titan_types::DType::F32)
            || self.beta_dtype.is_some_and(|dtype| dtype != titan_types::DType::F32)
        {
            return Err(KernelError::InvalidAbi("CUDA GroupNorm requires F32 input, output, gamma, and beta".into()));
        }
        if !self.input_contiguous || !self.output_contiguous {
            return Err(KernelError::InvalidAbi("CUDA GroupNorm requires contiguous NCHW input and output".into()));
        }
        if !self.epsilon.is_finite() || self.epsilon < 0.0 {
            return Err(KernelError::InvalidAbi("CUDA GroupNorm epsilon must be finite and non-negative".into()));
        }
        Ok(())
    }
}

/// Static contract required by the CUDA F32 last-axis LayerNorm lowering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerNormF32Descriptor {
    /// Number of logical rows.
    pub rows: u32,
    /// Number of features in the reduced last axis.
    pub cols: u32,
    /// Input element type.
    pub input_dtype: titan_types::DType,
    /// Output element type.
    pub output_dtype: titan_types::DType,
    /// Optional gamma element type.
    pub gamma_dtype: Option<titan_types::DType>,
    /// Optional beta element type.
    pub beta_dtype: Option<titan_types::DType>,
    /// Whether input storage is contiguous.
    pub input_contiguous: bool,
    /// Whether output storage is contiguous.
    pub output_contiguous: bool,
    /// Epsilon added to the variance.
    pub epsilon: f32,
}

impl LayerNormF32Descriptor {
    /// Validates the last-axis contiguous LayerNorm contract before lowering or launch.
    pub fn validate(&self) -> Result<(), KernelError> {
        if self.rows == 0 || self.cols == 0 {
            return Err(KernelError::InvalidAbi("CUDA LayerNorm rows and cols must be non-zero".into()));
        }
        if self.input_dtype != titan_types::DType::F32
            || self.output_dtype != titan_types::DType::F32
            || self.gamma_dtype.is_some_and(|dtype| dtype != titan_types::DType::F32)
            || self.beta_dtype.is_some_and(|dtype| dtype != titan_types::DType::F32)
        {
            return Err(KernelError::InvalidAbi("CUDA LayerNorm requires F32 input, output, gamma, and beta".into()));
        }
        if !self.input_contiguous || !self.output_contiguous {
            return Err(KernelError::InvalidAbi("CUDA LayerNorm requires contiguous input and output".into()));
        }
        if !self.epsilon.is_finite() || self.epsilon < 0.0 {
            return Err(KernelError::InvalidAbi("CUDA LayerNorm epsilon must be finite and non-negative".into()));
        }
        Ok(())
    }
}

/// Static contract required by the CUDA F32 broadcast-add lowering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BroadcastAddF32Descriptor {
    /// Left input shape padded on the leading side to four dimensions.
    pub lhs_shape: [u32; 4],
    /// Right input shape padded on the leading side to four dimensions.
    pub rhs_shape: [u32; 4],
    /// Output shape padded on the leading side to four dimensions.
    pub output_shape: [u32; 4],
    /// Number of logical dimensions before leading-one padding.
    pub rank: u8,
    /// Left input element type.
    pub lhs_dtype: titan_types::DType,
    /// Right input element type.
    pub rhs_dtype: titan_types::DType,
    /// Output element type.
    pub output_dtype: titan_types::DType,
    /// Whether the left input uses row-major contiguous storage.
    pub lhs_contiguous: bool,
    /// Whether the right input uses row-major contiguous storage.
    pub rhs_contiguous: bool,
    /// Whether the output uses row-major contiguous storage.
    pub output_contiguous: bool,
}

impl BroadcastAddF32Descriptor {
    /// Validates F32 contiguous equal-rank broadcasting before PTX lowering or launch.
    pub fn validate(&self) -> Result<(), KernelError> {
        if !(1..=4).contains(&self.rank) {
            return Err(KernelError::InvalidAbi("CUDA broadcast add supports ranks one through four".into()));
        }
        if self.lhs_dtype != titan_types::DType::F32
            || self.rhs_dtype != titan_types::DType::F32
            || self.output_dtype != titan_types::DType::F32
        {
            return Err(KernelError::InvalidAbi("CUDA broadcast add requires F32 inputs and output".into()));
        }
        if !self.lhs_contiguous || !self.rhs_contiguous || !self.output_contiguous {
            return Err(KernelError::InvalidAbi("CUDA broadcast add requires contiguous inputs and output".into()));
        }
        for ((lhs, rhs), output) in self.lhs_shape.into_iter().zip(self.rhs_shape).zip(self.output_shape) {
            if lhs == 0 || rhs == 0 || output == 0 {
                return Err(KernelError::InvalidAbi("CUDA broadcast add dimensions must be non-zero".into()));
            }
            let expected = if lhs == rhs {
                lhs
            }
            else if lhs == 1 {
                rhs
            }
            else if rhs == 1 {
                lhs
            }
            else {
                return Err(KernelError::InvalidAbi("CUDA broadcast add dimensions must match or equal one".into()));
            };
            if output != expected {
                return Err(KernelError::InvalidAbi("CUDA broadcast add output shape mismatch".into()));
            }
        }
        Ok(())
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

/// Fixed ABI for NCHW f32 Conv2D with OIHW weights and an optional bias.
pub fn conv2d_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
        ],
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Fixed ABI for F32 BHTD scaled dot-product attention without masks or causal mode.
pub fn scaled_dot_product_attention_f32_abi() -> KernelAbi {
    KernelAbi {
        version: 1,
        args: vec![
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: false, alignment: 4 },
            AbiArg::Buffer { dtype: titan_types::DType::F32, writable: true, alignment: 4 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
            AbiArg::Scalar { dtype: titan_types::DType::I32 },
        ],
        launch: titan_kernel::LaunchConfig { block_size: 128, vector_width: 1, pipeline_depth: 1 },
        workspace_bytes: 0,
    }
}

/// Static contract required by CUDA F32 BHTD scaled dot-product attention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScaledDotProductAttentionF32Descriptor {
    pub query_shape: [u32; 4],
    pub key_shape: [u32; 4],
    pub value_shape: [u32; 4],
    pub output_shape: [u32; 4],
    pub query_dtype: titan_types::DType,
    pub key_dtype: titan_types::DType,
    pub value_dtype: titan_types::DType,
    pub output_dtype: titan_types::DType,
    pub query_contiguous: bool,
    pub key_contiguous: bool,
    pub value_contiguous: bool,
    pub output_contiguous: bool,
    pub has_mask: bool,
    pub causal: bool,
}

impl ScaledDotProductAttentionF32Descriptor {
    /// Validates the fixed F32 contiguous BHTD attention contract before lowering or launch.
    pub fn validate(&self) -> Result<(), KernelError> {
        if self.has_mask || self.causal {
            return Err(KernelError::Unsupported(
                "CUDA scaled dot-product attention does not implement masks or causal mode".into(),
            ));
        }
        let [batch, heads, query_tokens, depth] = self.query_shape;
        let [key_batch, key_heads, key_tokens, key_depth] = self.key_shape;
        let [value_batch, value_heads, value_tokens, value_depth] = self.value_shape;
        if batch == 0 || heads == 0 || query_tokens == 0 || key_tokens == 0 || depth == 0 {
            return Err(KernelError::InvalidAbi("CUDA attention B, H, Tq, Tk, and D must be non-zero".into()));
        }
        if self.query_dtype != titan_types::DType::F32
            || self.key_dtype != titan_types::DType::F32
            || self.value_dtype != titan_types::DType::F32
            || self.output_dtype != titan_types::DType::F32
        {
            return Err(KernelError::InvalidAbi("CUDA attention requires F32 Q, K, V, and output".into()));
        }
        if !self.query_contiguous || !self.key_contiguous || !self.value_contiguous || !self.output_contiguous {
            return Err(KernelError::InvalidAbi("CUDA attention requires contiguous BHTD Q, K, V, and output".into()));
        }
        if [key_batch, key_heads, key_depth] != [batch, heads, depth]
            || [value_batch, value_heads, value_tokens, value_depth] != [batch, heads, key_tokens, depth]
            || self.output_shape != [batch, heads, query_tokens, depth]
        {
            return Err(KernelError::InvalidAbi(
                "CUDA attention requires Q[B,H,Tq,D], K/V[B,H,Tk,D], and output[B,H,Tq,D]".into(),
            ));
        }
        Ok(())
    }
}

/// Static contract required by the NCHW CUDA F32 Conv2D lowering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv2dF32Descriptor {
    pub input_shape: [u32; 4],
    pub weight_shape: [u32; 4],
    pub bias_shape: Option<[u32; 1]>,
    pub output_shape: [u32; 4],
    pub input_dtype: titan_types::DType,
    pub weight_dtype: titan_types::DType,
    pub bias_dtype: Option<titan_types::DType>,
    pub output_dtype: titan_types::DType,
    pub input_contiguous: bool,
    pub weight_contiguous: bool,
    pub bias_contiguous: Option<bool>,
    pub output_contiguous: bool,
    pub stride_h: u32,
    pub stride_w: u32,
    pub pad_h: u32,
    pub pad_w: u32,
    pub dilation_h: u32,
    pub dilation_w: u32,
    pub groups: u32,
}

impl Conv2dF32Descriptor {
    /// Validates the supported contiguous F32 NCHW/OIHW Conv2D contract.
    pub fn validate(&self) -> Result<(), KernelError> {
        let [batch, channels, input_h, input_w] = self.input_shape;
        let [output_channels, weight_channels, kernel_h, kernel_w] = self.weight_shape;
        if batch == 0
            || channels == 0
            || input_h == 0
            || input_w == 0
            || output_channels == 0
            || weight_channels == 0
            || kernel_h == 0
            || kernel_w == 0
        {
            return Err(KernelError::InvalidAbi("CUDA Conv2D dimensions must be non-zero".into()));
        }
        if self.stride_h == 0 || self.stride_w == 0 || self.dilation_h == 0 || self.dilation_w == 0 || self.groups == 0 {
            return Err(KernelError::InvalidAbi("CUDA Conv2D stride, dilation, and groups must be non-zero".into()));
        }
        if self.input_dtype != titan_types::DType::F32
            || self.weight_dtype != titan_types::DType::F32
            || self.bias_dtype.is_some_and(|dtype| dtype != titan_types::DType::F32)
            || self.output_dtype != titan_types::DType::F32
        {
            return Err(KernelError::InvalidAbi("CUDA Conv2D requires F32 input, weight, optional bias, and output".into()));
        }
        if !self.input_contiguous
            || !self.weight_contiguous
            || !self.output_contiguous
            || self.bias_contiguous.is_some_and(|contiguous| !contiguous)
        {
            return Err(KernelError::InvalidAbi(
                "CUDA Conv2D requires contiguous NCHW input, OIHW weight, optional bias, and output".into(),
            ));
        }
        if channels % self.groups != 0 || output_channels % self.groups != 0 || weight_channels != channels / self.groups {
            return Err(KernelError::InvalidAbi(
                "CUDA Conv2D groups must divide input/output channels and weight must be OIHW".into(),
            ));
        }
        if self.bias_shape.is_some_and(|shape| shape != [output_channels]) {
            return Err(KernelError::InvalidAbi("CUDA Conv2D bias must have shape [output_channels]".into()));
        }
        let effective_h = (kernel_h - 1)
            .checked_mul(self.dilation_h)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| KernelError::InvalidAbi("CUDA Conv2D kernel height overflows".into()))?;
        let effective_w = (kernel_w - 1)
            .checked_mul(self.dilation_w)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| KernelError::InvalidAbi("CUDA Conv2D kernel width overflows".into()))?;
        let padded_h = input_h
            .checked_add(
                self.pad_h
                    .checked_mul(2)
                    .ok_or_else(|| KernelError::InvalidAbi("CUDA Conv2D padding height overflows".into()))?,
            )
            .ok_or_else(|| KernelError::InvalidAbi("CUDA Conv2D padded height overflows".into()))?;
        let padded_w = input_w
            .checked_add(
                self.pad_w
                    .checked_mul(2)
                    .ok_or_else(|| KernelError::InvalidAbi("CUDA Conv2D padding width overflows".into()))?,
            )
            .ok_or_else(|| KernelError::InvalidAbi("CUDA Conv2D padded width overflows".into()))?;
        if padded_h < effective_h || padded_w < effective_w {
            return Err(KernelError::InvalidAbi("CUDA Conv2D kernel geometry exceeds padded input".into()));
        }
        let output_h = (padded_h - effective_h) / self.stride_h + 1;
        let output_w = (padded_w - effective_w) / self.stride_w + 1;
        if self.output_shape != [batch, output_channels, output_h, output_w] {
            return Err(KernelError::InvalidAbi("CUDA Conv2D output must match NCHW kernel geometry".into()));
        }
        Ok(())
    }
}

/// Static contract required by the row-major CUDA GEMM lowering.
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
    /// Validates the fixed F32 contiguous row-major GEMM contract before lowering or launch.
    pub fn validate(&self) -> Result<(), KernelError> {
        if self.transpose_lhs || self.transpose_rhs {
            return Err(KernelError::Unsupported("CUDA GEMM transpose attributes are not implemented".into()));
        }
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(KernelError::InvalidAbi("CUDA GEMM M, N, and K must be non-zero".into()));
        }
        if self.lhs_dtype != titan_types::DType::F32
            || self.rhs_dtype != titan_types::DType::F32
            || self.output_dtype != titan_types::DType::F32
        {
            return Err(KernelError::InvalidAbi("CUDA GEMM requires F32 A, B, and C buffers".into()));
        }
        if !self.lhs_contiguous || !self.rhs_contiguous || !self.output_contiguous {
            return Err(KernelError::InvalidAbi("CUDA GEMM requires contiguous row-major A, B, and C buffers".into()));
        }
        if self.lhs_shape != [self.m, self.k] || self.rhs_shape != [self.k, self.n] || self.output_shape != [self.m, self.n] {
            return Err(KernelError::InvalidAbi("CUDA GEMM shapes must be A[M,K], B[K,N], and C[M,N]".into()));
        }
        Ok(())
    }
}

/// CUDA compiler for the supported structured Titan Kernel IR subset.
///
/// This compiler is the only API in the repository that emits CUDA PTX. Its
/// output keeps the existing Driver JIT launch metadata contract beside the
/// opaque, NUL-terminated bytes consumed by `DeviceSession::load`.
#[derive(Debug, Default, Clone, Copy)]
pub struct CudaCompiler;

/// A CUDA artifact and the Driver launch contract required to load it.
#[derive(Clone, Debug)]
pub struct CudaArtifact {
    ptx: Vec<u8>,
    abi_hash: AbiHash,
    metadata: KernelLaunchMetadata,
}

impl CudaArtifact {
    /// Returns the NUL-terminated PTX consumed by the CUDA Driver JIT.
    pub fn ptx(&self) -> &[u8] {
        &self.ptx
    }

    /// Returns the ABI hash checked by `DeviceSession::launch`.
    pub fn abi_hash(&self) -> &AbiHash {
        &self.abi_hash
    }

    /// Returns the retained Driver JIT launch metadata.
    pub fn metadata(&self) -> &KernelLaunchMetadata {
        &self.metadata
    }

    fn into_ptx(self) -> Vec<u8> {
        self.ptx
    }
}

impl CudaCompiler {
    /// Lowers supported Titan Kernel IR into a Driver-loadable PTX artifact.
    pub fn compile_artifact(
        &self,
        ir: &KernelModule,
        abi: &KernelAbi,
        fingerprint: &DeviceFingerprint,
    ) -> Result<CudaArtifact, KernelError> {
        let lowered = ptx::lower(ir, abi, fingerprint)?;
        let entry = KernelId(lowered.entry().to_owned());
        let metadata = abi.launch_metadata(&entry)?;
        Ok(CudaArtifact { ptx: lowered.into_bytes(), abi_hash: abi.abi_hash(), metadata })
    }
}

impl TargetCompiler for CudaCompiler {
    fn target(&self) -> KernelTarget {
        KernelTarget::CudaPtx
    }

    fn compile(&self, ir: &KernelModule, abi: &KernelAbi, fingerprint: &DeviceFingerprint) -> Result<Vec<u8>, KernelError> {
        self.compile_artifact(ir, abi, fingerprint).map(CudaArtifact::into_ptx)
    }
}

#[derive(Debug)]
struct CudaBuffer {
    id: u64,
    allocation: Arc<CudaAllocation>,
}

impl Buffer for CudaBuffer {
    fn device(&self) -> DeviceId {
        self.allocation.device()
    }

    fn byte_len(&self) -> usize {
        self.allocation.byte_len()
    }

    fn identity(&self) -> u64 {
        self.id
    }
}

#[derive(Debug)]
struct CudaStream {
    context: Arc<CudaContext>,
    raw: CuStreamHandle,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl Stream for CudaStream {
    fn device(&self) -> DeviceId {
        self.context.device()
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if self.context.activate().is_ok() {
            if let Ok(destroy) =
                symbol::<CuStreamDestroy>(&self.context.library, b"cuStreamDestroy_v2\0", "resolve_cuStreamDestroy")
            {
                unsafe {
                    let _ = destroy(self.raw);
                }
            }
        }
    }
}

#[derive(Debug)]
struct CudaEvent {
    context: Arc<CudaContext>,
    raw: CuEventHandle,
    staging: Option<Arc<[u8]>>,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl Event for CudaEvent {
    fn device(&self) -> DeviceId {
        self.context.device()
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if self.context.activate().is_ok() {
            if self.staging.is_some() {
                if let Ok(synchronize) =
                    symbol::<CuEventSynchronize>(&self.context.library, b"cuEventSynchronize\0", "resolve_cuEventSynchronize")
                {
                    unsafe {
                        let _ = synchronize(self.raw);
                    }
                }
            }
            if let Ok(destroy) =
                symbol::<CuEventDestroy>(&self.context.library, b"cuEventDestroy_v2\0", "resolve_cuEventDestroy")
            {
                unsafe {
                    let _ = destroy(self.raw);
                }
            }
        }
    }
}

#[derive(Debug)]
struct CudaKernel {
    context: Arc<CudaContext>,
    module: CuModule,
    function: CuFunction,
    abi_hash: AbiHash,
    kernel_id: KernelId,
    metadata: KernelLaunchMetadata,
}

unsafe impl Send for CudaKernel {}
unsafe impl Sync for CudaKernel {}

impl LoadedKernel for CudaKernel {
    fn device(&self) -> DeviceId {
        self.context.device()
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

impl Drop for CudaKernel {
    fn drop(&mut self) {
        if self.context.activate().is_ok() {
            if let Ok(unload) = symbol::<CuModuleUnload>(&self.context.library, b"cuModuleUnload\0", "resolve_cuModuleUnload") {
                unsafe {
                    let _ = unload(self.module);
                }
            }
        }
    }
}

#[derive(Debug)]
struct CudaSession {
    context: Arc<CudaContext>,
    fingerprint: DeviceFingerprint,
    buffers: Mutex<HashMap<u64, Arc<CudaBuffer>>>,
    streams: Mutex<HashMap<usize, Arc<CudaStream>>>,
    events: Mutex<HashMap<usize, Arc<CudaEvent>>>,
}

impl CudaSession {
    fn buffer(&self, buffer: &dyn Buffer) -> Result<Arc<CudaBuffer>, HalError> {
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

    fn stream(&self, stream: &dyn Stream) -> Result<Arc<CudaStream>, HalError> {
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

    fn event(&self, event: &dyn Event) -> Result<Arc<CudaEvent>, HalError> {
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

    fn create_raw_event(&self, stream: Option<&CudaStream>, staging: Option<Arc<[u8]>>) -> Result<Arc<dyn Event>, HalError> {
        self.context.activate()?;
        let create: Symbol<CuEventCreate> = symbol(&self.context.library, b"cuEventCreate\0", "resolve_cuEventCreate")?;
        let mut raw = std::ptr::null_mut();
        unsafe { check("cuEventCreate", create(&mut raw, 0))? };
        if let Some(stream) = stream {
            let record: Symbol<CuEventRecord> = symbol(&self.context.library, b"cuEventRecord\0", "resolve_cuEventRecord")?;
            if let Err(failure) = unsafe { check("cuEventRecord", record(raw, stream.raw)) } {
                if let Ok(destroy) =
                    symbol::<CuEventDestroy>(&self.context.library, b"cuEventDestroy_v2\0", "resolve_cuEventDestroy")
                {
                    unsafe {
                        let _ = destroy(raw);
                    }
                }
                return Err(failure);
            }
        }
        let event = Arc::new(CudaEvent { context: self.context.clone(), raw, staging });
        self.events
            .lock()
            .map_err(|_| error("event", "registry poisoned"))?
            .insert(Arc::as_ptr(&event) as usize, event.clone());
        Ok(event)
    }

    fn record(&self, stream: &CudaStream, staging: Option<Arc<[u8]>>) -> Result<Arc<dyn Event>, HalError> {
        self.create_raw_event(Some(stream), staging)
    }
}

impl DeviceSession for CudaSession {
    fn device(&self) -> DeviceId {
        self.context.device()
    }

    fn fingerprint(&self) -> &DeviceFingerprint {
        &self.fingerprint
    }

    fn allocate(&self, bytes: usize, _alignment: usize) -> Result<Arc<dyn Buffer>, HalError> {
        let buffer = Arc::new(CudaBuffer {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            allocation: Arc::new(self.context.allocate(bytes)?),
        });
        self.buffers.lock().map_err(|_| error("allocate", "registry poisoned"))?.insert(buffer.id, buffer.clone());
        Ok(buffer)
    }

    fn upload(&self, stream: &dyn Stream, dst: &dyn Buffer, src: &[u8]) -> Result<Arc<dyn Event>, HalError> {
        let stream = self.stream(stream)?;
        let destination = self.buffer(dst)?;
        if src.len() > destination.byte_len() {
            return Err(error("upload", "source exceeds allocation"));
        }
        let staging = Arc::<[u8]>::from(src.to_vec());
        self.context.activate()?;
        let copy: Symbol<CuMemcpyHtoDAsync> =
            symbol(&self.context.library, b"cuMemcpyHtoDAsync_v2\0", "resolve_cuMemcpyHtoDAsync")?;
        unsafe {
            check("cuMemcpyHtoDAsync", copy(destination.allocation.raw(), staging.as_ptr().cast(), staging.len(), stream.raw))?
        };
        match self.record(&stream, Some(staging.clone())) {
            Ok(event) => Ok(event),
            Err(failure) => {
                // A failed event-record must not release source staging while
                // the asynchronous H2D transfer may still be consuming it.
                let synchronize: Symbol<CuStreamSynchronize> =
                    symbol(&self.context.library, b"cuStreamSynchronize\0", "resolve_cuStreamSynchronize")?;
                unsafe { check("cuStreamSynchronize", synchronize(stream.raw))? };
                Err(failure)
            }
        }
    }

    fn download(&self, stream: &dyn Stream, src: &dyn Buffer, dst: &mut [u8]) -> Result<Arc<dyn Event>, HalError> {
        let stream = self.stream(stream)?;
        let source = self.buffer(src)?;
        if dst.len() > source.byte_len() {
            return Err(error("download", "destination exceeds allocation"));
        }
        self.context.activate()?;
        let copy: Symbol<CuMemcpyDtoHAsync> =
            symbol(&self.context.library, b"cuMemcpyDtoHAsync_v2\0", "resolve_cuMemcpyDtoHAsync")?;
        unsafe { check("cuMemcpyDtoHAsync", copy(dst.as_mut_ptr().cast(), source.allocation.raw(), dst.len(), stream.raw))? };
        // The HAL's borrowed host destination cannot outlive this call, so D2H
        // must complete before returning even though it runs through the stream.
        let synchronize: Symbol<CuStreamSynchronize> =
            symbol(&self.context.library, b"cuStreamSynchronize\0", "resolve_cuStreamSynchronize")?;
        unsafe { check("cuStreamSynchronize", synchronize(stream.raw))? };
        self.record(&stream, None)
    }

    fn copy(&self, stream: &dyn Stream, dst: &dyn Buffer, src: &dyn Buffer, bytes: usize) -> Result<Arc<dyn Event>, HalError> {
        let stream = self.stream(stream)?;
        let destination = self.buffer(dst)?;
        let source = self.buffer(src)?;
        if bytes > source.byte_len() || bytes > destination.byte_len() {
            return Err(error("copy", "transfer exceeds allocation"));
        }
        self.context.activate()?;
        let copy: Symbol<CuMemcpyDtoDAsync> =
            symbol(&self.context.library, b"cuMemcpyDtoDAsync_v2\0", "resolve_cuMemcpyDtoDAsync")?;
        unsafe { check("cuMemcpyDtoDAsync", copy(destination.allocation.raw(), source.allocation.raw(), bytes, stream.raw))? };
        self.record(&stream, None)
    }

    fn create_stream(&self) -> Result<Arc<dyn Stream>, HalError> {
        self.context.activate()?;
        let create: Symbol<CuStreamCreate> = symbol(&self.context.library, b"cuStreamCreate\0", "resolve_cuStreamCreate")?;
        let mut raw = std::ptr::null_mut();
        unsafe { check("cuStreamCreate", create(&mut raw, 0))? };
        let stream = Arc::new(CudaStream { context: self.context.clone(), raw });
        self.streams
            .lock()
            .map_err(|_| error("stream", "registry poisoned"))?
            .insert(Arc::as_ptr(&stream) as usize, stream.clone());
        Ok(stream)
    }

    fn create_event(&self) -> Result<Arc<dyn Event>, HalError> {
        self.create_raw_event(None, None)
    }

    fn load(
        &self,
        artifact: &[u8],
        abi_hash: &AbiHash,
        metadata: KernelLaunchMetadata,
    ) -> Result<Arc<dyn LoadedKernel>, HalError> {
        let ptx = ptx::PtxArtifact::from_driver_bytes(artifact).map_err(|detail| error("load", detail))?;
        let entry =
            std::ffi::CString::new(metadata.entry.as_str()).map_err(|_| error("load", "kernel entry contains a NUL byte"))?;
        self.context.activate()?;
        let load: Symbol<CuModuleLoadDataEx> =
            symbol(&self.context.library, b"cuModuleLoadDataEx\0", "resolve_cuModuleLoadDataEx")?;
        let get_function: Symbol<CuModuleGetFunction> =
            symbol(&self.context.library, b"cuModuleGetFunction\0", "resolve_cuModuleGetFunction")?;
        let mut module = std::ptr::null_mut();
        let mut function = std::ptr::null_mut();
        let mut info_log = vec![0u8; 8192];
        let mut error_log = vec![0u8; 8192];
        let mut option_keys = [
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        ];
        let mut option_values = [
            info_log.as_mut_ptr().cast::<c_void>(),
            info_log.len() as usize as *mut c_void,
            error_log.as_mut_ptr().cast::<c_void>(),
            error_log.len() as usize as *mut c_void,
        ];
        let decode_log = |buffer: &[u8]| {
            let bytes = &buffer[..buffer.len()];
            let trimmed = bytes.split(|byte| *byte == 0).next().unwrap_or(&[]);
            String::from_utf8_lossy(trimmed).trim().to_string()
        };
        unsafe {
            let status = load(
                &mut module,
                ptx.as_ptr().cast(),
                option_keys.len() as u32,
                option_keys.as_mut_ptr(),
                option_values.as_mut_ptr(),
            );
            if status != CUDA_SUCCESS {
                let info = decode_log(&info_log);
                let error_log = decode_log(&error_log);
                let mut detail = format!("CUDA driver status {status}");
                if !info.is_empty() {
                    detail.push_str(&format!("\nJIT info log:\n{info}"));
                }
                if !error_log.is_empty() {
                    detail.push_str(&format!("\nJIT error log:\n{error_log}"));
                }
                return Err(error("cuModuleLoadDataEx", detail));
            }
            if let Err(failure) = check("cuModuleGetFunction", get_function(&mut function, module, entry.as_ptr())) {
                if let Ok(unload) =
                    symbol::<CuModuleUnload>(&self.context.library, b"cuModuleUnload\0", "resolve_cuModuleUnload")
                {
                    let _ = unload(module);
                }
                return Err(failure);
            }
        }
        Ok(Arc::new(CudaKernel {
            context: self.context.clone(),
            module,
            function,
            abi_hash: abi_hash.clone(),
            kernel_id: KernelId(metadata.entry.clone()),
            metadata,
        }))
    }

    fn launch(
        &self,
        stream: &dyn Stream,
        kernel: &dyn LoadedKernel,
        args: &EncodedLaunchArgs,
        geometry: &LaunchGeometry,
    ) -> Result<Arc<dyn Event>, HalError> {
        let stream = self.stream(stream)?;
        if kernel.device() != self.device() {
            return Err(error("launch", "cross-device kernel"));
        }
        let kernel = kernel.as_any().downcast_ref::<CudaKernel>().ok_or_else(|| error("launch", "foreign loaded kernel"))?;
        if !Arc::ptr_eq(&kernel.context, &self.context) {
            return Err(error("launch", "kernel belongs to another CUDA session"));
        }
        if args.canonical_abi() != kernel.abi_hash.0.as_bytes() {
            return Err(error("launch", "encoded arguments do not match kernel ABI"));
        }

        let mut cursor = 0usize;
        let mut values = Vec::with_capacity(kernel.metadata.arguments.len());
        let mut required_slots = Vec::new();
        for kind in &kernel.metadata.arguments {
            let bytes = match kind {
                LaunchArgKind::Buffer => {
                    let end = cursor.checked_add(4).ok_or_else(|| error("launch", "buffer slot overflow"))?;
                    let slot =
                        args.payload().get(cursor..end).ok_or_else(|| error("launch", "truncated buffer slot")).and_then(
                            |bytes| {
                                bytes.try_into().map(u32::from_le_bytes).map_err(|_| error("launch", "invalid buffer slot"))
                            },
                        )?;
                    cursor = end;
                    required_slots.push(slot);
                    let binding = args
                        .bindings()
                        .get(&slot)
                        .ok_or_else(|| error("launch", format!("missing buffer binding for slot {slot}")))?;
                    self.buffer(binding.buffer.as_ref())?.allocation.raw().to_ne_bytes().to_vec()
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
                    value
                }
                LaunchArgKind::Shape { rank } => {
                    let length = *rank as usize * 8;
                    let end = cursor.checked_add(length).ok_or_else(|| error("launch", "shape overflow"))?;
                    let value = args.payload().get(cursor..end).ok_or_else(|| error("launch", "truncated shape"))?.to_vec();
                    cursor = end;
                    value
                }
            };
            values.push(bytes);
        }
        if cursor != args.payload().len() {
            return Err(error("launch", "trailing ABI bytes"));
        }
        args.validate_for(self.device(), required_slots.iter())?;
        let mut parameters = values.iter_mut().map(|value| value.as_mut_ptr().cast::<c_void>()).collect::<Vec<_>>();
        self.context.activate()?;
        let launch: Symbol<CuLaunchKernel> = symbol(&self.context.library, b"cuLaunchKernel\0", "resolve_cuLaunchKernel")?;
        unsafe {
            check(
                "cuLaunchKernel",
                launch(
                    kernel.function,
                    geometry.grid[0],
                    geometry.grid[1],
                    geometry.grid[2],
                    geometry.block[0],
                    geometry.block[1],
                    geometry.block[2],
                    geometry.shared_bytes,
                    stream.raw,
                    parameters.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )?
        };
        self.record(&stream, None)
    }

    fn poll(&self, event: &dyn Event) -> Result<bool, HalError> {
        let event = self.event(event)?;
        self.context.activate()?;
        let query: Symbol<CuEventQuery> = symbol(&self.context.library, b"cuEventQuery\0", "resolve_cuEventQuery")?;
        match unsafe { query(event.raw) } {
            CUDA_SUCCESS => Ok(true),
            CUDA_ERROR_NOT_READY => Ok(false),
            status => Err(status_error("cuEventQuery", status)),
        }
    }

    fn wait(&self, event: &dyn Event) -> Result<(), HalError> {
        let event = self.event(event)?;
        self.context.activate()?;
        let synchronize: Symbol<CuEventSynchronize> =
            symbol(&self.context.library, b"cuEventSynchronize\0", "resolve_cuEventSynchronize")?;
        unsafe { check("cuEventSynchronize", synchronize(event.raw)) }
    }
}

impl BackendDriver for CudaDriver {
    fn id(&self) -> BackendId {
        BackendId::Cuda
    }

    fn enumerate(&self) -> Result<Vec<DeviceFingerprint>, HalError> {
        Ok(self.discovery.devices().to_vec())
    }

    fn open(&self, device: DeviceId) -> Result<Arc<dyn DeviceSession>, HalError> {
        if device.backend != BackendId::Cuda {
            return Err(error("open", "non-CUDA device passed to CUDA driver"));
        }
        let fingerprint = self
            .discovery
            .devices()
            .get(device.ordinal as usize)
            .cloned()
            .ok_or_else(|| error("open", "CUDA device ordinal is unavailable"))?;
        Ok(Arc::new(CudaSession {
            context: self.discovery.open_primary_context(device.ordinal)?,
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

fn activate(library: &Library, context: CuContext) -> Result<(), HalError> {
    let set_current: Symbol<CuCtxSetCurrent> = symbol(library, b"cuCtxSetCurrent\0", "resolve_cuCtxSetCurrent")?;
    unsafe { check("cuCtxSetCurrent", set_current(context)) }
}

fn symbol<'library, T>(
    library: &'library Library,
    name: &[u8],
    operation: &'static str,
) -> Result<Symbol<'library, T>, HalError> {
    unsafe { library.get(name) }.map_err(|source| error(operation, source.to_string()))
}

fn check(operation: &'static str, status: CuResult) -> Result<(), HalError> {
    if status == CUDA_SUCCESS { Ok(()) } else { Err(status_error(operation, status)) }
}

fn status_error(operation: &'static str, status: CuResult) -> HalError {
    error(operation, format!("CUDA driver status {status}"))
}

fn error(operation: &'static str, detail: impl Into<String>) -> HalError {
    HalError { operation, detail: detail.into() }
}
