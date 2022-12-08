#![warn(missing_docs)]
//! Device-aware tensors and backend-independent reference operators.

use std::{fmt, sync::Arc};
use titan_hal::{Buffer, DeviceSession, Event};
use titan_types::{BackendId, DType, DeviceId, Layout};

/// A registered runtime device.
#[derive(Clone)]
pub struct Device {
    id: DeviceId,
    session: Arc<dyn DeviceSession>,
}

/// Type-erased cloneable tensor reference used by graph/runtime protocols.
#[derive(Clone)]
pub struct TensorHandle {
    device: DeviceId,
    dtype: DType,
    shape: Vec<usize>,
    strides: Vec<i64>,
    storage: Option<Storage>,
}
impl fmt::Debug for TensorHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorHandle")
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .finish()
    }
}
impl PartialEq for TensorHandle {
    fn eq(&self, other: &Self) -> bool {
        self.device == other.device && self.dtype == other.dtype && self.shape == other.shape && self.strides == other.strides
    }
}
impl Eq for TensorHandle {}
impl std::hash::Hash for TensorHandle {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.device.hash(state);
        self.dtype.hash(state);
        self.shape.hash(state);
        self.strides.hash(state);
    }
}
impl TensorHandle {
    /// Returns the runtime device.
    pub fn device(&self) -> DeviceId {
        self.device
    }
    /// Returns the element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    /// Returns a dynamic shape view.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Returns element strides for dynamic runtime validation.
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }
    /// Returns the owning session when this handle retains backend storage.
    pub fn session(&self) -> Option<&Arc<dyn DeviceSession>> {
        match self.storage.as_ref()? {
            Storage::Device { session, .. } => Some(session),
            Storage::Host(_) => None,
        }
    }
    /// Returns the opaque storage buffer retained by this handle.
    pub fn buffer(&self) -> Option<Arc<dyn Buffer>> {
        match self.storage.as_ref()? {
            Storage::Device { buffer, .. } => Some(buffer.clone()),
            Storage::Host(_) => None,
        }
    }
    /// Downloads a dynamic f32 handle for runtime results.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>, TensorError> {
        if self.dtype != DType::F32 {
            return Err(TensorError::DTypeMismatch);
        }
        let count = self.shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d)).ok_or(TensorError::ShapeOverflow)?;
        match self.storage.as_ref().ok_or(TensorError::MissingStorage)? {
            Storage::Host(bytes) => decode_f32(bytes, count),
            Storage::Device { buffer, session } => {
                let mut bytes = vec![0u8; count * 4];
                let stream = session.create_stream().map_err(TensorError::Hal)?;
                let event = session.download(stream.as_ref(), buffer.as_ref(), &mut bytes).map_err(TensorError::Hal)?;
                session.wait(event.as_ref()).map_err(TensorError::Hal)?;
                decode_f32(&bytes, count)
            }
        }
    }

    /// Allocates a dynamic f32 output using an existing backend session.
    pub fn allocate_f32(session: Arc<dyn DeviceSession>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let count = shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d)).ok_or(TensorError::ShapeOverflow)?;
        let buffer = session.allocate(count * 4, 4).map_err(TensorError::Hal)?;
        Ok(Self {
            device: session.device(),
            dtype: DType::F32,
            strides: dynamic_contiguous_strides(&shape),
            shape,
            storage: Some(Storage::Device { buffer, session }),
        })
    }
    /// Allocates and uploads dynamic f32 data through an existing session.
    pub fn from_f32_vec(session: Arc<dyn DeviceSession>, shape: Vec<usize>, values: &[f32]) -> Result<Self, TensorError> {
        let count = shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d)).ok_or(TensorError::ShapeOverflow)?;
        if values.len() != count {
            return Err(TensorError::ElementCount { expected: count, actual: values.len() });
        }
        let handle = Self::allocate_f32(session.clone(), shape)?;
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) };
        let stream = session.create_stream().map_err(TensorError::Hal)?;
        let event = session
            .upload(stream.as_ref(), handle.buffer().ok_or(TensorError::MissingStorage)?.as_ref(), bytes)
            .map_err(TensorError::Hal)?;
        session.wait(event.as_ref()).map_err(TensorError::Hal)?;
        Ok(handle)
    }
}
impl Device {
    /// Creates a device from a backend session.
    pub fn from_session(session: Arc<dyn DeviceSession>) -> Self {
        Self { id: session.device(), session }
    }
    /// Returns its stable identity.
    pub fn id(&self) -> DeviceId {
        self.id
    }
    /// Returns the backend session.
    pub fn session(&self) -> &Arc<dyn DeviceSession> {
        &self.session
    }
}
impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device").field("id", &self.id).finish()
    }
}

/// Supported tensor element type.
pub trait Element: Copy + Default + Send + Sync + 'static {
    const DTYPE: DType;
}
impl Element for f32 {
    const DTYPE: DType = DType::F32;
}

/// Unified tensor type. Backend selection is runtime data, never a type parameter.
#[derive(Clone, Debug)]
pub struct Tensor<T: Element, const R: usize> {
    data: Vec<T>,
    storage: Storage,
    shape: [usize; R],
    strides: [i64; R],
    device: DeviceId,
    layout: Layout,
    pending_write: Option<Arc<dyn Event>>,
}

#[derive(Clone)]
enum Storage {
    Host(Arc<[u8]>),
    Device { buffer: Arc<dyn Buffer>, session: Arc<dyn DeviceSession> },
}

impl fmt::Debug for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Storage::Host(bytes) => f.debug_tuple("Host").field(&bytes.len()).finish(),
            Storage::Device { buffer, .. } => {
                f.debug_tuple("Device").field(&buffer.device()).field(&buffer.byte_len()).finish()
            }
        }
    }
}

impl<T: Element + PartialEq, const R: usize> PartialEq for Tensor<T, R> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape && self.device == other.device && self.layout == other.layout
    }
}

impl<T: Element, const R: usize> Tensor<T, R> {
    /// Produces an opaque runtime handle without exposing native storage.
    pub fn handle(&self) -> TensorHandle {
        TensorHandle {
            device: self.device,
            dtype: T::DTYPE,
            shape: self.shape.to_vec(),
            strides: self.strides.to_vec(),
            storage: Some(self.storage.clone()),
        }
    }
    fn from_host(shape: [usize; R], data: Vec<T>) -> Result<Self, TensorError> {
        let expected = shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d)).ok_or(TensorError::ShapeOverflow)?;
        if data.len() != expected {
            return Err(TensorError::ElementCount { expected, actual: data.len() });
        }
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>()) };
        Ok(Self {
            data,
            storage: Storage::Host(Arc::from(bytes)),
            shape,
            strides: contiguous_strides(&shape),
            device: DeviceId { backend: BackendId::Cpu, ordinal: 0 },
            layout: Layout::Contiguous,
            pending_write: None,
        })
    }
    /// Creates a tensor on an explicit device and uploads its bytes into session-owned storage.
    pub fn from_slice(device: &Device, shape: [usize; R], data: &[T]) -> Result<Self, TensorError> {
        let mut tensor = Self::from_host(shape, data.to_vec())?;
        tensor.device = device.id();
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>()) };
        let buffer = device.session().allocate(bytes.len(), std::mem::align_of::<T>()).map_err(TensorError::Hal)?;
        let stream = device.session().create_stream().map_err(TensorError::Hal)?;
        let event = device.session().upload(stream.as_ref(), buffer.as_ref(), bytes).map_err(TensorError::Hal)?;
        tensor.storage = Storage::Device { buffer, session: device.session().clone() };
        tensor.pending_write = Some(event);
        Ok(tensor)
    }
    /// Allocates a zero-filled CPU reference tensor.
    pub fn zeros(device: &Device, shape: [usize; R]) -> Result<Self, TensorError> {
        Self::from_slice(device, shape, &vec![T::default(); shape.iter().product()])
    }
    /// Returns the device identity.
    pub fn device(&self) -> DeviceId {
        self.device
    }
    /// Returns the fixed-rank shape.
    pub fn shape(&self) -> [usize; R] {
        self.shape
    }
    /// Returns element strides.
    pub fn strides(&self) -> [i64; R] {
        self.strides
    }
    /// Explicitly synchronizes and downloads tensor contents.
    pub fn to_vec(&self) -> Result<Vec<T>, TensorError> {
        if let Some(event) = &self.pending_write {
            if let Storage::Device { session, .. } = &self.storage {
                session.wait(event.as_ref()).map_err(TensorError::Hal)?;
            }
        }
        match &self.storage {
            Storage::Host(_) => Ok(self.data.clone()),
            Storage::Device { buffer, session } => {
                let mut bytes = vec![0u8; self.data.len() * std::mem::size_of::<T>()];
                let stream = session.create_stream().map_err(TensorError::Hal)?;
                let event = session.download(stream.as_ref(), buffer.as_ref(), &mut bytes).map_err(TensorError::Hal)?;
                session.wait(event.as_ref()).map_err(TensorError::Hal)?;
                Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, self.data.len()).to_vec() })
            }
        }
    }
}

fn contiguous_strides<const R: usize>(shape: &[usize; R]) -> [i64; R] {
    let mut out = [0; R];
    let mut step = 1i64;
    for i in (0..R).rev() {
        out[i] = step;
        step = step.saturating_mul(shape[i] as i64);
    }
    out
}

/// Tensor operation failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorError {
    ElementCount { expected: usize, actual: usize },
    ShapeOverflow,
    ShapeMismatch,
    DTypeMismatch,
    MissingStorage,
    UnsupportedDevice(DeviceId),
    Hal(titan_hal::HalError),
}
impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for TensorError {}

/// Convenience alias for the sole currently implemented scalar element type.
pub type F32Tensor<const R: usize> = Tensor<f32, R>;

fn dynamic_contiguous_strides(shape: &[usize]) -> Vec<i64> {
    let mut out = vec![0; shape.len()];
    let mut step = 1i64;
    for i in (0..shape.len()).rev() {
        out[i] = step;
        step = step.saturating_mul(shape[i] as i64);
    }
    out
}

fn decode_f32(bytes: &[u8], count: usize) -> Result<Vec<f32>, TensorError> {
    if bytes.len() != count * 4 {
        return Err(TensorError::ElementCount { expected: count * 4, actual: bytes.len() });
    }
    Ok(bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect())
}
