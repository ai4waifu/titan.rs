use titan_hal::BackendDriver;
use titan_tensor::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = std::sync::Arc::new(titan_backend_cpu::CpuDriver::default())
        .open(titan_types::DeviceId { backend: titan_types::BackendId::Cpu, ordinal: 0 })?;
    let device = Device::from_session(session);
    let tensor = Tensor::from_slice(&device, [2, 2], &[1.0f32, 2.0, 3.0, 4.0])?;
    println!("device={:?} shape={:?} values={:?}", tensor.device(), tensor.shape(), tensor.to_vec()?);
    Ok(())
}
