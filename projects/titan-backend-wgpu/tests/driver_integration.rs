use titan_backend_wgpu::{
    GemmF32Descriptor, WgpuCompiler, WgpuDriver, elementwise_add_f32_abi, gemm_f32_abi,
};
use titan_hal::{BackendDriver, BufferBinding, EncodedLaunchArgs};
use titan_kernel::{BasicBlock, BlockId, KernelArg, KernelModule, TargetCompiler};
use titan_types::{DType, KernelId};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32")))
        .collect()
}

fn vector_add_ir(abi: titan_kernel::KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("elementwise.add.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn gemm_ir(abi: titan_kernel::KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("gemm.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cpu_gemm(lhs: &[f32], rhs: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut output = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0;
            for t in 0..k {
                acc += lhs[row * k + t] * rhs[t * n + col];
            }
            output[row * n + col] = acc;
        }
    }
    output
}

#[test]
fn driver_wgsl_add_matches_cpu_reference_element_by_element() {
    let driver = WgpuDriver::open().expect("open WebGPU driver");
    let devices = driver.enumerate().expect("enumerate WebGPU devices");
    let Some(fingerprint) = devices.first() else {
        eprintln!("SKIP: WebGPU reported no adapter");
        return;
    };
    let session = driver.open(fingerprint.device).expect("open WebGPU session");
    let stream = session.create_stream().expect("create WebGPU stream");
    let lhs = vec![1.0, 2.0, 3.0, 4.0];
    let rhs = vec![10.0, 20.0, 30.0, 40.0];
    let expected: Vec<f32> = lhs.iter().zip(&rhs).map(|(left, right)| left + right).collect();
    let bytes = lhs.len() * std::mem::size_of::<f32>();

    let lhs_device = session.allocate(bytes, 4).expect("allocate lhs");
    let rhs_device = session.allocate(bytes, 4).expect("allocate rhs");
    let out_device = session.allocate(bytes, 4).expect("allocate out");
    session
        .wait(session.upload(stream.as_ref(), lhs_device.as_ref(), &f32_bytes(&lhs)).expect("upload lhs").as_ref())
        .expect("wait upload lhs");
    session
        .wait(session.upload(stream.as_ref(), rhs_device.as_ref(), &f32_bytes(&rhs)).expect("upload rhs").as_ref())
        .expect("wait upload rhs");

    let abi = elementwise_add_f32_abi();
    let compiler = WgpuCompiler;
    assert_eq!(compiler.target(), titan_kernel::KernelTarget::WgpuWgsl);
    let artifact = compiler
        .compile_artifact(&vector_add_ir(abi.clone()), &abi, session.fingerprint())
        .expect("lower structured add IR into WGSL");
    assert_eq!(artifact.wgsl().last(), Some(&0));
    let kernel = session
        .load(artifact.wgsl(), artifact.abi_hash(), artifact.metadata().clone())
        .expect("load WGSL artifact and retain metadata contract");
    let mut payload = Vec::new();
    payload.extend_from_slice(&11_u32.to_le_bytes());
    payload.extend_from_slice(&12_u32.to_le_bytes());
    payload.extend_from_slice(&13_u32.to_le_bytes());
    payload.extend_from_slice(&4_u32.to_le_bytes());
    payload.extend_from_slice(&(lhs.len() as u32).to_le_bytes());
    let args = EncodedLaunchArgs::try_new(
        payload,
        abi.canonical_bytes(),
        [
            BufferBinding::new(11, lhs_device.clone(), session.device()),
            BufferBinding::new(12, rhs_device.clone(), session.device()),
            BufferBinding::new(13, out_device.clone(), session.device()),
        ],
    )
    .expect("construct encoded launch bindings");
    let event = session
        .launch(
            stream.as_ref(),
            kernel.as_ref(),
            &args,
            &titan_hal::LaunchGeometry { grid: [lhs.len().div_ceil(128) as u32, 1, 1], block: [128, 1, 1], shared_bytes: 0 },
        )
        .expect("launch WGSL with supplied stream");
    session.wait(event.as_ref()).expect("wait WebGPU event");

    let mut output = vec![0_u8; bytes];
    let downloaded = session.download(stream.as_ref(), out_device.as_ref(), &mut output).expect("download output");
    session.wait(downloaded.as_ref()).expect("wait download");
    assert_eq!(f32_values(&output), expected, "WebGPU output must match CPU reference element by element");
}

#[test]
fn driver_wgsl_gemm_matches_cpu_reference_element_by_element() {
    let driver = WgpuDriver::open().expect("open WebGPU driver");
    let devices = driver.enumerate().expect("enumerate WebGPU devices");
    let Some(fingerprint) = devices.first() else {
        eprintln!("SKIP: WebGPU reported no adapter");
        return;
    };
    let session = driver.open(fingerprint.device).expect("open WebGPU session");
    let stream = session.create_stream().expect("create WebGPU stream");
    let (m, n, k) = (2usize, 3usize, 4usize);
    GemmF32Descriptor {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        lhs_shape: [m as u32, k as u32],
        rhs_shape: [k as u32, n as u32],
        output_shape: [m as u32, n as u32],
        lhs_dtype: DType::F32,
        rhs_dtype: DType::F32,
        output_dtype: DType::F32,
        lhs_contiguous: true,
        rhs_contiguous: true,
        output_contiguous: true,
        transpose_lhs: false,
        transpose_rhs: false,
    }
    .validate()
    .expect("validate contiguous row-major F32 GEMM descriptor");
    let lhs = vec![1., 2., 3., 4., 5., 6., 7., 8.];
    let rhs = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    let expected = cpu_gemm(&lhs, &rhs, m, n, k);
    let lhs_device = session.allocate(lhs.len() * 4, 4).expect("allocate lhs");
    let rhs_device = session.allocate(rhs.len() * 4, 4).expect("allocate rhs");
    let output_device = session.allocate(expected.len() * 4, 4).expect("allocate output");
    session
        .wait(session.upload(stream.as_ref(), lhs_device.as_ref(), &f32_bytes(&lhs)).expect("upload lhs").as_ref())
        .expect("wait upload lhs");
    session
        .wait(session.upload(stream.as_ref(), rhs_device.as_ref(), &f32_bytes(&rhs)).expect("upload rhs").as_ref())
        .expect("wait upload rhs");
    let abi = gemm_f32_abi();
    let artifact = WgpuCompiler
        .compile_artifact(&gemm_ir(abi.clone()), &abi, session.fingerprint())
        .expect("lower structured GEMM IR into WGSL");
    let kernel = session
        .load(artifact.wgsl(), artifact.abi_hash(), artifact.metadata().clone())
        .expect("load GEMM WGSL");
    let args = abi
        .encode(&[
            KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: false, alignment: 4, buffer: lhs_device },
            KernelArg::Buffer { slot: 2, dtype: DType::F32, writable: false, alignment: 4, buffer: rhs_device },
            KernelArg::Buffer { slot: 3, dtype: DType::F32, writable: true, alignment: 4, buffer: output_device.clone() },
            KernelArg::Scalar { dtype: DType::I32, bytes: (m as i32).to_le_bytes().to_vec() },
            KernelArg::Scalar { dtype: DType::I32, bytes: (n as i32).to_le_bytes().to_vec() },
            KernelArg::Scalar { dtype: DType::I32, bytes: (k as i32).to_le_bytes().to_vec() },
        ])
        .expect("encode F32 contiguous GEMM bindings for one WebGPU session");
    let event = session
        .launch(
            stream.as_ref(),
            kernel.as_ref(),
            &args,
            &titan_hal::LaunchGeometry { grid: [(m * n).div_ceil(128) as u32, 1, 1], block: [128, 1, 1], shared_bytes: 0 },
        )
        .expect("launch GEMM on the supplied stream");
    session.wait(event.as_ref()).expect("wait GEMM event");
    let mut output = vec![0_u8; expected.len() * 4];
    let downloaded = session.download(stream.as_ref(), output_device.as_ref(), &mut output).expect("download GEMM output");
    session.wait(downloaded.as_ref()).expect("wait download");
    assert_eq!(f32_values(&output), expected, "WebGPU GEMM output must match CPU reference element by element");
}

#[test]
fn gemm_descriptor_rejects_transpose_dtype_layout_and_shape_contracts() {
    let mut descriptor = GemmF32Descriptor {
        m: 2,
        n: 3,
        k: 4,
        lhs_shape: [2, 4],
        rhs_shape: [4, 3],
        output_shape: [2, 3],
        lhs_dtype: DType::F32,
        rhs_dtype: DType::F32,
        output_dtype: DType::F32,
        lhs_contiguous: true,
        rhs_contiguous: true,
        output_contiguous: true,
        transpose_lhs: false,
        transpose_rhs: false,
    };
    assert!(descriptor.validate().is_ok());
    descriptor.transpose_lhs = true;
    assert!(descriptor.validate().is_err());
}
