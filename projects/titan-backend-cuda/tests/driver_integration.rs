use titan_backend_cuda::{
    Conv2dF32Descriptor, CudaCompiler, CudaDriver, GemmF32Descriptor, ScaledDotProductAttentionF32Descriptor, conv2d_f32_abi,
    elementwise_add_f32_abi, gemm_f32_abi, scaled_dot_product_attention_f32_abi, softmax_f32_abi,
};
use titan_hal::{BackendDriver, BufferBinding, EncodedLaunchArgs};
use titan_kernel::{AddressSpace, BasicBlock, BlockId, Instruction, IrType, KernelArg, KernelModule, TargetCompiler, ValueId};
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

fn no_cuda_environment(error: &titan_hal::HalError) -> bool {
    error.operation == "load_driver"
        || ((error.operation == "cuInit" || error.operation == "cuDeviceGetCount") && error.detail.contains("status 100"))
}

fn vector_add_ir(abi: titan_kernel::KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("elementwise.add.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock {
            id: BlockId(0),
            params: vec![],
            instructions: vec![
                (
                    ValueId(0),
                    Instruction::Parameter {
                        index: 0,
                        ty: IrType::Pointer { address_space: AddressSpace::Global, dtype: DType::F32 },
                    },
                ),
                (
                    ValueId(1),
                    Instruction::Parameter {
                        index: 1,
                        ty: IrType::Pointer { address_space: AddressSpace::Global, dtype: DType::F32 },
                    },
                ),
                (
                    ValueId(2),
                    Instruction::Parameter {
                        index: 2,
                        ty: IrType::Pointer { address_space: AddressSpace::Global, dtype: DType::F32 },
                    },
                ),
                (ValueId(3), Instruction::Parameter { index: 3, ty: IrType::I32 }),
                (ValueId(4), Instruction::Load { ptr: ValueId(0), ty: IrType::F32 }),
                (ValueId(5), Instruction::Load { ptr: ValueId(1), ty: IrType::F32 }),
                (ValueId(6), Instruction::Add { lhs: ValueId(4), rhs: ValueId(5) }),
                (ValueId(7), Instruction::Store { ptr: ValueId(2), value: ValueId(6) }),
            ],
        }],
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

fn conv2d_ir(abi: titan_kernel::KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("conv2d.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn scaled_dot_product_attention_ir(abi: titan_kernel::KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("scaled_dot_product_attention.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn softmax_ir(abi: titan_kernel::KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("softmax.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn cpu_gemm(lhs: &[f32], rhs: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut output = vec![0.0; m * n];
    for row in 0..m {
        for column in 0..n {
            for inner in 0..k {
                output[row * n + column] += lhs[row * k + inner] * rhs[inner * n + column];
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn cpu_conv2d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    channels: usize,
    input_h: usize,
    input_w: usize,
    output_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    output_h: usize,
    output_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; batch * output_channels * output_h * output_w];
    let channels_per_group = channels / groups;
    let outputs_per_group = output_channels / groups;
    for n in 0..batch {
        for output_channel in 0..output_channels {
            let group = output_channel / outputs_per_group;
            for output_y in 0..output_h {
                for output_x in 0..output_w {
                    let output_index = ((n * output_channels + output_channel) * output_h + output_y) * output_w + output_x;
                    let mut sum = bias.map_or(0.0, |values| values[output_channel]);
                    for channel in 0..channels_per_group {
                        for kernel_y in 0..kernel_h {
                            for kernel_x in 0..kernel_w {
                                let input_y = output_y * stride_h + kernel_y * dilation_h;
                                let input_x = output_x * stride_w + kernel_x * dilation_w;
                                if input_y < pad_h || input_x < pad_w {
                                    continue;
                                }
                                let input_y = input_y - pad_h;
                                let input_x = input_x - pad_w;
                                if input_y >= input_h || input_x >= input_w {
                                    continue;
                                }
                                let input_channel = group * channels_per_group + channel;
                                let input_index = ((n * channels + input_channel) * input_h + input_y) * input_w + input_x;
                                let weight_index = ((output_channel * channels_per_group + channel) * kernel_h + kernel_y)
                                    * kernel_w
                                    + kernel_x;
                                sum += input[input_index] * weight[weight_index];
                            }
                        }
                    }
                    output[output_index] = sum;
                }
            }
        }
    }
    output
}

fn cpu_scaled_dot_product_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    heads: usize,
    query_tokens: usize,
    key_tokens: usize,
    depth: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; batch * heads * query_tokens * depth];
    let scale = (depth as f32).sqrt();
    for n in 0..batch {
        for head in 0..heads {
            for query_token in 0..query_tokens {
                let query_offset = ((n * heads + head) * query_tokens + query_token) * depth;
                let mut scores = Vec::with_capacity(key_tokens);
                for key_token in 0..key_tokens {
                    let key_offset = ((n * heads + head) * key_tokens + key_token) * depth;
                    let score = (0..depth).map(|lane| query[query_offset + lane] * key[key_offset + lane]).sum::<f32>() / scale;
                    scores.push(score);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator = scores.iter().map(|score| (score - maximum).exp()).sum::<f32>();
                for lane in 0..depth {
                    let mut result = 0.0;
                    for key_token in 0..key_tokens {
                        let value_offset = ((n * heads + head) * key_tokens + key_token) * depth;
                        result += ((scores[key_token] - maximum).exp() / denominator) * value[value_offset + lane];
                    }
                    output[query_offset + lane] = result;
                }
            }
        }
    }
    output
}

fn cpu_last_axis_softmax(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0; rows * cols];
    for row in 0..rows {
        let start = row * cols;
        let row_values = &input[start..start + cols];
        let maximum = row_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator = row_values.iter().map(|value| (value - maximum).exp()).sum::<f32>();
        for column in 0..cols {
            output[start + column] = (input[start + column] - maximum).exp() / denominator;
        }
    }
    output
}

#[test]
fn driver_ptx_vector_add_matches_cpu_reference() {
    let driver = match CudaDriver::open() {
        Ok(driver) => driver,
        Err(error) if no_cuda_environment(&error) => {
            eprintln!("SKIP: CUDA Driver API unavailable: {error}");
            return;
        }
        Err(error) => panic!("opening CUDA Driver API failed: {error}"),
    };
    let devices = driver.enumerate().expect("enumerate CUDA devices");
    let Some(fingerprint) = devices.first()
    else {
        eprintln!("SKIP: CUDA Driver API reported no GPU");
        return;
    };
    let session = driver.open(fingerprint.device).expect("open CUDA primary context");
    let stream = session.create_stream().expect("create CUDA stream");
    drop(session.create_event().expect("create CUDA event"));

    let lhs = vec![1.25_f32, -2.0, 3.5, 0.0, 99.0, -16.25, 8.0, 0.75, 4.0, -7.0, 2.5, 12.0, -1.0];
    let rhs = vec![-0.25_f32, 4.0, 1.5, 8.0, -9.0, 0.25, -3.0, 5.25, -4.0, 7.5, 0.5, -2.0, 11.0];
    let expected = lhs.iter().zip(&rhs).map(|(left, right)| left + right).collect::<Vec<_>>();
    let bytes = lhs.len() * std::mem::size_of::<f32>();

    let lhs_device = session.allocate(bytes, 4).expect("allocate lhs");
    let lhs_copy = session.allocate(bytes, 4).expect("allocate lhs copy");
    let rhs_device = session.allocate(bytes, 4).expect("allocate rhs");
    let out_device = session.allocate(bytes, 4).expect("allocate out");
    session
        .wait(session.upload(stream.as_ref(), lhs_device.as_ref(), &f32_bytes(&lhs)).expect("H2D lhs").as_ref())
        .expect("wait H2D lhs");
    session
        .wait(session.upload(stream.as_ref(), rhs_device.as_ref(), &f32_bytes(&rhs)).expect("H2D rhs").as_ref())
        .expect("wait H2D rhs");
    session
        .wait(session.copy(stream.as_ref(), lhs_copy.as_ref(), lhs_device.as_ref(), bytes).expect("D2D lhs").as_ref())
        .expect("wait D2D lhs");

    let abi = elementwise_add_f32_abi();
    let compiler = CudaCompiler;
    assert_eq!(compiler.target(), titan_kernel::KernelTarget::CudaPtx);
    let artifact = compiler
        .compile_artifact(&vector_add_ir(abi.clone()), &abi, session.fingerprint())
        .expect("lower structured add IR into typed PTX");
    assert_eq!(artifact.ptx().last(), Some(&0));
    let kernel = session
        .load(artifact.ptx(), artifact.abi_hash(), artifact.metadata().clone())
        .expect("Driver JIT loads compiler artifact and retains metadata contract");
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
            BufferBinding::new(11, lhs_copy.clone(), session.device()),
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
        .expect("launch PTX with supplied stream");
    let _complete_before_wait = session.poll(event.as_ref()).expect("query CUDA event");
    session.wait(event.as_ref()).expect("wait CUDA event");

    let mut output = vec![0_u8; bytes];
    let downloaded = session.download(stream.as_ref(), out_device.as_ref(), &mut output).expect("D2H output");
    session.wait(downloaded.as_ref()).expect("wait D2H output");
    assert_eq!(f32_values(&output), expected, "CUDA PTX output must match CPU reference element by element");
}

#[test]
fn driver_ptx_softmax_matches_cpu_stable_last_axis_reference() {
    let driver = match CudaDriver::open() {
        Ok(driver) => driver,
        Err(error) if no_cuda_environment(&error) => {
            eprintln!("SKIP: CUDA Driver API unavailable: {error}");
            return;
        }
        Err(error) => panic!("opening CUDA Driver API failed: {error}"),
    };
    let devices = driver.enumerate().expect("enumerate CUDA devices");
    let Some(fingerprint) = devices.first()
    else {
        eprintln!("SKIP: CUDA Driver API reported no GPU");
        return;
    };
    let session = driver.open(fingerprint.device).expect("open CUDA primary context");
    let stream = session.create_stream().expect("create CUDA stream");
    let rows = 2usize;
    let cols = 3usize;
    let input = vec![1001.0, 1000.0, 999.0, -1000.0, -999.0, -1001.0];
    let expected = cpu_last_axis_softmax(&input, rows, cols);
    let input_device = session.allocate(input.len() * 4, 4).expect("allocate softmax input");
    let output_device = session.allocate(expected.len() * 4, 4).expect("allocate softmax output");
    session
        .wait(session.upload(stream.as_ref(), input_device.as_ref(), &f32_bytes(&input)).expect("H2D softmax input").as_ref())
        .expect("wait H2D softmax input");
    let abi = softmax_f32_abi();
    let artifact = CudaCompiler
        .compile_artifact(&softmax_ir(abi.clone()), &abi, session.fingerprint())
        .expect("lower structured softmax IR into PTX");
    let kernel =
        session.load(artifact.ptx(), artifact.abi_hash(), artifact.metadata().clone()).expect("Driver JIT loads softmax PTX");
    let args = abi
        .encode(&[
            KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: false, alignment: 4, buffer: input_device },
            KernelArg::Buffer { slot: 2, dtype: DType::F32, writable: true, alignment: 4, buffer: output_device.clone() },
            KernelArg::Scalar { dtype: DType::I32, bytes: (rows as i32).to_le_bytes().to_vec() },
            KernelArg::Scalar { dtype: DType::I32, bytes: (cols as i32).to_le_bytes().to_vec() },
        ])
        .expect("encode F32 last-axis softmax bindings for one CUDA session");
    let event = session
        .launch(
            stream.as_ref(),
            kernel.as_ref(),
            &args,
            &titan_hal::LaunchGeometry { grid: [rows.div_ceil(128) as u32, 1, 1], block: [128, 1, 1], shared_bytes: 0 },
        )
        .expect("launch softmax on the supplied stream");
    session.wait(event.as_ref()).expect("wait softmax event");
    let mut output = vec![0_u8; expected.len() * 4];
    let downloaded = session.download(stream.as_ref(), output_device.as_ref(), &mut output).expect("D2H softmax output");
    session.wait(downloaded.as_ref()).expect("wait D2H softmax output");
    let actual = f32_values(&output);
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(error <= 2e-4, "softmax index {index} mismatch: cuda={actual} cpu={expected} error={error}");
    }
    for row in 0..rows {
        let sum: f32 = actual[row * cols..(row + 1) * cols].iter().sum();
        assert!((sum - 1.0).abs() <= 2e-4, "softmax row {row} sum {sum} is not close to 1");
    }
}

#[test]
fn driver_ptx_gemm_matches_cpu_reference_element_by_element() {
    let driver = match CudaDriver::open() {
        Ok(driver) => driver,
        Err(error) if no_cuda_environment(&error) => {
            eprintln!("SKIP: CUDA Driver API unavailable: {error}");
            return;
        }
        Err(error) => panic!("opening CUDA Driver API failed: {error}"),
    };
    let devices = driver.enumerate().expect("enumerate CUDA devices");
    let Some(fingerprint) = devices.first()
    else {
        eprintln!("SKIP: CUDA Driver API reported no GPU");
        return;
    };
    let session = driver.open(fingerprint.device).expect("open CUDA primary context");
    let stream = session.create_stream().expect("create CUDA stream");
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
        .wait(session.upload(stream.as_ref(), lhs_device.as_ref(), &f32_bytes(&lhs)).expect("H2D lhs").as_ref())
        .expect("wait H2D lhs");
    session
        .wait(session.upload(stream.as_ref(), rhs_device.as_ref(), &f32_bytes(&rhs)).expect("H2D rhs").as_ref())
        .expect("wait H2D rhs");
    let abi = gemm_f32_abi();
    let artifact = CudaCompiler
        .compile_artifact(&gemm_ir(abi.clone()), &abi, session.fingerprint())
        .expect("lower structured GEMM IR into PTX");
    let kernel =
        session.load(artifact.ptx(), artifact.abi_hash(), artifact.metadata().clone()).expect("Driver JIT loads GEMM PTX");
    let args = abi
        .encode(&[
            KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: false, alignment: 4, buffer: lhs_device },
            KernelArg::Buffer { slot: 2, dtype: DType::F32, writable: false, alignment: 4, buffer: rhs_device },
            KernelArg::Buffer { slot: 3, dtype: DType::F32, writable: true, alignment: 4, buffer: output_device.clone() },
            KernelArg::Scalar { dtype: DType::I32, bytes: (m as i32).to_le_bytes().to_vec() },
            KernelArg::Scalar { dtype: DType::I32, bytes: (n as i32).to_le_bytes().to_vec() },
            KernelArg::Scalar { dtype: DType::I32, bytes: (k as i32).to_le_bytes().to_vec() },
        ])
        .expect("encode F32 contiguous GEMM bindings for one CUDA session");
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
    let downloaded = session.download(stream.as_ref(), output_device.as_ref(), &mut output).expect("D2H GEMM output");
    session.wait(downloaded.as_ref()).expect("wait D2H GEMM output");
    assert_eq!(f32_values(&output), expected, "CUDA GEMM must match CPU reference for every output element");
}

#[test]
fn driver_ptx_conv2d_matches_cpu_reference_for_grouped_padded_strided_input() {
    let driver = match CudaDriver::open() {
        Ok(driver) => driver,
        Err(error) if no_cuda_environment(&error) => {
            eprintln!("SKIP: CUDA Driver API unavailable: {error}");
            return;
        }
        Err(error) => panic!("opening CUDA Driver API failed: {error}"),
    };
    let devices = driver.enumerate().expect("enumerate CUDA devices");
    let Some(fingerprint) = devices.first()
    else {
        eprintln!("SKIP: CUDA Driver API reported no GPU");
        return;
    };
    let session = driver.open(fingerprint.device).expect("open CUDA primary context");
    let stream = session.create_stream().expect("create CUDA stream");
    let (batch, channels, input_h, input_w) = (1usize, 2usize, 4usize, 4usize);
    let (output_channels, kernel_h, kernel_w) = (2usize, 3usize, 3usize);
    let (stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups) =
        (2usize, 2usize, 1usize, 1usize, 1usize, 1usize, 2usize);
    let (output_h, output_w) = (2usize, 2usize);
    let descriptor = Conv2dF32Descriptor {
        input_shape: [batch as u32, channels as u32, input_h as u32, input_w as u32],
        weight_shape: [output_channels as u32, (channels / groups) as u32, kernel_h as u32, kernel_w as u32],
        bias_shape: Some([output_channels as u32]),
        output_shape: [batch as u32, output_channels as u32, output_h as u32, output_w as u32],
        input_dtype: DType::F32,
        weight_dtype: DType::F32,
        bias_dtype: Some(DType::F32),
        output_dtype: DType::F32,
        input_contiguous: true,
        weight_contiguous: true,
        bias_contiguous: Some(true),
        output_contiguous: true,
        stride_h: stride_h as u32,
        stride_w: stride_w as u32,
        pad_h: pad_h as u32,
        pad_w: pad_w as u32,
        dilation_h: dilation_h as u32,
        dilation_w: dilation_w as u32,
        groups: groups as u32,
    };
    descriptor.validate().expect("validate grouped F32 NCHW/OIHW Conv2D descriptor");
    let input = vec![
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., -1., -2., -3., -4., -5., -6., -7., -8., -9.,
        -10., -11., -12., -13., -14., -15., -16.,
    ];
    let weight = vec![1., 0., -1., 1., 0., -1., 1., 0., -1., 0.5, 0., -0.5, 0.5, 0., -0.5, 0.5, 0., -0.5];
    let bias = vec![0.25, -0.75];
    let expected = cpu_conv2d(
        &input,
        &weight,
        Some(&bias),
        batch,
        channels,
        input_h,
        input_w,
        output_channels,
        kernel_h,
        kernel_w,
        output_h,
        output_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
    );
    let input_device = session.allocate(input.len() * 4, 4).expect("allocate input");
    let weight_device = session.allocate(weight.len() * 4, 4).expect("allocate OIHW weight");
    let bias_device = session.allocate(bias.len() * 4, 4).expect("allocate bias");
    let output_device = session.allocate(expected.len() * 4, 4).expect("allocate NCHW output");
    for (buffer, values, name) in [
        (input_device.as_ref(), f32_bytes(&input), "input"),
        (weight_device.as_ref(), f32_bytes(&weight), "weight"),
        (bias_device.as_ref(), f32_bytes(&bias), "bias"),
    ] {
        let event = session.upload(stream.as_ref(), buffer, &values).unwrap_or_else(|error| panic!("H2D {name}: {error}"));
        session.wait(event.as_ref()).unwrap_or_else(|error| panic!("wait H2D {name}: {error}"));
    }
    let abi = conv2d_f32_abi();
    let artifact = CudaCompiler
        .compile_artifact(&conv2d_ir(abi.clone()), &abi, session.fingerprint())
        .expect("lower structured Conv2D IR into PTX");
    let kernel =
        session.load(artifact.ptx(), artifact.abi_hash(), artifact.metadata().clone()).expect("Driver JIT loads Conv2D PTX");
    let mut arguments = vec![
        KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: false, alignment: 4, buffer: input_device },
        KernelArg::Buffer { slot: 2, dtype: DType::F32, writable: false, alignment: 4, buffer: weight_device },
        KernelArg::Buffer { slot: 3, dtype: DType::F32, writable: false, alignment: 4, buffer: bias_device },
        KernelArg::Buffer { slot: 4, dtype: DType::F32, writable: true, alignment: 4, buffer: output_device.clone() },
    ];
    for value in [
        batch,
        channels,
        input_h,
        input_w,
        output_channels,
        kernel_h,
        kernel_w,
        output_h,
        output_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        1,
    ] {
        arguments.push(KernelArg::Scalar { dtype: DType::I32, bytes: (value as i32).to_le_bytes().to_vec() });
    }
    let args = abi.encode(&arguments).expect("encode F32 Conv2D arguments for one CUDA session");
    let complete = session
        .launch(
            stream.as_ref(),
            kernel.as_ref(),
            &args,
            &titan_hal::LaunchGeometry {
                grid: [expected.len().div_ceil(128) as u32, 1, 1],
                block: [128, 1, 1],
                shared_bytes: 0,
            },
        )
        .expect("launch Conv2D on the supplied stream");
    session.wait(complete.as_ref()).expect("wait Conv2D event");
    let mut output = vec![0_u8; expected.len() * 4];
    let copied = session.download(stream.as_ref(), output_device.as_ref(), &mut output).expect("D2H Conv2D output");
    session.wait(copied.as_ref()).expect("wait D2H Conv2D output");
    assert_eq!(f32_values(&output), expected, "CUDA Conv2D must match the CPU reference for every output element");
}

#[test]
fn driver_ptx_scaled_dot_product_attention_matches_cpu_reference_with_distinct_query_and_key_lengths() {
    let driver = match CudaDriver::open() {
        Ok(driver) => driver,
        Err(error) if no_cuda_environment(&error) => {
            eprintln!("SKIP: CUDA Driver API unavailable: {error}");
            return;
        }
        Err(error) => panic!("opening CUDA Driver API failed: {error}"),
    };
    let devices = driver.enumerate().expect("enumerate CUDA devices");
    let Some(fingerprint) = devices.first()
    else {
        eprintln!("SKIP: CUDA Driver API reported no GPU");
        return;
    };
    let session = driver.open(fingerprint.device).expect("open CUDA primary context");
    let stream = session.create_stream().expect("create CUDA stream");
    let (batch, heads, query_tokens, key_tokens, depth) = (1usize, 1usize, 2usize, 3usize, 2usize);
    let descriptor = ScaledDotProductAttentionF32Descriptor {
        query_shape: [batch as u32, heads as u32, query_tokens as u32, depth as u32],
        key_shape: [batch as u32, heads as u32, key_tokens as u32, depth as u32],
        value_shape: [batch as u32, heads as u32, key_tokens as u32, depth as u32],
        output_shape: [batch as u32, heads as u32, query_tokens as u32, depth as u32],
        query_dtype: DType::F32,
        key_dtype: DType::F32,
        value_dtype: DType::F32,
        output_dtype: DType::F32,
        query_contiguous: true,
        key_contiguous: true,
        value_contiguous: true,
        output_contiguous: true,
        has_mask: false,
        causal: false,
    };
    descriptor.validate().expect("validate BHTD F32 attention descriptor");
    let query = vec![1.0, -0.5, 0.25, 1.5];
    let key = vec![0.5, 1.0, -1.0, 0.25, 0.75, -0.5];
    let value = vec![2.0, -1.0, 0.5, 3.0, -2.0, 1.5];
    let expected = cpu_scaled_dot_product_attention(&query, &key, &value, batch, heads, query_tokens, key_tokens, depth);
    let query_device = session.allocate(query.len() * 4, 4).expect("allocate Q");
    let key_device = session.allocate(key.len() * 4, 4).expect("allocate K");
    let value_device = session.allocate(value.len() * 4, 4).expect("allocate V");
    let output_device = session.allocate(expected.len() * 4, 4).expect("allocate attention output");
    for (buffer, values, name) in [
        (query_device.as_ref(), f32_bytes(&query), "Q"),
        (key_device.as_ref(), f32_bytes(&key), "K"),
        (value_device.as_ref(), f32_bytes(&value), "V"),
    ] {
        let event = session.upload(stream.as_ref(), buffer, &values).unwrap_or_else(|error| panic!("H2D {name}: {error}"));
        session.wait(event.as_ref()).unwrap_or_else(|error| panic!("wait H2D {name}: {error}"));
    }
    let abi = scaled_dot_product_attention_f32_abi();
    let artifact = CudaCompiler
        .compile_artifact(&scaled_dot_product_attention_ir(abi.clone()), &abi, session.fingerprint())
        .expect("lower structured attention IR into PTX");
    let kernel =
        session.load(artifact.ptx(), artifact.abi_hash(), artifact.metadata().clone()).expect("Driver JIT loads attention PTX");
    let mut arguments = vec![
        KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: false, alignment: 4, buffer: query_device },
        KernelArg::Buffer { slot: 2, dtype: DType::F32, writable: false, alignment: 4, buffer: key_device },
        KernelArg::Buffer { slot: 3, dtype: DType::F32, writable: false, alignment: 4, buffer: value_device },
        KernelArg::Buffer { slot: 4, dtype: DType::F32, writable: true, alignment: 4, buffer: output_device.clone() },
    ];
    for scalar in [batch, heads, query_tokens, key_tokens, depth] {
        arguments.push(KernelArg::Scalar { dtype: DType::I32, bytes: (scalar as i32).to_le_bytes().to_vec() });
    }
    let args = abi.encode(&arguments).expect("encode BHTD F32 attention arguments for one CUDA session");
    let completion = session
        .launch(
            stream.as_ref(),
            kernel.as_ref(),
            &args,
            &titan_hal::LaunchGeometry {
                grid: [expected.len().div_ceil(128) as u32, 1, 1],
                block: [128, 1, 1],
                shared_bytes: 0,
            },
        )
        .expect("launch attention on the supplied stream");
    session.wait(completion.as_ref()).expect("wait attention event");
    let mut output = vec![0_u8; expected.len() * 4];
    let copied = session.download(stream.as_ref(), output_device.as_ref(), &mut output).expect("D2H attention output");
    session.wait(copied.as_ref()).expect("wait D2H attention output");
    for (actual, expected) in f32_values(&output).into_iter().zip(expected) {
        assert!((actual - expected).abs() <= 2e-4, "CUDA attention output {actual} differs from CPU reference {expected}");
    }
}

#[test]
fn attention_descriptor_rejects_mask_dtype_layout_and_shape_contracts() {
    let valid = ScaledDotProductAttentionF32Descriptor {
        query_shape: [1, 1, 2, 2],
        key_shape: [1, 1, 3, 2],
        value_shape: [1, 1, 3, 2],
        output_shape: [1, 1, 2, 2],
        query_dtype: DType::F32,
        key_dtype: DType::F32,
        value_dtype: DType::F32,
        output_dtype: DType::F32,
        query_contiguous: true,
        key_contiguous: true,
        value_contiguous: true,
        output_contiguous: true,
        has_mask: false,
        causal: false,
    };
    assert!(valid.validate().is_ok());
    let mut mask = valid;
    mask.has_mask = true;
    assert!(mask.validate().unwrap_err().to_string().contains("masks"));
    let mut causal = valid;
    causal.causal = true;
    assert!(causal.validate().unwrap_err().to_string().contains("causal"));
    let mut dtype = valid;
    dtype.value_dtype = DType::I32;
    assert!(dtype.validate().unwrap_err().to_string().contains("F32"));
    let mut layout = valid;
    layout.key_contiguous = false;
    assert!(layout.validate().unwrap_err().to_string().contains("contiguous"));
    let mut shape = valid;
    shape.value_shape = [1, 1, 2, 2];
    assert!(shape.validate().unwrap_err().to_string().contains("Q[B,H,Tq,D]"));
}

#[test]
fn driver_attention_rejects_buffer_from_another_cuda_session() {
    let driver = match CudaDriver::open() {
        Ok(driver) => driver,
        Err(error) if no_cuda_environment(&error) => {
            eprintln!("SKIP: CUDA Driver API unavailable: {error}");
            return;
        }
        Err(error) => panic!("opening CUDA Driver API failed: {error}"),
    };
    let devices = driver.enumerate().expect("enumerate CUDA devices");
    let Some(fingerprint) = devices.first()
    else {
        eprintln!("SKIP: CUDA Driver API reported no GPU");
        return;
    };
    let session = driver.open(fingerprint.device).expect("open first CUDA session");
    let foreign_session = driver.open(fingerprint.device).expect("open second CUDA session");
    let stream = session.create_stream().expect("create CUDA stream");
    let query = session.allocate(4, 4).expect("allocate Q");
    let key = session.allocate(4, 4).expect("allocate K");
    let value = session.allocate(4, 4).expect("allocate V");
    let foreign_output = foreign_session.allocate(4, 4).expect("allocate foreign output");
    let abi = scaled_dot_product_attention_f32_abi();
    let artifact = CudaCompiler
        .compile_artifact(&scaled_dot_product_attention_ir(abi.clone()), &abi, session.fingerprint())
        .expect("lower structured attention IR into PTX");
    let kernel = session.load(artifact.ptx(), artifact.abi_hash(), artifact.metadata().clone()).expect("load attention PTX");
    let mut arguments = vec![
        KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: false, alignment: 4, buffer: query },
        KernelArg::Buffer { slot: 2, dtype: DType::F32, writable: false, alignment: 4, buffer: key },
        KernelArg::Buffer { slot: 3, dtype: DType::F32, writable: false, alignment: 4, buffer: value },
        KernelArg::Buffer { slot: 4, dtype: DType::F32, writable: true, alignment: 4, buffer: foreign_output },
    ];
    for scalar in [1_i32; 5] {
        arguments.push(KernelArg::Scalar { dtype: DType::I32, bytes: scalar.to_le_bytes().to_vec() });
    }
    let args = abi.encode(&arguments).expect("encode attention buffers on the CUDA device");
    let error = session
        .launch(
            stream.as_ref(),
            kernel.as_ref(),
            &args,
            &titan_hal::LaunchGeometry { grid: [1, 1, 1], block: [128, 1, 1], shared_bytes: 0 },
        )
        .unwrap_err();
    assert_eq!(error.operation, "buffer");
    assert!(error.detail.contains("foreign buffer"));
}

#[test]
fn conv2d_descriptor_rejects_dtype_layout_groups_and_geometry_contracts() {
    let valid = Conv2dF32Descriptor {
        input_shape: [1, 2, 4, 4],
        weight_shape: [2, 1, 3, 3],
        bias_shape: Some([2]),
        output_shape: [1, 2, 2, 2],
        input_dtype: DType::F32,
        weight_dtype: DType::F32,
        bias_dtype: Some(DType::F32),
        output_dtype: DType::F32,
        input_contiguous: true,
        weight_contiguous: true,
        bias_contiguous: Some(true),
        output_contiguous: true,
        stride_h: 2,
        stride_w: 2,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 2,
    };
    assert!(valid.validate().is_ok());
    let mut wrong_dtype = valid;
    wrong_dtype.weight_dtype = DType::I32;
    assert!(wrong_dtype.validate().unwrap_err().to_string().contains("F32"));
    let mut strided = valid;
    strided.output_contiguous = false;
    assert!(strided.validate().unwrap_err().to_string().contains("contiguous"));
    let mut wrong_groups = valid;
    wrong_groups.groups = 3;
    assert!(wrong_groups.validate().unwrap_err().to_string().contains("groups"));
    let mut wrong_output = valid;
    wrong_output.output_shape = [1, 2, 3, 3];
    assert!(wrong_output.validate().unwrap_err().to_string().contains("output"));
    let mut zero_stride = valid;
    zero_stride.stride_h = 0;
    assert!(zero_stride.validate().unwrap_err().to_string().contains("stride"));
    let mut no_bias = valid;
    no_bias.bias_shape = None;
    no_bias.bias_dtype = None;
    no_bias.bias_contiguous = None;
    assert!(no_bias.validate().is_ok());
}

#[test]
fn driver_conv2d_rejects_buffer_from_another_cuda_session() {
    let driver = match CudaDriver::open() {
        Ok(driver) => driver,
        Err(error) if no_cuda_environment(&error) => {
            eprintln!("SKIP: CUDA Driver API unavailable: {error}");
            return;
        }
        Err(error) => panic!("opening CUDA Driver API failed: {error}"),
    };
    let devices = driver.enumerate().expect("enumerate CUDA devices");
    let Some(fingerprint) = devices.first()
    else {
        eprintln!("SKIP: CUDA Driver API reported no GPU");
        return;
    };
    let session = driver.open(fingerprint.device).expect("open first CUDA session");
    let foreign_session = driver.open(fingerprint.device).expect("open second CUDA session");
    let stream = session.create_stream().expect("create CUDA stream");
    let input = session.allocate(4, 4).expect("allocate input");
    let weight = session.allocate(4, 4).expect("allocate weight");
    let unused_bias = session.allocate(4, 4).expect("allocate unused bias buffer");
    let foreign_output = foreign_session.allocate(4, 4).expect("allocate foreign output");
    let abi = conv2d_f32_abi();
    let artifact = CudaCompiler
        .compile_artifact(&conv2d_ir(abi.clone()), &abi, session.fingerprint())
        .expect("lower structured Conv2D IR into PTX");
    let kernel = session.load(artifact.ptx(), artifact.abi_hash(), artifact.metadata().clone()).expect("load Conv2D PTX");
    let mut arguments = vec![
        KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: false, alignment: 4, buffer: input },
        KernelArg::Buffer { slot: 2, dtype: DType::F32, writable: false, alignment: 4, buffer: weight },
        KernelArg::Buffer { slot: 3, dtype: DType::F32, writable: false, alignment: 4, buffer: unused_bias },
        KernelArg::Buffer { slot: 4, dtype: DType::F32, writable: true, alignment: 4, buffer: foreign_output },
    ];
    for value in [1usize, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0] {
        arguments.push(KernelArg::Scalar { dtype: DType::I32, bytes: (value as i32).to_le_bytes().to_vec() });
    }
    let args = abi.encode(&arguments).expect("encode Conv2D buffers on the CUDA device");
    let error = session
        .launch(
            stream.as_ref(),
            kernel.as_ref(),
            &args,
            &titan_hal::LaunchGeometry { grid: [1, 1, 1], block: [128, 1, 1], shared_bytes: 0 },
        )
        .unwrap_err();
    assert_eq!(error.operation, "buffer");
    assert!(error.detail.contains("foreign buffer"));
}

#[test]
fn gemm_descriptor_rejects_transpose_dtype_layout_and_shape_contracts() {
    let valid = GemmF32Descriptor {
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
    assert!(valid.validate().is_ok());
    let mut transposed = valid;
    transposed.transpose_rhs = true;
    assert!(transposed.validate().unwrap_err().to_string().contains("transpose"));
    let mut wrong_dtype = valid;
    wrong_dtype.output_dtype = DType::I32;
    assert!(wrong_dtype.validate().unwrap_err().to_string().contains("F32"));
    let mut strided = valid;
    strided.lhs_contiguous = false;
    assert!(strided.validate().unwrap_err().to_string().contains("contiguous"));
    let mut wrong_shape = valid;
    wrong_shape.rhs_shape = [3, 4];
    assert!(wrong_shape.validate().unwrap_err().to_string().contains("shapes"));
}

#[test]
fn driver_gemm_rejects_buffer_from_another_cuda_session() {
    let driver = match CudaDriver::open() {
        Ok(driver) => driver,
        Err(error) if no_cuda_environment(&error) => {
            eprintln!("SKIP: CUDA Driver API unavailable: {error}");
            return;
        }
        Err(error) => panic!("opening CUDA Driver API failed: {error}"),
    };
    let devices = driver.enumerate().expect("enumerate CUDA devices");
    let Some(fingerprint) = devices.first()
    else {
        eprintln!("SKIP: CUDA Driver API reported no GPU");
        return;
    };
    let session = driver.open(fingerprint.device).expect("open first CUDA session");
    let foreign_session = driver.open(fingerprint.device).expect("open second CUDA session");
    let stream = session.create_stream().expect("create CUDA stream");
    let lhs = session.allocate(4, 4).expect("allocate lhs");
    let rhs = session.allocate(4, 4).expect("allocate rhs");
    let foreign_output = foreign_session.allocate(4, 4).expect("allocate foreign output");
    let abi = gemm_f32_abi();
    let artifact = CudaCompiler
        .compile_artifact(&gemm_ir(abi.clone()), &abi, session.fingerprint())
        .expect("lower structured GEMM IR into PTX");
    let kernel = session.load(artifact.ptx(), artifact.abi_hash(), artifact.metadata().clone()).expect("load GEMM PTX");
    let args = abi
        .encode(&[
            KernelArg::Buffer { slot: 1, dtype: DType::F32, writable: false, alignment: 4, buffer: lhs },
            KernelArg::Buffer { slot: 2, dtype: DType::F32, writable: false, alignment: 4, buffer: rhs },
            KernelArg::Buffer { slot: 3, dtype: DType::F32, writable: true, alignment: 4, buffer: foreign_output },
            KernelArg::Scalar { dtype: DType::I32, bytes: 1_i32.to_le_bytes().to_vec() },
            KernelArg::Scalar { dtype: DType::I32, bytes: 1_i32.to_le_bytes().to_vec() },
            KernelArg::Scalar { dtype: DType::I32, bytes: 1_i32.to_le_bytes().to_vec() },
        ])
        .expect("encode buffers on the CUDA device");
    let error = session
        .launch(
            stream.as_ref(),
            kernel.as_ref(),
            &args,
            &titan_hal::LaunchGeometry { grid: [1, 1, 1], block: [128, 1, 1], shared_bytes: 0 },
        )
        .unwrap_err();
    assert_eq!(error.operation, "buffer");
    assert!(error.detail.contains("foreign buffer"));
}
