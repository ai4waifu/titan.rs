use std::time::Duration;
use titan_distributed::{Checkpoint, Collective, LocalRing, Strategy};
use titan_graph::{CommandBuffer, Op};
use titan_hal::Cpu;
use titan_kernel::{KernelTarget, LaunchConfig, MatmulKernel};
use titan_model::{DeploymentManifest, DeploymentTarget, Linear, Module, OnnxModel};
use titan_runtime::Runtime;
use titan_tensor::{Tensor, squared_matmul_grad};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cpu = Cpu;
    let x = Tensor::from_vec(cpu, [2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    let mut weights = Tensor::from_vec(cpu, [2, 1], vec![0.5, -0.25])?;
    let mut runtime = Runtime::open("target/titan/autotune.tune");
    let generated = MatmulKernel::new("training-matmul", LaunchConfig::default()).compile(KernelTarget::Ptx);
    println!("kernel target={:?}", generated.target);
    let mut graph = CommandBuffer::new();
    graph.push(Op::Multiply);
    graph.push(Op::Add);
    println!("graph ops={:?}", graph.submit(|ops| ops).join().expect("graph worker panicked"));
    let linear = Linear::<2, 1, Cpu>::from_weights(cpu, vec![0.5, -0.25])?;
    println!("module output={:?}", linear.forward(&x, &mut runtime, cpu)?.as_slice());

    for step in 0..3 {
        let prediction = runtime.matmul(&x, &weights, cpu)?;
        let loss: f32 = prediction.as_slice().iter().map(|v| v * v).sum();
        let grad = squared_matmul_grad(&x, &weights, cpu)?;
        for (w, g) in weights.as_mut_slice().iter_mut().zip(grad.as_slice()) {
            *w -= 0.01 * g;
        }
        println!("step={step} loss={loss:.6}");
    }
    let total = LocalRing.all_reduce_sum(&[vec![1.0, 2.0], vec![3.0, 4.0]])?;
    println!("all_reduce={total:?}");
    let checkpoint = Checkpoint { step: 3, weights: weights.as_slice().to_vec(), strategy: Strategy::Zero { stage: 1 } };
    std::fs::write("target/titan/checkpoint.titan", checkpoint.encode())?;
    let restored = Checkpoint::decode(&std::fs::read_to_string("target/titan/checkpoint.titan")?)?;
    println!("checkpoint step={} strategy={:?}", restored.step, restored.strategy);
    let promoted = titan_autotune_feedback(&mut runtime);
    println!("telemetry_feedback={promoted}");
    let manifest = DeploymentManifest { model: "linear-demo".into(), target: DeploymentTarget::Native, backend: "cpu".into() };
    std::fs::write("target/titan/deployment.manifest", manifest.encode())?;
    std::fs::write("target/titan/model.onnx.txt", OnnxModel::linear("linear-demo").encode())?;
    for (operator, elapsed) in runtime.telemetry().summary() {
        println!("telemetry {operator}: {} us", elapsed.as_micros());
    }
    Ok(())
}

fn titan_autotune_feedback(runtime: &mut Runtime) -> bool {
    runtime.record_autotune_feedback(64, Duration::from_nanos(1), Duration::from_nanos(2))
}
