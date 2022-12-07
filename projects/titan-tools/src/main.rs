use std::path::PathBuf;
use titan_distributed::checksum;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "tt", about = "Titan.rs developer toolchain")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Prints runtime artifacts and diagnostic signals.
    Debug {
        #[arg(long, default_value = "target/titan")]
        output: PathBuf,
    },
    /// Describes a cluster launch topology without requiring a scheduler.
    Cluster {
        #[arg(long, default_value_t = 1)]
        nodes: u32,
        #[arg(long, default_value_t = 0)]
        rank: u32,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Debug { output } => {
            println!("tt.debug.version=1");
            for name in ["autotune.tune", "checkpoint.titan", "deployment.manifest", "model.onnx.txt"] {
                let path = output.join(name);
                println!("artifact.{name}={}", if path.is_file() { "ready" } else { "missing" });
            }
            let manifest = output.join("checkpoint.titan");
            if let Ok(bytes) = std::fs::read(&manifest) {
                println!("checkpoint.checksum={}", checksum(&bytes));
                println!("checkpoint.recovery=ready");
            } else {
                println!("checkpoint.recovery=unavailable");
            }
        }
        Command::Cluster { nodes, rank } => {
            if nodes == 0 || rank >= nodes {
                return Err(format!("rank {rank} must be less than nodes {nodes}").into());
            }
            println!("tt.cluster.version=1");
            println!("cluster.rank={rank}");
            println!("cluster.world={nodes}");
            println!("cluster.transport=local");
            println!("cluster.description=local deterministic topology; no remote connections");
        }
    }
    Ok(())
}
