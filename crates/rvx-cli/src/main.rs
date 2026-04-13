use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use rvx_virt::{ArtifactBundle, VirtMachineBuilder, VmConfig, VmExit};

#[derive(Parser, Debug)]
#[command(name = "rvx")]
struct Cli {
    #[arg(long)]
    firmware: PathBuf,
    #[arg(long)]
    kernel: PathBuf,
    #[arg(long)]
    initrd: Option<PathBuf>,
    #[arg(long)]
    drive: Option<PathBuf>,
    #[arg(long)]
    append: Option<String>,
    #[arg(long, default_value_t = 4)]
    smp: u32,
    #[arg(long, default_value_t = 1024)]
    mem: u64,
    #[arg(long, default_value_t = true)]
    nographic: bool,
    #[arg(long)]
    trace: bool,
    #[arg(long)]
    time_limit_ms: Option<u64>,
    #[arg(long)]
    perf_json: Option<PathBuf>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("rvx: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    if cli.smp == 0 {
        bail!("--smp must be greater than zero");
    }

    let config = VmConfig {
        hart_count: cli.smp,
        ram_bytes: cli.mem * 1024 * 1024,
        nographic: cli.nographic,
        trace: cli.trace,
        time_limit_ms: cli.time_limit_ms,
        ..VmConfig::default()
    };
    let artifacts = ArtifactBundle {
        firmware: Some(cli.firmware),
        kernel: cli.kernel,
        initrd: cli.initrd,
        append: cli.append,
        drive: cli.drive,
    };

    let result = VirtMachineBuilder::new(config, artifacts)
        .build()?
        .run()
        .context("vm execution failed")?;

    if let Some(path) = cli.perf_json {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::to_vec_pretty(&result)?)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }

    match result.exit {
        VmExit::PowerOff => Ok(()),
        VmExit::Reset => {
            eprintln!("rvx: guest requested reset");
            Ok(())
        }
        VmExit::TimedOut => bail!("execution timed out"),
        VmExit::Halted => Ok(()),
        VmExit::BufferFull { hart_id } => bail!("translation buffer exhausted on hart {hart_id}"),
        VmExit::GuestTrap { hart_id, exit_code } => {
            bail!("guest exited from hart {hart_id} with code {exit_code:#x}")
        }
    }
}
