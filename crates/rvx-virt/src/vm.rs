use std::io::{self, Read};
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    mpsc,
    OnceLock,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use rvx_core::{Bus, ShutdownReason};
use rvx_riscv::{
    BlockBuilderCpu, BlockExecution, BlockKey, BlockStatus, BlockTerminatorKind, CompiledBlock,
    Cpu, DecodedInstruction, Exception, JitEngine, JitInstruction, MSTATUS_FS_MASK, MmuStats,
    PrivilegeLevel, Trap, TrapInfo, UnjittableBlock, decode, decode_compressed,
    jit_max_block_instructions,
};
use serde::Serialize;

use crate::model::{MachineModel, TestBundle};

const SBI_EXT_BASE: u64 = 0x10;
const SBI_EXT_TIME: u64 = 0x5449_4d45;
const SBI_EXT_IPI: u64 = 0x0073_5049;
const SBI_EXT_RFENCE: u64 = 0x5246_4e43;
const SBI_EXT_HSM: u64 = 0x0048_534d;
const SBI_EXT_SRST: u64 = 0x5352_5354;
const SBI_EXT_DBCN: u64 = 0x4442_434e;
const SBI_EXT_BASE_GET_SPEC_VERSION: u64 = 0;
const SBI_EXT_BASE_GET_IMP_ID: u64 = 1;
const SBI_EXT_BASE_GET_IMP_VERSION: u64 = 2;
const SBI_EXT_BASE_PROBE_EXT: u64 = 3;
const SBI_EXT_BASE_GET_MVENDORID: u64 = 4;
const SBI_EXT_BASE_GET_MARCHID: u64 = 5;
const SBI_EXT_BASE_GET_MIMPID: u64 = 6;
const SBI_EXT_TIME_SET_TIMER: u64 = 0;
const SBI_EXT_IPI_SEND_IPI: u64 = 0;
const SBI_EXT_RFENCE_REMOTE_FENCE_I: u64 = 0;
const SBI_EXT_RFENCE_REMOTE_SFENCE_VMA: u64 = 1;
const SBI_EXT_RFENCE_REMOTE_SFENCE_VMA_ASID: u64 = 2;
const SBI_EXT_HSM_HART_START: u64 = 0;
const SBI_EXT_HSM_HART_STOP: u64 = 1;
const SBI_EXT_HSM_HART_STATUS: u64 = 2;
const SBI_EXT_SRST_RESET: u64 = 0;
const SBI_EXT_DBCN_CONSOLE_WRITE: u64 = 0;
const SBI_EXT_DBCN_CONSOLE_READ: u64 = 1;
const SBI_EXT_DBCN_CONSOLE_WRITE_BYTE: u64 = 2;
const SBI_EXT_0_1_SET_TIMER: u64 = 0;
const SBI_EXT_0_1_CONSOLE_PUTCHAR: u64 = 1;
const SBI_EXT_0_1_CONSOLE_GETCHAR: u64 = 2;
const SBI_EXT_0_1_CLEAR_IPI: u64 = 3;
const SBI_EXT_0_1_SEND_IPI: u64 = 4;
const SBI_EXT_0_1_REMOTE_FENCE_I: u64 = 5;
const SBI_EXT_0_1_REMOTE_SFENCE_VMA: u64 = 6;
const SBI_EXT_0_1_REMOTE_SFENCE_VMA_ASID: u64 = 7;
const SBI_EXT_0_1_SHUTDOWN: u64 = 8;
const SBI_SPEC_V2: u64 = 2 << 24;
const SBI_IMPL_ID_RVX: u64 = 0x5256_58;
const SBI_IMPL_VERSION_RVX: u64 = 0x0001_0000;
const SBI_ERR_NOT_SUPPORTED: i64 = -2;
const SBI_ERR_INVALID_PARAM: i64 = -3;
const SBI_ERR_ALREADY_STARTED: i64 = -7;
const SBI_HSM_STATE_STARTED: u64 = 0;
const SBI_HSM_STATE_STOPPED: u64 = 1;
const SBI_SRST_RESET_TYPE_SHUTDOWN: u64 = 0;
const SBI_SRST_RESET_TYPE_COLD_REBOOT: u64 = 1;
const SBI_SRST_RESET_TYPE_WARM_REBOOT: u64 = 2;
const IRQ_SSIP: u64 = 1;
const IRQ_STIP: u64 = 5;
const IRQ_SEIP: u64 = 9;
const JIT_BLOCK_CACHE_SIZE: usize = 1 << 14;
const JIT_BLOCK_CACHE_WAYS: usize = 4;
const JIT_BLOCK_CACHE_SETS: usize = JIT_BLOCK_CACHE_SIZE / JIT_BLOCK_CACHE_WAYS;
const JIT_BLOCK_BURST_RETIRED_LIMIT: u64 = 4096;

#[derive(Debug, Clone)]
pub struct ArtifactBundle {
    pub firmware: Option<PathBuf>,
    pub kernel: PathBuf,
    pub initrd: Option<PathBuf>,
    pub append: Option<String>,
    pub drive: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct VmConfig {
    pub hart_count: u32,
    pub ram_bytes: u64,
    pub nographic: bool,
    pub trace: bool,
    pub code_buffer_bytes: usize,
    pub time_limit_ms: Option<u64>,
}

impl Default for VmConfig {
    fn default() -> Self {
        Self {
            hart_count: 4,
            ram_bytes: 1024 * 1024 * 1024,
            nographic: true,
            trace: false,
            code_buffer_bytes: 0,
            time_limit_ms: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PerfReport {
    pub started_at_unix_ms: u128,
    pub elapsed_ms: u128,
    pub hart_count: u32,
}

#[derive(Debug, Clone, Serialize)]
pub enum VmExit {
    PowerOff,
    Reset,
    GuestTrap { hart_id: usize, exit_code: usize },
    BufferFull { hart_id: usize },
    TimedOut,
    Halted,
}

#[derive(Debug, Clone, Serialize)]
pub struct MmuStatsReport {
    pub tlb_hits: u64,
    pub tlb_misses: u64,
    pub page_walks: u64,
    pub flushes: u64,
    pub jit_cache_fills: u64,
    pub translated_load_hits: u64,
    pub translated_load_misses: u64,
    pub translated_store_hits: u64,
    pub translated_store_misses: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HartSnapshot {
    pub hart_id: usize,
    pub started: bool,
    pub wfi: bool,
    pub software_interrupt: bool,
    pub pc: u64,
    pub privilege: &'static str,
    pub ra: u64,
    pub sp: u64,
    pub gp: u64,
    pub tp: u64,
    pub a: [u64; 8],
    pub mstatus: u64,
    pub satp: u64,
    pub pending_mip: u64,
    pub mip: u64,
    pub mie: u64,
    pub sepc: u64,
    pub scause: u64,
    pub stval: u64,
    pub mmu: MmuStatsReport,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeReport {
    pub retired_instructions: u64,
    pub compiled_blocks: usize,
    pub jitted_blocks_executed: u64,
    pub block_lookups: BlockLookupReport,
    pub vm_managed: VmManagedReport,
    pub block_chains: BlockChainReport,
    pub block_terminators: BlockTerminatorReport,
    pub snapshots: Vec<HartSnapshot>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BlockLookupReport {
    pub requests: u64,
    pub hart_cache_hits: u64,
    pub hart_cache_misses: u64,
    pub hart_cache_collisions: u64,
    pub compiled_table_hits: u64,
    pub compiled_table_misses: u64,
    pub compiled_new: u64,
    pub unjittable_hits: u64,
    pub unjittable_new: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BlockChainReport {
    pub attempts: u64,
    pub hits: u64,
    pub misses: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct VmManagedReport {
    pub total: u64,
    pub ecall: u64,
    pub sret: u64,
    pub mret: u64,
    pub sfence_vma: u64,
    pub sfence_vma_global: u64,
    pub sfence_vma_asid: u64,
    pub sfence_vma_page: u64,
    pub sfence_vma_page_asid: u64,
    pub wfi: u64,
    pub fence: u64,
    pub fence_i: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BlockTerminatorReport {
    pub fallthrough: u64,
    pub jal: u64,
    pub jalr: u64,
    pub branch: u64,
    pub csr: u64,
    pub atomic: u64,
    pub ebreak: u64,
    pub other: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct VmResult {
    pub exit: VmExit,
    pub perf: PerfReport,
    pub runtime: RuntimeReport,
}

pub struct VirtMachineBuilder {
    config: VmConfig,
    artifacts: ArtifactBundle,
}

impl VirtMachineBuilder {
    pub fn new(config: VmConfig, artifacts: ArtifactBundle) -> Self {
        Self { config, artifacts }
    }

    pub fn build(self) -> Result<VirtMachine> {
        VirtMachine::new(self.config, self.artifacts)
    }
}

struct HartState {
    cpu: Cpu,
    started: bool,
    timer_deadline: Option<u64>,
    software_interrupt: bool,
    wfi: bool,
    block_cache: Box<[[Option<BlockCacheEntry>; JIT_BLOCK_CACHE_WAYS]; JIT_BLOCK_CACHE_SETS]>,
}

#[derive(Clone, Copy)]
struct CachedBlockRef {
    key: BlockKey,
    entry:
        unsafe extern "C" fn(*mut Cpu, *const Bus, *mut TrapInfo, u64, u64, *mut u8, u64, u64) -> u64,
    terminator: BlockTerminatorKind,
}

#[derive(Clone, Copy)]
enum BlockCacheEntry {
    Compiled(CachedBlockRef),
    VmManaged {
        key: BlockKey,
        block: UnjittableBlock,
    },
}

impl BlockCacheEntry {
    fn key(self) -> BlockKey {
        match self {
            Self::Compiled(cached) => cached.key,
            Self::VmManaged { key, .. } => key,
        }
    }
}

#[derive(Clone, Copy)]
enum BlockLookupOutcome {
    Compiled(CachedBlockRef),
    VmManaged(UnjittableBlock),
    Interpreter,
}

enum JitStepOutcome {
    Completed(bool),
    VmManaged(UnjittableBlock),
    Interpreter,
}

enum ParallelJitStepOutcome {
    Completed(ParallelStep),
    VmManaged(UnjittableBlock),
    Interpreter,
}

struct HostConsole {
    rx: mpsc::Receiver<u8>,
    _tty_mode: Option<HostTtyMode>,
}

struct HostTtyMode {
    original: libc::termios,
}

impl HostTtyMode {
    fn for_stdin() -> io::Result<Option<Self>> {
        // Piped stdin in tests should stay untouched; only interactive TTYs need no-echo mode.
        if unsafe { libc::isatty(libc::STDIN_FILENO) } != 1 {
            return Ok(None);
        }

        let mut termios = MaybeUninit::<libc::termios>::uninit();
        if unsafe { libc::tcgetattr(libc::STDIN_FILENO, termios.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let original = unsafe { termios.assume_init() };
        let mut configured = original;
        configured.c_lflag &= !(libc::ECHO | libc::ECHONL | libc::ICANON | libc::IEXTEN);
        configured.c_cc[libc::VMIN] = 1;
        configured.c_cc[libc::VTIME] = 0;
        if unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &configured) } != 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(Some(Self { original }))
    }
}

impl Drop for HostTtyMode {
    fn drop(&mut self) {
        let _ = unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.original) };
    }
}

impl HostConsole {
    fn spawn(enabled: bool) -> Option<Self> {
        if !enabled {
            return None;
        }

        let tty_mode = match HostTtyMode::for_stdin() {
            Ok(mode) => mode,
            Err(err) => {
                eprintln!("rvx: failed to configure host tty for nographic mode: {err}");
                None
            }
        };
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let stdin = io::stdin();
            let mut handle = stdin.lock();
            let mut buf = [0u8; 256];
            loop {
                match handle.read(&mut buf) {
                    Ok(0) => break,
                    Ok(count) => {
                        for &byte in &buf[..count] {
                            if tx.send(byte).is_err() {
                                return;
                            }
                        }
                    }
                    Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
        });
        Some(Self {
            rx,
            _tty_mode: tty_mode,
        })
    }

    fn drain_into_uart(&self, model: &MachineModel) -> bool {
        let mut received = false;
        let mut uart = model.uart.lock().unwrap();
        for byte in self.rx.try_iter().take(256) {
            uart.0.receive(byte);
            received = true;
        }
        received
    }
}

fn write_model_console(model: &MachineModel, bytes: &[u8]) {
    let mut uart = model.uart.lock().unwrap();
    for &byte in bytes {
        uart.0.write(0, byte);
    }
}

pub struct VirtMachine {
    config: VmConfig,
    artifacts: ArtifactBundle,
    model: MachineModel,
    harts: Vec<HartState>,
    host_console: Option<HostConsole>,
    pending_exit: Option<VmExit>,
    retired_instructions: u64,
    time_ticks: u64,
    compiled_blocks: usize,
    jitted_blocks_executed: u64,
    block_lookups: BlockLookupReport,
    vm_managed: VmManagedReport,
    block_chains: BlockChainReport,
    block_terminators: BlockTerminatorReport,
}

impl VirtMachine {
    pub fn new(config: VmConfig, artifacts: ArtifactBundle) -> Result<Self> {
        if artifacts.drive.is_some() {
            bail!("virtio block is not implemented on the first-party runtime yet");
        }

        let mut harts = Vec::with_capacity(config.hart_count as usize);
        for hart_id in 0..config.hart_count as usize {
            let mut cpu = Cpu::new();
            cpu.csr.mhartid = hart_id as u64;
            cpu.csr.mstatus |= MSTATUS_FS_MASK;
            harts.push(HartState {
                cpu,
                started: hart_id == 0,
                timer_deadline: None,
                software_interrupt: false,
                wfi: false,
                block_cache: Box::new([[None; JIT_BLOCK_CACHE_WAYS]; JIT_BLOCK_CACHE_SETS]),
            });
        }

        let host_console = HostConsole::spawn(config.nographic);

        Ok(Self {
            model: MachineModel::new(config.hart_count as usize, config.ram_bytes)?,
            config,
            artifacts,
            harts,
            host_console,
            pending_exit: None,
            retired_instructions: 0,
            time_ticks: 0,
            compiled_blocks: 0,
            jitted_blocks_executed: 0,
            block_lookups: BlockLookupReport::default(),
            vm_managed: VmManagedReport::default(),
            block_chains: BlockChainReport::default(),
            block_terminators: BlockTerminatorReport::default(),
        })
    }

    pub fn run(mut self) -> Result<VmResult> {
        let started = Instant::now();
        let started_at_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

        let bundle = TestBundle {
            firmware: self
                .artifacts
                .firmware
                .clone()
                .context("missing firmware path")?,
            kernel: self.artifacts.kernel.clone(),
            initrd: self.artifacts.initrd.clone(),
            append: self.artifacts.append.clone(),
        };
        let boot = self.model.load_bundle(&bundle)?;
        self.reset_harts(boot.kernel_entry, boot.fdt_addr);

        let time_limit = self.config.time_limit_ms.map(Duration::from_millis);

        let experimental_parallel = std::env::var_os("RVX_EXPERIMENTAL_PARALLEL").is_some();
        if self.harts.len() > 1 && experimental_parallel {
            return self.run_parallel(started, started_at_unix_ms, time_limit);
        }
        self.model.bus().set_concurrent_access(false);

        let mut jit = JitEngine::new()?;

        loop {
            self.flush_console_output();
            self.poll_console_input();

            if let Some(reason) = self.model.shutdown_reason() {
                let exit = match reason {
                    ShutdownReason::Pass => VmExit::PowerOff,
                    ShutdownReason::Reset => VmExit::Reset,
                    ShutdownReason::Fail(code) => VmExit::GuestTrap {
                        hart_id: 0,
                        exit_code: code as usize,
                    },
                };
                return Ok(self.finish(exit, started, started_at_unix_ms));
            }

            if let Some(exit) = self.pending_exit.clone() {
                return Ok(self.finish(exit, started, started_at_unix_ms));
            }

            if let Some(limit) = time_limit
                && started.elapsed() >= limit
            {
                return Ok(self.finish(VmExit::TimedOut, started, started_at_unix_ms));
            }

            let mut made_progress = false;
            for hart_id in 0..self.harts.len() {
                if !self.harts[hart_id].started {
                    continue;
                }
                let retired_before = self.retired_instructions;
                let progress = self.step_hart(hart_id, self.time_ticks, &mut jit)?;
                self.advance_virtual_time(retired_before, progress);
                made_progress |= progress;
                if self.pending_exit.is_some() {
                    break;
                }
            }

            if !made_progress {
                if self.harts.iter().all(|hart| !hart.started) {
                    return Ok(self.finish(VmExit::Halted, started, started_at_unix_ms));
                }
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    }

    fn finish(self, exit: VmExit, started: Instant, started_at_unix_ms: u128) -> VmResult {
        self.flush_console_output();
        VmResult {
            exit,
            perf: PerfReport {
                started_at_unix_ms,
                elapsed_ms: started.elapsed().as_millis(),
                hart_count: self.config.hart_count,
            },
            runtime: RuntimeReport {
                retired_instructions: self.retired_instructions,
                compiled_blocks: self.compiled_blocks,
                jitted_blocks_executed: self.jitted_blocks_executed,
                block_lookups: self.block_lookups,
                vm_managed: self.vm_managed,
                block_chains: self.block_chains,
                block_terminators: self.block_terminators,
                snapshots: self
                    .harts
                    .iter()
                    .enumerate()
                    .map(|(hart_id, hart)| HartSnapshot {
                        hart_id,
                        started: hart.started,
                        wfi: hart.wfi,
                        software_interrupt: hart.software_interrupt,
                        pc: hart.cpu.pc,
                        privilege: privilege_name(hart.cpu.privilege),
                        ra: hart.cpu.read_x(1),
                        sp: hart.cpu.read_x(2),
                        gp: hart.cpu.read_x(3),
                        tp: hart.cpu.read_x(4),
                        a: [
                            hart.cpu.read_x(10),
                            hart.cpu.read_x(11),
                            hart.cpu.read_x(12),
                            hart.cpu.read_x(13),
                            hart.cpu.read_x(14),
                            hart.cpu.read_x(15),
                            hart.cpu.read_x(16),
                            hart.cpu.read_x(17),
                        ],
                        mstatus: hart.cpu.csr.mstatus,
                        satp: hart.cpu.csr.satp,
                        pending_mip: hart.cpu.pending_mip,
                        mip: hart.cpu.csr.mip,
                        mie: hart.cpu.csr.mie,
                        sepc: hart.cpu.csr.sepc,
                        scause: hart.cpu.csr.scause,
                        stval: hart.cpu.csr.stval,
                        mmu: mmu_stats_report(hart.cpu.mmu.stats()),
                    })
                    .collect(),
            },
        }
    }

    fn reset_harts(&mut self, kernel_entry: u64, fdt_addr: u64) {
        for (hart_id, hart) in self.harts.iter_mut().enumerate() {
            hart.cpu.pc = if hart_id == 0 { kernel_entry } else { 0 };
            hart.cpu.privilege = if hart_id == 0 {
                PrivilegeLevel::Supervisor
            } else {
                PrivilegeLevel::Machine
            };
            hart.cpu.write_x(10, hart_id as u64);
            hart.cpu.write_x(11, fdt_addr);
            hart.cpu.mmu.flush();
            hart.timer_deadline = None;
            hart.software_interrupt = false;
            hart.wfi = false;
        }
    }

    fn poll_console_input(&mut self) -> bool {
        self.host_console
            .as_ref()
            .is_some_and(|console| console.drain_into_uart(&self.model))
    }

    fn flush_console_output(&self) {
        self.model.uart.lock().unwrap().0.flush_output();
    }

    fn run_parallel(
        self,
        started: Instant,
        started_at_unix_ms: u128,
        time_limit: Option<Duration>,
    ) -> Result<VmResult> {
        self.model.bus().set_concurrent_access(true);
        ParallelVm::from_machine(self).run(started, started_at_unix_ms, time_limit)
    }

    fn step_hart(&mut self, hart_id: usize, now_ticks: u64, jit: &mut JitEngine) -> Result<bool> {
        self.refresh_hart_interrupts(hart_id, now_ticks);
        let observed_software_interrupt = self.harts[hart_id].software_interrupt;
        {
            let hart = &mut self.harts[hart_id];
            hart.cpu.csr.time = now_ticks;
            hart.cpu.sync_mip();
            if hart.wfi && hart.cpu.csr.mip == 0 {
                return Ok(false);
            }
            if hart.wfi {
                if self.config.trace && (hart.cpu.csr.mip & (1 << IRQ_SSIP)) != 0 {
                    eprintln!(
                        "rvx: hart {hart_id} woke from wfi mip=0x{:016x} mie=0x{:016x} mstatus=0x{:016x}",
                        hart.cpu.csr.mip, hart.cpu.csr.mie, hart.cpu.csr.mstatus,
                    );
                }
                hart.wfi = false;
            }
            if take_pending_interrupt(&mut hart.cpu, hart_id, self.config.trace) {
                return Ok(true);
            }
        }
        let made_progress = match self.step_hart_jit(hart_id, jit)? {
            JitStepOutcome::Completed(result) => result,
            JitStepOutcome::VmManaged(entry) => self.execute_vm_managed_entry(hart_id, entry)?,
            JitStepOutcome::Interpreter => self.step_hart_interpreter(hart_id)?,
        };
        self.sync_hart_interrupt_sources_from_cpu(hart_id, observed_software_interrupt);
        Ok(made_progress)
    }

    fn step_hart_jit(&mut self, hart_id: usize, jit: &mut JitEngine) -> Result<JitStepOutcome> {
        let burst_limit = jit_block_burst_retired_limit();
        let mut trap_info = TrapInfo::default();
        let chaining_enabled = jit.block_chaining_enabled();
        let (bus_ptr, ram_base, ram_len, ram_ptr) = {
            let bus = self.model.bus();
            let ram_base = bus.ram().base();
            let ram_len = bus.ram().len();
            let ram_ptr = bus.ram().as_mut_ptr();
            (bus as *const Bus, ram_base, ram_len, ram_ptr)
        };
        let mut burst_progress = false;
        let mut burst_retired = 0u64;
        let mut chained_block: Option<CachedBlockRef> = None;

        loop {
            let cached = if let Some(cached) = chained_block.take() {
                self.block_chains.attempts = self.block_chains.attempts.wrapping_add(1);
                self.block_chains.hits = self.block_chains.hits.wrapping_add(1);
                cached
            } else {
                let key = {
                    let hart = &self.harts[hart_id];
                    BlockKey::for_cpu(&hart.cpu)
                };
                match self.lookup_or_compile_block(hart_id, key, jit)? {
                    BlockLookupOutcome::Compiled(cached) => cached,
                    BlockLookupOutcome::VmManaged(entry) => {
                        return if burst_progress {
                            Ok(JitStepOutcome::Completed(true))
                        } else {
                            Ok(JitStepOutcome::VmManaged(entry))
                        };
                    }
                    BlockLookupOutcome::Interpreter => {
                        return if burst_progress {
                            Ok(JitStepOutcome::Completed(true))
                        } else {
                            Ok(JitStepOutcome::Interpreter)
                        };
                    }
                }
            };

            trap_info.next_block = 0;
            let retired_base = if chaining_enabled { burst_retired } else { 0 };
            let packed = unsafe {
                (cached.entry)(
                    &mut self.harts[hart_id].cpu as *mut Cpu,
                    bus_ptr,
                    &mut trap_info as *mut TrapInfo,
                    ram_base,
                    ram_len,
                    ram_ptr,
                    retired_base,
                    burst_limit,
                )
            };
            let retired_before = burst_retired;
            let result = BlockExecution::from_packed(packed);
            self.jitted_blocks_executed = self.jitted_blocks_executed.wrapping_add(1);
            increment_block_terminator(&mut self.block_terminators, cached.terminator);
            let retired_delta = if chaining_enabled {
                result.retired as u64 - retired_before
            } else {
                result.retired as u64
            };
            self.retired_instructions = self
                .retired_instructions
                .wrapping_add(retired_delta);
            self.harts[hart_id].cpu.csr.cycle = self.harts[hart_id]
                .cpu
                .csr
                .cycle
                .wrapping_add(retired_delta);
            burst_retired = if chaining_enabled {
                result.retired as u64
            } else {
                burst_retired.wrapping_add(result.retired as u64)
            };
            burst_progress |= retired_delta != 0;
            if result.status != BlockStatus::Trap
                && take_pending_interrupt(&mut self.harts[hart_id].cpu, hart_id, self.config.trace)
            {
                return Ok(JitStepOutcome::Completed(true));
            }

            match result.status {
                BlockStatus::Continue => {
                    if chaining_enabled
                        && cached.terminator == BlockTerminatorKind::Jalr
                        && retired_delta != 0
                        && burst_retired < burst_limit
                    {
                        let target_key = BlockKey::for_cpu(&self.harts[hart_id].cpu);
                        match self.lookup_or_compile_block(hart_id, target_key, jit)? {
                            BlockLookupOutcome::Compiled(target_cached)
                                if jit.patch_jalr_chain(cached.key, target_key) =>
                            {
                                self.block_chains.misses =
                                    self.block_chains.misses.wrapping_add(1);
                                chained_block = Some(target_cached);
                                continue;
                            }
                            BlockLookupOutcome::VmManaged(_) => {
                                return Ok(JitStepOutcome::Completed(true));
                            }
                            _ => {}
                        }
                    }
                    if retired_delta == 0 || burst_retired >= burst_limit {
                        return Ok(JitStepOutcome::Completed(burst_progress));
                    }
                }
                BlockStatus::Chain => {
                    if trap_info.next_block == 0 || burst_retired >= burst_limit {
                        self.block_chains.misses = self.block_chains.misses.wrapping_add(1);
                        return Ok(JitStepOutcome::Completed(burst_progress));
                    }
                    let block = unsafe { &*(trap_info.next_block as usize as *const CompiledBlock) };
                    chained_block = Some(CachedBlockRef {
                        key: block.key,
                        entry: block.entry(),
                        terminator: block.terminator(),
                    });
                }
                BlockStatus::Trap => {
                    self.harts[hart_id].cpu.csr.cycle =
                        self.harts[hart_id].cpu.csr.cycle.wrapping_add(1);
                    if self.config.trace {
                        eprintln!(
                            "rvx: trap hart={hart_id} pc=0x{:016x} cause=0x{:016x} tval=0x{:016x}",
                            self.harts[hart_id].cpu.pc, trap_info.cause, trap_info.tval,
                        );
                    }
                    let trap = Trap::from_cause(trap_info.cause, trap_info.tval)
                        .context("invalid trap reported by jitted block")?;
                    self.harts[hart_id].cpu.take_trap(trap);
                    return Ok(JitStepOutcome::Completed(true));
                }
            }
        }
    }

    fn lookup_or_compile_block(
        &mut self,
        hart_id: usize,
        key: BlockKey,
        jit: &mut JitEngine,
    ) -> Result<BlockLookupOutcome> {
        self.block_lookups.requests = self.block_lookups.requests.wrapping_add(1);
        let cache_set_index = block_cache_set_index(key);
        {
            let cache_set = &mut self.harts[hart_id].block_cache[cache_set_index];
            let mut hit_index = None;
            let mut occupied = false;
            for (index, slot) in cache_set.iter().enumerate() {
                if let Some(cached) = slot {
                    occupied = true;
                    if cached.key() == key {
                        hit_index = Some(index);
                        break;
                    }
                }
            }
            if let Some(index) = hit_index {
                self.block_lookups.hart_cache_hits =
                    self.block_lookups.hart_cache_hits.wrapping_add(1);
                if index != 0 {
                    cache_set.swap(0, index);
                }
                return Ok(match cache_set[0].unwrap() {
                    BlockCacheEntry::Compiled(cached) => BlockLookupOutcome::Compiled(cached),
                    BlockCacheEntry::VmManaged { block, .. } => BlockLookupOutcome::VmManaged(block),
                });
            }
            if occupied {
                self.block_lookups.hart_cache_collisions =
                    self.block_lookups.hart_cache_collisions.wrapping_add(1);
            }
        }
        self.block_lookups.hart_cache_misses = self.block_lookups.hart_cache_misses.wrapping_add(1);

        let cached = if let Some(block) = jit.block(key) {
            self.block_lookups.compiled_table_hits =
                self.block_lookups.compiled_table_hits.wrapping_add(1);
            CachedBlockRef {
                key,
                entry: block.entry(),
                terminator: block.terminator(),
            }
        } else {
            self.block_lookups.compiled_table_misses =
                self.block_lookups.compiled_table_misses.wrapping_add(1);
            if let Some(block) = jit.unjittable_block(key) {
                self.block_lookups.unjittable_hits =
                    self.block_lookups.unjittable_hits.wrapping_add(1);
                let cache_set = &mut self.harts[hart_id].block_cache[cache_set_index];
                for index in (1..JIT_BLOCK_CACHE_WAYS).rev() {
                    cache_set[index] = cache_set[index - 1];
                }
                cache_set[0] = Some(BlockCacheEntry::VmManaged { key, block });
                return Ok(BlockLookupOutcome::VmManaged(block));
            }
            let instructions = match self.build_jit_block(hart_id)? {
                Some(instructions) => instructions,
                None => {
                    if let Some(block) =
                        current_vm_managed_entry(&self.harts[hart_id].cpu, self.model.bus())
                    {
                        jit.note_unjittable_block(key, block);
                        self.block_lookups.unjittable_new =
                            self.block_lookups.unjittable_new.wrapping_add(1);
                        let cache_set = &mut self.harts[hart_id].block_cache[cache_set_index];
                        for index in (1..JIT_BLOCK_CACHE_WAYS).rev() {
                            cache_set[index] = cache_set[index - 1];
                        }
                        cache_set[0] = Some(BlockCacheEntry::VmManaged { key, block });
                        return Ok(BlockLookupOutcome::VmManaged(block));
                    }
                    return Ok(BlockLookupOutcome::Interpreter);
                }
            };
            let before = jit.block_count();
            let (entry, terminator) = {
                let block = jit.compile_block(key, instructions)?;
                (block.entry(), block.terminator())
            };
            let after = jit.block_count();
            self.compiled_blocks += after - before;
            self.block_lookups.compiled_new = self
                .block_lookups
                .compiled_new
                .wrapping_add((after - before) as u64);
            CachedBlockRef {
                key,
                entry,
                terminator,
            }
        };

        let cache_set = &mut self.harts[hart_id].block_cache[cache_set_index];
        for index in (1..JIT_BLOCK_CACHE_WAYS).rev() {
            cache_set[index] = cache_set[index - 1];
        }
        cache_set[0] = Some(BlockCacheEntry::Compiled(cached));
        Ok(BlockLookupOutcome::Compiled(cached))
    }

    fn build_jit_block(&mut self, hart_id: usize) -> Result<Option<Vec<JitInstruction>>> {
        let mut cpu = self.harts[hart_id].cpu.block_builder();
        let max_block_instructions = jit_max_block_instructions();
        let mut instructions = Vec::with_capacity(max_block_instructions);

        for _ in 0..max_block_instructions {
            let raw16 = match cpu.fetch_u16(self.model.bus()) {
                Ok(raw) => raw,
                Err(_) => return Ok((!instructions.is_empty()).then_some(instructions)),
            };

            let (insn, insn_bytes) = if (raw16 & 0x3) != 0x3 {
                match decode_compressed(raw16) {
                    Ok(insn) => (insn, 2u8),
                    Err(_) => return Ok((!instructions.is_empty()).then_some(instructions)),
                }
            } else {
                let raw = match cpu.fetch_u32(self.model.bus()) {
                    Ok(raw) => raw,
                    Err(_) => return Ok((!instructions.is_empty()).then_some(instructions)),
                };
                match decode(raw) {
                    Ok(insn) => (insn, 4u8),
                    Err(_) => return Ok((!instructions.is_empty()).then_some(instructions)),
                }
            };

            if is_vm_managed_instruction(insn) {
                return Ok((!instructions.is_empty()).then_some(instructions));
            }

            instructions.push(JitInstruction {
                decoded: insn,
                instruction_bytes: insn_bytes,
            });
            if let Some(next_pc) = traced_successor_pc(cpu.pc, insn, insn_bytes) {
                cpu.pc = next_pc;
                continue;
            }
            if ends_jit_block(insn) {
                break;
            }
            cpu.pc = cpu.pc.wrapping_add(insn_bytes as u64);
        }

        Ok((!instructions.is_empty()).then_some(instructions))
    }

    fn step_hart_interpreter(&mut self, hart_id: usize) -> Result<bool> {
        self.harts[hart_id].cpu.csr.cycle = self.harts[hart_id].cpu.csr.cycle.wrapping_add(1);
        let raw16 = {
            let hart = &mut self.harts[hart_id];
            hart.cpu.fetch_u16(self.model.bus())
        };
        let raw16 = match raw16 {
            Ok(raw) => raw,
            Err(trap) => {
                self.harts[hart_id].cpu.take_trap(trap);
                return Ok(true);
            }
        };

        let (insn, insn_bytes, illegal_tval) = if (raw16 & 0x3) != 0x3 {
            match decode_compressed(raw16) {
                Ok(insn) => (insn, 2u8, raw16 as u64),
                Err(_) => {
                    if self.config.trace {
                        eprintln!(
                            "rvx: illegal decode hart={hart_id} pc=0x{:016x} raw16=0x{:04x}",
                            self.harts[hart_id].cpu.pc, raw16,
                        );
                    }
                    self.harts[hart_id]
                        .cpu
                        .take_trap(Trap::exception(Exception::IllegalInstruction, raw16 as u64));
                    return Ok(true);
                }
            }
        } else {
            let raw = {
                let hart = &mut self.harts[hart_id];
                hart.cpu.fetch_u32(self.model.bus())
            };
            let raw = match raw {
                Ok(raw) => raw,
                Err(trap) => {
                    self.harts[hart_id].cpu.take_trap(trap);
                    return Ok(true);
                }
            };
            match decode(raw) {
                Ok(insn) => (insn, 4u8, raw as u64),
                Err(_) => {
                    if self.config.trace {
                        eprintln!(
                            "rvx: illegal decode hart={hart_id} pc=0x{:016x} raw32=0x{:08x}",
                            self.harts[hart_id].cpu.pc, raw,
                        );
                    }
                    self.harts[hart_id]
                        .cpu
                        .take_trap(Trap::exception(Exception::IllegalInstruction, raw as u64));
                    return Ok(true);
                }
            }
        };

        match insn {
            DecodedInstruction::Ecall => {
                let privilege = self.harts[hart_id].cpu.privilege;
                if privilege == PrivilegeLevel::Supervisor {
                    self.handle_supervisor_ecall(hart_id)?;
                } else {
                    let trap = Trap::exception(
                        if privilege == PrivilegeLevel::User {
                            Exception::UserEnvCall
                        } else {
                            Exception::MachineEnvCall
                        },
                        0,
                    );
                    self.harts[hart_id].cpu.take_trap(trap);
                }
                self.retired_instructions = self.retired_instructions.wrapping_add(1);
                self.harts[hart_id].cpu.csr.instret =
                    self.harts[hart_id].cpu.csr.instret.wrapping_add(1);
                return Ok(true);
            }
            DecodedInstruction::Sret => {
                let cpu = &mut self.harts[hart_id].cpu;
                if let Err(trap) = cpu.execute_sret() {
                    cpu.take_trap(trap);
                }
                self.retired_instructions = self.retired_instructions.wrapping_add(1);
                cpu.csr.instret = cpu.csr.instret.wrapping_add(1);
                return Ok(true);
            }
            DecodedInstruction::Mret => {
                let cpu = &mut self.harts[hart_id].cpu;
                if let Err(trap) = cpu.execute_mret() {
                    cpu.take_trap(trap);
                }
                self.retired_instructions = self.retired_instructions.wrapping_add(1);
                cpu.csr.instret = cpu.csr.instret.wrapping_add(1);
                return Ok(true);
            }
            DecodedInstruction::SfenceVma { rs1, rs2 } => {
                let cpu = &mut self.harts[hart_id].cpu;
                let addr = (rs1 != 0).then(|| cpu.read_x(rs1));
                let asid = (rs2 != 0).then(|| cpu.read_x(rs2) as u16);
                cpu.mmu.flush_vma(addr, asid);
                cpu.pc = cpu.pc.wrapping_add(insn_bytes as u64);
                self.retired_instructions = self.retired_instructions.wrapping_add(1);
                cpu.csr.instret = cpu.csr.instret.wrapping_add(1);
                return Ok(true);
            }
            DecodedInstruction::Wfi => {
                self.harts[hart_id].wfi = true;
                let cpu = &mut self.harts[hart_id].cpu;
                cpu.pc = cpu.pc.wrapping_add(insn_bytes as u64);
                self.retired_instructions = self.retired_instructions.wrapping_add(1);
                cpu.csr.instret = cpu.csr.instret.wrapping_add(1);
                return Ok(false);
            }
            _ => {}
        }

        let result = {
            let hart = &mut self.harts[hart_id];
            hart.cpu.execute(insn, insn_bytes, self.model.bus())
        };
        match result {
            Ok(_) => {
                if instruction_may_write_memory(insn) {
                    self.invalidate_reservations();
                }
                self.retired_instructions = self.retired_instructions.wrapping_add(1);
                Ok(true)
            }
            Err(trap) => {
                if self.config.trace {
                    eprintln!(
                        "rvx: trap hart={hart_id} pc=0x{:016x} cause=0x{:016x} tval=0x{:016x}",
                        self.harts[hart_id].cpu.pc,
                        trap.cause(),
                        if trap.tval == 0 {
                            illegal_tval
                        } else {
                            trap.tval
                        },
                    );
                }
                self.harts[hart_id].cpu.take_trap(trap);
                Ok(true)
            }
        }
    }

    fn execute_vm_managed_entry(
        &mut self,
        hart_id: usize,
        entry: UnjittableBlock,
    ) -> Result<bool> {
        self.harts[hart_id].cpu.csr.cycle = self.harts[hart_id].cpu.csr.cycle.wrapping_add(1);
        match entry.decoded {
            DecodedInstruction::Ecall => {
                self.vm_managed.total = self.vm_managed.total.wrapping_add(1);
                self.vm_managed.ecall = self.vm_managed.ecall.wrapping_add(1);
                let privilege = self.harts[hart_id].cpu.privilege;
                if privilege == PrivilegeLevel::Supervisor {
                    self.handle_supervisor_ecall(hart_id)?;
                } else {
                    let trap = Trap::exception(
                        if privilege == PrivilegeLevel::User {
                            Exception::UserEnvCall
                        } else {
                            Exception::MachineEnvCall
                        },
                        0,
                    );
                    self.harts[hart_id].cpu.take_trap(trap);
                }
                self.retired_instructions = self.retired_instructions.wrapping_add(1);
                self.harts[hart_id].cpu.csr.instret =
                    self.harts[hart_id].cpu.csr.instret.wrapping_add(1);
                Ok(true)
            }
            DecodedInstruction::Sret => {
                self.vm_managed.total = self.vm_managed.total.wrapping_add(1);
                self.vm_managed.sret = self.vm_managed.sret.wrapping_add(1);
                let cpu = &mut self.harts[hart_id].cpu;
                if let Err(trap) = cpu.execute_sret() {
                    cpu.take_trap(trap);
                }
                self.retired_instructions = self.retired_instructions.wrapping_add(1);
                cpu.csr.instret = cpu.csr.instret.wrapping_add(1);
                Ok(true)
            }
            DecodedInstruction::Mret => {
                self.vm_managed.total = self.vm_managed.total.wrapping_add(1);
                self.vm_managed.mret = self.vm_managed.mret.wrapping_add(1);
                let cpu = &mut self.harts[hart_id].cpu;
                if let Err(trap) = cpu.execute_mret() {
                    cpu.take_trap(trap);
                }
                self.retired_instructions = self.retired_instructions.wrapping_add(1);
                cpu.csr.instret = cpu.csr.instret.wrapping_add(1);
                Ok(true)
            }
            DecodedInstruction::SfenceVma { rs1, rs2 } => {
                self.vm_managed.total = self.vm_managed.total.wrapping_add(1);
                self.vm_managed.sfence_vma = self.vm_managed.sfence_vma.wrapping_add(1);
                let cpu = &mut self.harts[hart_id].cpu;
                let addr = (rs1 != 0).then(|| cpu.read_x(rs1));
                let asid = (rs2 != 0).then(|| cpu.read_x(rs2) as u16);
                record_vm_managed_sfence_shape(&mut self.vm_managed, addr, asid);
                cpu.mmu.flush_vma(addr, asid);
                cpu.pc = cpu.pc.wrapping_add(entry.instruction_bytes as u64);
                self.retired_instructions = self.retired_instructions.wrapping_add(1);
                cpu.csr.instret = cpu.csr.instret.wrapping_add(1);
                Ok(true)
            }
            DecodedInstruction::Wfi => {
                self.vm_managed.total = self.vm_managed.total.wrapping_add(1);
                self.vm_managed.wfi = self.vm_managed.wfi.wrapping_add(1);
                self.harts[hart_id].wfi = true;
                let cpu = &mut self.harts[hart_id].cpu;
                cpu.pc = cpu.pc.wrapping_add(entry.instruction_bytes as u64);
                self.retired_instructions = self.retired_instructions.wrapping_add(1);
                cpu.csr.instret = cpu.csr.instret.wrapping_add(1);
                Ok(false)
            }
            DecodedInstruction::Fence => {
                self.vm_managed.total = self.vm_managed.total.wrapping_add(1);
                self.vm_managed.fence = self.vm_managed.fence.wrapping_add(1);
                self.model.bus().memory_barrier();
                let cpu = &mut self.harts[hart_id].cpu;
                cpu.pc = cpu.pc.wrapping_add(entry.instruction_bytes as u64);
                self.retired_instructions = self.retired_instructions.wrapping_add(1);
                cpu.csr.instret = cpu.csr.instret.wrapping_add(1);
                Ok(true)
            }
            DecodedInstruction::FenceI => {
                self.vm_managed.total = self.vm_managed.total.wrapping_add(1);
                self.vm_managed.fence_i = self.vm_managed.fence_i.wrapping_add(1);
                self.model.bus().memory_barrier();
                let cpu = &mut self.harts[hart_id].cpu;
                cpu.pc = cpu.pc.wrapping_add(entry.instruction_bytes as u64);
                self.retired_instructions = self.retired_instructions.wrapping_add(1);
                cpu.csr.instret = cpu.csr.instret.wrapping_add(1);
                Ok(true)
            }
            _ => unreachable!("cached unjittable block must be vm-managed"),
        }
    }

    fn refresh_hart_interrupts(&mut self, hart_id: usize, now_ticks: u64) {
        let hart = &mut self.harts[hart_id];
        hart.cpu.pending_mip &= !((1 << IRQ_SSIP) | (1 << IRQ_STIP) | (1 << IRQ_SEIP));
        if hart.software_interrupt {
            hart.cpu.pending_mip |= 1 << IRQ_SSIP;
        }
        if hart
            .timer_deadline
            .is_some_and(|deadline| now_ticks >= deadline)
        {
            hart.cpu.pending_mip |= 1 << IRQ_STIP;
        }
        if self.model.supervisor_external_pending(hart_id) {
            hart.cpu.pending_mip |= 1 << IRQ_SEIP;
        }
    }

    fn sync_hart_interrupt_sources_from_cpu(
        &mut self,
        hart_id: usize,
        observed_software_interrupt: bool,
    ) {
        let hart = &mut self.harts[hart_id];
        if observed_software_interrupt && (hart.cpu.csr.mip & (1 << IRQ_SSIP)) == 0 {
            hart.software_interrupt = false;
        }
    }

    fn handle_supervisor_ecall(&mut self, hart_id: usize) -> Result<()> {
        let ext = self.harts[hart_id].cpu.read_x(17);
        let fid = self.harts[hart_id].cpu.read_x(16);
        let args = [
            self.harts[hart_id].cpu.read_x(10),
            self.harts[hart_id].cpu.read_x(11),
            self.harts[hart_id].cpu.read_x(12),
            self.harts[hart_id].cpu.read_x(13),
            self.harts[hart_id].cpu.read_x(14),
            self.harts[hart_id].cpu.read_x(15),
        ];

        let (error, value) = match ext {
            SBI_EXT_BASE => self.sbi_base(fid, args),
            SBI_EXT_TIME => self.sbi_time(hart_id, fid, args),
            SBI_EXT_IPI => self.sbi_ipi(fid, args),
            SBI_EXT_RFENCE => self.sbi_rfence(fid, args),
            SBI_EXT_HSM => self.sbi_hsm(hart_id, fid, args),
            SBI_EXT_SRST => self.sbi_srst(fid, args),
            SBI_EXT_DBCN => self.sbi_dbcn(fid, args)?,
            SBI_EXT_0_1_SET_TIMER
            | SBI_EXT_0_1_CONSOLE_PUTCHAR
            | SBI_EXT_0_1_CONSOLE_GETCHAR
            | SBI_EXT_0_1_CLEAR_IPI
            | SBI_EXT_0_1_SEND_IPI
            | SBI_EXT_0_1_REMOTE_FENCE_I
            | SBI_EXT_0_1_REMOTE_SFENCE_VMA
            | SBI_EXT_0_1_REMOTE_SFENCE_VMA_ASID
            | SBI_EXT_0_1_SHUTDOWN => self.sbi_legacy(hart_id, ext, args)?,
            _ => (SBI_ERR_NOT_SUPPORTED, 0),
        };

        let cpu = &mut self.harts[hart_id].cpu;
        cpu.write_x(10, error as u64);
        cpu.write_x(11, value);
        cpu.pc = cpu.pc.wrapping_add(4);
        Ok(())
    }

    fn sbi_base(&self, fid: u64, args: [u64; 6]) -> (i64, u64) {
        match fid {
            SBI_EXT_BASE_GET_SPEC_VERSION => (0, SBI_SPEC_V2),
            SBI_EXT_BASE_GET_IMP_ID => (0, SBI_IMPL_ID_RVX),
            SBI_EXT_BASE_GET_IMP_VERSION => (0, SBI_IMPL_VERSION_RVX),
            SBI_EXT_BASE_PROBE_EXT => (
                0,
                u64::from(matches!(
                    args[0],
                    SBI_EXT_BASE
                        | SBI_EXT_TIME
                        | SBI_EXT_IPI
                        | SBI_EXT_RFENCE
                        | SBI_EXT_HSM
                        | SBI_EXT_SRST
                        | SBI_EXT_DBCN
                )),
            ),
            SBI_EXT_BASE_GET_MVENDORID | SBI_EXT_BASE_GET_MARCHID | SBI_EXT_BASE_GET_MIMPID => {
                (0, 0)
            }
            _ => (SBI_ERR_NOT_SUPPORTED, 0),
        }
    }

    fn sbi_time(&mut self, hart_id: usize, fid: u64, args: [u64; 6]) -> (i64, u64) {
        if fid != SBI_EXT_TIME_SET_TIMER {
            return (SBI_ERR_NOT_SUPPORTED, 0);
        }
        self.harts[hart_id].timer_deadline = Some(args[0]);
        self.harts[hart_id].cpu.pending_mip &= !(1 << IRQ_STIP);
        (0, 0)
    }

    fn sbi_ipi(&mut self, fid: u64, args: [u64; 6]) -> (i64, u64) {
        if fid != SBI_EXT_IPI_SEND_IPI {
            return (SBI_ERR_NOT_SUPPORTED, 0);
        }
        self.apply_hart_mask(args[0], args[1], |hart| hart.software_interrupt = true);
        (0, 0)
    }

    fn sbi_rfence(&mut self, fid: u64, args: [u64; 6]) -> (i64, u64) {
        match fid {
            SBI_EXT_RFENCE_REMOTE_FENCE_I => (0, 0),
            SBI_EXT_RFENCE_REMOTE_SFENCE_VMA => {
                self.apply_hart_mask(args[0], args[1], |hart| {
                    hart.cpu.mmu.flush_range(args[2], args[3], None)
                });
                (0, 0)
            }
            SBI_EXT_RFENCE_REMOTE_SFENCE_VMA_ASID => {
                let asid = args[4] as u16;
                self.apply_hart_mask(args[0], args[1], |hart| {
                    hart.cpu.mmu.flush_range(args[2], args[3], Some(asid))
                });
                (0, 0)
            }
            _ => (SBI_ERR_NOT_SUPPORTED, 0),
        }
    }

    fn sbi_hsm(&mut self, current_hart: usize, fid: u64, args: [u64; 6]) -> (i64, u64) {
        match fid {
            SBI_EXT_HSM_HART_START => {
                let hart_id = args[0] as usize;
                let Some(hart) = self.harts.get_mut(hart_id) else {
                    return (SBI_ERR_INVALID_PARAM, 0);
                };
                if hart.started {
                    return (SBI_ERR_ALREADY_STARTED, 0);
                }
                hart.started = true;
                hart.cpu.pc = args[1];
                hart.cpu.privilege = PrivilegeLevel::Supervisor;
                hart.cpu.write_x(10, hart_id as u64);
                hart.cpu.write_x(11, args[2]);
                hart.cpu.pending_mip = 0;
                hart.cpu.mmu.flush();
                (0, 0)
            }
            SBI_EXT_HSM_HART_STOP => {
                self.harts[current_hart].started = false;
                (0, 0)
            }
            SBI_EXT_HSM_HART_STATUS => {
                let hart_id = args[0] as usize;
                let Some(hart) = self.harts.get(hart_id) else {
                    return (SBI_ERR_INVALID_PARAM, 0);
                };
                (
                    0,
                    if hart.started {
                        SBI_HSM_STATE_STARTED
                    } else {
                        SBI_HSM_STATE_STOPPED
                    },
                )
            }
            _ => (SBI_ERR_NOT_SUPPORTED, 0),
        }
    }

    fn sbi_srst(&mut self, fid: u64, args: [u64; 6]) -> (i64, u64) {
        if fid != SBI_EXT_SRST_RESET {
            return (SBI_ERR_NOT_SUPPORTED, 0);
        }
        self.pending_exit = Some(match args[0] {
            SBI_SRST_RESET_TYPE_SHUTDOWN => VmExit::PowerOff,
            SBI_SRST_RESET_TYPE_COLD_REBOOT | SBI_SRST_RESET_TYPE_WARM_REBOOT => VmExit::Reset,
            _ => VmExit::Reset,
        });
        (0, 0)
    }

    fn sbi_dbcn(&mut self, fid: u64, args: [u64; 6]) -> Result<(i64, u64)> {
        match fid {
            SBI_EXT_DBCN_CONSOLE_WRITE => {
                let count = args[0] as usize;
                let base = args[1];
                let mut buf = vec![0u8; count];
                for (index, byte) in buf.iter_mut().enumerate() {
                    *byte = self.model.bus().read_u8(base + index as u64).map_err(|_| {
                        anyhow::anyhow!("dbcn read fault at 0x{:016x}", base + index as u64)
                    })?;
                }
                write_model_console(&self.model, &buf);
                Ok((0, count as u64))
            }
            SBI_EXT_DBCN_CONSOLE_WRITE_BYTE => {
                write_model_console(&self.model, &[args[0] as u8]);
                Ok((0, 0))
            }
            SBI_EXT_DBCN_CONSOLE_READ => Ok((0, 0)),
            _ => Ok((SBI_ERR_NOT_SUPPORTED, 0)),
        }
    }

    fn sbi_legacy(&mut self, hart_id: usize, ext: u64, args: [u64; 6]) -> Result<(i64, u64)> {
        let result = match ext {
            SBI_EXT_0_1_SET_TIMER => self.sbi_time(hart_id, SBI_EXT_TIME_SET_TIMER, args),
            SBI_EXT_0_1_CONSOLE_PUTCHAR => {
                write_model_console(&self.model, &[args[0] as u8]);
                (0, 0)
            }
            SBI_EXT_0_1_CONSOLE_GETCHAR => (0, u64::MAX),
            SBI_EXT_0_1_CLEAR_IPI => {
                self.harts[hart_id].software_interrupt = false;
                (0, 0)
            }
            SBI_EXT_0_1_SEND_IPI => (0, 0),
            SBI_EXT_0_1_REMOTE_FENCE_I
            | SBI_EXT_0_1_REMOTE_SFENCE_VMA
            | SBI_EXT_0_1_REMOTE_SFENCE_VMA_ASID => (0, 0),
            SBI_EXT_0_1_SHUTDOWN => {
                self.pending_exit = Some(VmExit::PowerOff);
                (0, 0)
            }
            _ => (SBI_ERR_NOT_SUPPORTED, 0),
        };
        Ok(result)
    }

    fn apply_hart_mask(&mut self, hmask: u64, hbase: u64, mut f: impl FnMut(&mut HartState)) {
        let base = hbase as usize;
        for bit in 0..64usize {
            if ((hmask >> bit) & 1) == 0 {
                continue;
            }
            if let Some(hart) = self.harts.get_mut(base + bit) {
                f(hart);
            }
        }
    }

    fn advance_virtual_time(&mut self, retired_before: u64, made_progress: bool) {
        let retired_delta = self.retired_instructions.wrapping_sub(retired_before);
        let tick_delta = retired_delta.max(u64::from(made_progress));
        self.time_ticks = self.time_ticks.wrapping_add(tick_delta);
    }

    fn invalidate_reservations(&mut self) {
        let _ = self.model.bus().invalidate_reservations();
    }
}

struct ParallelVm {
    config: VmConfig,
    model: Arc<MachineModel>,
    harts: Vec<Mutex<HartState>>,
    jit_enabled: bool,
    host_console: Mutex<Option<HostConsole>>,
    pending_exit: Mutex<Option<VmExit>>,
    retired_instructions: AtomicU64,
    time_ticks: AtomicU64,
    compiled_blocks: AtomicUsize,
    jitted_blocks_executed: AtomicU64,
    block_lookup_requests: AtomicU64,
    block_lookup_hart_cache_hits: AtomicU64,
    block_lookup_hart_cache_misses: AtomicU64,
    block_lookup_hart_cache_collisions: AtomicU64,
    block_lookup_table_hits: AtomicU64,
    block_lookup_table_misses: AtomicU64,
    block_lookup_compiled_new: AtomicU64,
    block_lookup_unjittable_hits: AtomicU64,
    block_lookup_unjittable_new: AtomicU64,
    vm_managed_total: AtomicU64,
    vm_managed_ecall: AtomicU64,
    vm_managed_sret: AtomicU64,
    vm_managed_mret: AtomicU64,
    vm_managed_sfence_vma: AtomicU64,
    vm_managed_sfence_vma_global: AtomicU64,
    vm_managed_sfence_vma_asid: AtomicU64,
    vm_managed_sfence_vma_page: AtomicU64,
    vm_managed_sfence_vma_page_asid: AtomicU64,
    vm_managed_wfi: AtomicU64,
    vm_managed_fence: AtomicU64,
    vm_managed_fence_i: AtomicU64,
    stop: AtomicBool,
}

#[derive(Debug, Clone, Copy, Default)]
struct ParallelStep {
    progress: bool,
    retired_delta: u64,
}

impl ParallelVm {
    fn from_machine(machine: VirtMachine) -> Self {
        Self {
            config: machine.config,
            model: Arc::new(machine.model),
            harts: machine.harts.into_iter().map(Mutex::new).collect(),
            jit_enabled: std::env::var_os("RVX_PARALLEL_DISABLE_JIT").is_none(),
            host_console: Mutex::new(machine.host_console),
            pending_exit: Mutex::new(machine.pending_exit),
            retired_instructions: AtomicU64::new(machine.retired_instructions),
            time_ticks: AtomicU64::new(machine.time_ticks),
            compiled_blocks: AtomicUsize::new(machine.compiled_blocks),
            jitted_blocks_executed: AtomicU64::new(machine.jitted_blocks_executed),
            block_lookup_requests: AtomicU64::new(machine.block_lookups.requests),
            block_lookup_hart_cache_hits: AtomicU64::new(machine.block_lookups.hart_cache_hits),
            block_lookup_hart_cache_misses: AtomicU64::new(machine.block_lookups.hart_cache_misses),
            block_lookup_hart_cache_collisions: AtomicU64::new(
                machine.block_lookups.hart_cache_collisions,
            ),
            block_lookup_table_hits: AtomicU64::new(machine.block_lookups.compiled_table_hits),
            block_lookup_table_misses: AtomicU64::new(machine.block_lookups.compiled_table_misses),
            block_lookup_compiled_new: AtomicU64::new(machine.block_lookups.compiled_new),
            block_lookup_unjittable_hits: AtomicU64::new(machine.block_lookups.unjittable_hits),
            block_lookup_unjittable_new: AtomicU64::new(machine.block_lookups.unjittable_new),
            vm_managed_total: AtomicU64::new(machine.vm_managed.total),
            vm_managed_ecall: AtomicU64::new(machine.vm_managed.ecall),
            vm_managed_sret: AtomicU64::new(machine.vm_managed.sret),
            vm_managed_mret: AtomicU64::new(machine.vm_managed.mret),
            vm_managed_sfence_vma: AtomicU64::new(machine.vm_managed.sfence_vma),
            vm_managed_sfence_vma_global: AtomicU64::new(machine.vm_managed.sfence_vma_global),
            vm_managed_sfence_vma_asid: AtomicU64::new(machine.vm_managed.sfence_vma_asid),
            vm_managed_sfence_vma_page: AtomicU64::new(machine.vm_managed.sfence_vma_page),
            vm_managed_sfence_vma_page_asid: AtomicU64::new(
                machine.vm_managed.sfence_vma_page_asid,
            ),
            vm_managed_wfi: AtomicU64::new(machine.vm_managed.wfi),
            vm_managed_fence: AtomicU64::new(machine.vm_managed.fence),
            vm_managed_fence_i: AtomicU64::new(machine.vm_managed.fence_i),
            stop: AtomicBool::new(false),
        }
    }

    fn run(
        self,
        started: Instant,
        started_at_unix_ms: u128,
        time_limit: Option<Duration>,
    ) -> Result<VmResult> {
        let shared = Arc::new(self);
        let shared_jit = Arc::new(Mutex::new(JitEngine::new_parallel()?));
        let mut handles = Vec::with_capacity(shared.harts.len());

        for hart_id in 0..shared.harts.len() {
            let shared = Arc::clone(&shared);
            let shared_jit = Arc::clone(&shared_jit);
            handles.push(std::thread::spawn(move || {
                parallel_worker(shared, shared_jit, hart_id, started, time_limit)
            }));
        }

        for handle in handles {
            match handle.join() {
                Ok(result) => result?,
                Err(_) => bail!("hart worker thread panicked"),
            }
        }

        shared.flush_console_output();

        let exit = if let Some(reason) = shared.model.shutdown_reason() {
            match reason {
                ShutdownReason::Pass => VmExit::PowerOff,
                ShutdownReason::Reset => VmExit::Reset,
                ShutdownReason::Fail(code) => VmExit::GuestTrap {
                    hart_id: 0,
                    exit_code: code as usize,
                },
            }
        } else if let Some(exit) = shared.pending_exit.lock().unwrap().clone() {
            exit
        } else if let Some(limit) = time_limit {
            if started.elapsed() >= limit {
                VmExit::TimedOut
            } else if shared
                .harts
                .iter()
                .all(|hart| !hart.lock().unwrap().started)
            {
                VmExit::Halted
            } else {
                VmExit::TimedOut
            }
        } else if shared
            .harts
            .iter()
            .all(|hart| !hart.lock().unwrap().started)
        {
            VmExit::Halted
        } else {
            VmExit::TimedOut
        };

        Ok(VmResult {
            exit,
            perf: PerfReport {
                started_at_unix_ms,
                elapsed_ms: started.elapsed().as_millis(),
                hart_count: shared.config.hart_count,
            },
            runtime: RuntimeReport {
                retired_instructions: shared.retired_instructions.load(Ordering::Relaxed),
                compiled_blocks: shared.compiled_blocks.load(Ordering::Relaxed),
                jitted_blocks_executed: shared.jitted_blocks_executed.load(Ordering::Relaxed),
                block_lookups: BlockLookupReport {
                    requests: shared.block_lookup_requests.load(Ordering::Relaxed),
                    hart_cache_hits: shared
                        .block_lookup_hart_cache_hits
                        .load(Ordering::Relaxed),
                    hart_cache_misses: shared
                        .block_lookup_hart_cache_misses
                        .load(Ordering::Relaxed),
                    hart_cache_collisions: shared
                        .block_lookup_hart_cache_collisions
                        .load(Ordering::Relaxed),
                    compiled_table_hits: shared.block_lookup_table_hits.load(Ordering::Relaxed),
                    compiled_table_misses: shared
                        .block_lookup_table_misses
                        .load(Ordering::Relaxed),
                    compiled_new: shared.block_lookup_compiled_new.load(Ordering::Relaxed),
                    unjittable_hits: shared
                        .block_lookup_unjittable_hits
                        .load(Ordering::Relaxed),
                    unjittable_new: shared
                        .block_lookup_unjittable_new
                        .load(Ordering::Relaxed),
                },
                vm_managed: VmManagedReport {
                    total: shared.vm_managed_total.load(Ordering::Relaxed),
                    ecall: shared.vm_managed_ecall.load(Ordering::Relaxed),
                    sret: shared.vm_managed_sret.load(Ordering::Relaxed),
                    mret: shared.vm_managed_mret.load(Ordering::Relaxed),
                    sfence_vma: shared.vm_managed_sfence_vma.load(Ordering::Relaxed),
                    sfence_vma_global: shared
                        .vm_managed_sfence_vma_global
                        .load(Ordering::Relaxed),
                    sfence_vma_asid: shared
                        .vm_managed_sfence_vma_asid
                        .load(Ordering::Relaxed),
                    sfence_vma_page: shared
                        .vm_managed_sfence_vma_page
                        .load(Ordering::Relaxed),
                    sfence_vma_page_asid: shared
                        .vm_managed_sfence_vma_page_asid
                        .load(Ordering::Relaxed),
                    wfi: shared.vm_managed_wfi.load(Ordering::Relaxed),
                    fence: shared.vm_managed_fence.load(Ordering::Relaxed),
                    fence_i: shared.vm_managed_fence_i.load(Ordering::Relaxed),
                },
                block_chains: BlockChainReport::default(),
                block_terminators: BlockTerminatorReport::default(),
                snapshots: shared
                    .harts
                    .iter()
                    .enumerate()
                    .map(|(hart_id, hart)| {
                        let hart = hart.lock().unwrap();
                        HartSnapshot {
                            hart_id,
                            started: hart.started,
                            wfi: hart.wfi,
                            software_interrupt: hart.software_interrupt,
                            pc: hart.cpu.pc,
                            privilege: privilege_name(hart.cpu.privilege),
                            ra: hart.cpu.read_x(1),
                            sp: hart.cpu.read_x(2),
                            gp: hart.cpu.read_x(3),
                            tp: hart.cpu.read_x(4),
                            a: [
                                hart.cpu.read_x(10),
                                hart.cpu.read_x(11),
                                hart.cpu.read_x(12),
                                hart.cpu.read_x(13),
                                hart.cpu.read_x(14),
                                hart.cpu.read_x(15),
                                hart.cpu.read_x(16),
                                hart.cpu.read_x(17),
                            ],
                            mstatus: hart.cpu.csr.mstatus,
                            satp: hart.cpu.csr.satp,
                            pending_mip: hart.cpu.pending_mip,
                            mip: hart.cpu.csr.mip,
                            mie: hart.cpu.csr.mie,
                            sepc: hart.cpu.csr.sepc,
                            scause: hart.cpu.csr.scause,
                            stval: hart.cpu.csr.stval,
                            mmu: mmu_stats_report(hart.cpu.mmu.stats()),
                        }
                    })
                    .collect(),
            },
        })
    }

    fn poll_console_input(&self) -> bool {
        let console = self.host_console.lock().unwrap();
        console
            .as_ref()
            .is_some_and(|console| console.drain_into_uart(&self.model))
    }

    fn flush_console_output(&self) {
        self.model.uart.lock().unwrap().0.flush_output();
    }

    fn advance_virtual_time(&self, retired_delta: u64, made_progress: bool) {
        let tick_delta = retired_delta.max(u64::from(made_progress));
        self.time_ticks.fetch_add(tick_delta, Ordering::Relaxed);
    }
}

fn parallel_worker(
    shared: Arc<ParallelVm>,
    shared_jit: Arc<Mutex<JitEngine>>,
    hart_id: usize,
    started: Instant,
    time_limit: Option<Duration>,
) -> Result<()> {
    let parallel_step_quantum = parallel_step_quantum();

    loop {
        if parallel_should_stop(&shared, started, time_limit) {
            return Ok(());
        }
        if hart_id == 0 {
            shared.flush_console_output();
            shared.poll_console_input();
        }

        let mut made_progress = false;
        for _ in 0..parallel_step_quantum {
            if parallel_should_stop(&shared, started, time_limit) {
                return Ok(());
            }
            let step = parallel_step_hart(&shared, hart_id, &shared_jit)?;
            shared.advance_virtual_time(step.retired_delta, step.progress);
            made_progress |= step.progress;
            if !step.progress {
                break;
            }
        }

        if !made_progress {
            if !parallel_fast_forward_time(&shared) {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    }
}

fn parallel_step_quantum() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("RVX_PARALLEL_STEP_QUANTUM")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(256)
    })
}

fn jit_block_burst_retired_limit() -> u64 {
    static VALUE: OnceLock<u64> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("RVX_JIT_BLOCK_BURST_RETIRED_LIMIT")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(JIT_BLOCK_BURST_RETIRED_LIMIT)
    })
}

fn parallel_should_stop(
    shared: &ParallelVm,
    started: Instant,
    time_limit: Option<Duration>,
) -> bool {
    if shared.stop.load(Ordering::Relaxed) {
        return true;
    }

    let should_stop = shared.model.shutdown_reason().is_some()
        || shared.pending_exit.lock().unwrap().is_some()
        || shared
            .harts
            .iter()
            .all(|hart| !hart.lock().unwrap().started)
        || time_limit.is_some_and(|limit| started.elapsed() >= limit);
    if should_stop {
        shared.stop.store(true, Ordering::Relaxed);
    }
    should_stop
}

fn parallel_step_hart(
    shared: &ParallelVm,
    hart_id: usize,
    shared_jit: &Mutex<JitEngine>,
) -> Result<ParallelStep> {
    let now_ticks = shared.time_ticks.load(Ordering::Relaxed);
    let observed_software_interrupt;
    {
        let mut hart = shared.harts[hart_id].lock().unwrap();
        if !hart.started {
            return Ok(ParallelStep::default());
        }
        parallel_refresh_hart_interrupts(shared, hart_id, now_ticks, &mut hart);
        observed_software_interrupt = hart.software_interrupt;
        hart.cpu.csr.time = now_ticks;
        hart.cpu.sync_mip();
        if hart.wfi && hart.cpu.csr.mip == 0 {
            return Ok(ParallelStep::default());
        }
        if hart.wfi {
            if shared.config.trace && (hart.cpu.csr.mip & (1 << IRQ_SSIP)) != 0 {
                eprintln!(
                    "rvx: hart {hart_id} woke from wfi mip=0x{:016x} mie=0x{:016x} mstatus=0x{:016x}",
                    hart.cpu.csr.mip, hart.cpu.csr.mie, hart.cpu.csr.mstatus,
                );
            }
            hart.wfi = false;
        }
        if take_pending_interrupt(&mut hart.cpu, hart_id, shared.config.trace) {
            return Ok(ParallelStep {
                progress: true,
                retired_delta: 0,
            });
        }
    }

    let step = if shared.jit_enabled {
        match parallel_step_hart_jit(shared, hart_id, shared_jit)? {
            ParallelJitStepOutcome::Completed(step) => step,
            ParallelJitStepOutcome::VmManaged(entry) => {
                parallel_execute_vm_managed_entry(shared, hart_id, entry)?
            }
            ParallelJitStepOutcome::Interpreter => {
                parallel_step_hart_interpreter(shared, hart_id)?
            }
        }
    } else {
        parallel_step_hart_interpreter(shared, hart_id)?
    };
    parallel_sync_hart_interrupt_sources(shared, hart_id, observed_software_interrupt);
    Ok(step)
}

fn parallel_step_hart_jit(
    shared: &ParallelVm,
    hart_id: usize,
    shared_jit: &Mutex<JitEngine>,
) -> Result<ParallelJitStepOutcome> {
    shared
        .block_lookup_requests
        .fetch_add(1, Ordering::Relaxed);
    let (key, cached_entry, occupied) = {
        let mut hart = shared.harts[hart_id].lock().unwrap();
        let key = BlockKey::for_cpu(&hart.cpu);
        let (cached_entry, occupied) = lookup_hart_block_cache(&mut hart.block_cache, key);
        (key, cached_entry, occupied)
    };
    if let Some(cached_entry) = cached_entry {
        shared
            .block_lookup_hart_cache_hits
            .fetch_add(1, Ordering::Relaxed);
        match cached_entry {
            BlockCacheEntry::Compiled(cached) => {
                return parallel_execute_cached_block(shared, hart_id, cached);
            }
            BlockCacheEntry::VmManaged { block, .. } => {
                return Ok(ParallelJitStepOutcome::VmManaged(block));
            }
        }
    }
    if occupied {
        shared
            .block_lookup_hart_cache_collisions
            .fetch_add(1, Ordering::Relaxed);
    }
    shared
        .block_lookup_hart_cache_misses
        .fetch_add(1, Ordering::Relaxed);

    let cached = {
        let jit = shared_jit.lock().unwrap();
        if let Some(block) = jit.block(key) {
            shared
                .block_lookup_table_hits
                .fetch_add(1, Ordering::Relaxed);
            CachedBlockRef {
                key,
                entry: block.entry(),
                terminator: block.terminator(),
            }
        } else {
            shared
                .block_lookup_table_misses
                .fetch_add(1, Ordering::Relaxed);
            if let Some(block) = jit.unjittable_block(key) {
                shared
                    .block_lookup_unjittable_hits
                    .fetch_add(1, Ordering::Relaxed);
                drop(jit);
                let mut hart = shared.harts[hart_id].lock().unwrap();
                insert_hart_block_cache(
                    &mut hart.block_cache,
                    BlockCacheEntry::VmManaged { key, block },
                );
                return Ok(ParallelJitStepOutcome::VmManaged(block));
            }
            drop(jit);
            let cpu_template = {
                let hart = shared.harts[hart_id].lock().unwrap();
                hart.cpu.block_builder()
            };
            let instructions = match parallel_build_jit_block(shared, cpu_template)? {
                Some(instructions) => instructions,
                None => {
                    let block = {
                        let hart = shared.harts[hart_id].lock().unwrap();
                        current_vm_managed_entry(&hart.cpu, shared.model.bus())
                    };
                    if let Some(block) = block {
                        let mut jit = shared_jit.lock().unwrap();
                        jit.note_unjittable_block(key, block);
                        shared
                            .block_lookup_unjittable_new
                            .fetch_add(1, Ordering::Relaxed);
                        drop(jit);
                        let mut hart = shared.harts[hart_id].lock().unwrap();
                        insert_hart_block_cache(
                            &mut hart.block_cache,
                            BlockCacheEntry::VmManaged { key, block },
                        );
                        return Ok(ParallelJitStepOutcome::VmManaged(block));
                    }
                    return Ok(ParallelJitStepOutcome::Interpreter);
                }
            };
            let mut jit = shared_jit.lock().unwrap();
            let before = jit.block_count();
            let (entry, terminator) = {
                let block = jit.compile_block(key, instructions)?;
                (block.entry(), block.terminator())
            };
            let after = jit.block_count();
            shared
                .compiled_blocks
                .fetch_add(after - before, Ordering::Relaxed);
            shared
                .block_lookup_compiled_new
                .fetch_add((after - before) as u64, Ordering::Relaxed);
            CachedBlockRef {
                key,
                entry,
                terminator,
            }
        }
    };
    {
        let mut hart = shared.harts[hart_id].lock().unwrap();
        insert_hart_block_cache(&mut hart.block_cache, BlockCacheEntry::Compiled(cached));
    }

    parallel_execute_cached_block(shared, hart_id, cached)
}

fn parallel_execute_cached_block(
    shared: &ParallelVm,
    hart_id: usize,
    cached: CachedBlockRef,
) -> Result<ParallelJitStepOutcome> {
    let mut hart = shared.harts[hart_id].lock().unwrap();
    let mut trap_info = TrapInfo::default();
    let bus = shared.model.bus();
    let packed = unsafe {
        (cached.entry)(
            &mut hart.cpu as *mut Cpu,
            bus as *const Bus,
            &mut trap_info as *mut TrapInfo,
            bus.ram().base(),
            bus.ram().len(),
            bus.ram().as_mut_ptr(),
            0,
            u64::MAX,
        )
    };
    let result = BlockExecution::from_packed(packed);
    shared
        .jitted_blocks_executed
        .fetch_add(1, Ordering::Relaxed);
    shared
        .retired_instructions
        .fetch_add(result.retired as u64, Ordering::Relaxed);
    hart.cpu.csr.cycle = hart.cpu.csr.cycle.wrapping_add(result.retired as u64);
    if result.status != BlockStatus::Trap
        && take_pending_interrupt(&mut hart.cpu, hart_id, shared.config.trace)
    {
        return Ok(ParallelJitStepOutcome::Completed(ParallelStep {
            progress: true,
            retired_delta: result.retired as u64,
        }));
    }

    match result.status {
        BlockStatus::Continue => Ok(ParallelJitStepOutcome::Completed(ParallelStep {
            progress: result.retired != 0,
            retired_delta: result.retired as u64,
        })),
        BlockStatus::Chain => Ok(ParallelJitStepOutcome::Completed(ParallelStep {
            progress: result.retired != 0,
            retired_delta: result.retired as u64,
        })),
        BlockStatus::Trap => {
            hart.cpu.csr.cycle = hart.cpu.csr.cycle.wrapping_add(1);
            if shared.config.trace {
                eprintln!(
                    "rvx: trap hart={hart_id} pc=0x{:016x} cause=0x{:016x} tval=0x{:016x}",
                    hart.cpu.pc, trap_info.cause, trap_info.tval,
                );
            }
            let trap = Trap::from_cause(trap_info.cause, trap_info.tval)
                .context("invalid trap reported by jitted block")?;
            hart.cpu.take_trap(trap);
            Ok(ParallelJitStepOutcome::Completed(ParallelStep {
                progress: true,
                retired_delta: result.retired as u64,
            }))
        }
    }
}

fn parallel_build_jit_block(
    shared: &ParallelVm,
    mut cpu: BlockBuilderCpu,
) -> Result<Option<Vec<JitInstruction>>> {
    let max_block_instructions = jit_max_block_instructions();
    let mut instructions = Vec::with_capacity(max_block_instructions);

    for _ in 0..max_block_instructions {
        let raw16 = match cpu.fetch_u16(shared.model.bus()) {
            Ok(raw) => raw,
            Err(_) => return Ok((!instructions.is_empty()).then_some(instructions)),
        };

        let (insn, insn_bytes) = if (raw16 & 0x3) != 0x3 {
            match decode_compressed(raw16) {
                Ok(insn) => (insn, 2u8),
                Err(_) => return Ok((!instructions.is_empty()).then_some(instructions)),
            }
        } else {
            let raw = match cpu.fetch_u32(shared.model.bus()) {
                Ok(raw) => raw,
                Err(_) => return Ok((!instructions.is_empty()).then_some(instructions)),
            };
            match decode(raw) {
                Ok(insn) => (insn, 4u8),
                Err(_) => return Ok((!instructions.is_empty()).then_some(instructions)),
            }
        };

        if is_parallel_vm_managed_instruction(insn) {
            return Ok((!instructions.is_empty()).then_some(instructions));
        }

        instructions.push(JitInstruction {
            decoded: insn,
            instruction_bytes: insn_bytes,
        });
        if let Some(next_pc) = traced_successor_pc(cpu.pc, insn, insn_bytes) {
            cpu.pc = next_pc;
            continue;
        }
        if ends_jit_block(insn) {
            break;
        }
        cpu.pc = cpu.pc.wrapping_add(insn_bytes as u64);
    }

    Ok((!instructions.is_empty()).then_some(instructions))
}

fn take_pending_interrupt(cpu: &mut Cpu, hart_id: usize, trace: bool) -> bool {
    let Some(trap) = cpu.pending_interrupt() else {
        return false;
    };
    cpu.csr.cycle = cpu.csr.cycle.wrapping_add(1);
    if trace {
        eprintln!(
            "rvx: async interrupt hart={hart_id} pc=0x{:016x} cause=0x{:016x}",
            cpu.pc,
            trap.cause(),
        );
    }
    cpu.take_trap(trap);
    true
}

fn block_cache_set_index(key: BlockKey) -> usize {
    let mixed = key.pc
        ^ key.satp.rotate_left(13)
        ^ ((key.privilege as u64) << 3)
        ^ ((key.data_privilege as u64) << 7)
        ^ key.mstatus_vm.rotate_left(29);
    (mixed as usize) & (JIT_BLOCK_CACHE_SETS - 1)
}

fn lookup_hart_block_cache(
    cache: &mut [[Option<BlockCacheEntry>; JIT_BLOCK_CACHE_WAYS]; JIT_BLOCK_CACHE_SETS],
    key: BlockKey,
) -> (Option<BlockCacheEntry>, bool) {
    let cache_set = &mut cache[block_cache_set_index(key)];
    let mut hit_index = None;
    let mut occupied = false;
    for (index, slot) in cache_set.iter().enumerate() {
        if let Some(entry) = slot {
            occupied = true;
            if entry.key() == key {
                hit_index = Some(index);
                break;
            }
        }
    }
    if let Some(index) = hit_index {
        if index != 0 {
            cache_set.swap(0, index);
        }
        (cache_set[0], occupied)
    } else {
        (None, occupied)
    }
}

fn insert_hart_block_cache(
    cache: &mut [[Option<BlockCacheEntry>; JIT_BLOCK_CACHE_WAYS]; JIT_BLOCK_CACHE_SETS],
    entry: BlockCacheEntry,
) {
    let cache_set = &mut cache[block_cache_set_index(entry.key())];
    for index in (1..JIT_BLOCK_CACHE_WAYS).rev() {
        cache_set[index] = cache_set[index - 1];
    }
    cache_set[0] = Some(entry);
}

fn parallel_step_hart_interpreter(shared: &ParallelVm, hart_id: usize) -> Result<ParallelStep> {
    let mut hart = shared.harts[hart_id].lock().unwrap();
    hart.cpu.csr.cycle = hart.cpu.csr.cycle.wrapping_add(1);
    let raw16 = match hart.cpu.fetch_u16(shared.model.bus()) {
        Ok(raw) => raw,
        Err(trap) => {
            hart.cpu.take_trap(trap);
            return Ok(ParallelStep {
                progress: true,
                retired_delta: 0,
            });
        }
    };

    let (insn, insn_bytes, illegal_tval) = if (raw16 & 0x3) != 0x3 {
        match decode_compressed(raw16) {
            Ok(insn) => (insn, 2u8, raw16 as u64),
            Err(_) => {
                if shared.config.trace {
                    eprintln!(
                        "rvx: illegal decode hart={hart_id} pc=0x{:016x} raw16=0x{:04x}",
                        hart.cpu.pc, raw16,
                    );
                }
                hart.cpu
                    .take_trap(Trap::exception(Exception::IllegalInstruction, raw16 as u64));
                return Ok(ParallelStep {
                    progress: true,
                    retired_delta: 0,
                });
            }
        }
    } else {
        let raw = match hart.cpu.fetch_u32(shared.model.bus()) {
            Ok(raw) => raw,
            Err(trap) => {
                hart.cpu.take_trap(trap);
                return Ok(ParallelStep {
                    progress: true,
                    retired_delta: 0,
                });
            }
        };
        match decode(raw) {
            Ok(insn) => (insn, 4u8, raw as u64),
            Err(_) => {
                if shared.config.trace {
                    eprintln!(
                        "rvx: illegal decode hart={hart_id} pc=0x{:016x} raw32=0x{:08x}",
                        hart.cpu.pc, raw,
                    );
                }
                hart.cpu
                    .take_trap(Trap::exception(Exception::IllegalInstruction, raw as u64));
                return Ok(ParallelStep {
                    progress: true,
                    retired_delta: 0,
                });
            }
        }
    };

    match insn {
        DecodedInstruction::Ecall => {
            let privilege = hart.cpu.privilege;
            if privilege == PrivilegeLevel::Supervisor {
                let ext = hart.cpu.read_x(17);
                let fid = hart.cpu.read_x(16);
                let args = [
                    hart.cpu.read_x(10),
                    hart.cpu.read_x(11),
                    hart.cpu.read_x(12),
                    hart.cpu.read_x(13),
                    hart.cpu.read_x(14),
                    hart.cpu.read_x(15),
                ];
                drop(hart);
                let (error, value) =
                    parallel_handle_supervisor_ecall(shared, hart_id, ext, fid, args)?;
                let mut hart = shared.harts[hart_id].lock().unwrap();
                hart.cpu.write_x(10, error as u64);
                hart.cpu.write_x(11, value);
                hart.cpu.pc = hart.cpu.pc.wrapping_add(4);
                hart.cpu.csr.instret = hart.cpu.csr.instret.wrapping_add(1);
            } else {
                let trap = Trap::exception(
                    if privilege == PrivilegeLevel::User {
                        Exception::UserEnvCall
                    } else {
                        Exception::MachineEnvCall
                    },
                    0,
                );
                hart.cpu.take_trap(trap);
                hart.cpu.csr.instret = hart.cpu.csr.instret.wrapping_add(1);
            }
            shared.retired_instructions.fetch_add(1, Ordering::Relaxed);
            return Ok(ParallelStep {
                progress: true,
                retired_delta: 1,
            });
        }
        DecodedInstruction::Sret => {
            if let Err(trap) = hart.cpu.execute_sret() {
                hart.cpu.take_trap(trap);
            }
            hart.cpu.csr.instret = hart.cpu.csr.instret.wrapping_add(1);
            shared.retired_instructions.fetch_add(1, Ordering::Relaxed);
            return Ok(ParallelStep {
                progress: true,
                retired_delta: 1,
            });
        }
        DecodedInstruction::Mret => {
            if let Err(trap) = hart.cpu.execute_mret() {
                hart.cpu.take_trap(trap);
            }
            hart.cpu.csr.instret = hart.cpu.csr.instret.wrapping_add(1);
            shared.retired_instructions.fetch_add(1, Ordering::Relaxed);
            return Ok(ParallelStep {
                progress: true,
                retired_delta: 1,
            });
        }
        DecodedInstruction::SfenceVma { rs1, rs2 } => {
            let addr = (rs1 != 0).then(|| hart.cpu.read_x(rs1));
            let asid = (rs2 != 0).then(|| hart.cpu.read_x(rs2) as u16);
            hart.cpu.mmu.flush_vma(addr, asid);
            hart.cpu.pc = hart.cpu.pc.wrapping_add(insn_bytes as u64);
            hart.cpu.csr.instret = hart.cpu.csr.instret.wrapping_add(1);
            shared.retired_instructions.fetch_add(1, Ordering::Relaxed);
            return Ok(ParallelStep {
                progress: true,
                retired_delta: 1,
            });
        }
        DecodedInstruction::Wfi => {
            hart.wfi = true;
            hart.cpu.pc = hart.cpu.pc.wrapping_add(insn_bytes as u64);
            hart.cpu.csr.instret = hart.cpu.csr.instret.wrapping_add(1);
            shared.retired_instructions.fetch_add(1, Ordering::Relaxed);
            return Ok(ParallelStep {
                progress: false,
                retired_delta: 1,
            });
        }
        _ => {}
    }

    let result = hart.cpu.execute(insn, insn_bytes, shared.model.bus());
    match result {
        Ok(_) => {
            if instruction_may_write_memory(insn) {
                let _ = shared.model.bus().invalidate_reservations();
            }
            shared.retired_instructions.fetch_add(1, Ordering::Relaxed);
            Ok(ParallelStep {
                progress: true,
                retired_delta: 1,
            })
        }
        Err(trap) => {
            if shared.config.trace {
                eprintln!(
                    "rvx: trap hart={hart_id} pc=0x{:016x} cause=0x{:016x} tval=0x{:016x}",
                    hart.cpu.pc,
                    trap.cause(),
                    if trap.tval == 0 {
                        illegal_tval
                    } else {
                        trap.tval
                    },
                );
            }
            hart.cpu.take_trap(trap);
            Ok(ParallelStep {
                progress: true,
                retired_delta: 0,
            })
        }
    }
}

fn parallel_execute_vm_managed_entry(
    shared: &ParallelVm,
    hart_id: usize,
    entry: UnjittableBlock,
) -> Result<ParallelStep> {
    let mut hart = shared.harts[hart_id].lock().unwrap();
    hart.cpu.csr.cycle = hart.cpu.csr.cycle.wrapping_add(1);

    match entry.decoded {
        DecodedInstruction::Ecall => {
            shared.vm_managed_total.fetch_add(1, Ordering::Relaxed);
            shared.vm_managed_ecall.fetch_add(1, Ordering::Relaxed);
            let privilege = hart.cpu.privilege;
            if privilege == PrivilegeLevel::Supervisor {
                let ext = hart.cpu.read_x(17);
                let fid = hart.cpu.read_x(16);
                let args = [
                    hart.cpu.read_x(10),
                    hart.cpu.read_x(11),
                    hart.cpu.read_x(12),
                    hart.cpu.read_x(13),
                    hart.cpu.read_x(14),
                    hart.cpu.read_x(15),
                ];
                drop(hart);
                let (error, value) =
                    parallel_handle_supervisor_ecall(shared, hart_id, ext, fid, args)?;
                let mut hart = shared.harts[hart_id].lock().unwrap();
                hart.cpu.write_x(10, error as u64);
                hart.cpu.write_x(11, value);
                hart.cpu.pc = hart.cpu.pc.wrapping_add(4);
                hart.cpu.csr.instret = hart.cpu.csr.instret.wrapping_add(1);
            } else {
                let trap = Trap::exception(
                    if privilege == PrivilegeLevel::User {
                        Exception::UserEnvCall
                    } else {
                        Exception::MachineEnvCall
                    },
                    0,
                );
                hart.cpu.take_trap(trap);
                hart.cpu.csr.instret = hart.cpu.csr.instret.wrapping_add(1);
            }
            shared.retired_instructions.fetch_add(1, Ordering::Relaxed);
            Ok(ParallelStep {
                progress: true,
                retired_delta: 1,
            })
        }
        DecodedInstruction::Sret => {
            shared.vm_managed_total.fetch_add(1, Ordering::Relaxed);
            shared.vm_managed_sret.fetch_add(1, Ordering::Relaxed);
            if let Err(trap) = hart.cpu.execute_sret() {
                hart.cpu.take_trap(trap);
            }
            hart.cpu.csr.instret = hart.cpu.csr.instret.wrapping_add(1);
            shared.retired_instructions.fetch_add(1, Ordering::Relaxed);
            Ok(ParallelStep {
                progress: true,
                retired_delta: 1,
            })
        }
        DecodedInstruction::Mret => {
            shared.vm_managed_total.fetch_add(1, Ordering::Relaxed);
            shared.vm_managed_mret.fetch_add(1, Ordering::Relaxed);
            if let Err(trap) = hart.cpu.execute_mret() {
                hart.cpu.take_trap(trap);
            }
            hart.cpu.csr.instret = hart.cpu.csr.instret.wrapping_add(1);
            shared.retired_instructions.fetch_add(1, Ordering::Relaxed);
            Ok(ParallelStep {
                progress: true,
                retired_delta: 1,
            })
        }
        DecodedInstruction::SfenceVma { rs1, rs2 } => {
            shared.vm_managed_total.fetch_add(1, Ordering::Relaxed);
            shared
                .vm_managed_sfence_vma
                .fetch_add(1, Ordering::Relaxed);
            let addr = (rs1 != 0).then(|| hart.cpu.read_x(rs1));
            let asid = (rs2 != 0).then(|| hart.cpu.read_x(rs2) as u16);
            record_parallel_vm_managed_sfence_shape(shared, addr, asid);
            hart.cpu.mmu.flush_vma(addr, asid);
            hart.cpu.pc = hart.cpu.pc.wrapping_add(entry.instruction_bytes as u64);
            hart.cpu.csr.instret = hart.cpu.csr.instret.wrapping_add(1);
            shared.retired_instructions.fetch_add(1, Ordering::Relaxed);
            Ok(ParallelStep {
                progress: true,
                retired_delta: 1,
            })
        }
        DecodedInstruction::Wfi => {
            shared.vm_managed_total.fetch_add(1, Ordering::Relaxed);
            shared.vm_managed_wfi.fetch_add(1, Ordering::Relaxed);
            hart.wfi = true;
            hart.cpu.pc = hart.cpu.pc.wrapping_add(entry.instruction_bytes as u64);
            hart.cpu.csr.instret = hart.cpu.csr.instret.wrapping_add(1);
            shared.retired_instructions.fetch_add(1, Ordering::Relaxed);
            Ok(ParallelStep {
                progress: false,
                retired_delta: 1,
            })
        }
        DecodedInstruction::Fence => {
            shared.vm_managed_total.fetch_add(1, Ordering::Relaxed);
            shared.vm_managed_fence.fetch_add(1, Ordering::Relaxed);
            shared.model.bus().memory_barrier();
            hart.cpu.pc = hart.cpu.pc.wrapping_add(entry.instruction_bytes as u64);
            hart.cpu.csr.instret = hart.cpu.csr.instret.wrapping_add(1);
            shared.retired_instructions.fetch_add(1, Ordering::Relaxed);
            Ok(ParallelStep {
                progress: true,
                retired_delta: 1,
            })
        }
        DecodedInstruction::FenceI => {
            shared.vm_managed_total.fetch_add(1, Ordering::Relaxed);
            shared.vm_managed_fence_i.fetch_add(1, Ordering::Relaxed);
            shared.model.bus().memory_barrier();
            hart.cpu.pc = hart.cpu.pc.wrapping_add(entry.instruction_bytes as u64);
            hart.cpu.csr.instret = hart.cpu.csr.instret.wrapping_add(1);
            shared.retired_instructions.fetch_add(1, Ordering::Relaxed);
            Ok(ParallelStep {
                progress: true,
                retired_delta: 1,
            })
        }
        _ => unreachable!("cached unjittable block must be vm-managed"),
    }
}

fn parallel_refresh_hart_interrupts(
    shared: &ParallelVm,
    hart_id: usize,
    now_ticks: u64,
    hart: &mut HartState,
) {
    hart.cpu.pending_mip &= !((1 << IRQ_SSIP) | (1 << IRQ_STIP) | (1 << IRQ_SEIP));
    if hart.software_interrupt {
        hart.cpu.pending_mip |= 1 << IRQ_SSIP;
    }
    if hart
        .timer_deadline
        .is_some_and(|deadline| now_ticks >= deadline)
    {
        hart.cpu.pending_mip |= 1 << IRQ_STIP;
    }
    if shared.model.supervisor_external_pending(hart_id) {
        hart.cpu.pending_mip |= 1 << IRQ_SEIP;
    }
}

fn parallel_sync_hart_interrupt_sources(
    shared: &ParallelVm,
    hart_id: usize,
    observed_software_interrupt: bool,
) {
    let mut hart = shared.harts[hart_id].lock().unwrap();
    if observed_software_interrupt && (hart.cpu.csr.mip & (1 << IRQ_SSIP)) == 0 {
        hart.software_interrupt = false;
    }
}

fn parallel_handle_supervisor_ecall(
    shared: &ParallelVm,
    hart_id: usize,
    ext: u64,
    fid: u64,
    args: [u64; 6],
) -> Result<(i64, u64)> {
    match ext {
        SBI_EXT_BASE => Ok(sbi_base(fid, args)),
        SBI_EXT_TIME => Ok(parallel_sbi_time(shared, hart_id, fid, args)),
        SBI_EXT_IPI => Ok(parallel_sbi_ipi(shared, fid, args)),
        SBI_EXT_RFENCE => Ok(parallel_sbi_rfence(shared, fid, args)),
        SBI_EXT_HSM => Ok(parallel_sbi_hsm(shared, hart_id, fid, args)),
        SBI_EXT_SRST => Ok(parallel_sbi_srst(shared, fid, args)),
        SBI_EXT_DBCN => parallel_sbi_dbcn(shared, fid, args),
        SBI_EXT_0_1_SET_TIMER
        | SBI_EXT_0_1_CONSOLE_PUTCHAR
        | SBI_EXT_0_1_CONSOLE_GETCHAR
        | SBI_EXT_0_1_CLEAR_IPI
        | SBI_EXT_0_1_SEND_IPI
        | SBI_EXT_0_1_REMOTE_FENCE_I
        | SBI_EXT_0_1_REMOTE_SFENCE_VMA
        | SBI_EXT_0_1_REMOTE_SFENCE_VMA_ASID
        | SBI_EXT_0_1_SHUTDOWN => parallel_sbi_legacy(shared, hart_id, ext, args),
        _ => Ok((SBI_ERR_NOT_SUPPORTED, 0)),
    }
}

fn sbi_base(fid: u64, args: [u64; 6]) -> (i64, u64) {
    match fid {
        SBI_EXT_BASE_GET_SPEC_VERSION => (0, SBI_SPEC_V2),
        SBI_EXT_BASE_GET_IMP_ID => (0, SBI_IMPL_ID_RVX),
        SBI_EXT_BASE_GET_IMP_VERSION => (0, SBI_IMPL_VERSION_RVX),
        SBI_EXT_BASE_PROBE_EXT => (
            0,
            u64::from(matches!(
                args[0],
                SBI_EXT_BASE
                    | SBI_EXT_TIME
                    | SBI_EXT_IPI
                    | SBI_EXT_RFENCE
                    | SBI_EXT_HSM
                    | SBI_EXT_SRST
                    | SBI_EXT_DBCN
            )),
        ),
        SBI_EXT_BASE_GET_MVENDORID | SBI_EXT_BASE_GET_MARCHID | SBI_EXT_BASE_GET_MIMPID => (0, 0),
        _ => (SBI_ERR_NOT_SUPPORTED, 0),
    }
}

fn parallel_sbi_time(shared: &ParallelVm, hart_id: usize, fid: u64, args: [u64; 6]) -> (i64, u64) {
    if fid != SBI_EXT_TIME_SET_TIMER {
        return (SBI_ERR_NOT_SUPPORTED, 0);
    }
    let mut hart = shared.harts[hart_id].lock().unwrap();
    hart.timer_deadline = Some(args[0]);
    hart.cpu.pending_mip &= !(1 << IRQ_STIP);
    (0, 0)
}

fn parallel_sbi_ipi(shared: &ParallelVm, fid: u64, args: [u64; 6]) -> (i64, u64) {
    if fid != SBI_EXT_IPI_SEND_IPI {
        return (SBI_ERR_NOT_SUPPORTED, 0);
    }
    if shared.config.trace {
        eprintln!("rvx: sbi ipi send mask=0x{:016x} base={}", args[0], args[1]);
    }
    parallel_apply_hart_mask(shared, args[0], args[1], |hart| {
        hart.software_interrupt = true
    });
    (0, 0)
}

fn parallel_sbi_rfence(shared: &ParallelVm, fid: u64, args: [u64; 6]) -> (i64, u64) {
    match fid {
        SBI_EXT_RFENCE_REMOTE_FENCE_I => (0, 0),
        SBI_EXT_RFENCE_REMOTE_SFENCE_VMA => {
            parallel_apply_hart_mask(shared, args[0], args[1], |hart| {
                hart.cpu.mmu.flush_range(args[2], args[3], None)
            });
            (0, 0)
        }
        SBI_EXT_RFENCE_REMOTE_SFENCE_VMA_ASID => {
            let asid = args[4] as u16;
            parallel_apply_hart_mask(shared, args[0], args[1], |hart| {
                hart.cpu.mmu.flush_range(args[2], args[3], Some(asid))
            });
            (0, 0)
        }
        _ => (SBI_ERR_NOT_SUPPORTED, 0),
    }
}

fn parallel_sbi_hsm(
    shared: &ParallelVm,
    current_hart: usize,
    fid: u64,
    args: [u64; 6],
) -> (i64, u64) {
    match fid {
        SBI_EXT_HSM_HART_START => {
            let hart_id = args[0] as usize;
            let Some(target) = shared.harts.get(hart_id) else {
                return (SBI_ERR_INVALID_PARAM, 0);
            };
            let mut hart = target.lock().unwrap();
            if hart.started {
                if shared.config.trace {
                    eprintln!("rvx: sbi hsm start hart={hart_id} ignored: already started");
                }
                return (SBI_ERR_ALREADY_STARTED, 0);
            }
            if shared.config.trace {
                eprintln!(
                    "rvx: sbi hsm start hart={hart_id} pc=0x{:016x} opaque=0x{:016x}",
                    args[1], args[2],
                );
            }
            hart.started = true;
            hart.wfi = false;
            hart.cpu.pc = args[1];
            hart.cpu.privilege = PrivilegeLevel::Supervisor;
            hart.cpu.write_x(10, hart_id as u64);
            hart.cpu.write_x(11, args[2]);
            hart.cpu.pending_mip = 0;
            hart.cpu.mmu.flush();
            (0, 0)
        }
        SBI_EXT_HSM_HART_STOP => {
            if shared.config.trace {
                eprintln!("rvx: sbi hsm stop hart={current_hart}");
            }
            shared.harts[current_hart].lock().unwrap().started = false;
            (0, 0)
        }
        SBI_EXT_HSM_HART_STATUS => {
            let hart_id = args[0] as usize;
            let Some(target) = shared.harts.get(hart_id) else {
                return (SBI_ERR_INVALID_PARAM, 0);
            };
            let hart = target.lock().unwrap();
            if shared.config.trace {
                eprintln!(
                    "rvx: sbi hsm status hart={hart_id} -> {}",
                    if hart.started { "started" } else { "stopped" }
                );
            }
            (
                0,
                if hart.started {
                    SBI_HSM_STATE_STARTED
                } else {
                    SBI_HSM_STATE_STOPPED
                },
            )
        }
        _ => (SBI_ERR_NOT_SUPPORTED, 0),
    }
}

fn parallel_sbi_srst(shared: &ParallelVm, fid: u64, args: [u64; 6]) -> (i64, u64) {
    if fid != SBI_EXT_SRST_RESET {
        return (SBI_ERR_NOT_SUPPORTED, 0);
    }
    *shared.pending_exit.lock().unwrap() = Some(match args[0] {
        SBI_SRST_RESET_TYPE_SHUTDOWN => VmExit::PowerOff,
        SBI_SRST_RESET_TYPE_COLD_REBOOT | SBI_SRST_RESET_TYPE_WARM_REBOOT => VmExit::Reset,
        _ => VmExit::Reset,
    });
    (0, 0)
}

fn parallel_sbi_dbcn(shared: &ParallelVm, fid: u64, args: [u64; 6]) -> Result<(i64, u64)> {
    match fid {
        SBI_EXT_DBCN_CONSOLE_WRITE => {
            let count = args[0] as usize;
            let base = args[1];
            let mut buf = vec![0u8; count];
            for (index, byte) in buf.iter_mut().enumerate() {
                *byte = shared
                    .model
                    .bus()
                    .read_u8(base + index as u64)
                    .map_err(|_| {
                        anyhow::anyhow!("dbcn read fault at 0x{:016x}", base + index as u64)
                    })?;
            }
            write_model_console(&shared.model, &buf);
            Ok((0, count as u64))
        }
        SBI_EXT_DBCN_CONSOLE_WRITE_BYTE => {
            write_model_console(&shared.model, &[args[0] as u8]);
            Ok((0, 0))
        }
        SBI_EXT_DBCN_CONSOLE_READ => Ok((0, 0)),
        _ => Ok((SBI_ERR_NOT_SUPPORTED, 0)),
    }
}

fn parallel_sbi_legacy(
    shared: &ParallelVm,
    hart_id: usize,
    ext: u64,
    args: [u64; 6],
) -> Result<(i64, u64)> {
    let result = match ext {
        SBI_EXT_0_1_SET_TIMER => parallel_sbi_time(shared, hart_id, SBI_EXT_TIME_SET_TIMER, args),
        SBI_EXT_0_1_CONSOLE_PUTCHAR => {
            write_model_console(&shared.model, &[args[0] as u8]);
            (0, 0)
        }
        SBI_EXT_0_1_CONSOLE_GETCHAR => (0, u64::MAX),
        SBI_EXT_0_1_CLEAR_IPI => {
            shared.harts[hart_id].lock().unwrap().software_interrupt = false;
            (0, 0)
        }
        SBI_EXT_0_1_SEND_IPI => (0, 0),
        SBI_EXT_0_1_REMOTE_FENCE_I
        | SBI_EXT_0_1_REMOTE_SFENCE_VMA
        | SBI_EXT_0_1_REMOTE_SFENCE_VMA_ASID => (0, 0),
        SBI_EXT_0_1_SHUTDOWN => {
            *shared.pending_exit.lock().unwrap() = Some(VmExit::PowerOff);
            (0, 0)
        }
        _ => (SBI_ERR_NOT_SUPPORTED, 0),
    };
    Ok(result)
}

fn parallel_apply_hart_mask(
    shared: &ParallelVm,
    hmask: u64,
    hbase: u64,
    mut f: impl FnMut(&mut HartState),
) {
    let base = hbase as usize;
    for bit in 0..64usize {
        if ((hmask >> bit) & 1) == 0 {
            continue;
        }
        if let Some(hart) = shared.harts.get(base + bit) {
            f(&mut hart.lock().unwrap());
        }
    }
}

fn parallel_fast_forward_time(shared: &ParallelVm) -> bool {
    let now_ticks = shared.time_ticks.load(Ordering::Relaxed);
    let mut any_started = false;
    let mut next_deadline: Option<u64> = None;

    for (hart_id, hart_lock) in shared.harts.iter().enumerate() {
        let mut hart = hart_lock.lock().unwrap();
        if !hart.started {
            continue;
        }
        any_started = true;
        parallel_refresh_hart_interrupts(shared, hart_id, now_ticks, &mut hart);
        hart.cpu.csr.time = now_ticks;
        if hart.cpu.pending_interrupt().is_some() || !hart.wfi {
            return false;
        }
        if let Some(deadline) = hart.timer_deadline.filter(|deadline| *deadline > now_ticks) {
            next_deadline = Some(match next_deadline {
                Some(current) => current.min(deadline),
                None => deadline,
            });
        }
    }

    let Some(next_deadline) = next_deadline else {
        return false;
    };
    if !any_started || next_deadline <= now_ticks {
        return false;
    }

    let mut current = now_ticks;
    loop {
        match shared.time_ticks.compare_exchange_weak(
            current,
            next_deadline,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => return true,
            Err(observed) if observed >= next_deadline => return true,
            Err(observed) => current = observed,
        }
    }
}

fn privilege_name(privilege: PrivilegeLevel) -> &'static str {
    match privilege {
        PrivilegeLevel::User => "user",
        PrivilegeLevel::Supervisor => "supervisor",
        PrivilegeLevel::Machine => "machine",
    }
}

fn record_vm_managed_sfence_shape(report: &mut VmManagedReport, addr: Option<u64>, asid: Option<u16>) {
    match (addr, asid) {
        (None, None) => report.sfence_vma_global = report.sfence_vma_global.wrapping_add(1),
        (None, Some(_)) => report.sfence_vma_asid = report.sfence_vma_asid.wrapping_add(1),
        (Some(_), None) => report.sfence_vma_page = report.sfence_vma_page.wrapping_add(1),
        (Some(_), Some(_)) => {
            report.sfence_vma_page_asid = report.sfence_vma_page_asid.wrapping_add(1)
        }
    }
}

fn record_parallel_vm_managed_sfence_shape(
    shared: &ParallelVm,
    addr: Option<u64>,
    asid: Option<u16>,
) {
    match (addr, asid) {
        (None, None) => {
            shared
                .vm_managed_sfence_vma_global
                .fetch_add(1, Ordering::Relaxed);
        }
        (None, Some(_)) => {
            shared
                .vm_managed_sfence_vma_asid
                .fetch_add(1, Ordering::Relaxed);
        }
        (Some(_), None) => {
            shared
                .vm_managed_sfence_vma_page
                .fetch_add(1, Ordering::Relaxed);
        }
        (Some(_), Some(_)) => {
            shared
                .vm_managed_sfence_vma_page_asid
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn mmu_stats_report(stats: MmuStats) -> MmuStatsReport {
    MmuStatsReport {
        tlb_hits: stats.tlb_hits,
        tlb_misses: stats.tlb_misses,
        page_walks: stats.page_walks,
        flushes: stats.flushes,
        jit_cache_fills: stats.jit_cache_fills,
        translated_load_hits: stats.translated_load_hits,
        translated_load_misses: stats.translated_load_misses,
        translated_store_hits: stats.translated_store_hits,
        translated_store_misses: stats.translated_store_misses,
    }
}

fn increment_block_terminator(
    report: &mut BlockTerminatorReport,
    terminator: BlockTerminatorKind,
) {
    match terminator {
        BlockTerminatorKind::Fallthrough => {
            report.fallthrough = report.fallthrough.wrapping_add(1)
        }
        BlockTerminatorKind::Jal => report.jal = report.jal.wrapping_add(1),
        BlockTerminatorKind::Jalr => report.jalr = report.jalr.wrapping_add(1),
        BlockTerminatorKind::Branch => report.branch = report.branch.wrapping_add(1),
        BlockTerminatorKind::Csr => report.csr = report.csr.wrapping_add(1),
        BlockTerminatorKind::Atomic => report.atomic = report.atomic.wrapping_add(1),
        BlockTerminatorKind::Ebreak => report.ebreak = report.ebreak.wrapping_add(1),
        BlockTerminatorKind::Other => report.other = report.other.wrapping_add(1),
    }
}

fn is_vm_managed_instruction(insn: DecodedInstruction) -> bool {
    matches!(
        insn,
        DecodedInstruction::Ecall
            | DecodedInstruction::Wfi
            | DecodedInstruction::Mret
            | DecodedInstruction::Sret
            | DecodedInstruction::SfenceVma { .. }
    )
}

fn is_parallel_vm_managed_instruction(insn: DecodedInstruction) -> bool {
    matches!(
        insn,
        DecodedInstruction::Ecall
            | DecodedInstruction::Wfi
            | DecodedInstruction::Mret
            | DecodedInstruction::Sret
            | DecodedInstruction::SfenceVma { .. }
    )
}

fn current_vm_managed_entry(cpu: &Cpu, bus: &Bus) -> Option<UnjittableBlock> {
    let mut cpu = cpu.block_builder();
    let raw16 = match cpu.fetch_u16(bus) {
        Ok(raw) => raw,
        Err(_) => return None,
    };

    let (decoded, instruction_bytes) = if (raw16 & 0x3) != 0x3 {
        match decode_compressed(raw16) {
            Ok(insn) => (insn, 2u8),
            Err(_) => return None,
        }
    } else {
        let raw = match cpu.fetch_u32(bus) {
            Ok(raw) => raw,
            Err(_) => return None,
        };
        match decode(raw) {
            Ok(insn) => (insn, 4u8),
            Err(_) => return None,
        }
    };

    is_vm_managed_instruction(decoded).then_some(UnjittableBlock {
        decoded,
        instruction_bytes,
    })
}

fn ends_jit_block(insn: DecodedInstruction) -> bool {
    matches!(
        insn,
        DecodedInstruction::Jal { .. }
            | DecodedInstruction::Jalr { .. }
            | DecodedInstruction::Branch { .. }
            | DecodedInstruction::Ebreak
            | DecodedInstruction::FenceI
            | DecodedInstruction::Atomic { .. }
            | DecodedInstruction::Csrrw { .. }
            | DecodedInstruction::Csrrs { .. }
            | DecodedInstruction::Csrrc { .. }
            | DecodedInstruction::Csrrwi { .. }
            | DecodedInstruction::Csrrsi { .. }
            | DecodedInstruction::Csrrci { .. }
    )
}

fn traced_successor_pc(pc: u64, insn: DecodedInstruction, instruction_bytes: u8) -> Option<u64> {
    match insn {
        // Follow direct jumps that do not write a return address; this avoids tracing through
        // normal function calls while still fusing compiler-generated jumps and tail-calls.
        DecodedInstruction::Jal { rd: 0, imm } => Some(pc.wrapping_add_signed(imm)),
        // Static branch prediction keeps trace building cheap while still fusing hot loops.
        DecodedInstruction::Branch { imm, .. } => Some(if imm < 0 {
            pc.wrapping_add_signed(imm)
        } else {
            pc.wrapping_add(instruction_bytes as u64)
        }),
        _ => None,
    }
}

fn instruction_may_write_memory(insn: DecodedInstruction) -> bool {
    match insn {
        DecodedInstruction::Store { .. } | DecodedInstruction::FloatStore { .. } => true,
        DecodedInstruction::Atomic { op, .. } => {
            !matches!(op, rvx_riscv::AtomicOp::LrW | rvx_riscv::AtomicOp::LrD)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::RAM_BASE;

    fn test_machine() -> VirtMachine {
        VirtMachine::new(
            VmConfig {
                hart_count: 1,
                ram_bytes: 64 * 1024 * 1024,
                nographic: false,
                trace: false,
                code_buffer_bytes: 0,
                time_limit_ms: None,
            },
            ArtifactBundle {
                firmware: Some(PathBuf::from("unused-fw")),
                kernel: PathBuf::from("unused-kernel"),
                initrd: None,
                append: None,
                drive: None,
            },
        )
        .expect("failed to build test machine")
    }

    #[test]
    fn single_threaded_wfi_parks_hart_until_interrupt() {
        let mut vm = test_machine();
        vm.model
            .bus()
            .load(RAM_BASE, &0x1050_0073u32.to_le_bytes())
            .expect("failed to load wfi");
        vm.harts[0].cpu.pc = RAM_BASE;
        vm.harts[0].cpu.privilege = PrivilegeLevel::Supervisor;

        let mut jit = JitEngine::new_with_options(true, false).expect("failed to create jit");

        let first_progress = vm.step_hart(0, 0, &mut jit).expect("first step failed");
        assert!(!first_progress);
        assert!(vm.harts[0].wfi);
        assert_eq!(vm.harts[0].cpu.pc, RAM_BASE + 4);
        assert_eq!(vm.harts[0].cpu.csr.instret, 1);

        let parked_pc = vm.harts[0].cpu.pc;
        let parked_instret = vm.harts[0].cpu.csr.instret;
        let second_progress = vm.step_hart(0, 0, &mut jit).expect("second step failed");
        assert!(!second_progress);
        assert!(vm.harts[0].wfi);
        assert_eq!(vm.harts[0].cpu.pc, parked_pc);
        assert_eq!(vm.harts[0].cpu.csr.instret, parked_instret);
    }

    #[test]
    fn traced_successor_pc_follows_static_jumps_and_predicted_branches() {
        assert_eq!(
            super::traced_successor_pc(
                0x2000,
                DecodedInstruction::Branch {
                    kind: rvx_riscv::BranchKind::Eq,
                    rs1: 1,
                    rs2: 2,
                    imm: -16,
                },
                4,
            ),
            Some(0x1ff0)
        );
        assert_eq!(
            super::traced_successor_pc(
                0x3000,
                DecodedInstruction::Branch {
                    kind: rvx_riscv::BranchKind::Eq,
                    rs1: 1,
                    rs2: 2,
                    imm: 28,
                },
                4,
            ),
            Some(0x3004)
        );
        assert_eq!(
            super::traced_successor_pc(
                0x4000,
                DecodedInstruction::Jal { rd: 0, imm: 12 },
                4,
            ),
            Some(0x400c)
        );
        assert_eq!(
            super::traced_successor_pc(
                0x5000,
                DecodedInstruction::Jal { rd: 1, imm: 12 },
                4,
            ),
            None
        );
        assert_eq!(
            super::traced_successor_pc(
                0x6000,
                DecodedInstruction::Jalr {
                    rd: 1,
                    rs1: 2,
                    imm: 12,
                },
                4,
            ),
            None
        );
    }

    #[test]
    fn single_threaded_jit_keeps_fence_in_compiled_blocks() {
        assert!(!super::is_vm_managed_instruction(DecodedInstruction::Fence));
        assert!(!super::is_vm_managed_instruction(DecodedInstruction::FenceI));
        assert!(!super::ends_jit_block(DecodedInstruction::Fence));
        assert!(super::ends_jit_block(DecodedInstruction::FenceI));
    }
}
