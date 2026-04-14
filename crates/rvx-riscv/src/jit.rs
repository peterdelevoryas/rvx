use std::cell::Cell;
use std::ffi::c_void;
use std::mem::{self, offset_of};
use std::ptr;
use std::sync::OnceLock;

use anyhow::Result;
use cranelift::codegen::ir::{BlockArg, FuncRef, MemFlags};
use cranelift::codegen::settings::{self, Configurable};
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};
use rustc_hash::FxHashMap;
use rvx_core::Bus;

use crate::csr::{MSTATUS_MXR, MSTATUS_SUM};
use crate::mmu::{
    JIT_CACHE_ENTRY_CONTEXT_TAG_OFFSET, JIT_CACHE_ENTRY_GUEST_PAGE_OFFSET,
    JIT_CACHE_ENTRY_HOST_ADDEND_OFFSET, JIT_CACHE_ENTRY_SIZE, JIT_CACHE_SIZE_MASK,
    MMU_JIT_SUPERVISOR_MXR_READ_OFFSET, MMU_JIT_SUPERVISOR_READ_OFFSET,
    MMU_JIT_SUPERVISOR_SUM_MXR_READ_OFFSET, MMU_JIT_SUPERVISOR_SUM_READ_OFFSET,
    MMU_JIT_SUPERVISOR_SUM_WRITE_OFFSET, MMU_JIT_SUPERVISOR_WRITE_OFFSET,
    MMU_JIT_USER_READ_OFFSET, MMU_JIT_USER_WRITE_OFFSET, MMU_STATS_TRANSLATED_LOAD_HITS_OFFSET,
    MMU_STATS_TRANSLATED_LOAD_MISSES_OFFSET, MMU_STATS_TRANSLATED_STORE_HITS_OFFSET,
    MMU_STATS_TRANSLATED_STORE_MISSES_OFFSET, PAGE_SIZE, SATP_MODE_BARE, SATP_MODE_SHIFT,
};
use crate::{
    BranchKind, Cpu, CsrFile, DecodedInstruction, LoadKind, PrivilegeLevel, StepOutcome,
    StoreKind,
};

mod x64;

const BLOCK_STATUS_CONTINUE: i64 = 0;
const BLOCK_STATUS_TRAP: i64 = 1;
const BLOCK_STATUS_CHAIN: i64 = 2;
pub const DEFAULT_JIT_MAX_BLOCK_INSTRUCTIONS: usize = 32;
const CPU_X_OFFSET: i32 = offset_of!(Cpu, x) as i32;
const CPU_PC_OFFSET: i32 = offset_of!(Cpu, pc) as i32;
const CPU_CSR_OFFSET: i32 = offset_of!(Cpu, csr) as i32;
const CPU_MMU_OFFSET: i32 = offset_of!(Cpu, mmu) as i32;
const CSR_INSTRET_OFFSET: i32 = (CPU_CSR_OFFSET as usize + offset_of!(CsrFile, instret)) as i32;
const TRAP_INFO_CAUSE_OFFSET: i32 = offset_of!(TrapInfo, cause) as i32;
const TRAP_INFO_NEXT_BLOCK_OFFSET: i32 = offset_of!(TrapInfo, next_block) as i32;

type CompiledBlockEntry =
    unsafe extern "C" fn(*mut Cpu, *const Bus, *mut TrapInfo, u64, u64, *mut u8, u64, u64) -> u64;

pub type ChainLookupCallback = unsafe extern "C" fn(*mut c_void, *mut Cpu, *const Bus) -> u64;

thread_local! {
    static CURRENT_GPR_CACHE: Cell<*mut GprCache> = const { Cell::new(ptr::null_mut()) };
    static CURRENT_CHAIN_LOOKUP_CTX: Cell<*mut c_void> = const { Cell::new(ptr::null_mut()) };
    static CURRENT_CHAIN_LOOKUP_CALLBACK: Cell<Option<ChainLookupCallback>> = const { Cell::new(None) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockKey {
    pub pc: u64,
    pub satp: u64,
    pub privilege: PrivilegeLevel,
    pub data_privilege: PrivilegeLevel,
    pub mstatus_vm: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct JitInstruction {
    pub decoded: DecodedInstruction,
    pub instruction_bytes: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnjittableBlock {
    pub decoded: DecodedInstruction,
    pub instruction_bytes: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TrapInfo {
    pub cause: u64,
    pub tval: u64,
    pub next_block: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockStatus {
    Continue,
    Trap,
    Chain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockExecution {
    pub status: BlockStatus,
    pub retired: u32,
}

#[repr(C)]
pub struct CompiledBlock {
    pub key: BlockKey,
    pub instructions: Box<[JitInstruction]>,
    entry: CompiledBlockEntry,
    chain_entry: u64,
    _code_owner: CompiledCodeOwner,
    chain_data: ChainData,
    writes_memory: bool,
    terminator: BlockTerminatorKind,
}

struct GprCache {
    vars: [Variable; 32],
    loaded: [bool; 32],
    dirty: [bool; 32],
}

#[derive(Clone, Copy)]
struct GprCacheState {
    loaded: [bool; 32],
    dirty: [bool; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockTerminatorKind {
    Fallthrough,
    Jal,
    Jalr,
    Branch,
    Csr,
    Atomic,
    Ebreak,
    Other,
}

pub struct ChainLookupGuard {
    previous_ctx: *mut c_void,
    previous_callback: Option<ChainLookupCallback>,
}

struct StaticChainSlot {
    target: BlockKey,
    block: Box<u64>,
}

struct JalrChainCache {
    target_pc: Box<u64>,
    block: Box<u64>,
}

enum ChainData {
    None,
    StaticSingle {
        slot: StaticChainSlot,
    },
    StaticBranch {
        taken_pc: u64,
        taken: StaticChainSlot,
        fallthrough: StaticChainSlot,
    },
    Jalr(JalrChainCache),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChainCompileInfo {
    None,
    StaticSingle { slot_addr: u64 },
    Branch {
        taken_pc: u64,
        taken_slot_addr: u64,
        fallthrough_slot_addr: u64,
    },
    Jalr {
        target_pc_addr: u64,
        block_slot_addr: u64,
    },
}

#[allow(dead_code)]
enum CompiledCodeOwner {
    Module,
    #[cfg(all(target_arch = "x86_64", unix))]
    X64(x64::ExecutableBlock),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JitBackend {
    Cranelift,
    X64Template,
}

pub struct JitEngine {
    module: JITModule,
    ctx: cranelift::codegen::Context,
    builder_ctx: FunctionBuilderContext,
    step_func: FuncId,
    invalidate_reservations_func: FuncId,
    memory_barrier_func: FuncId,
    translated_load_func: FuncId,
    translated_store_func: FuncId,
    blocks: FxHashMap<BlockKey, Box<CompiledBlock>>,
    unjittable_blocks: FxHashMap<BlockKey, UnjittableBlock>,
    pending_chain_links: FxHashMap<BlockKey, Vec<usize>>,
    next_block_id: u64,
    allow_fast_mem: bool,
    block_chaining: bool,
    helper_only: bool,
    collect_mmu_stats: bool,
    preferred_backend: JitBackend,
}

impl BlockKey {
    pub fn for_cpu(cpu: &Cpu) -> Self {
        Self {
            pc: cpu.pc,
            satp: cpu.csr.satp,
            privilege: cpu.privilege,
            data_privilege: cpu.effective_data_privilege(),
            mstatus_vm: cpu.data_mstatus_vm(),
        }
    }
}

pub fn jit_max_block_instructions() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("RVX_JIT_MAX_BLOCK_INSTRUCTIONS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_JIT_MAX_BLOCK_INSTRUCTIONS)
    })
}

fn preferred_backend(block_chaining: bool, helper_only: bool) -> JitBackend {
    if block_chaining || helper_only {
        return JitBackend::Cranelift;
    }
    match std::env::var("RVX_JIT_BACKEND").ok().as_deref() {
        Some("cranelift") => JitBackend::Cranelift,
        Some("x64") | Some("x86_64") => JitBackend::X64Template,
        _ => {
            #[cfg(all(target_arch = "x86_64", unix))]
            {
                JitBackend::X64Template
            }
            #[cfg(not(all(target_arch = "x86_64", unix)))]
            {
                JitBackend::Cranelift
            }
        }
    }
}

impl BlockExecution {
    pub fn from_packed(packed: u64) -> Self {
        let status = match (packed & 0xffff_ffff) as u32 {
            0 => BlockStatus::Continue,
            1 => BlockStatus::Trap,
            2 => BlockStatus::Chain,
            other => panic!("unknown compiled block status {other}"),
        };
        Self {
            status,
            retired: (packed >> 32) as u32,
        }
    }
}

impl CompiledBlock {
    pub fn entry(&self) -> CompiledBlockEntry {
        self.entry
    }

    pub fn chain_token(&self) -> u64 {
        self as *const Self as u64
    }

    pub fn writes_memory(&self) -> bool {
        self.writes_memory
    }

    pub fn chain_entry(&self) -> u64 {
        self.chain_entry
    }

    pub fn terminator(&self) -> BlockTerminatorKind {
        self.terminator
    }
}

impl ChainData {
    fn none() -> Self {
        Self::None
    }

    fn static_single(target: BlockKey) -> Self {
        Self::StaticSingle {
            slot: StaticChainSlot {
                target,
                block: Box::new(0),
            },
        }
    }

    fn static_branch(taken_pc: u64, taken: BlockKey, fallthrough: BlockKey) -> Self {
        Self::StaticBranch {
            taken_pc,
            taken: StaticChainSlot {
                target: taken,
                block: Box::new(0),
            },
            fallthrough: StaticChainSlot {
                target: fallthrough,
                block: Box::new(0),
            },
        }
    }

    fn jalr() -> Self {
        Self::Jalr(JalrChainCache {
            target_pc: Box::new(0),
            block: Box::new(0),
        })
    }

    fn compile_info(&self) -> ChainCompileInfo {
        match self {
            ChainData::None => ChainCompileInfo::None,
            ChainData::StaticSingle { slot } => ChainCompileInfo::StaticSingle {
                slot_addr: slot.block.as_ref() as *const u64 as u64,
            },
            ChainData::StaticBranch {
                taken_pc,
                taken,
                fallthrough,
            } => ChainCompileInfo::Branch {
                taken_pc: *taken_pc,
                taken_slot_addr: taken.block.as_ref() as *const u64 as u64,
                fallthrough_slot_addr: fallthrough.block.as_ref() as *const u64 as u64,
            },
            ChainData::Jalr(cache) => ChainCompileInfo::Jalr {
                target_pc_addr: cache.target_pc.as_ref() as *const u64 as u64,
                block_slot_addr: cache.block.as_ref() as *const u64 as u64,
            },
        }
    }

    fn static_slot_refs(&mut self) -> Vec<(BlockKey, *mut u64)> {
        match self {
            ChainData::None | ChainData::Jalr(_) => Vec::new(),
            ChainData::StaticSingle { slot } => {
                vec![(slot.target, slot.block.as_mut() as *mut u64)]
            }
            ChainData::StaticBranch {
                taken,
                fallthrough,
                ..
            } => vec![
                (taken.target, taken.block.as_mut() as *mut u64),
                (fallthrough.target, fallthrough.block.as_mut() as *mut u64),
            ],
        }
    }

    fn patch_jalr(&mut self, target_pc: u64, block: u64) -> bool {
        let ChainData::Jalr(cache) = self else {
            return false;
        };
        *cache.target_pc = target_pc;
        *cache.block = block;
        true
    }
}

impl Drop for ChainLookupGuard {
    fn drop(&mut self) {
        CURRENT_CHAIN_LOOKUP_CTX.with(|slot| slot.set(self.previous_ctx));
        CURRENT_CHAIN_LOOKUP_CALLBACK.with(|slot| slot.set(self.previous_callback));
    }
}

impl GprCache {
    #[allow(dead_code)]
    fn new(builder: &mut FunctionBuilder) -> Self {
        let vars = std::array::from_fn(|_| builder.declare_var(types::I64));
        Self {
            vars,
            loaded: [false; 32],
            dirty: [false; 32],
        }
    }

    fn load(&mut self, builder: &mut FunctionBuilder, cpu: Value, reg: u8) -> Value {
        if reg == 0 {
            return builder.ins().iconst(types::I64, 0);
        }
        let idx = reg as usize;
        if !self.loaded[idx] {
            let value = builder.ins().load(
                types::I64,
                MemFlags::new(),
                cpu,
                CPU_X_OFFSET + reg as i32 * 8,
            );
            builder.def_var(self.vars[idx], value);
            self.loaded[idx] = true;
        }
        builder.use_var(self.vars[idx])
    }

    fn store(&mut self, builder: &mut FunctionBuilder, reg: u8, value: Value) {
        if reg == 0 {
            return;
        }
        let idx = reg as usize;
        builder.def_var(self.vars[idx], value);
        self.loaded[idx] = true;
        self.dirty[idx] = true;
    }

    fn flush(&mut self, builder: &mut FunctionBuilder, cpu: Value) {
        for reg in 1..32u8 {
            let idx = reg as usize;
            if !(self.loaded[idx] && self.dirty[idx]) {
                continue;
            }
            let value = builder.use_var(self.vars[idx]);
            builder
                .ins()
                .store(MemFlags::new(), value, cpu, CPU_X_OFFSET + reg as i32 * 8);
            self.dirty[idx] = false;
        }
    }

    fn snapshot(&self) -> GprCacheState {
        GprCacheState {
            loaded: self.loaded,
            dirty: self.dirty,
        }
    }

    fn restore(&mut self, state: GprCacheState) {
        self.loaded = state.loaded;
        self.dirty = state.dirty;
    }

    fn invalidate(&mut self) {
        self.loaded = [false; 32];
        self.dirty = [false; 32];
    }
}

pub unsafe fn install_chain_lookup(
    ctx: *mut c_void,
    callback: ChainLookupCallback,
) -> ChainLookupGuard {
    let previous_ctx = CURRENT_CHAIN_LOOKUP_CTX.with(|slot| slot.replace(ctx));
    let previous_callback =
        CURRENT_CHAIN_LOOKUP_CALLBACK.with(|slot| slot.replace(Some(callback)));
    ChainLookupGuard {
        previous_ctx,
        previous_callback,
    }
}

impl JitEngine {
    pub fn new() -> Result<Self> {
        Self::new_with_options(true, std::env::var_os("RVX_JIT_HELPER_ONLY").is_some())
    }

    pub fn new_with_fast_mem(allow_fast_mem: bool) -> Result<Self> {
        Self::new_with_options(
            allow_fast_mem,
            std::env::var_os("RVX_JIT_HELPER_ONLY").is_some(),
        )
    }

    pub fn new_parallel() -> Result<Self> {
        Self::new_with_options_internal(
            false,
            std::env::var_os("RVX_PARALLEL_INLINE_JIT").is_none(),
            false,
        )
    }

    pub fn new_with_options(allow_fast_mem: bool, helper_only: bool) -> Result<Self> {
        Self::new_with_options_internal(
            allow_fast_mem,
            helper_only,
            std::env::var_os("RVX_JIT_BLOCK_CHAINING").is_some(),
        )
    }

    fn new_with_options_internal(
        allow_fast_mem: bool,
        helper_only: bool,
        block_chaining: bool,
    ) -> Result<Self> {
        let mut flag_builder = settings::builder();
        // Faster compile throughput matters more than marginal code quality during guest boot.
        flag_builder.set(
            "regalloc_algorithm",
            &std::env::var("RVX_JIT_REGALLOC").unwrap_or_else(|_| "single_pass".to_string()),
        )?;
        if std::env::var_os("RVX_JIT_ENABLE_VERIFIER").is_none() {
            flag_builder.set("enable_verifier", "false")?;
        }
        flag_builder.set("use_colocated_libcalls", "false")?;
        flag_builder.set("is_pic", "false")?;
        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {msg}");
        });
        let isa = isa_builder.finish(settings::Flags::new(flag_builder))?;
        let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
        builder.symbol(
            "rvx_jit_exec_instruction",
            rvx_jit_exec_instruction as *const u8,
        );
        builder.symbol(
            "rvx_jit_invalidate_reservations",
            rvx_jit_invalidate_reservations as *const u8,
        );
        builder.symbol("rvx_jit_memory_barrier", rvx_jit_memory_barrier as *const u8);
        builder.symbol(
            "rvx_jit_translated_load_slow",
            rvx_jit_translated_load_slow as *const u8,
        );
        builder.symbol(
            "rvx_jit_translated_store_slow",
            rvx_jit_translated_store_slow as *const u8,
        );
        builder.symbol("rvx_jit_chain_lookup", rvx_jit_chain_lookup as *const u8);
        let mut module = JITModule::new(builder);

        let mut step_sig = module.make_signature();
        for _ in 0..4 {
            step_sig.params.push(AbiParam::new(types::I64));
        }
        step_sig.returns.push(AbiParam::new(types::I32));
        let step_func =
            module.declare_function("rvx_jit_exec_instruction", Linkage::Import, &step_sig)?;
        let mut invalidate_sig = module.make_signature();
        invalidate_sig.params.push(AbiParam::new(types::I64));
        let invalidate_reservations_func = module.declare_function(
            "rvx_jit_invalidate_reservations",
            Linkage::Import,
            &invalidate_sig,
        )?;
        let memory_barrier_func =
            module.declare_function("rvx_jit_memory_barrier", Linkage::Import, &invalidate_sig)?;
        let mut translated_load_sig = module.make_signature();
        for _ in 0..5 {
            translated_load_sig.params.push(AbiParam::new(types::I64));
        }
        translated_load_sig.returns.push(AbiParam::new(types::I64));
        let translated_load_func = module.declare_function(
            "rvx_jit_translated_load_slow",
            Linkage::Import,
            &translated_load_sig,
        )?;
        let mut translated_store_sig = module.make_signature();
        for _ in 0..6 {
            translated_store_sig.params.push(AbiParam::new(types::I64));
        }
        let translated_store_func = module.declare_function(
            "rvx_jit_translated_store_slow",
            Linkage::Import,
            &translated_store_sig,
        )?;
        let ctx = module.make_context();
        let builder_ctx = FunctionBuilderContext::new();
        let preferred_backend = preferred_backend(block_chaining, helper_only);
        Ok(Self {
            module,
            ctx,
            builder_ctx,
            step_func,
            invalidate_reservations_func,
            memory_barrier_func,
            translated_load_func,
            translated_store_func,
            blocks: FxHashMap::default(),
            unjittable_blocks: FxHashMap::default(),
            pending_chain_links: FxHashMap::default(),
            next_block_id: 0,
            allow_fast_mem,
            block_chaining,
            helper_only,
            collect_mmu_stats: std::env::var_os("RVX_MMU_STATS").is_some(),
            preferred_backend,
        })
    }

    pub fn block(&self, key: BlockKey) -> Option<&CompiledBlock> {
        self.blocks.get(&key).map(Box::as_ref)
    }

    pub fn unjittable_block(&self, key: BlockKey) -> Option<UnjittableBlock> {
        self.unjittable_blocks.get(&key).copied()
    }

    pub fn note_unjittable_block(&mut self, key: BlockKey, block: UnjittableBlock) {
        self.unjittable_blocks.insert(key, block);
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn block_chaining_enabled(&self) -> bool {
        self.block_chaining
    }

    fn prepare_chain(&self, key: BlockKey, instructions: &[JitInstruction]) -> ChainData {
        if !self.block_chaining {
            return ChainData::none();
        }
        match chain_targets(key, instructions) {
            ChainTargets::None => ChainData::none(),
            ChainTargets::StaticSingle(target) => ChainData::static_single(target),
            ChainTargets::StaticBranch {
                taken_pc,
                taken,
                fallthrough,
            } => ChainData::static_branch(taken_pc, taken, fallthrough),
            ChainTargets::Jalr => ChainData::jalr(),
        }
    }

    fn resolve_pending_chain_links(&mut self, key: BlockKey, block: u64) {
        if let Some(slots) = self.pending_chain_links.remove(&key) {
            for slot in slots {
                unsafe {
                    *(slot as *mut u64) = block;
                }
            }
        }
    }

    fn register_block_chain_links(&mut self, key: BlockKey) {
        let slot_refs = {
            let block = self
                .blocks
                .get_mut(&key)
                .expect("compiled block must be present before registering links");
            block.chain_data.static_slot_refs()
        };
        for (target, slot_ptr) in slot_refs {
            let target_block = self
                .blocks
                .get(&target)
                .map(|block| block.chain_token());
            if let Some(block) = target_block {
                unsafe {
                    *slot_ptr = block;
                }
            } else {
                self.pending_chain_links
                    .entry(target)
                    .or_default()
                    .push(slot_ptr as usize);
            }
        }
    }

    pub fn patch_jalr_chain(&mut self, source: BlockKey, target: BlockKey) -> bool {
        let Some(target_block) = self.blocks.get(&target).map(|block| block.chain_token()) else {
            return false;
        };
        let Some(source_block) = self.blocks.get_mut(&source) else {
            return false;
        };
        source_block.chain_data.patch_jalr(target.pc, target_block)
    }

    pub fn compile_block(
        &mut self,
        key: BlockKey,
        instructions: Vec<JitInstruction>,
    ) -> Result<&CompiledBlock> {
        if !self.blocks.contains_key(&key) {
            let instructions = instructions.into_boxed_slice();
            let chain_data = self.prepare_chain(key, &instructions);
            let (entry, chain_entry, code_owner) =
                self.compile_entry(key, &instructions, chain_data.compile_info())?;
            let writes_memory = instructions
                .iter()
                .any(|instruction| instruction_writes_memory(instruction.decoded));
            let terminator = block_terminator_kind(&instructions);
            let block = Box::new(CompiledBlock {
                key,
                instructions,
                entry,
                chain_entry,
                _code_owner: code_owner,
                chain_data,
                writes_memory,
                terminator,
            });
            self.blocks.insert(key, block);
            let chain_token = self
                .blocks
                .get(&key)
                .expect("compiled block must be cached after insertion")
                .chain_token();
            self.resolve_pending_chain_links(key, chain_token);
            self.register_block_chain_links(key);
        }
        Ok(self
            .blocks
            .get(&key)
            .expect("compiled block must be cached"))
    }

    fn compile_entry(
        &mut self,
        key: BlockKey,
        instructions: &[JitInstruction],
        chain_info: ChainCompileInfo,
    ) -> Result<(CompiledBlockEntry, u64, CompiledCodeOwner)> {
        let block_id = self.next_block_id;
        self.next_block_id = self.next_block_id.wrapping_add(1);
        if matches!(self.preferred_backend, JitBackend::X64Template) {
            if let Some((entry, code)) = x64::try_compile_block(key, instructions)? {
                return Ok((entry, entry as usize as u64, CompiledCodeOwner::X64(code)));
            }
        }
        let (entry, chain_entry) = self.compile_systemv_entry(block_id, key, instructions, chain_info)?;
        Ok((entry, chain_entry, CompiledCodeOwner::Module))
    }

    fn compile_systemv_entry(
        &mut self,
        block_id: u64,
        key: BlockKey,
        instructions: &[JitInstruction],
        chain_info: ChainCompileInfo,
    ) -> Result<(CompiledBlockEntry, u64)> {
        let mut ctx = self.module.make_context();
        mem::swap(&mut ctx, &mut self.ctx);
        self.module.clear_context(&mut ctx);
        for _ in 0..8 {
            ctx.func.signature.params.push(AbiParam::new(types::I64));
        }
        ctx.func.signature.returns.push(AbiParam::new(types::I64));

        let mut builder_ctx = FunctionBuilderContext::new();
        mem::swap(&mut builder_ctx, &mut self.builder_ctx);
        self.build_block_function(&mut ctx.func, key, instructions, chain_info, &mut builder_ctx);
        mem::swap(&mut builder_ctx, &mut self.builder_ctx);

        let name = format!("rvx_block_{block_id}");
        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &ctx.func.signature)?;
        self.module.define_function(func_id, &mut ctx)?;
        self.module.clear_context(&mut ctx);
        mem::swap(&mut ctx, &mut self.ctx);
        self.module.finalize_definitions()?;
        let code = self.module.get_finalized_function(func_id);
        Ok((
            unsafe { mem::transmute::<_, CompiledBlockEntry>(code) },
            code as usize as u64,
        ))
    }

    fn build_block_function(
        &mut self,
        func: &mut cranelift::codegen::ir::Function,
        key: BlockKey,
        instructions: &[JitInstruction],
        chain_info: ChainCompileInfo,
        builder_ctx: &mut FunctionBuilderContext,
    ) {
        let mut builder = FunctionBuilder::new(func, builder_ctx);
        let entry_block = builder.create_block();
        let exit_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.append_block_param(exit_block, types::I32);
        builder.append_block_param(exit_block, types::I64);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let cpu = builder.block_params(entry_block)[0];
        let bus = builder.block_params(entry_block)[1];
        let trap = builder.block_params(entry_block)[2];
        let ram_base = builder.block_params(entry_block)[3];
        let ram_len = builder.block_params(entry_block)[4];
        let ram_ptr = builder.block_params(entry_block)[5];
        let retired_base = builder.block_params(entry_block)[6];
        let retired_limit = builder.block_params(entry_block)[7];
        let step_func = self
            .module
            .declare_func_in_func(self.step_func, &mut builder.func);
        let invalidate_reservations_func = self
            .module
            .declare_func_in_func(self.invalidate_reservations_func, &mut builder.func);
        let memory_barrier_func = self
            .module
            .declare_func_in_func(self.memory_barrier_func, &mut builder.func);
        let translated_load_func = self
            .module
            .declare_func_in_func(self.translated_load_func, &mut builder.func);
        let translated_store_func = self
            .module
            .declare_func_in_func(self.translated_store_func, &mut builder.func);
        let direct_fast_mem = self.allow_fast_mem
            && (key.data_privilege == PrivilegeLevel::Machine
                || ((key.satp >> SATP_MODE_SHIFT as u64) & 0xf) == SATP_MODE_BARE);
        let translated_fast_mem = self.allow_fast_mem
            && !direct_fast_mem
            && matches!(
                key.data_privilege,
                PrivilegeLevel::User | PrivilegeLevel::Supervisor
            );
        let helper_only = self.helper_only;
        let collect_mmu_stats = self.collect_mmu_stats;
        let retired_var = builder.declare_var(types::I64);
        builder.def_var(retired_var, retired_base);
        let pc_var = builder.declare_var(types::I64);
        let pc = builder
            .ins()
            .load(types::I64, MemFlags::new(), cpu, CPU_PC_OFFSET);
        builder.def_var(pc_var, pc);
        let instret_var = builder.declare_var(types::I64);
        let instret = builder
            .ins()
            .load(types::I64, MemFlags::new(), cpu, CSR_INSTRET_OFFSET);
        builder.def_var(instret_var, instret);

        for (index, instruction) in instructions.iter().enumerate() {
            let is_last_instruction = index + 1 == instructions.len();
            let instruction_ptr = instruction as *const JitInstruction;
            if helper_only {
                emit_helper_instruction(
                    &mut builder,
                    cpu,
                    bus,
                    trap,
                    step_func,
                    pc_var,
                    instret_var,
                    retired_var,
                    exit_block,
                    instruction_ptr,
                );
                continue;
            }
            match instruction.decoded {
                DecodedInstruction::Lui { rd, imm } => {
                    let value = builder.ins().iconst(types::I64, imm);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Auipc { rd, imm } => {
                    let pc = builder.use_var(pc_var);
                    let value = builder.ins().iadd_imm(pc, imm);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Jal { rd, imm } => {
                    let pc = builder.use_var(pc_var);
                    let next_pc = builder.ins().iadd_imm(pc, instruction.instruction_bytes as i64);
                    store_x(&mut builder, cpu, rd, next_pc);
                    let target = builder.ins().iadd_imm(pc, imm);
                    builder.def_var(pc_var, target);
                    let instret = builder.use_var(instret_var);
                    let instret = builder.ins().iadd_imm(instret, 1);
                    builder.def_var(instret_var, instret);
                    let retired = builder.use_var(retired_var);
                    let retired = builder.ins().iadd_imm(retired, 1);
                    builder.def_var(retired_var, retired);
                }
                DecodedInstruction::Jalr { rd, rs1, imm } => {
                    let pc = builder.use_var(pc_var);
                    let next_pc = builder.ins().iadd_imm(pc, instruction.instruction_bytes as i64);
                    let base = load_x(&mut builder, cpu, rs1);
                    let target = builder.ins().iadd_imm(base, imm);
                    let target = builder.ins().band_imm(target, !1);
                    store_x(&mut builder, cpu, rd, next_pc);
                    builder.def_var(pc_var, target);
                    let instret = builder.use_var(instret_var);
                    let instret = builder.ins().iadd_imm(instret, 1);
                    builder.def_var(instret_var, instret);
                    let retired = builder.use_var(retired_var);
                    let retired = builder.ins().iadd_imm(retired, 1);
                    builder.def_var(retired_var, retired);
                }
                DecodedInstruction::Branch { kind, rs1, rs2, imm } => {
                    if is_last_instruction {
                        let pc = builder.use_var(pc_var);
                        let lhs = load_x(&mut builder, cpu, rs1);
                        let rhs = load_x(&mut builder, cpu, rs2);
                        let taken = emit_branch_condition(&mut builder, kind, lhs, rhs);
                        let target = builder.ins().iadd_imm(pc, imm);
                        let fallthrough =
                            builder.ins().iadd_imm(pc, instruction.instruction_bytes as i64);
                        let next_pc = builder.ins().select(taken, target, fallthrough);
                        builder.def_var(pc_var, next_pc);
                        let instret = builder.use_var(instret_var);
                        let instret = builder.ins().iadd_imm(instret, 1);
                        builder.def_var(instret_var, instret);
                        let retired = builder.use_var(retired_var);
                        let retired = builder.ins().iadd_imm(retired, 1);
                        builder.def_var(retired_var, retired);
                    } else {
                        emit_traced_branch_or_exit(
                            &mut builder,
                            cpu,
                            pc_var,
                            instret_var,
                            retired_var,
                            exit_block,
                            kind,
                            rs1,
                            rs2,
                            imm,
                            instruction.instruction_bytes,
                            imm < 0,
                        );
                    }
                }
                DecodedInstruction::Addi { rd, rs1, imm } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let value = builder.ins().iadd_imm(lhs, imm);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Slti { rd, rs1, imm } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = builder.ins().iconst(types::I64, imm);
                    let cmp = builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs);
                    let value = bool_to_i64(&mut builder, cmp);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Sltiu { rd, rs1, imm } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = builder.ins().iconst(types::I64, imm);
                    let cmp = builder.ins().icmp(IntCC::UnsignedLessThan, lhs, rhs);
                    let value = bool_to_i64(&mut builder, cmp);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Xori { rd, rs1, imm } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = builder.ins().iconst(types::I64, imm);
                    let value = builder.ins().bxor(lhs, rhs);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Ori { rd, rs1, imm } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = builder.ins().iconst(types::I64, imm);
                    let value = builder.ins().bor(lhs, rhs);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Andi { rd, rs1, imm } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = builder.ins().iconst(types::I64, imm);
                    let value = builder.ins().band(lhs, rhs);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Slli { rd, rs1, shamt } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = builder.ins().iconst(types::I64, shamt as i64);
                    let value = builder.ins().ishl(lhs, rhs);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Srli { rd, rs1, shamt } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = builder.ins().iconst(types::I64, shamt as i64);
                    let value = builder.ins().ushr(lhs, rhs);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Srai { rd, rs1, shamt } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = builder.ins().iconst(types::I64, shamt as i64);
                    let value = builder.ins().sshr(lhs, rhs);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Addiw { rd, rs1, imm } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let sum = builder.ins().iadd_imm(lhs, imm);
                    let value = sext_w(&mut builder, sum);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Slliw { rd, rs1, shamt } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let lhs = truncate_w(&mut builder, lhs);
                    let rhs = builder.ins().iconst(types::I32, shamt as i64);
                    let value = builder.ins().ishl(lhs, rhs);
                    let value = builder.ins().sextend(types::I64, value);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Srliw { rd, rs1, shamt } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let lhs = truncate_w(&mut builder, lhs);
                    let rhs = builder.ins().iconst(types::I32, shamt as i64);
                    let value = builder.ins().ushr(lhs, rhs);
                    let value = builder.ins().sextend(types::I64, value);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Sraiw { rd, rs1, shamt } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let lhs = truncate_w(&mut builder, lhs);
                    let rhs = builder.ins().iconst(types::I32, shamt as i64);
                    let value = builder.ins().sshr(lhs, rhs);
                    let value = builder.ins().sextend(types::I64, value);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Add { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let value = builder.ins().iadd(lhs, rhs);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Sub { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let value = builder.ins().isub(lhs, rhs);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Sll { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let rhs = shift_amount(&mut builder, rhs, 0x3f);
                    let value = builder.ins().ishl(lhs, rhs);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Slt { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let cmp = builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs);
                    let value = bool_to_i64(&mut builder, cmp);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Sltu { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let cmp = builder.ins().icmp(IntCC::UnsignedLessThan, lhs, rhs);
                    let value = bool_to_i64(&mut builder, cmp);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Xor { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let value = builder.ins().bxor(lhs, rhs);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Srl { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let rhs = shift_amount(&mut builder, rhs, 0x3f);
                    let value = builder.ins().ushr(lhs, rhs);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Sra { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let rhs = shift_amount(&mut builder, rhs, 0x3f);
                    let value = builder.ins().sshr(lhs, rhs);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Or { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let value = builder.ins().bor(lhs, rhs);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::And { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let value = builder.ins().band(lhs, rhs);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Addw { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let lhs = truncate_w(&mut builder, lhs);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let rhs = truncate_w(&mut builder, rhs);
                    let value = builder.ins().iadd(lhs, rhs);
                    let value = builder.ins().sextend(types::I64, value);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Subw { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let lhs = truncate_w(&mut builder, lhs);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let rhs = truncate_w(&mut builder, rhs);
                    let value = builder.ins().isub(lhs, rhs);
                    let value = builder.ins().sextend(types::I64, value);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Sllw { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let lhs = truncate_w(&mut builder, lhs);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let rhs = shift_amount(&mut builder, rhs, 0x1f);
                    let rhs = builder.ins().ireduce(types::I32, rhs);
                    let value = builder.ins().ishl(lhs, rhs);
                    let value = builder.ins().sextend(types::I64, value);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Srlw { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let lhs = truncate_w(&mut builder, lhs);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let rhs = shift_amount(&mut builder, rhs, 0x1f);
                    let rhs = builder.ins().ireduce(types::I32, rhs);
                    let value = builder.ins().ushr(lhs, rhs);
                    let value = builder.ins().sextend(types::I64, value);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Sraw { rd, rs1, rs2 } => {
                    let lhs = load_x(&mut builder, cpu, rs1);
                    let lhs = truncate_w(&mut builder, lhs);
                    let rhs = load_x(&mut builder, cpu, rs2);
                    let rhs = shift_amount(&mut builder, rhs, 0x1f);
                    let rhs = builder.ins().ireduce(types::I32, rhs);
                    let value = builder.ins().sshr(lhs, rhs);
                    let value = builder.ins().sextend(types::I64, value);
                    store_x(&mut builder, cpu, rd, value);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                DecodedInstruction::Load { kind, rd, rs1, imm } if direct_fast_mem => {
                    emit_fast_load_or_helper(
                        &mut builder,
                        cpu,
                        bus,
                        trap,
                        ram_base,
                        ram_len,
                        ram_ptr,
                        step_func,
                        pc_var,
                        instret_var,
                        retired_var,
                        exit_block,
                        instruction_ptr,
                        instruction.instruction_bytes,
                        kind,
                        rd,
                        rs1,
                        imm,
                    );
                }
                DecodedInstruction::Load { kind, rd, rs1, imm } if translated_fast_mem => {
                    emit_translated_load_or_helper(
                        &mut builder,
                        cpu,
                        bus,
                        trap,
                        translated_load_func,
                        pc_var,
                        instret_var,
                        retired_var,
                        exit_block,
                        instruction.instruction_bytes,
                        kind,
                        rd,
                        rs1,
                        imm,
                        key.satp,
                        key.mstatus_vm,
                        key.data_privilege,
                        collect_mmu_stats,
                    );
                }
                DecodedInstruction::Store {
                    kind,
                    rs1,
                    rs2,
                    imm,
                } if direct_fast_mem => {
                    emit_fast_store_or_helper(
                        &mut builder,
                        cpu,
                        bus,
                        trap,
                        ram_base,
                        ram_len,
                        ram_ptr,
                        step_func,
                        invalidate_reservations_func,
                        pc_var,
                        instret_var,
                        retired_var,
                        exit_block,
                        instruction_ptr,
                        instruction.instruction_bytes,
                        kind,
                        rs1,
                        rs2,
                        imm,
                    );
                }
                DecodedInstruction::Store {
                    kind,
                    rs1,
                    rs2,
                    imm,
                } if translated_fast_mem => {
                    emit_translated_store_or_helper(
                        &mut builder,
                        cpu,
                        bus,
                        trap,
                        translated_store_func,
                        invalidate_reservations_func,
                        pc_var,
                        instret_var,
                        retired_var,
                        exit_block,
                        instruction.instruction_bytes,
                        kind,
                        rs1,
                        rs2,
                        imm,
                        key.satp,
                        key.mstatus_vm,
                        key.data_privilege,
                        collect_mmu_stats,
                    );
                }
                DecodedInstruction::Fence | DecodedInstruction::FenceI => {
                    builder.ins().call(memory_barrier_func, &[bus]);
                    finish_inline_instruction(
                        &mut builder,
                        pc_var,
                        instret_var,
                        retired_var,
                        instruction.instruction_bytes,
                    );
                }
                _ => {
                    emit_helper_instruction(
                        &mut builder,
                        cpu,
                        bus,
                        trap,
                        step_func,
                        pc_var,
                        instret_var,
                        retired_var,
                        exit_block,
                        instruction_ptr,
                    );
                }
            }
        }

        sync_cpu_state(&mut builder, cpu, pc_var, instret_var);
        CURRENT_GPR_CACHE.with(|slot| slot.set(ptr::null_mut()));
        let continue_status = builder.ins().iconst(types::I32, BLOCK_STATUS_CONTINUE);
        let retired = builder.use_var(retired_var);
        let done_args = [BlockArg::Value(continue_status), BlockArg::Value(retired)];
        builder.ins().jump(exit_block, &done_args);

        builder.switch_to_block(exit_block);
        let status = builder.block_params(exit_block)[0];
        let retired = builder.block_params(exit_block)[1];
        if !matches!(chain_info, ChainCompileInfo::None) {
            emit_static_chain_or_return(
                &mut builder,
                chain_info,
                cpu,
                trap,
                status,
                retired,
                retired_limit,
            );
        }
        let status = builder.ins().uextend(types::I64, status);
        let retired = builder.ins().ishl_imm(retired, 32);
        let packed = builder.ins().bor(retired, status);
        builder.ins().return_(&[packed]);
        builder.seal_all_blocks();
        builder.finalize();
    }

}

fn load_x(builder: &mut FunctionBuilder, cpu: Value, reg: u8) -> Value {
    CURRENT_GPR_CACHE.with(|slot| {
        let cache = slot.get();
        if cache.is_null() {
            if reg == 0 {
                builder.ins().iconst(types::I64, 0)
            } else {
                builder.ins().load(
                    types::I64,
                    MemFlags::new(),
                    cpu,
                    CPU_X_OFFSET + reg as i32 * 8,
                )
            }
        } else {
            unsafe { (&mut *cache).load(builder, cpu, reg) }
        }
    })
}

fn store_x(builder: &mut FunctionBuilder, cpu: Value, reg: u8, value: Value) {
    CURRENT_GPR_CACHE.with(|slot| {
        let cache = slot.get();
        if cache.is_null() {
            if reg != 0 {
                builder
                    .ins()
                    .store(MemFlags::new(), value, cpu, CPU_X_OFFSET + reg as i32 * 8);
            }
        } else {
            unsafe { (&mut *cache).store(builder, reg, value) };
        }
    });
}

fn flush_cached_xs(builder: &mut FunctionBuilder, cpu: Value) {
    CURRENT_GPR_CACHE.with(|slot| {
        let cache = slot.get();
        if !cache.is_null() {
            unsafe { (&mut *cache).flush(builder, cpu) };
        }
    });
}

fn snapshot_cached_xs() -> Option<GprCacheState> {
    CURRENT_GPR_CACHE.with(|slot| {
        let cache = slot.get();
        if cache.is_null() {
            None
        } else {
            Some(unsafe { (&*cache).snapshot() })
        }
    })
}

fn restore_cached_xs(state: Option<GprCacheState>) {
    CURRENT_GPR_CACHE.with(|slot| {
        let cache = slot.get();
        if cache.is_null() {
            return;
        }
        match state {
            Some(state) => unsafe { (&mut *cache).restore(state) },
            None => unsafe { (&mut *cache).invalidate() },
        }
    });
}

fn invalidate_cached_xs() {
    CURRENT_GPR_CACHE.with(|slot| {
        let cache = slot.get();
        if !cache.is_null() {
            unsafe { (&mut *cache).invalidate() };
        }
    });
}

fn truncate_w(builder: &mut FunctionBuilder, value: Value) -> Value {
    builder.ins().ireduce(types::I32, value)
}

fn sext_w(builder: &mut FunctionBuilder, value: Value) -> Value {
    let value = builder.ins().ireduce(types::I32, value);
    builder.ins().sextend(types::I64, value)
}

fn instruction_writes_memory(instruction: DecodedInstruction) -> bool {
    match instruction {
        DecodedInstruction::Store { .. } | DecodedInstruction::FloatStore { .. } => true,
        DecodedInstruction::Atomic { op, .. } => {
            !matches!(op, crate::AtomicOp::LrW | crate::AtomicOp::LrD)
        }
        _ => false,
    }
}

enum ChainTargets {
    None,
    StaticSingle(BlockKey),
    StaticBranch {
        taken_pc: u64,
        taken: BlockKey,
        fallthrough: BlockKey,
    },
    Jalr,
}

fn chain_targets(key: BlockKey, instructions: &[JitInstruction]) -> ChainTargets {
    let Some((last, prefix)) = instructions.split_last() else {
        return ChainTargets::None;
    };

    let last_pc = prefix.iter().fold(key.pc, |pc, instruction| {
        pc.wrapping_add(instruction.instruction_bytes as u64)
    });
    let fallthrough_pc = last_pc.wrapping_add(last.instruction_bytes as u64);

    match last.decoded {
        DecodedInstruction::Jal { imm, .. } => ChainTargets::StaticSingle(BlockKey {
            pc: last_pc.wrapping_add_signed(imm),
            ..key
        }),
        DecodedInstruction::Branch { imm, .. } => {
            let taken_pc = last_pc.wrapping_add_signed(imm);
            ChainTargets::StaticBranch {
                taken_pc,
                taken: BlockKey { pc: taken_pc, ..key },
                fallthrough: BlockKey {
                    pc: fallthrough_pc,
                    ..key
                },
            }
        }
        DecodedInstruction::Jalr { .. } => ChainTargets::Jalr,
        _ if block_terminator_kind(instructions) == BlockTerminatorKind::Fallthrough => {
            ChainTargets::StaticSingle(BlockKey {
                pc: fallthrough_pc,
                ..key
            })
        }
        _ => ChainTargets::None,
    }
}

fn block_terminator_kind(instructions: &[JitInstruction]) -> BlockTerminatorKind {
    let Some(last) = instructions.last() else {
        return BlockTerminatorKind::Fallthrough;
    };
    match last.decoded {
        DecodedInstruction::Jal { .. } => BlockTerminatorKind::Jal,
        DecodedInstruction::Jalr { .. } => BlockTerminatorKind::Jalr,
        DecodedInstruction::Branch { .. } => BlockTerminatorKind::Branch,
        DecodedInstruction::Atomic { .. } => BlockTerminatorKind::Atomic,
        DecodedInstruction::Ebreak => BlockTerminatorKind::Ebreak,
        DecodedInstruction::Csrrw { .. }
        | DecodedInstruction::Csrrs { .. }
        | DecodedInstruction::Csrrc { .. }
        | DecodedInstruction::Csrrwi { .. }
        | DecodedInstruction::Csrrsi { .. }
        | DecodedInstruction::Csrrci { .. } => BlockTerminatorKind::Csr,
        _ if instructions.len() >= jit_max_block_instructions() => BlockTerminatorKind::Fallthrough,
        _ => BlockTerminatorKind::Other,
    }
}

fn shift_amount(builder: &mut FunctionBuilder, value: Value, mask: u64) -> Value {
    let mask = builder.ins().iconst(types::I64, mask as i64);
    builder.ins().band(value, mask)
}

fn bool_to_i64(builder: &mut FunctionBuilder, value: Value) -> Value {
    let one = builder.ins().iconst(types::I64, 1);
    let zero = builder.ins().iconst(types::I64, 0);
    builder.ins().select(value, one, zero)
}

fn emit_branch_condition(
    builder: &mut FunctionBuilder,
    kind: BranchKind,
    lhs: Value,
    rhs: Value,
) -> Value {
    let cc = match kind {
        BranchKind::Eq => IntCC::Equal,
        BranchKind::Ne => IntCC::NotEqual,
        BranchKind::Lt => IntCC::SignedLessThan,
        BranchKind::Ge => IntCC::SignedGreaterThanOrEqual,
        BranchKind::Ltu => IntCC::UnsignedLessThan,
        BranchKind::Geu => IntCC::UnsignedGreaterThanOrEqual,
    };
    builder.ins().icmp(cc, lhs, rhs)
}

fn increment_mmu_counter(builder: &mut FunctionBuilder, cpu: Value, offset: i32) {
    let value = builder
        .ins()
        .load(types::I64, MemFlags::new(), cpu, CPU_MMU_OFFSET + offset);
    let value = builder.ins().iadd_imm(value, 1);
    builder
        .ins()
        .store(MemFlags::new(), value, cpu, CPU_MMU_OFFSET + offset);
}

fn emit_memory_trap_guard(
    builder: &mut FunctionBuilder,
    trap: Value,
    retired_var: Variable,
    exit_block: Block,
) {
    let success_block = builder.create_block();
    let trap_cause = builder
        .ins()
        .load(types::I64, MemFlags::new(), trap, TRAP_INFO_CAUSE_OFFSET);
    let success = builder.ins().icmp_imm(IntCC::Equal, trap_cause, 0);
    let trap_status = builder.ins().iconst(types::I32, BLOCK_STATUS_TRAP);
    let retired = builder.use_var(retired_var);
    let trap_args = [BlockArg::Value(trap_status), BlockArg::Value(retired)];
    builder
        .ins()
        .brif(success, success_block, &[], exit_block, &trap_args);
    builder.switch_to_block(success_block);
    builder.seal_block(success_block);
}

fn sync_cpu_state(
    builder: &mut FunctionBuilder,
    cpu: Value,
    pc_var: Variable,
    instret_var: Variable,
) {
    flush_cached_xs(builder, cpu);
    let pc = builder.use_var(pc_var);
    builder.ins().store(MemFlags::new(), pc, cpu, CPU_PC_OFFSET);
    let instret = builder.use_var(instret_var);
    builder
        .ins()
        .store(MemFlags::new(), instret, cpu, CSR_INSTRET_OFFSET);
}

fn load_kind_code(kind: LoadKind) -> i64 {
    match kind {
        LoadKind::Byte => 0,
        LoadKind::Half => 1,
        LoadKind::Word => 2,
        LoadKind::Double => 3,
        LoadKind::ByteUnsigned => 4,
        LoadKind::HalfUnsigned => 5,
        LoadKind::WordUnsigned => 6,
    }
}

fn store_kind_code(kind: StoreKind) -> i64 {
    match kind {
        StoreKind::Byte => 0,
        StoreKind::Half => 1,
        StoreKind::Word => 2,
        StoreKind::Double => 3,
    }
}

fn finish_inline_instruction(
    builder: &mut FunctionBuilder,
    pc_var: Variable,
    instret_var: Variable,
    retired_var: Variable,
    instruction_bytes: u8,
) {
    let pc = builder.use_var(pc_var);
    let next_pc = builder.ins().iadd_imm(pc, instruction_bytes as i64);
    builder.def_var(pc_var, next_pc);
    let instret = builder.use_var(instret_var);
    let instret = builder.ins().iadd_imm(instret, 1);
    builder.def_var(instret_var, instret);
    let retired = builder.use_var(retired_var);
    let retired = builder.ins().iadd_imm(retired, 1);
    builder.def_var(retired_var, retired);
}


fn emit_exit_continue(
    builder: &mut FunctionBuilder,
    cpu: Value,
    exit_block: Block,
    pc: Value,
    instret: Value,
    retired: Value,
) {
    flush_cached_xs(builder, cpu);
    builder.ins().store(MemFlags::new(), pc, cpu, CPU_PC_OFFSET);
    builder
        .ins()
        .store(MemFlags::new(), instret, cpu, CSR_INSTRET_OFFSET);
    let status = builder.ins().iconst(types::I32, BLOCK_STATUS_CONTINUE);
    let exit_args = [BlockArg::Value(status), BlockArg::Value(retired)];
    builder.ins().jump(exit_block, &exit_args);
}

#[allow(clippy::too_many_arguments)]
fn emit_static_chain_or_return(
    builder: &mut FunctionBuilder,
    chain_info: ChainCompileInfo,
    cpu: Value,
    trap: Value,
    status: Value,
    retired: Value,
    retired_limit: Value,
) {
    let return_block = builder.create_block();
    let chain_budget_block = builder.create_block();
    let chain_status = builder.ins().iconst(types::I32, BLOCK_STATUS_CHAIN);

    let is_continue = builder
        .ins()
        .icmp_imm(IntCC::Equal, status, BLOCK_STATUS_CONTINUE);
    builder
        .ins()
        .brif(is_continue, chain_budget_block, &[], return_block, &[]);

    builder.switch_to_block(chain_budget_block);
    builder.seal_block(chain_budget_block);
    let within_budget = builder
        .ins()
        .icmp(IntCC::UnsignedLessThan, retired, retired_limit);
    match chain_info {
        ChainCompileInfo::None => {
            let _ = within_budget;
            builder.ins().jump(return_block, &[]);
        }
        ChainCompileInfo::StaticSingle { slot_addr } => {
            let chain_lookup_block = builder.create_block();
            builder
                .ins()
                .brif(within_budget, chain_lookup_block, &[], return_block, &[]);

            builder.switch_to_block(chain_lookup_block);
            builder.seal_block(chain_lookup_block);
            let slot_addr = builder.ins().iconst(types::I64, slot_addr as i64);
            let callee = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), slot_addr, 0);
            let miss = builder.ins().icmp_imm(IntCC::Equal, callee, 0);
            let chain_hit_block = builder.create_block();
            builder
                .ins()
                .brif(miss, return_block, &[], chain_hit_block, &[]);

            builder.switch_to_block(chain_hit_block);
            builder.seal_block(chain_hit_block);
            builder.ins().store(
                MemFlags::trusted(),
                callee,
                trap,
                TRAP_INFO_NEXT_BLOCK_OFFSET,
            );
            let status = builder.ins().uextend(types::I64, chain_status);
            let retired = builder.ins().ishl_imm(retired, 32);
            let packed = builder.ins().bor(retired, status);
            builder.ins().return_(&[packed]);
        }
        ChainCompileInfo::Branch {
            taken_pc,
            taken_slot_addr,
            fallthrough_slot_addr,
        } => {
            let choose_slot_block = builder.create_block();
            let taken_slot_block = builder.create_block();
            let fallthrough_slot_block = builder.create_block();
            let chain_hit_block = builder.create_block();
            builder.append_block_param(chain_hit_block, types::I64);

            builder
                .ins()
                .brif(within_budget, choose_slot_block, &[], return_block, &[]);

            builder.switch_to_block(choose_slot_block);
            builder.seal_block(choose_slot_block);
            let next_pc = builder
                .ins()
                .load(types::I64, MemFlags::new(), cpu, CPU_PC_OFFSET);
            let is_taken = builder.ins().icmp_imm(IntCC::Equal, next_pc, taken_pc as i64);
            builder.ins().brif(
                is_taken,
                taken_slot_block,
                &[],
                fallthrough_slot_block,
                &[],
            );

            builder.switch_to_block(taken_slot_block);
            builder.seal_block(taken_slot_block);
            let taken_slot_addr = builder.ins().iconst(types::I64, taken_slot_addr as i64);
            let taken_callee = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), taken_slot_addr, 0);
            let taken_miss = builder.ins().icmp_imm(IntCC::Equal, taken_callee, 0);
            builder.ins().brif(
                taken_miss,
                return_block,
                &[],
                chain_hit_block,
                &[BlockArg::Value(taken_callee)],
            );

            builder.switch_to_block(fallthrough_slot_block);
            builder.seal_block(fallthrough_slot_block);
            let fallthrough_slot_addr =
                builder.ins().iconst(types::I64, fallthrough_slot_addr as i64);
            let fallthrough_callee = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), fallthrough_slot_addr, 0);
            let fallthrough_miss = builder.ins().icmp_imm(IntCC::Equal, fallthrough_callee, 0);
            builder.ins().brif(
                fallthrough_miss,
                return_block,
                &[],
                chain_hit_block,
                &[BlockArg::Value(fallthrough_callee)],
            );

            builder.switch_to_block(chain_hit_block);
            builder.seal_block(chain_hit_block);
            let callee = builder.block_params(chain_hit_block)[0];
            builder.ins().store(
                MemFlags::trusted(),
                callee,
                trap,
                TRAP_INFO_NEXT_BLOCK_OFFSET,
            );
            let status = builder.ins().uextend(types::I64, chain_status);
            let retired = builder.ins().ishl_imm(retired, 32);
            let packed = builder.ins().bor(retired, status);
            builder.ins().return_(&[packed]);
        }
        ChainCompileInfo::Jalr {
            target_pc_addr,
            block_slot_addr,
        } => {
            let chain_lookup_block = builder.create_block();
            let entry_check_block = builder.create_block();
            let chain_hit_block = builder.create_block();

            builder
                .ins()
                .brif(within_budget, chain_lookup_block, &[], return_block, &[]);

            builder.switch_to_block(chain_lookup_block);
            builder.seal_block(chain_lookup_block);
            let next_pc = builder
                .ins()
                .load(types::I64, MemFlags::new(), cpu, CPU_PC_OFFSET);
            let target_pc_addr = builder.ins().iconst(types::I64, target_pc_addr as i64);
            let cached_target_pc = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), target_pc_addr, 0);
            let target_match = builder.ins().icmp(IntCC::Equal, next_pc, cached_target_pc);
            builder
                .ins()
                .brif(target_match, entry_check_block, &[], return_block, &[]);

            builder.switch_to_block(entry_check_block);
            builder.seal_block(entry_check_block);
            let block_slot_addr = builder.ins().iconst(types::I64, block_slot_addr as i64);
            let callee = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), block_slot_addr, 0);
            let miss = builder.ins().icmp_imm(IntCC::Equal, callee, 0);
            builder
                .ins()
                .brif(miss, return_block, &[], chain_hit_block, &[]);

            builder.switch_to_block(chain_hit_block);
            builder.seal_block(chain_hit_block);
            builder.ins().store(
                MemFlags::trusted(),
                callee,
                trap,
                TRAP_INFO_NEXT_BLOCK_OFFSET,
            );
            let status = builder.ins().uextend(types::I64, chain_status);
            let retired = builder.ins().ishl_imm(retired, 32);
            let packed = builder.ins().bor(retired, status);
            builder.ins().return_(&[packed]);
        }
    }

    builder.switch_to_block(return_block);
    builder.seal_block(return_block);
}

#[allow(clippy::too_many_arguments)]
fn emit_traced_branch_or_exit(
    builder: &mut FunctionBuilder,
    cpu: Value,
    pc_var: Variable,
    instret_var: Variable,
    retired_var: Variable,
    exit_block: Block,
    kind: BranchKind,
    rs1: u8,
    rs2: u8,
    imm: i64,
    instruction_bytes: u8,
    predict_taken: bool,
) {
    let pc = builder.use_var(pc_var);
    let lhs = load_x(builder, cpu, rs1);
    let rhs = load_x(builder, cpu, rs2);
    let taken = emit_branch_condition(builder, kind, lhs, rhs);
    let target = builder.ins().iadd_imm(pc, imm);
    let fallthrough = builder.ins().iadd_imm(pc, instruction_bytes as i64);
    let instret = builder.use_var(instret_var);
    let instret = builder.ins().iadd_imm(instret, 1);
    let retired = builder.use_var(retired_var);
    let retired = builder.ins().iadd_imm(retired, 1);

    let continue_block = builder.create_block();
    builder.append_block_param(continue_block, types::I64);
    builder.append_block_param(continue_block, types::I64);
    builder.append_block_param(continue_block, types::I64);
    let exit_now_block = builder.create_block();
    builder.append_block_param(exit_now_block, types::I64);
    builder.append_block_param(exit_now_block, types::I64);
    builder.append_block_param(exit_now_block, types::I64);

    let continue_pc = if predict_taken { target } else { fallthrough };
    let exit_pc = if predict_taken { fallthrough } else { target };
    let continue_args = [
        BlockArg::Value(continue_pc),
        BlockArg::Value(instret),
        BlockArg::Value(retired),
    ];
    let exit_args = [
        BlockArg::Value(exit_pc),
        BlockArg::Value(instret),
        BlockArg::Value(retired),
    ];
    let cache_state = snapshot_cached_xs();
    if predict_taken {
        builder
            .ins()
            .brif(taken, continue_block, &continue_args, exit_now_block, &exit_args);
    } else {
        builder
            .ins()
            .brif(taken, exit_now_block, &exit_args, continue_block, &continue_args);
    }

    builder.switch_to_block(exit_now_block);
    restore_cached_xs(cache_state);
    builder.seal_block(exit_now_block);
    let exit_pc = builder.block_params(exit_now_block)[0];
    let exit_instret = builder.block_params(exit_now_block)[1];
    let exit_retired = builder.block_params(exit_now_block)[2];
    emit_exit_continue(builder, cpu, exit_block, exit_pc, exit_instret, exit_retired);

    builder.switch_to_block(continue_block);
    restore_cached_xs(cache_state);
    builder.seal_block(continue_block);
    let continue_pc = builder.block_params(continue_block)[0];
    let continue_instret = builder.block_params(continue_block)[1];
    let continue_retired = builder.block_params(continue_block)[2];
    builder.def_var(pc_var, continue_pc);
    builder.def_var(instret_var, continue_instret);
    builder.def_var(retired_var, continue_retired);
}

#[allow(clippy::too_many_arguments)]
fn emit_helper_instruction(
    builder: &mut FunctionBuilder,
    cpu: Value,
    bus: Value,
    trap: Value,
    step_func: FuncRef,
    pc_var: Variable,
    instret_var: Variable,
    retired_var: Variable,
    exit_block: Block,
    instruction_ptr: *const JitInstruction,
) {
    sync_cpu_state(builder, cpu, pc_var, instret_var);
    let instruction_ptr = builder.ins().iconst(types::I64, instruction_ptr as i64);
    let call = builder
        .ins()
        .call(step_func, &[cpu, bus, trap, instruction_ptr]);
    let status = builder.inst_results(call)[0];
    let continue_block = builder.create_block();
    let retired = builder.use_var(retired_var);
    let is_continue = builder
        .ins()
        .icmp_imm(IntCC::Equal, status, BLOCK_STATUS_CONTINUE);
    let trap_args = [BlockArg::Value(status), BlockArg::Value(retired)];
    builder
        .ins()
        .brif(is_continue, continue_block, &[], exit_block, &trap_args);

    builder.switch_to_block(continue_block);
    builder.seal_block(continue_block);
    invalidate_cached_xs();
    let retired = builder.use_var(retired_var);
    let retired = builder.ins().iadd_imm(retired, 1);
    builder.def_var(retired_var, retired);
    let pc = builder
        .ins()
        .load(types::I64, MemFlags::new(), cpu, CPU_PC_OFFSET);
    builder.def_var(pc_var, pc);
    let instret = builder
        .ins()
        .load(types::I64, MemFlags::new(), cpu, CSR_INSTRET_OFFSET);
    builder.def_var(instret_var, instret);
}

#[allow(clippy::too_many_arguments)]
fn emit_fast_load_or_helper(
    builder: &mut FunctionBuilder,
    cpu: Value,
    bus: Value,
    trap: Value,
    ram_base: Value,
    ram_len: Value,
    ram_ptr: Value,
    step_func: FuncRef,
    pc_var: Variable,
    instret_var: Variable,
    retired_var: Variable,
    exit_block: Block,
    instruction_ptr: *const JitInstruction,
    instruction_bytes: u8,
    kind: LoadKind,
    rd: u8,
    rs1: u8,
    imm: i64,
) {
    let (size, ty, sign_extend) = match kind {
        LoadKind::Byte => (1, types::I8, true),
        LoadKind::Half => (2, types::I16, true),
        LoadKind::Word => (4, types::I32, true),
        LoadKind::Double => (8, types::I64, false),
        LoadKind::ByteUnsigned => (1, types::I8, false),
        LoadKind::HalfUnsigned => (2, types::I16, false),
        LoadKind::WordUnsigned => (4, types::I32, false),
    };

    let base = load_x(builder, cpu, rs1);
    let addr = builder.ins().iadd_imm(base, imm);
    let offset = builder.ins().isub(addr, ram_base);
    let max_offset = builder.ins().iadd_imm(ram_len, -(size as i64));
    let in_bounds = builder
        .ins()
        .icmp(IntCC::UnsignedLessThanOrEqual, offset, max_offset);
    let fast_block = builder.create_block();
    let slow_block = builder.create_block();
    let continue_block = builder.create_block();
    let cache_state = snapshot_cached_xs();
    builder
        .ins()
        .brif(in_bounds, fast_block, &[], slow_block, &[]);

    builder.switch_to_block(fast_block);
    restore_cached_xs(cache_state);
    builder.seal_block(fast_block);
    let host = builder.ins().iadd(ram_ptr, offset);
    let mut value = builder.ins().load(ty, MemFlags::new(), host, 0);
    value = match (ty, sign_extend) {
        (types::I64, _) => value,
        (_, true) => builder.ins().sextend(types::I64, value),
        (_, false) => builder.ins().uextend(types::I64, value),
    };
    store_x(builder, cpu, rd, value);
    finish_inline_instruction(builder, pc_var, instret_var, retired_var, instruction_bytes);
    flush_cached_xs(builder, cpu);
    builder.ins().jump(continue_block, &[]);

    builder.switch_to_block(slow_block);
    restore_cached_xs(cache_state);
    builder.seal_block(slow_block);
    emit_helper_instruction(
        builder,
        cpu,
        bus,
        trap,
        step_func,
        pc_var,
        instret_var,
        retired_var,
        exit_block,
        instruction_ptr,
    );
    flush_cached_xs(builder, cpu);
    builder.ins().jump(continue_block, &[]);

    builder.switch_to_block(continue_block);
    invalidate_cached_xs();
    builder.seal_block(continue_block);
}

#[allow(clippy::too_many_arguments)]
fn emit_fast_store_or_helper(
    builder: &mut FunctionBuilder,
    cpu: Value,
    bus: Value,
    trap: Value,
    ram_base: Value,
    ram_len: Value,
    ram_ptr: Value,
    step_func: FuncRef,
    invalidate_reservations_func: FuncRef,
    pc_var: Variable,
    instret_var: Variable,
    retired_var: Variable,
    exit_block: Block,
    instruction_ptr: *const JitInstruction,
    instruction_bytes: u8,
    kind: StoreKind,
    rs1: u8,
    rs2: u8,
    imm: i64,
) {
    let (size, ty) = match kind {
        StoreKind::Byte => (1, types::I8),
        StoreKind::Half => (2, types::I16),
        StoreKind::Word => (4, types::I32),
        StoreKind::Double => (8, types::I64),
    };

    let base = load_x(builder, cpu, rs1);
    let addr = builder.ins().iadd_imm(base, imm);
    let offset = builder.ins().isub(addr, ram_base);
    let max_offset = builder.ins().iadd_imm(ram_len, -(size as i64));
    let in_bounds = builder
        .ins()
        .icmp(IntCC::UnsignedLessThanOrEqual, offset, max_offset);
    let fast_block = builder.create_block();
    let slow_block = builder.create_block();
    let continue_block = builder.create_block();
    let cache_state = snapshot_cached_xs();
    builder
        .ins()
        .brif(in_bounds, fast_block, &[], slow_block, &[]);

    builder.switch_to_block(fast_block);
    restore_cached_xs(cache_state);
    builder.seal_block(fast_block);
    let host = builder.ins().iadd(ram_ptr, offset);
    let value = load_x(builder, cpu, rs2);
    let value = match ty {
        types::I8 => builder.ins().ireduce(types::I8, value),
        types::I16 => builder.ins().ireduce(types::I16, value),
        types::I32 => builder.ins().ireduce(types::I32, value),
        types::I64 => value,
        _ => unreachable!(),
    };
    builder.ins().store(MemFlags::new(), value, host, 0);
    builder.ins().call(invalidate_reservations_func, &[bus]);
    finish_inline_instruction(builder, pc_var, instret_var, retired_var, instruction_bytes);
    flush_cached_xs(builder, cpu);
    builder.ins().jump(continue_block, &[]);

    builder.switch_to_block(slow_block);
    restore_cached_xs(cache_state);
    builder.seal_block(slow_block);
    emit_helper_instruction(
        builder,
        cpu,
        bus,
        trap,
        step_func,
        pc_var,
        instret_var,
        retired_var,
        exit_block,
        instruction_ptr,
    );
    flush_cached_xs(builder, cpu);
    builder.ins().jump(continue_block, &[]);

    builder.switch_to_block(continue_block);
    invalidate_cached_xs();
    builder.seal_block(continue_block);
}

#[allow(clippy::too_many_arguments)]
fn emit_translated_load_or_helper(
    builder: &mut FunctionBuilder,
    cpu: Value,
    bus: Value,
    trap: Value,
    translated_load_func: FuncRef,
    pc_var: Variable,
    instret_var: Variable,
    retired_var: Variable,
    exit_block: Block,
    instruction_bytes: u8,
    kind: LoadKind,
    rd: u8,
    rs1: u8,
    imm: i64,
    satp: u64,
    mstatus_vm: u64,
    privilege: PrivilegeLevel,
    collect_mmu_stats: bool,
) {
    let (size, ty, sign_extend) = match kind {
        LoadKind::Byte => (1, types::I8, true),
        LoadKind::Half => (2, types::I16, true),
        LoadKind::Word => (4, types::I32, true),
        LoadKind::Double => (8, types::I64, false),
        LoadKind::ByteUnsigned => (1, types::I8, false),
        LoadKind::HalfUnsigned => (2, types::I16, false),
        LoadKind::WordUnsigned => (4, types::I32, false),
    };

    let base = load_x(builder, cpu, rs1);
    let addr = builder.ins().iadd_imm(base, imm);
    let (host, hit) =
        emit_translated_host_ptr(builder, cpu, addr, size, satp, mstatus_vm, privilege, false);

    let fast_block = builder.create_block();
    let slow_block = builder.create_block();
    let continue_block = builder.create_block();
    let cache_state = snapshot_cached_xs();
    builder.ins().brif(hit, fast_block, &[], slow_block, &[]);

    builder.switch_to_block(fast_block);
    restore_cached_xs(cache_state);
    builder.seal_block(fast_block);
    if collect_mmu_stats {
        increment_mmu_counter(builder, cpu, MMU_STATS_TRANSLATED_LOAD_HITS_OFFSET);
    }
    let mut value = builder.ins().load(ty, MemFlags::new(), host, 0);
    value = match (ty, sign_extend) {
        (types::I64, _) => value,
        (_, true) => builder.ins().sextend(types::I64, value),
        (_, false) => builder.ins().uextend(types::I64, value),
    };
    store_x(builder, cpu, rd, value);
    finish_inline_instruction(builder, pc_var, instret_var, retired_var, instruction_bytes);
    builder.ins().jump(continue_block, &[]);

    builder.switch_to_block(slow_block);
    restore_cached_xs(cache_state);
    builder.seal_block(slow_block);
    if collect_mmu_stats {
        increment_mmu_counter(builder, cpu, MMU_STATS_TRANSLATED_LOAD_MISSES_OFFSET);
    }
    sync_cpu_state(builder, cpu, pc_var, instret_var);
    let kind_code = builder.ins().iconst(types::I64, load_kind_code(kind));
    let call = builder
        .ins()
        .call(translated_load_func, &[cpu, bus, trap, addr, kind_code]);
    let value = builder.inst_results(call)[0];
    emit_memory_trap_guard(builder, trap, retired_var, exit_block);
    restore_cached_xs(cache_state);
    store_x(builder, cpu, rd, value);
    finish_inline_instruction(builder, pc_var, instret_var, retired_var, instruction_bytes);
    builder.ins().jump(continue_block, &[]);

    builder.switch_to_block(continue_block);
    builder.seal_block(continue_block);
}

#[allow(clippy::too_many_arguments)]
fn emit_translated_store_or_helper(
    builder: &mut FunctionBuilder,
    cpu: Value,
    bus: Value,
    trap: Value,
    translated_store_func: FuncRef,
    invalidate_reservations_func: FuncRef,
    pc_var: Variable,
    instret_var: Variable,
    retired_var: Variable,
    exit_block: Block,
    instruction_bytes: u8,
    kind: StoreKind,
    rs1: u8,
    rs2: u8,
    imm: i64,
    satp: u64,
    mstatus_vm: u64,
    privilege: PrivilegeLevel,
    collect_mmu_stats: bool,
) {
    let (size, ty) = match kind {
        StoreKind::Byte => (1, types::I8),
        StoreKind::Half => (2, types::I16),
        StoreKind::Word => (4, types::I32),
        StoreKind::Double => (8, types::I64),
    };

    let base = load_x(builder, cpu, rs1);
    let addr = builder.ins().iadd_imm(base, imm);
    let (host, hit) =
        emit_translated_host_ptr(builder, cpu, addr, size, satp, mstatus_vm, privilege, true);

    let fast_block = builder.create_block();
    let slow_block = builder.create_block();
    let continue_block = builder.create_block();
    let cache_state = snapshot_cached_xs();
    builder.ins().brif(hit, fast_block, &[], slow_block, &[]);

    builder.switch_to_block(fast_block);
    restore_cached_xs(cache_state);
    builder.seal_block(fast_block);
    if collect_mmu_stats {
        increment_mmu_counter(builder, cpu, MMU_STATS_TRANSLATED_STORE_HITS_OFFSET);
    }
    let value = load_x(builder, cpu, rs2);
    let value = match ty {
        types::I8 => builder.ins().ireduce(types::I8, value),
        types::I16 => builder.ins().ireduce(types::I16, value),
        types::I32 => builder.ins().ireduce(types::I32, value),
        types::I64 => value,
        _ => unreachable!(),
    };
    builder.ins().store(MemFlags::new(), value, host, 0);
    builder.ins().call(invalidate_reservations_func, &[bus]);
    finish_inline_instruction(builder, pc_var, instret_var, retired_var, instruction_bytes);
    builder.ins().jump(continue_block, &[]);

    builder.switch_to_block(slow_block);
    restore_cached_xs(cache_state);
    builder.seal_block(slow_block);
    if collect_mmu_stats {
        increment_mmu_counter(builder, cpu, MMU_STATS_TRANSLATED_STORE_MISSES_OFFSET);
    }
    sync_cpu_state(builder, cpu, pc_var, instret_var);
    let value = load_x(builder, cpu, rs2);
    let kind_code = builder.ins().iconst(types::I64, store_kind_code(kind));
    builder.ins().call(
        translated_store_func,
        &[cpu, bus, trap, addr, value, kind_code],
    );
    emit_memory_trap_guard(builder, trap, retired_var, exit_block);
    restore_cached_xs(cache_state);
    finish_inline_instruction(builder, pc_var, instret_var, retired_var, instruction_bytes);
    builder.ins().jump(continue_block, &[]);

    builder.switch_to_block(continue_block);
    builder.seal_block(continue_block);
}

fn emit_translated_host_ptr(
    builder: &mut FunctionBuilder,
    cpu: Value,
    addr: Value,
    size: i64,
    satp: u64,
    mstatus_vm: u64,
    privilege: PrivilegeLevel,
    is_write: bool,
) -> (Value, Value) {
    let cache_offset = match (
        privilege,
        is_write,
        mstatus_vm & (MSTATUS_SUM | MSTATUS_MXR),
    ) {
        (PrivilegeLevel::User, false, _) => MMU_JIT_USER_READ_OFFSET,
        (PrivilegeLevel::User, true, _) => MMU_JIT_USER_WRITE_OFFSET,
        (PrivilegeLevel::Supervisor, false, 0) => MMU_JIT_SUPERVISOR_READ_OFFSET,
        (PrivilegeLevel::Supervisor, false, MSTATUS_SUM) => MMU_JIT_SUPERVISOR_SUM_READ_OFFSET,
        (PrivilegeLevel::Supervisor, false, MSTATUS_MXR) => MMU_JIT_SUPERVISOR_MXR_READ_OFFSET,
        (PrivilegeLevel::Supervisor, false, _) => MMU_JIT_SUPERVISOR_SUM_MXR_READ_OFFSET,
        (PrivilegeLevel::Supervisor, true, mode) if (mode & MSTATUS_SUM) != 0 => {
            MMU_JIT_SUPERVISOR_SUM_WRITE_OFFSET
        }
        (PrivilegeLevel::Supervisor, true, _) => MMU_JIT_SUPERVISOR_WRITE_OFFSET,
        (PrivilegeLevel::Machine, _, _) => unreachable!(),
    };
    let vpn = builder.ins().ushr_imm(addr, 12);
    let vpn_hash = builder.ins().ushr_imm(vpn, 8);
    let cache_index = builder.ins().bxor(vpn, vpn_hash);
    let cache_index = builder
        .ins()
        .band_imm(cache_index, JIT_CACHE_SIZE_MASK as i64);
    let cache_entry_stride = builder
        .ins()
        .imul_imm(cache_index, JIT_CACHE_ENTRY_SIZE as i64);
    let cache_ptr = builder.ins().load(
        types::I64,
        MemFlags::new(),
        cpu,
        CPU_MMU_OFFSET + cache_offset,
    );
    let entry_ptr = builder.ins().iadd(cache_ptr, cache_entry_stride);
    let entry_guest_page = builder.ins().load(
        types::I64,
        MemFlags::new(),
        entry_ptr,
        JIT_CACHE_ENTRY_GUEST_PAGE_OFFSET,
    );
    let host_addend = builder.ins().load(
        types::I64,
        MemFlags::new(),
        entry_ptr,
        JIT_CACHE_ENTRY_HOST_ADDEND_OFFSET,
    );
    let entry_context_tag = builder.ins().load(
        types::I64,
        MemFlags::new(),
        entry_ptr,
        JIT_CACHE_ENTRY_CONTEXT_TAG_OFFSET,
    );
    let guest_page = builder.ins().band_imm(addr, -((PAGE_SIZE) as i64));
    let page_match = builder
        .ins()
        .icmp(IntCC::Equal, guest_page, entry_guest_page);
    let context_tag = builder.ins().iconst(types::I64, satp as i64);
    let context_match = builder
        .ins()
        .icmp(IntCC::Equal, context_tag, entry_context_tag);
    let page_offset = builder.ins().band_imm(addr, (PAGE_SIZE - 1) as i64);
    let max_page_offset = builder
        .ins()
        .iconst(types::I64, (PAGE_SIZE - size as u64) as i64);
    let in_page = builder
        .ins()
        .icmp(IntCC::UnsignedLessThanOrEqual, page_offset, max_page_offset);

    let page_match = bool_to_i64(builder, page_match);
    let context_match = bool_to_i64(builder, context_match);
    let in_page = bool_to_i64(builder, in_page);

    let hit_mask = page_match;
    let hit_mask = builder.ins().band(hit_mask, context_match);
    let hit_mask = builder.ins().band(hit_mask, in_page);
    let hit = builder.ins().icmp_imm(IntCC::NotEqual, hit_mask, 0);
    let host = builder.ins().iadd(addr, host_addend);
    (host, hit)
}

extern "C" fn rvx_jit_exec_instruction(
    cpu: *mut Cpu,
    bus: *const Bus,
    trap_info: *mut TrapInfo,
    instruction: *const JitInstruction,
) -> u32 {
    let cpu = unsafe { &mut *cpu };
    let bus = unsafe { &*bus };
    let instruction = unsafe { &*instruction };
    match cpu.execute(instruction.decoded, instruction.instruction_bytes, bus) {
        Ok(StepOutcome::Continue | StepOutcome::Fence | StepOutcome::Halted) => {
            if instruction_writes_memory(instruction.decoded) {
                let _ = bus.invalidate_reservations();
            }
            BLOCK_STATUS_CONTINUE as u32
        }
        Err(trap) => {
            unsafe {
                *trap_info = TrapInfo {
                    cause: trap.cause(),
                    tval: trap.tval,
                    next_block: 0,
                };
            }
            BLOCK_STATUS_TRAP as u32
        }
    }
}

extern "C" fn rvx_jit_invalidate_reservations(bus: *const Bus) {
    let _ = unsafe { &*bus }.invalidate_reservations();
}

extern "C" fn rvx_jit_memory_barrier(bus: *const Bus) {
    unsafe { &*bus }.memory_barrier();
}

extern "C" fn rvx_jit_chain_lookup(cpu: *mut Cpu, bus: *const Bus) -> u64 {
    let ctx = CURRENT_CHAIN_LOOKUP_CTX.with(|slot| slot.get());
    let callback = CURRENT_CHAIN_LOOKUP_CALLBACK.with(|slot| slot.get());
    match (ctx.is_null(), callback) {
        (false, Some(callback)) => unsafe { callback(ctx, cpu, bus) },
        _ => 0,
    }
}

extern "C" fn rvx_jit_translated_load_slow(
    cpu: *mut Cpu,
    bus: *const Bus,
    trap_info: *mut TrapInfo,
    addr: u64,
    kind: u64,
) -> u64 {
    let cpu = unsafe { &mut *cpu };
    let bus = unsafe { &*bus };
    let trap_info = unsafe { &mut *trap_info };
    *trap_info = TrapInfo::default();

    let result = match decode_load_kind(kind) {
        Some(LoadKind::Byte) => cpu.load_u8(bus, addr).map(sext8),
        Some(LoadKind::Half) => cpu.load_u16(bus, addr).map(sext16),
        Some(LoadKind::Word) => cpu.load_u32(bus, addr).map(sext32),
        Some(LoadKind::Double) => cpu.load_u64(bus, addr),
        Some(LoadKind::ByteUnsigned) => cpu.load_u8(bus, addr).map(|value| value as u64),
        Some(LoadKind::HalfUnsigned) => cpu.load_u16(bus, addr).map(|value| value as u64),
        Some(LoadKind::WordUnsigned) => cpu.load_u32(bus, addr).map(|value| value as u64),
        None => panic!("invalid translated load kind {kind}"),
    };
    match result {
        Ok(value) => value,
        Err(trap) => {
            *trap_info = TrapInfo {
                cause: trap.cause(),
                tval: trap.tval,
                next_block: 0,
            };
            0
        }
    }
}

extern "C" fn rvx_jit_translated_store_slow(
    cpu: *mut Cpu,
    bus: *const Bus,
    trap_info: *mut TrapInfo,
    addr: u64,
    value: u64,
    kind: u64,
) {
    let cpu = unsafe { &mut *cpu };
    let bus = unsafe { &*bus };
    let trap_info = unsafe { &mut *trap_info };
    *trap_info = TrapInfo::default();

    let result = match decode_store_kind(kind) {
        Some(StoreKind::Byte) => cpu.store_u8(bus, addr, value as u8),
        Some(StoreKind::Half) => cpu.store_u16(bus, addr, value as u16),
        Some(StoreKind::Word) => cpu.store_u32(bus, addr, value as u32),
        Some(StoreKind::Double) => cpu.store_u64(bus, addr, value),
        None => panic!("invalid translated store kind {kind}"),
    };
    match result {
        Ok(()) => {
            let _ = bus.invalidate_reservations();
        }
        Err(trap) => {
            *trap_info = TrapInfo {
                cause: trap.cause(),
                tval: trap.tval,
                next_block: 0,
            };
        }
    }
}

fn decode_load_kind(kind: u64) -> Option<LoadKind> {
    match kind {
        0 => Some(LoadKind::Byte),
        1 => Some(LoadKind::Half),
        2 => Some(LoadKind::Word),
        3 => Some(LoadKind::Double),
        4 => Some(LoadKind::ByteUnsigned),
        5 => Some(LoadKind::HalfUnsigned),
        6 => Some(LoadKind::WordUnsigned),
        _ => None,
    }
}

fn decode_store_kind(kind: u64) -> Option<StoreKind> {
    match kind {
        0 => Some(StoreKind::Byte),
        1 => Some(StoreKind::Half),
        2 => Some(StoreKind::Word),
        3 => Some(StoreKind::Double),
        _ => None,
    }
}

fn sext8(value: u8) -> u64 {
    (value as i8) as i64 as u64
}

fn sext16(value: u16) -> u64 {
    (value as i16) as i64 as u64
}

fn sext32(value: u32) -> u64 {
    (value as i32) as i64 as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::{CSR_INSTRET, CSR_SSTATUS, MSTATUS_MPRV};
    use crate::mmu::{JIT_CACHE_SIZE, PTE_R, PTE_U, PTE_V, PTE_W};
    use rvx_core::{Bus, Ram};

    const RAM_BASE: u64 = 0x8000_0000;
    const RAM_SIZE: u64 = 0x20_000;
    const ROOT_PT: u64 = RAM_BASE + 0x1_000;
    const MID_PT: u64 = RAM_BASE + 0x2_000;
    const LEAF_PT: u64 = RAM_BASE + 0x3_000;
    const DATA_PAGE: u64 = RAM_BASE + 0x4_000;
    const USER_VA: u64 = 0x0000_0000_4000_1000;
    const SUPERPAGE_VA: u64 = 0x0000_0000_4020_0000;
    const SUPERPAGE_OFFSET: u64 = 0x10_000;
    const SV39_MODE: u64 = 8;

    #[test]
    fn jitted_user_mode_translated_load_works() {
        let bus = setup_sv39_bus();
        bus.write_u32(DATA_PAGE, 0x1234_5678).unwrap();

        let mut cpu = translated_user_cpu();
        cpu.write_x(1, USER_VA);
        let mut jit = JitEngine::new_with_options_internal(true, false, true).unwrap();
        let block = compile_test_block(
            &mut jit,
            &cpu,
            DecodedInstruction::Load {
                kind: LoadKind::WordUnsigned,
                rd: 2,
                rs1: 1,
                imm: 0,
            },
        );

        let result = execute_block(block, &mut cpu, &bus);
        assert_eq!(result.status, BlockStatus::Continue);
        assert_eq!(result.retired, 1);
        assert_eq!(cpu.read_x(2), 0x1234_5678);
    }

    #[test]
    fn jitted_user_mode_translated_store_works() {
        let bus = setup_sv39_bus();

        let mut cpu = translated_user_cpu();
        cpu.write_x(1, USER_VA);
        cpu.write_x(2, 0xfeed_beef);
        let mut jit = JitEngine::new_with_options_internal(true, false, true).unwrap();
        let block = compile_test_block(
            &mut jit,
            &cpu,
            DecodedInstruction::Store {
                kind: StoreKind::Word,
                rs1: 1,
                rs2: 2,
                imm: 0,
            },
        );

        let result = execute_block(block, &mut cpu, &bus);
        assert_eq!(result.status, BlockStatus::Continue);
        assert_eq!(result.retired, 1);
        assert_eq!(bus.read_u32(DATA_PAGE).unwrap(), 0xfeed_beef);
    }

    #[test]
    fn jitted_supervisor_mode_translated_load_works() {
        let bus = setup_sv39_supervisor_bus();
        bus.write_u32(DATA_PAGE, 0xcafe_babe).unwrap();

        let mut cpu = translated_supervisor_cpu();
        cpu.write_x(1, USER_VA);
        let mut jit = JitEngine::new_with_options(true, false).unwrap();
        let block = compile_test_block(
            &mut jit,
            &cpu,
            DecodedInstruction::Load {
                kind: LoadKind::WordUnsigned,
                rd: 2,
                rs1: 1,
                imm: 0,
            },
        );

        let result = execute_block(block, &mut cpu, &bus);
        assert_eq!(result.status, BlockStatus::Continue);
        assert_eq!(result.retired, 1);
        assert_eq!(cpu.read_x(2), 0xcafe_babe);
    }

    #[test]
    fn jitted_supervisor_mode_translated_store_works() {
        let bus = setup_sv39_supervisor_bus();

        let mut cpu = translated_supervisor_cpu();
        cpu.write_x(1, USER_VA);
        cpu.write_x(2, 0x0ddc_0ffe);
        let mut jit = JitEngine::new_with_options(true, false).unwrap();
        let block = compile_test_block(
            &mut jit,
            &cpu,
            DecodedInstruction::Store {
                kind: StoreKind::Word,
                rs1: 1,
                rs2: 2,
                imm: 0,
            },
        );

        let result = execute_block(block, &mut cpu, &bus);
        assert_eq!(result.status, BlockStatus::Continue);
        assert_eq!(result.retired, 1);
        assert_eq!(bus.read_u32(DATA_PAGE).unwrap(), 0x0ddc_0ffe);
    }

    #[test]
    fn jitted_supervisor_mode_cache_respects_sum_bit() {
        let bus = setup_sv39_bus();
        bus.write_u32(DATA_PAGE, 0x1357_9bdf).unwrap();

        let mut cpu = translated_supervisor_cpu();
        cpu.csr.mstatus = MSTATUS_SUM;
        cpu.write_x(1, USER_VA);

        let mut jit = JitEngine::new_with_options(true, false).unwrap();
        let warm_block = compile_test_block(
            &mut jit,
            &cpu,
            DecodedInstruction::Load {
                kind: LoadKind::WordUnsigned,
                rd: 2,
                rs1: 1,
                imm: 0,
            },
        );
        let warm = execute_block_allow_trap(warm_block, &mut cpu, &bus);
        assert_eq!(warm.result.status, BlockStatus::Continue);
        assert_eq!(cpu.read_x(2), 0x1357_9bdf);

        cpu.csr.mstatus &= !MSTATUS_SUM;
        let fault_block = compile_test_block(
            &mut jit,
            &cpu,
            DecodedInstruction::Load {
                kind: LoadKind::WordUnsigned,
                rd: 3,
                rs1: 1,
                imm: 0,
            },
        );
        let fault = execute_block_allow_trap(fault_block, &mut cpu, &bus);
        assert_eq!(fault.result.status, BlockStatus::Trap);
        assert_eq!(fault.result.retired, 0);
        assert_eq!(fault.trap.cause, crate::Exception::LoadPageFault as u64);
        assert_eq!(fault.trap.tval, USER_VA);
    }

    #[test]
    fn jitted_block_stops_after_vm_state_changing_csr() {
        let bus = setup_sv39_bus();
        bus.write_u32(DATA_PAGE, 0x2468_ace0).unwrap();

        let mut cpu = translated_supervisor_cpu();
        cpu.csr.mstatus = MSTATUS_SUM;
        cpu.write_x(1, USER_VA);
        cpu.write_x(5, MSTATUS_SUM);

        let mut jit = JitEngine::new_with_options(true, false).unwrap();
        let block = jit
            .compile_block(
                BlockKey::for_cpu(&cpu),
                vec![
                    JitInstruction {
                        decoded: DecodedInstruction::Load {
                            kind: LoadKind::WordUnsigned,
                            rd: 2,
                            rs1: 1,
                            imm: 0,
                        },
                        instruction_bytes: 4,
                    },
                    JitInstruction {
                        decoded: DecodedInstruction::Csrrc {
                            rd: 0,
                            rs1: 5,
                            csr: CSR_SSTATUS,
                        },
                        instruction_bytes: 4,
                    },
                ],
            )
            .unwrap()
            .entry();

        let result = execute_block(block, &mut cpu, &bus);
        assert_eq!(result.status, BlockStatus::Continue);
        assert_eq!(result.retired, 2);
        assert_eq!(cpu.read_x(2), 0x2468_ace0);
        assert_eq!(cpu.csr.mstatus & MSTATUS_SUM, 0);

        let follow_on = compile_test_block(
            &mut jit,
            &cpu,
            DecodedInstruction::Load {
                kind: LoadKind::WordUnsigned,
                rd: 3,
                rs1: 1,
                imm: 0,
            },
        );
        let follow_on = execute_block_allow_trap(follow_on, &mut cpu, &bus);
        assert_eq!(follow_on.result.status, BlockStatus::Trap);
        assert_eq!(follow_on.trap.cause, crate::Exception::LoadPageFault as u64);
        assert_eq!(follow_on.trap.tval, USER_VA);
    }

    #[test]
    fn jitted_block_syncs_instret_before_helper_instruction() {
        let bus = Bus::new(Ram::new(RAM_BASE, RAM_SIZE).unwrap());

        let mut cpu = Cpu::new();
        cpu.pc = 0x1_000;
        cpu.csr.instret = 41;

        let mut jit = JitEngine::new_with_options(true, false).unwrap();
        let block = jit
            .compile_block(
                BlockKey::for_cpu(&cpu),
                vec![
                    JitInstruction {
                        decoded: DecodedInstruction::Addi {
                            rd: 1,
                            rs1: 0,
                            imm: 1,
                        },
                        instruction_bytes: 4,
                    },
                    JitInstruction {
                        decoded: DecodedInstruction::Csrrs {
                            rd: 2,
                            rs1: 0,
                            csr: CSR_INSTRET,
                        },
                        instruction_bytes: 4,
                    },
                ],
            )
            .unwrap()
            .entry();

        let result = execute_block(block, &mut cpu, &bus);
        assert_eq!(result.status, BlockStatus::Continue);
        assert_eq!(result.retired, 2);
        assert_eq!(cpu.pc, 0x1_008);
        assert_eq!(cpu.csr.instret, 43);
        assert_eq!(cpu.read_x(1), 1);
        assert_eq!(cpu.read_x(2), 42);
    }

    #[test]
    fn jitted_inline_jumps_and_branches_update_pc_without_helper_calls() {
        let bus = Bus::new(Ram::new(RAM_BASE, RAM_SIZE).unwrap());

        let mut cpu = Cpu::new();
        cpu.pc = 0x1_000;
        cpu.write_x(3, 0x2_000);
        cpu.write_x(4, 5);
        cpu.write_x(5, 5);

        let mut jit = JitEngine::new_with_options(true, false).unwrap();
        let jal = jit
            .compile_block(
                BlockKey::for_cpu(&cpu),
                vec![JitInstruction {
                    decoded: DecodedInstruction::Jal { rd: 1, imm: 16 },
                    instruction_bytes: 4,
                }],
            )
            .unwrap()
            .entry();
        let result = execute_block(jal, &mut cpu, &bus);
        assert_eq!(result.status, BlockStatus::Continue);
        assert_eq!(result.retired, 1);
        assert_eq!(cpu.pc, 0x1_010);
        assert_eq!(cpu.read_x(1), 0x1_004);

        cpu.pc = 0x3_000;
        let jalr = jit
            .compile_block(
                BlockKey::for_cpu(&cpu),
                vec![JitInstruction {
                    decoded: DecodedInstruction::Jalr {
                        rd: 2,
                        rs1: 3,
                        imm: 3,
                    },
                    instruction_bytes: 4,
                }],
            )
            .unwrap()
            .entry();
        let result = execute_block(jalr, &mut cpu, &bus);
        assert_eq!(result.status, BlockStatus::Continue);
        assert_eq!(result.retired, 1);
        assert_eq!(cpu.pc, 0x2_002);
        assert_eq!(cpu.read_x(2), 0x3_004);

        cpu.pc = 0x4_000;
        let branch = jit
            .compile_block(
                BlockKey::for_cpu(&cpu),
                vec![JitInstruction {
                    decoded: DecodedInstruction::Branch {
                        kind: BranchKind::Eq,
                        rs1: 4,
                        rs2: 5,
                        imm: 12,
                    },
                    instruction_bytes: 4,
                }],
            )
            .unwrap()
            .entry();
        let result = execute_block(branch, &mut cpu, &bus);
        assert_eq!(result.status, BlockStatus::Continue);
        assert_eq!(result.retired, 1);
        assert_eq!(cpu.pc, 0x4_00c);
    }

    #[test]
    fn jitted_blocks_can_chain_to_patched_static_successor() {
        let bus = Bus::new(Ram::new(RAM_BASE, RAM_SIZE).unwrap());

        let mut cpu = Cpu::new();
        cpu.pc = 0x1_000;

        let mut jit = JitEngine::new_with_options_internal(true, false, true).unwrap();
        let first = jit
            .compile_block(
                BlockKey::for_cpu(&cpu),
                vec![JitInstruction {
                    decoded: DecodedInstruction::Jal { rd: 1, imm: 4 },
                    instruction_bytes: 4,
                }],
            )
            .unwrap()
            .entry();

        let mut next_cpu = cpu.clone();
        next_cpu.pc = 0x1_004;
        jit.compile_block(
            BlockKey::for_cpu(&next_cpu),
            vec![JitInstruction {
                decoded: DecodedInstruction::Addi {
                    rd: 2,
                    rs1: 0,
                    imm: 7,
                },
                instruction_bytes: 4,
            }],
        )
        .unwrap();

        let result = execute_block(first, &mut cpu, &bus);
        assert_eq!(result.status, BlockStatus::Continue);
        assert_eq!(result.retired, 2);
        assert_eq!(cpu.pc, 0x1_008);
        assert_eq!(cpu.read_x(1), 0x1_004);
        assert_eq!(cpu.read_x(2), 7);
    }

    #[test]
    fn jitted_jalr_can_chain_through_patched_target_cache() {
        let bus = Bus::new(Ram::new(RAM_BASE, RAM_SIZE).unwrap());

        let mut cpu = Cpu::new();
        cpu.pc = 0x3_000;
        cpu.write_x(3, 0x2_000);

        let mut jit = JitEngine::new_with_options_internal(true, false, true).unwrap();
        let source_key = BlockKey::for_cpu(&cpu);
        let jalr = jit
            .compile_block(
                source_key,
                vec![JitInstruction {
                    decoded: DecodedInstruction::Jalr {
                        rd: 2,
                        rs1: 3,
                        imm: 3,
                    },
                    instruction_bytes: 4,
                }],
            )
            .unwrap()
            .entry();

        let mut target_cpu = cpu.clone();
        target_cpu.pc = 0x2_002;
        let target_key = BlockKey::for_cpu(&target_cpu);
        jit.compile_block(
            target_key,
            vec![JitInstruction {
                decoded: DecodedInstruction::Addi {
                    rd: 6,
                    rs1: 0,
                    imm: 42,
                },
                instruction_bytes: 4,
            }],
        )
        .unwrap();
        assert!(jit.patch_jalr_chain(source_key, target_key));

        let result = execute_block(jalr, &mut cpu, &bus);
        assert_eq!(result.status, BlockStatus::Continue);
        assert_eq!(result.retired, 2);
        assert_eq!(cpu.pc, 0x2_006);
        assert_eq!(cpu.read_x(2), 0x3_004);
        assert_eq!(cpu.read_x(6), 42);
    }

    #[test]
    fn jitted_branch_can_chain_to_both_static_successors() {
        let bus = Bus::new(Ram::new(RAM_BASE, RAM_SIZE).unwrap());

        let mut taken_cpu = Cpu::new();
        taken_cpu.pc = 0x2_000;
        taken_cpu.write_x(1, 1);
        taken_cpu.write_x(2, 1);

        let mut jit = JitEngine::new_with_options_internal(true, false, true).unwrap();
        let branch = jit
            .compile_block(
                BlockKey::for_cpu(&taken_cpu),
                vec![JitInstruction {
                    decoded: DecodedInstruction::Branch {
                        kind: BranchKind::Eq,
                        rs1: 1,
                        rs2: 2,
                        imm: 8,
                    },
                    instruction_bytes: 4,
                }],
            )
            .unwrap()
            .entry();

        let mut taken_target_cpu = taken_cpu.clone();
        taken_target_cpu.pc = 0x2_008;
        jit.compile_block(
            BlockKey::for_cpu(&taken_target_cpu),
            vec![JitInstruction {
                decoded: DecodedInstruction::Addi {
                    rd: 3,
                    rs1: 0,
                    imm: 11,
                },
                instruction_bytes: 4,
            }],
        )
        .unwrap();

        let mut fallthrough_cpu = taken_cpu.clone();
        fallthrough_cpu.write_x(2, 2);
        let mut fallthrough_target_cpu = fallthrough_cpu.clone();
        fallthrough_target_cpu.pc = 0x2_004;
        jit.compile_block(
            BlockKey::for_cpu(&fallthrough_target_cpu),
            vec![JitInstruction {
                decoded: DecodedInstruction::Addi {
                    rd: 4,
                    rs1: 0,
                    imm: 13,
                },
                instruction_bytes: 4,
            }],
        )
        .unwrap();

        let taken = execute_block(branch, &mut taken_cpu, &bus);
        assert_eq!(taken.status, BlockStatus::Continue);
        assert_eq!(taken.retired, 2);
        assert_eq!(taken_cpu.pc, 0x2_00c);
        assert_eq!(taken_cpu.read_x(3), 11);

        let fallthrough = execute_block(branch, &mut fallthrough_cpu, &bus);
        assert_eq!(fallthrough.status, BlockStatus::Continue);
        assert_eq!(fallthrough.retired, 2);
        assert_eq!(fallthrough_cpu.pc, 0x2_008);
        assert_eq!(fallthrough_cpu.read_x(4), 13);
    }

    #[test]
    fn jitted_traced_branch_continues_on_predicted_path_and_exits_on_other_path() {
        let bus = Bus::new(Ram::new(RAM_BASE, RAM_SIZE).unwrap());

        let mut cpu = Cpu::new();
        cpu.pc = 0x1_000;
        cpu.write_x(1, 1);
        cpu.write_x(2, 2);

        let mut jit = JitEngine::new_with_options(true, false).unwrap();
        let block = jit
            .compile_block(
                BlockKey::for_cpu(&cpu),
                vec![
                    JitInstruction {
                        decoded: DecodedInstruction::Branch {
                            kind: BranchKind::Eq,
                            rs1: 1,
                            rs2: 2,
                            imm: 12,
                        },
                        instruction_bytes: 4,
                    },
                    JitInstruction {
                        decoded: DecodedInstruction::Addi {
                            rd: 3,
                            rs1: 0,
                            imm: 7,
                        },
                        instruction_bytes: 4,
                    },
                ],
            )
            .unwrap()
            .entry();

        let result = execute_block(block, &mut cpu, &bus);
        assert_eq!(result.status, BlockStatus::Continue);
        assert_eq!(result.retired, 2);
        assert_eq!(cpu.pc, 0x1_008);
        assert_eq!(cpu.read_x(3), 7);

        cpu.pc = 0x1_000;
        cpu.csr.instret = 0;
        cpu.write_x(1, 5);
        cpu.write_x(2, 5);
        cpu.write_x(3, 0);
        let result = execute_block(block, &mut cpu, &bus);
        assert_eq!(result.status, BlockStatus::Continue);
        assert_eq!(result.retired, 1);
        assert_eq!(cpu.pc, 0x1_00c);
        assert_eq!(cpu.read_x(3), 0);
    }

    #[test]
    fn translated_fault_syncs_current_pc_after_prior_inline_instruction() {
        let bus = setup_sv39_bus();

        let mut cpu = translated_supervisor_cpu();
        cpu.csr.instret = 9;
        cpu.write_x(1, USER_VA);

        let mut jit = JitEngine::new_with_options(true, false).unwrap();
        let block = jit
            .compile_block(
                BlockKey::for_cpu(&cpu),
                vec![
                    JitInstruction {
                        decoded: DecodedInstruction::Addi {
                            rd: 3,
                            rs1: 0,
                            imm: 7,
                        },
                        instruction_bytes: 4,
                    },
                    JitInstruction {
                        decoded: DecodedInstruction::Load {
                            kind: LoadKind::WordUnsigned,
                            rd: 2,
                            rs1: 1,
                            imm: 0,
                        },
                        instruction_bytes: 4,
                    },
                ],
            )
            .unwrap()
            .entry();

        let fault = execute_block_allow_trap(block, &mut cpu, &bus);
        assert_eq!(fault.result.status, BlockStatus::Trap);
        assert_eq!(fault.result.retired, 1);
        assert_eq!(fault.trap.cause, crate::Exception::LoadPageFault as u64);
        assert_eq!(fault.trap.tval, USER_VA);
        assert_eq!(cpu.pc, 0x1_004);
        assert_eq!(cpu.csr.instret, 10);
        assert_eq!(cpu.read_x(3), 7);
    }

    #[test]
    fn jitted_supervisor_cache_uses_lookup_page_for_superpages() {
        let bus = Bus::new(Ram::new(RAM_BASE, RAM_SIZE).unwrap());
        map_sv39_superpage(&bus, SUPERPAGE_VA, RAM_BASE, PTE_R | PTE_W);
        bus.write_u32(RAM_BASE + SUPERPAGE_OFFSET, 0xdead_beef)
            .unwrap();

        let mut cpu = translated_supervisor_cpu();
        cpu.write_x(1, SUPERPAGE_VA + SUPERPAGE_OFFSET);
        let mut jit = JitEngine::new_with_options(true, false).unwrap();
        let block = compile_test_block(
            &mut jit,
            &cpu,
            DecodedInstruction::Load {
                kind: LoadKind::WordUnsigned,
                rd: 2,
                rs1: 1,
                imm: 0,
            },
        );

        let result = execute_block(block, &mut cpu, &bus);
        assert_eq!(result.status, BlockStatus::Continue);
        assert_eq!(cpu.read_x(2), 0xdead_beef);

        let vpn = (SUPERPAGE_VA + SUPERPAGE_OFFSET) >> 12;
        let idx = ((vpn ^ (vpn >> 8)) as usize) & (JIT_CACHE_SIZE - 1);
        let entry = cpu.mmu.jit_supervisor_read[idx];
        let guest_page = (SUPERPAGE_VA + SUPERPAGE_OFFSET) & !0xfff;
        let host_page = (bus.ram().as_mut_ptr() as usize as u64) + SUPERPAGE_OFFSET;
        assert_eq!(entry.guest_page, guest_page);
        assert_eq!(entry.host_addend, host_page.wrapping_sub(guest_page));
        assert_eq!(entry.context_tag, cpu.csr.satp);
    }

    #[test]
    fn block_key_tracks_effective_mprv_data_privilege() {
        let mut cpu = Cpu::new();
        cpu.privilege = PrivilegeLevel::Machine;
        cpu.pc = 0x1000;
        cpu.csr.satp = (SV39_MODE << SATP_MODE_SHIFT as u64) | ppn(ROOT_PT);
        cpu.csr.mstatus = MSTATUS_MPRV | MSTATUS_SUM | ((PrivilegeLevel::Supervisor as u64) << 11);

        let key = BlockKey::for_cpu(&cpu);
        assert_eq!(key.privilege, PrivilegeLevel::Machine);
        assert_eq!(key.data_privilege, PrivilegeLevel::Supervisor);
        assert_eq!(key.mstatus_vm, MSTATUS_SUM);
    }

    fn translated_user_cpu() -> Cpu {
        let mut cpu = Cpu::new();
        cpu.privilege = PrivilegeLevel::User;
        cpu.pc = 0x1_000;
        cpu.csr.satp = (SV39_MODE << SATP_MODE_SHIFT as u64) | ppn(ROOT_PT);
        cpu
    }

    fn translated_supervisor_cpu() -> Cpu {
        let mut cpu = Cpu::new();
        cpu.privilege = PrivilegeLevel::Supervisor;
        cpu.pc = 0x1_000;
        cpu.csr.satp = (SV39_MODE << SATP_MODE_SHIFT as u64) | ppn(ROOT_PT);
        cpu
    }

    fn compile_test_block(
        jit: &mut JitEngine,
        cpu: &Cpu,
        insn: DecodedInstruction,
    ) -> CompiledBlockEntry {
        let block = jit
            .compile_block(
                BlockKey::for_cpu(cpu),
                vec![JitInstruction {
                    decoded: insn,
                    instruction_bytes: 4,
                }],
            )
            .unwrap();
        block.entry()
    }

    fn execute_block(entry: CompiledBlockEntry, cpu: &mut Cpu, bus: &Bus) -> BlockExecution {
        let executed = execute_block_allow_trap(entry, cpu, bus);
        assert_eq!(executed.trap.cause, 0);
        assert_eq!(executed.trap.tval, 0);
        executed.result
    }

    struct ExecutedBlock {
        result: BlockExecution,
        trap: TrapInfo,
    }

    fn execute_block_allow_trap(
        entry: CompiledBlockEntry,
        cpu: &mut Cpu,
        bus: &Bus,
    ) -> ExecutedBlock {
        let mut trap = TrapInfo::default();
        let mut current = entry;
        let mut retired = 0u64;
        let packed = loop {
            trap.next_block = 0;
            let packed = unsafe {
                current(
                    cpu,
                    bus,
                    &mut trap,
                    bus.ram().base(),
                    bus.ram().len(),
                    bus.ram().as_mut_ptr(),
                    retired,
                    u64::MAX,
                )
            };
            let result = BlockExecution::from_packed(packed);
            retired = result.retired as u64;
            match result.status {
                BlockStatus::Chain if trap.next_block != 0 => {
                    let block = unsafe { &*(trap.next_block as usize as *const CompiledBlock) };
                    current = block.entry();
                }
                _ => break packed,
            }
        };
        ExecutedBlock {
            result: BlockExecution::from_packed(packed),
            trap,
        }
    }

    fn setup_sv39_bus() -> Bus {
        let bus = Bus::new(Ram::new(RAM_BASE, RAM_SIZE).unwrap());
        map_sv39_page(&bus, USER_VA, DATA_PAGE, PTE_R | PTE_W | PTE_U);
        bus
    }

    fn setup_sv39_supervisor_bus() -> Bus {
        let bus = Bus::new(Ram::new(RAM_BASE, RAM_SIZE).unwrap());
        map_sv39_page(&bus, USER_VA, DATA_PAGE, PTE_R | PTE_W);
        bus
    }

    fn map_sv39_page(bus: &Bus, va: u64, pa: u64, flags: u64) {
        let root_index = vpn_index(va, 2);
        let mid_index = vpn_index(va, 1);
        let leaf_index = vpn_index(va, 0);
        bus.write_u64(ROOT_PT + root_index * 8, pte(MID_PT, PTE_V))
            .unwrap();
        bus.write_u64(MID_PT + mid_index * 8, pte(LEAF_PT, PTE_V))
            .unwrap();
        bus.write_u64(LEAF_PT + leaf_index * 8, pte(pa, flags | PTE_V))
            .unwrap();
    }

    fn map_sv39_superpage(bus: &Bus, va: u64, pa: u64, flags: u64) {
        let root_index = vpn_index(va, 2);
        let mid_index = vpn_index(va, 1);
        bus.write_u64(ROOT_PT + root_index * 8, pte(MID_PT, PTE_V))
            .unwrap();
        bus.write_u64(MID_PT + mid_index * 8, pte(pa, flags | PTE_V))
            .unwrap();
    }

    fn vpn_index(addr: u64, level: u32) -> u64 {
        (addr >> (12 + level * 9)) & 0x1ff
    }

    fn pte(addr: u64, flags: u64) -> u64 {
        (ppn(addr) << 10) | flags
    }

    fn ppn(addr: u64) -> u64 {
        addr >> 12
    }
}
