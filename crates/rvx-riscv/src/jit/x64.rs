#[cfg(all(target_arch = "x86_64", unix))]
use std::mem;
#[cfg(all(target_arch = "x86_64", unix))]
use std::ptr::{self, NonNull};

#[cfg(all(target_arch = "x86_64", unix))]
use anyhow::{Context, Result};

#[cfg(all(target_arch = "x86_64", unix))]
use super::*;

#[cfg(all(target_arch = "x86_64", unix))]
pub(super) struct ExecutableBlock {
    ptr: NonNull<u8>,
    mapped_len: usize,
}

#[cfg(all(target_arch = "x86_64", unix))]
unsafe impl Send for ExecutableBlock {}

#[cfg(all(target_arch = "x86_64", unix))]
unsafe impl Sync for ExecutableBlock {}

#[cfg(all(target_arch = "x86_64", unix))]
impl Drop for ExecutableBlock {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr().cast(), self.mapped_len);
        }
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
#[allow(dead_code)]
#[derive(Clone, Copy)]
enum Reg {
    Rax = 0,
    Rcx = 1,
    Rdx = 2,
    Rbx = 3,
    Rsp = 4,
    Rbp = 5,
    Rsi = 6,
    Rdi = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

#[cfg(all(target_arch = "x86_64", unix))]
impl Reg {
    fn low3(self) -> u8 {
        (self as u8) & 7
    }

    fn high(self) -> bool {
        (self as u8) >= 8
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
#[derive(Clone, Copy)]
struct Label(usize);

#[cfg(all(target_arch = "x86_64", unix))]
struct Fixup {
    disp_pos: usize,
    label: Label,
}

#[cfg(all(target_arch = "x86_64", unix))]
struct Assembler {
    code: Vec<u8>,
    labels: Vec<Option<usize>>,
    fixups: Vec<Fixup>,
}

#[cfg(all(target_arch = "x86_64", unix))]
impl Assembler {
    fn new() -> Self {
        Self {
            code: Vec::with_capacity(256),
            labels: Vec::new(),
            fixups: Vec::new(),
        }
    }

    fn finish(mut self) -> Result<Vec<u8>> {
        for fixup in self.fixups {
            let target = self.labels[fixup.label.0].context("unbound x64 jit label")?;
            let disp_end = fixup.disp_pos + 4;
            let rel = (target as isize) - (disp_end as isize);
            let rel: i32 = rel
                .try_into()
                .context("x64 jit branch target out of range")?;
            self.code[fixup.disp_pos..disp_end].copy_from_slice(&rel.to_le_bytes());
        }
        Ok(self.code)
    }

    fn create_label(&mut self) -> Label {
        let label = Label(self.labels.len());
        self.labels.push(None);
        label
    }

    fn bind(&mut self, label: Label) {
        self.labels[label.0] = Some(self.code.len());
    }

    fn emit_u8(&mut self, byte: u8) {
        self.code.push(byte);
    }

    fn emit_u32(&mut self, value: u32) {
        self.code.extend_from_slice(&value.to_le_bytes());
    }

    fn emit_i32(&mut self, value: i32) {
        self.code.extend_from_slice(&value.to_le_bytes());
    }

    fn emit_u64(&mut self, value: u64) {
        self.code.extend_from_slice(&value.to_le_bytes());
    }

    fn rex(&mut self, w: bool, r: bool, x: bool, b: bool) {
        let rex = 0x40 | ((w as u8) << 3) | ((r as u8) << 2) | ((x as u8) << 1) | (b as u8);
        if rex != 0x40 {
            self.emit_u8(rex);
        }
    }

    fn modrm(&mut self, mode: u8, reg: u8, rm: u8) {
        self.emit_u8((mode << 6) | ((reg & 7) << 3) | (rm & 7));
    }

    fn push(&mut self, reg: Reg) {
        if reg.high() {
            self.emit_u8(0x41);
        }
        self.emit_u8(0x50 + reg.low3());
    }

    fn pop(&mut self, reg: Reg) {
        if reg.high() {
            self.emit_u8(0x41);
        }
        self.emit_u8(0x58 + reg.low3());
    }

    fn ret(&mut self) {
        self.emit_u8(0xc3);
    }

    fn mov_rr64(&mut self, dst: Reg, src: Reg) {
        self.rex(true, src.high(), false, dst.high());
        self.emit_u8(0x89);
        self.modrm(0b11, src.low3(), dst.low3());
    }

    fn mov_ri64(&mut self, dst: Reg, imm: u64) {
        self.rex(true, false, false, dst.high());
        self.emit_u8(0xb8 + dst.low3());
        self.emit_u64(imm);
    }

    fn mov_rm64(&mut self, dst: Reg, base: Reg, disp: i32) {
        self.rex(true, dst.high(), false, base.high());
        self.emit_u8(0x8b);
        self.modrm(0b10, dst.low3(), base.low3());
        self.emit_i32(disp);
    }

    fn mov_mr64(&mut self, base: Reg, disp: i32, src: Reg) {
        self.rex(true, src.high(), false, base.high());
        self.emit_u8(0x89);
        self.modrm(0b10, src.low3(), base.low3());
        self.emit_i32(disp);
    }

    fn xor_rr32(&mut self, dst: Reg, src: Reg) {
        self.rex(false, src.high(), false, dst.high());
        self.emit_u8(0x31);
        self.modrm(0b11, src.low3(), dst.low3());
    }

    fn alu_ri64(&mut self, subop: u8, dst: Reg, imm: i32) {
        self.rex(true, false, false, dst.high());
        self.emit_u8(0x81);
        self.modrm(0b11, subop, dst.low3());
        self.emit_i32(imm);
    }

    fn alu_ri32(&mut self, subop: u8, dst: Reg, imm: i32) {
        self.rex(false, false, false, dst.high());
        self.emit_u8(0x81);
        self.modrm(0b11, subop, dst.low3());
        self.emit_i32(imm);
    }

    fn alu_rr64(&mut self, opcode: u8, dst: Reg, src: Reg) {
        self.rex(true, src.high(), false, dst.high());
        self.emit_u8(opcode);
        self.modrm(0b11, src.low3(), dst.low3());
    }

    fn alu_rr32(&mut self, opcode: u8, dst: Reg, src: Reg) {
        self.rex(false, src.high(), false, dst.high());
        self.emit_u8(opcode);
        self.modrm(0b11, src.low3(), dst.low3());
    }

    fn add_mem_imm32(&mut self, base: Reg, disp: i32, imm: i32) {
        self.rex(true, false, false, base.high());
        self.emit_u8(0x81);
        self.modrm(0b10, 0, base.low3());
        self.emit_i32(disp);
        self.emit_i32(imm);
    }

    fn shift_imm64(&mut self, subop: u8, reg: Reg, imm: u8) {
        self.rex(true, false, false, reg.high());
        self.emit_u8(0xc1);
        self.modrm(0b11, subop, reg.low3());
        self.emit_u8(imm);
    }

    fn shift_imm32(&mut self, subop: u8, reg: Reg, imm: u8) {
        self.rex(false, false, false, reg.high());
        self.emit_u8(0xc1);
        self.modrm(0b11, subop, reg.low3());
        self.emit_u8(imm);
    }

    fn shift_cl64(&mut self, subop: u8, reg: Reg) {
        self.rex(true, false, false, reg.high());
        self.emit_u8(0xd3);
        self.modrm(0b11, subop, reg.low3());
    }

    fn shift_cl32(&mut self, subop: u8, reg: Reg) {
        self.rex(false, false, false, reg.high());
        self.emit_u8(0xd3);
        self.modrm(0b11, subop, reg.low3());
    }

    fn cmp_rr64(&mut self, lhs: Reg, rhs: Reg) {
        self.rex(true, rhs.high(), false, lhs.high());
        self.emit_u8(0x39);
        self.modrm(0b11, rhs.low3(), lhs.low3());
    }

    fn test_rr32(&mut self, lhs: Reg, rhs: Reg) {
        self.rex(false, rhs.high(), false, lhs.high());
        self.emit_u8(0x85);
        self.modrm(0b11, rhs.low3(), lhs.low3());
    }

    fn setcc_al(&mut self, cc: u8) {
        self.emit_u8(0x0f);
        self.emit_u8(0x90 | cc);
        self.emit_u8(0xc0);
    }

    fn movzx_rax_al(&mut self) {
        self.emit_u8(0x48);
        self.emit_u8(0x0f);
        self.emit_u8(0xb6);
        self.emit_u8(0xc0);
    }

    fn cdqe(&mut self) {
        self.emit_u8(0x48);
        self.emit_u8(0x98);
    }

    fn call_reg(&mut self, reg: Reg) {
        self.rex(false, false, false, reg.high());
        self.emit_u8(0xff);
        self.modrm(0b11, 2, reg.low3());
    }

    fn jcc(&mut self, cc: u8, label: Label) {
        self.emit_u8(0x0f);
        self.emit_u8(0x80 | cc);
        let disp_pos = self.code.len();
        self.emit_u32(0);
        self.fixups.push(Fixup { disp_pos, label });
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
const CC_NE: u8 = 0x5;
#[cfg(all(target_arch = "x86_64", unix))]
const CC_L: u8 = 0xc;
#[cfg(all(target_arch = "x86_64", unix))]
const CC_B: u8 = 0x2;

#[cfg(all(target_arch = "x86_64", unix))]
pub(super) fn try_compile_block(
    key: BlockKey,
    instructions: &[JitInstruction],
) -> Result<Option<(CompiledBlockEntry, ExecutableBlock)>> {
    let mut asm = Assembler::new();
    let continue_return = asm.create_label();
    let trap_return = asm.create_label();
    let helper_addr = rvx_jit_exec_instruction as *const () as usize as u64;

    emit_prologue(&mut asm, helper_addr);

    let mut current_pc = key.pc;
    for (index, instruction) in instructions.iter().enumerate() {
        let is_last = index + 1 == instructions.len();
        let expected_next_pc = trace_successor(current_pc, *instruction);
        if !emit_inline_instruction(&mut asm, current_pc, *instruction) {
            emit_helper_instruction(&mut asm, instruction as *const JitInstruction as u64);
            asm.test_rr32(Reg::Rax, Reg::Rax);
            asm.jcc(CC_NE, trap_return);
            asm.alu_ri64(0, Reg::R14, 1);
            if !is_last && instruction_may_change_pc(instruction.decoded) {
                asm.mov_rm64(Reg::Rax, Reg::Rbx, CPU_PC_OFFSET);
                asm.mov_ri64(Reg::Rdx, expected_next_pc);
                asm.cmp_rr64(Reg::Rax, Reg::Rdx);
                asm.jcc(CC_NE, continue_return);
            }
        }
        current_pc = expected_next_pc;
    }

    asm.bind(continue_return);
    emit_return(&mut asm, BLOCK_STATUS_CONTINUE as i32);
    asm.bind(trap_return);
    emit_return(&mut asm, BLOCK_STATUS_TRAP as i32);

    let code = asm.finish()?;
    let executable = map_executable(&code)?;
    let entry = unsafe { mem::transmute::<*const u8, CompiledBlockEntry>(executable.ptr.as_ptr()) };
    Ok(Some((entry, executable)))
}

#[cfg(all(target_arch = "x86_64", unix))]
fn emit_prologue(asm: &mut Assembler, helper_addr: u64) {
    asm.push(Reg::Rbx);
    asm.push(Reg::R12);
    asm.push(Reg::R13);
    asm.push(Reg::R14);
    asm.push(Reg::R15);
    asm.mov_rr64(Reg::Rbx, Reg::Rdi);
    asm.mov_rr64(Reg::R12, Reg::Rsi);
    asm.mov_rr64(Reg::R13, Reg::Rdx);
    asm.xor_rr32(Reg::R14, Reg::R14);
    asm.mov_ri64(Reg::R15, helper_addr);
}

#[cfg(all(target_arch = "x86_64", unix))]
fn emit_return(asm: &mut Assembler, status: i32) {
    asm.mov_rr64(Reg::Rax, Reg::R14);
    asm.shift_imm64(4, Reg::Rax, 32);
    if status != 0 {
        asm.alu_ri64(1, Reg::Rax, status);
    }
    asm.pop(Reg::R15);
    asm.pop(Reg::R14);
    asm.pop(Reg::R13);
    asm.pop(Reg::R12);
    asm.pop(Reg::Rbx);
    asm.ret();
}

#[cfg(all(target_arch = "x86_64", unix))]
fn emit_helper_instruction(asm: &mut Assembler, instruction_ptr: u64) {
    asm.mov_rr64(Reg::Rdi, Reg::Rbx);
    asm.mov_rr64(Reg::Rsi, Reg::R12);
    asm.mov_rr64(Reg::Rdx, Reg::R13);
    asm.mov_ri64(Reg::Rcx, instruction_ptr);
    asm.call_reg(Reg::R15);
}

#[cfg(all(target_arch = "x86_64", unix))]
fn emit_inline_instruction(asm: &mut Assembler, current_pc: u64, instruction: JitInstruction) -> bool {
    match instruction.decoded {
        DecodedInstruction::Lui { rd, imm } => {
            asm.mov_ri64(Reg::Rax, imm as u64);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Auipc { rd, imm } => {
            asm.mov_ri64(Reg::Rax, current_pc.wrapping_add_signed(imm));
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Addi { rd, rs1, imm } => {
            load_x(asm, Reg::Rax, rs1);
            asm.alu_ri64(0, Reg::Rax, imm as i32);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Slti { rd, rs1, imm } => {
            load_x(asm, Reg::Rax, rs1);
            asm.mov_ri64(Reg::Rdx, imm as u64);
            asm.cmp_rr64(Reg::Rax, Reg::Rdx);
            asm.setcc_al(CC_L);
            asm.movzx_rax_al();
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Sltiu { rd, rs1, imm } => {
            load_x(asm, Reg::Rax, rs1);
            asm.mov_ri64(Reg::Rdx, imm as u64);
            asm.cmp_rr64(Reg::Rax, Reg::Rdx);
            asm.setcc_al(CC_B);
            asm.movzx_rax_al();
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Xori { rd, rs1, imm } => {
            load_x(asm, Reg::Rax, rs1);
            asm.alu_ri64(6, Reg::Rax, imm as i32);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Ori { rd, rs1, imm } => {
            load_x(asm, Reg::Rax, rs1);
            asm.alu_ri64(1, Reg::Rax, imm as i32);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Andi { rd, rs1, imm } => {
            load_x(asm, Reg::Rax, rs1);
            asm.alu_ri64(4, Reg::Rax, imm as i32);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Slli { rd, rs1, shamt } => {
            load_x(asm, Reg::Rax, rs1);
            asm.shift_imm64(4, Reg::Rax, shamt);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Srli { rd, rs1, shamt } => {
            load_x(asm, Reg::Rax, rs1);
            asm.shift_imm64(5, Reg::Rax, shamt);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Srai { rd, rs1, shamt } => {
            load_x(asm, Reg::Rax, rs1);
            asm.shift_imm64(7, Reg::Rax, shamt);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Addiw { rd, rs1, imm } => {
            load_x(asm, Reg::Rax, rs1);
            asm.alu_ri32(0, Reg::Rax, imm as i32);
            asm.cdqe();
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Slliw { rd, rs1, shamt } => {
            load_x(asm, Reg::Rax, rs1);
            asm.shift_imm32(4, Reg::Rax, shamt);
            asm.cdqe();
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Srliw { rd, rs1, shamt } => {
            load_x(asm, Reg::Rax, rs1);
            asm.shift_imm32(5, Reg::Rax, shamt);
            asm.cdqe();
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Sraiw { rd, rs1, shamt } => {
            load_x(asm, Reg::Rax, rs1);
            asm.shift_imm32(7, Reg::Rax, shamt);
            asm.cdqe();
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Add { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rdx, rs2);
            asm.alu_rr64(0x01, Reg::Rax, Reg::Rdx);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Sub { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rdx, rs2);
            asm.alu_rr64(0x29, Reg::Rax, Reg::Rdx);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Sll { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rcx, rs2);
            asm.alu_ri32(4, Reg::Rcx, 63);
            asm.shift_cl64(4, Reg::Rax);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Slt { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rdx, rs2);
            asm.cmp_rr64(Reg::Rax, Reg::Rdx);
            asm.setcc_al(CC_L);
            asm.movzx_rax_al();
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Sltu { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rdx, rs2);
            asm.cmp_rr64(Reg::Rax, Reg::Rdx);
            asm.setcc_al(CC_B);
            asm.movzx_rax_al();
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Xor { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rdx, rs2);
            asm.alu_rr64(0x31, Reg::Rax, Reg::Rdx);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Srl { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rcx, rs2);
            asm.alu_ri32(4, Reg::Rcx, 63);
            asm.shift_cl64(5, Reg::Rax);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Sra { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rcx, rs2);
            asm.alu_ri32(4, Reg::Rcx, 63);
            asm.shift_cl64(7, Reg::Rax);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Or { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rdx, rs2);
            asm.alu_rr64(0x09, Reg::Rax, Reg::Rdx);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::And { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rdx, rs2);
            asm.alu_rr64(0x21, Reg::Rax, Reg::Rdx);
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Addw { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rdx, rs2);
            asm.alu_rr32(0x01, Reg::Rax, Reg::Rdx);
            asm.cdqe();
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Subw { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rdx, rs2);
            asm.alu_rr32(0x29, Reg::Rax, Reg::Rdx);
            asm.cdqe();
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Sllw { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rcx, rs2);
            asm.alu_ri32(4, Reg::Rcx, 31);
            asm.shift_cl32(4, Reg::Rax);
            asm.cdqe();
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Srlw { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rcx, rs2);
            asm.alu_ri32(4, Reg::Rcx, 31);
            asm.shift_cl32(5, Reg::Rax);
            asm.cdqe();
            store_x(asm, rd, Reg::Rax);
        }
        DecodedInstruction::Sraw { rd, rs1, rs2 } => {
            load_x(asm, Reg::Rax, rs1);
            load_x(asm, Reg::Rcx, rs2);
            asm.alu_ri32(4, Reg::Rcx, 31);
            asm.shift_cl32(7, Reg::Rax);
            asm.cdqe();
            store_x(asm, rd, Reg::Rax);
        }
        _ => return false,
    }

    finish_inline_instruction(asm, instruction.instruction_bytes);
    true
}

#[cfg(all(target_arch = "x86_64", unix))]
fn finish_inline_instruction(asm: &mut Assembler, instruction_bytes: u8) {
    asm.add_mem_imm32(Reg::Rbx, CPU_PC_OFFSET, instruction_bytes as i32);
    asm.add_mem_imm32(Reg::Rbx, CSR_INSTRET_OFFSET, 1);
    asm.alu_ri64(0, Reg::R14, 1);
}

#[cfg(all(target_arch = "x86_64", unix))]
fn load_x(asm: &mut Assembler, dst: Reg, reg: u8) {
    if reg == 0 {
        asm.xor_rr32(dst, dst);
    } else {
        asm.mov_rm64(dst, Reg::Rbx, CPU_X_OFFSET + reg as i32 * 8);
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
fn store_x(asm: &mut Assembler, reg: u8, src: Reg) {
    if reg != 0 {
        asm.mov_mr64(Reg::Rbx, CPU_X_OFFSET + reg as i32 * 8, src);
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
fn trace_successor(pc: u64, instruction: JitInstruction) -> u64 {
    match instruction.decoded {
        DecodedInstruction::Jal { rd: 0, imm } => pc.wrapping_add_signed(imm),
        DecodedInstruction::Branch { imm, .. } => {
            if imm < 0 {
                pc.wrapping_add_signed(imm)
            } else {
                pc.wrapping_add(instruction.instruction_bytes as u64)
            }
        }
        _ => pc.wrapping_add(instruction.instruction_bytes as u64),
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
fn instruction_may_change_pc(instruction: DecodedInstruction) -> bool {
    matches!(
        instruction,
        DecodedInstruction::Jal { .. }
            | DecodedInstruction::Jalr { .. }
            | DecodedInstruction::Branch { .. }
    )
}

#[cfg(all(target_arch = "x86_64", unix))]
fn map_executable(code: &[u8]) -> Result<ExecutableBlock> {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size: usize = page_size
        .try_into()
        .context("host page size does not fit usize")?;
    let mapped_len = code.len().max(1).next_multiple_of(page_size);
    let ptr = unsafe {
        libc::mmap(
            ptr::null_mut(),
            mapped_len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error()).context("mmap failed for x64 jit block");
    }
    unsafe {
        ptr::copy_nonoverlapping(code.as_ptr(), ptr.cast(), code.len());
    }
    if unsafe { libc::mprotect(ptr, mapped_len, libc::PROT_READ | libc::PROT_EXEC) } != 0 {
        let err = std::io::Error::last_os_error();
        unsafe {
            libc::munmap(ptr, mapped_len);
        }
        return Err(err).context("mprotect failed for x64 jit block");
    }
    Ok(ExecutableBlock {
        ptr: NonNull::new(ptr.cast()).expect("mmap returned null"),
        mapped_len,
    })
}

#[cfg(not(all(target_arch = "x86_64", unix)))]
pub(super) struct ExecutableBlock;

#[cfg(not(all(target_arch = "x86_64", unix)))]
pub(super) fn try_compile_block(
    _key: BlockKey,
    _instructions: &[JitInstruction],
) -> anyhow::Result<Option<(CompiledBlockEntry, ExecutableBlock)>> {
    Ok(None)
}
