use rvx_core::Bus;

use crate::cpu::{Cpu, PrivilegeLevel};
use crate::csr::{CSR_SATP, CsrOp};
use crate::decode::{
    AtomicOp, BranchKind, DecodedInstruction, FloatLoadKind, FloatStoreKind, LoadKind, StoreKind,
};
use crate::mmu::AccessType;
use crate::trap::{Exception, Trap};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    Continue,
    Fence,
    Halted,
}

impl Cpu {
    pub fn execute(
        &mut self,
        insn: DecodedInstruction,
        instruction_bytes: u8,
        bus: &Bus,
    ) -> std::result::Result<StepOutcome, Trap> {
        let pc = self.pc;
        let mut next_pc = pc.wrapping_add(instruction_bytes as u64);
        let outcome = match insn {
            DecodedInstruction::Lui { rd, imm } => {
                self.write_x(rd, imm as u64);
                StepOutcome::Continue
            }
            DecodedInstruction::Auipc { rd, imm } => {
                self.write_x(rd, pc.wrapping_add(imm as u64));
                StepOutcome::Continue
            }
            DecodedInstruction::Jal { rd, imm } => {
                self.write_x(rd, next_pc);
                next_pc = pc.wrapping_add_signed(imm);
                StepOutcome::Continue
            }
            DecodedInstruction::Jalr { rd, rs1, imm } => {
                let target = self.read_x(rs1).wrapping_add_signed(imm) & !1;
                self.write_x(rd, next_pc);
                next_pc = target;
                StepOutcome::Continue
            }
            DecodedInstruction::Branch {
                kind,
                rs1,
                rs2,
                imm,
            } => {
                if branch_taken(kind, self.read_x(rs1), self.read_x(rs2)) {
                    next_pc = pc.wrapping_add_signed(imm);
                }
                StepOutcome::Continue
            }
            DecodedInstruction::Load { kind, rd, rs1, imm } => {
                let addr = self.read_x(rs1).wrapping_add_signed(imm);
                let value = load(self, bus, kind, addr)?;
                self.write_x(rd, value);
                StepOutcome::Continue
            }
            DecodedInstruction::FloatLoad { kind, rd, rs1, imm } => {
                let addr = self.read_x(rs1).wrapping_add_signed(imm);
                let value = float_load(self, bus, kind, addr)?;
                self.write_f(rd, value);
                StepOutcome::Continue
            }
            DecodedInstruction::Store {
                kind,
                rs1,
                rs2,
                imm,
            } => {
                let addr = self.read_x(rs1).wrapping_add_signed(imm);
                store(self, bus, kind, addr, self.read_x(rs2))?;
                StepOutcome::Continue
            }
            DecodedInstruction::FloatStore {
                kind,
                rs1,
                rs2,
                imm,
            } => {
                let addr = self.read_x(rs1).wrapping_add_signed(imm);
                float_store(self, bus, kind, addr, self.read_f(rs2))?;
                StepOutcome::Continue
            }
            DecodedInstruction::Addi { rd, rs1, imm } => {
                self.write_x(rd, self.read_x(rs1).wrapping_add_signed(imm));
                StepOutcome::Continue
            }
            DecodedInstruction::Slti { rd, rs1, imm } => {
                self.write_x(rd, ((self.read_x(rs1) as i64) < imm) as u64);
                StepOutcome::Continue
            }
            DecodedInstruction::Sltiu { rd, rs1, imm } => {
                self.write_x(rd, (self.read_x(rs1) < imm as u64) as u64);
                StepOutcome::Continue
            }
            DecodedInstruction::Xori { rd, rs1, imm } => {
                self.write_x(rd, self.read_x(rs1) ^ imm as u64);
                StepOutcome::Continue
            }
            DecodedInstruction::Ori { rd, rs1, imm } => {
                self.write_x(rd, self.read_x(rs1) | imm as u64);
                StepOutcome::Continue
            }
            DecodedInstruction::Andi { rd, rs1, imm } => {
                self.write_x(rd, self.read_x(rs1) & imm as u64);
                StepOutcome::Continue
            }
            DecodedInstruction::Slli { rd, rs1, shamt } => {
                self.write_x(rd, self.read_x(rs1) << shamt);
                StepOutcome::Continue
            }
            DecodedInstruction::Srli { rd, rs1, shamt } => {
                self.write_x(rd, self.read_x(rs1) >> shamt);
                StepOutcome::Continue
            }
            DecodedInstruction::Srai { rd, rs1, shamt } => {
                self.write_x(rd, ((self.read_x(rs1) as i64) >> shamt) as u64);
                StepOutcome::Continue
            }
            DecodedInstruction::Addiw { rd, rs1, imm } => {
                self.write_x(rd, sext32(self.read_x(rs1).wrapping_add_signed(imm) as u32));
                StepOutcome::Continue
            }
            DecodedInstruction::Slliw { rd, rs1, shamt } => {
                self.write_x(rd, sext32((self.read_x(rs1) as u32) << shamt));
                StepOutcome::Continue
            }
            DecodedInstruction::Srliw { rd, rs1, shamt } => {
                self.write_x(rd, sext32((self.read_x(rs1) as u32) >> shamt));
                StepOutcome::Continue
            }
            DecodedInstruction::Sraiw { rd, rs1, shamt } => {
                self.write_x(rd, sext32(((self.read_x(rs1) as i32) >> shamt) as u32));
                StepOutcome::Continue
            }
            DecodedInstruction::Add { rd, rs1, rs2 } => {
                self.write_x(rd, self.read_x(rs1).wrapping_add(self.read_x(rs2)));
                StepOutcome::Continue
            }
            DecodedInstruction::Sub { rd, rs1, rs2 } => {
                self.write_x(rd, self.read_x(rs1).wrapping_sub(self.read_x(rs2)));
                StepOutcome::Continue
            }
            DecodedInstruction::Sll { rd, rs1, rs2 } => {
                self.write_x(rd, self.read_x(rs1) << (self.read_x(rs2) & 0x3f));
                StepOutcome::Continue
            }
            DecodedInstruction::Slt { rd, rs1, rs2 } => {
                self.write_x(
                    rd,
                    ((self.read_x(rs1) as i64) < (self.read_x(rs2) as i64)) as u64,
                );
                StepOutcome::Continue
            }
            DecodedInstruction::Sltu { rd, rs1, rs2 } => {
                self.write_x(rd, (self.read_x(rs1) < self.read_x(rs2)) as u64);
                StepOutcome::Continue
            }
            DecodedInstruction::Xor { rd, rs1, rs2 } => {
                self.write_x(rd, self.read_x(rs1) ^ self.read_x(rs2));
                StepOutcome::Continue
            }
            DecodedInstruction::Srl { rd, rs1, rs2 } => {
                self.write_x(rd, self.read_x(rs1) >> (self.read_x(rs2) & 0x3f));
                StepOutcome::Continue
            }
            DecodedInstruction::Sra { rd, rs1, rs2 } => {
                self.write_x(
                    rd,
                    ((self.read_x(rs1) as i64) >> (self.read_x(rs2) & 0x3f)) as u64,
                );
                StepOutcome::Continue
            }
            DecodedInstruction::Or { rd, rs1, rs2 } => {
                self.write_x(rd, self.read_x(rs1) | self.read_x(rs2));
                StepOutcome::Continue
            }
            DecodedInstruction::And { rd, rs1, rs2 } => {
                self.write_x(rd, self.read_x(rs1) & self.read_x(rs2));
                StepOutcome::Continue
            }
            DecodedInstruction::Addw { rd, rs1, rs2 } => {
                self.write_x(
                    rd,
                    sext32((self.read_x(rs1) as u32).wrapping_add(self.read_x(rs2) as u32)),
                );
                StepOutcome::Continue
            }
            DecodedInstruction::Subw { rd, rs1, rs2 } => {
                self.write_x(
                    rd,
                    sext32((self.read_x(rs1) as u32).wrapping_sub(self.read_x(rs2) as u32)),
                );
                StepOutcome::Continue
            }
            DecodedInstruction::Sllw { rd, rs1, rs2 } => {
                self.write_x(
                    rd,
                    sext32((self.read_x(rs1) as u32) << (self.read_x(rs2) & 0x1f)),
                );
                StepOutcome::Continue
            }
            DecodedInstruction::Srlw { rd, rs1, rs2 } => {
                self.write_x(
                    rd,
                    sext32((self.read_x(rs1) as u32) >> (self.read_x(rs2) & 0x1f)),
                );
                StepOutcome::Continue
            }
            DecodedInstruction::Sraw { rd, rs1, rs2 } => {
                self.write_x(
                    rd,
                    sext32(((self.read_x(rs1) as i32) >> (self.read_x(rs2) & 0x1f)) as u32),
                );
                StepOutcome::Continue
            }
            DecodedInstruction::Mul { rd, rs1, rs2 } => {
                self.write_x(rd, self.read_x(rs1).wrapping_mul(self.read_x(rs2)));
                StepOutcome::Continue
            }
            DecodedInstruction::Mulh { rd, rs1, rs2 } => {
                self.write_x(
                    rd,
                    (((self.read_x(rs1) as i128) * (self.read_x(rs2) as i128)) >> 64) as u64,
                );
                StepOutcome::Continue
            }
            DecodedInstruction::Mulhsu { rd, rs1, rs2 } => {
                let lhs = self.read_x(rs1) as i64 as i128;
                let rhs = self.read_x(rs2) as i128;
                self.write_x(rd, ((lhs * rhs) >> 64) as u64);
                StepOutcome::Continue
            }
            DecodedInstruction::Mulhu { rd, rs1, rs2 } => {
                self.write_x(
                    rd,
                    (((self.read_x(rs1) as u128) * (self.read_x(rs2) as u128)) >> 64) as u64,
                );
                StepOutcome::Continue
            }
            DecodedInstruction::Div { rd, rs1, rs2 } => {
                self.write_x(rd, div_signed(self.read_x(rs1), self.read_x(rs2)));
                StepOutcome::Continue
            }
            DecodedInstruction::Divu { rd, rs1, rs2 } => {
                self.write_x(rd, div_unsigned(self.read_x(rs1), self.read_x(rs2)));
                StepOutcome::Continue
            }
            DecodedInstruction::Rem { rd, rs1, rs2 } => {
                self.write_x(rd, rem_signed(self.read_x(rs1), self.read_x(rs2)));
                StepOutcome::Continue
            }
            DecodedInstruction::Remu { rd, rs1, rs2 } => {
                self.write_x(rd, rem_unsigned(self.read_x(rs1), self.read_x(rs2)));
                StepOutcome::Continue
            }
            DecodedInstruction::Mulw { rd, rs1, rs2 } => {
                self.write_x(
                    rd,
                    sext32((self.read_x(rs1) as u32).wrapping_mul(self.read_x(rs2) as u32)),
                );
                StepOutcome::Continue
            }
            DecodedInstruction::Divw { rd, rs1, rs2 } => {
                self.write_x(
                    rd,
                    sext32(divw_signed(self.read_x(rs1), self.read_x(rs2)) as u32),
                );
                StepOutcome::Continue
            }
            DecodedInstruction::Divuw { rd, rs1, rs2 } => {
                self.write_x(
                    rd,
                    sext32(divw_unsigned(self.read_x(rs1), self.read_x(rs2))),
                );
                StepOutcome::Continue
            }
            DecodedInstruction::Remw { rd, rs1, rs2 } => {
                self.write_x(
                    rd,
                    sext32(remw_signed(self.read_x(rs1), self.read_x(rs2)) as u32),
                );
                StepOutcome::Continue
            }
            DecodedInstruction::Remuw { rd, rs1, rs2 } => {
                self.write_x(
                    rd,
                    sext32(remw_unsigned(self.read_x(rs1), self.read_x(rs2))),
                );
                StepOutcome::Continue
            }
            DecodedInstruction::Fence | DecodedInstruction::FenceI => {
                bus.memory_barrier();
                StepOutcome::Fence
            }
            DecodedInstruction::Ecall => {
                let exc = match self.privilege {
                    PrivilegeLevel::User => Exception::UserEnvCall,
                    PrivilegeLevel::Supervisor => Exception::SupervisorEnvCall,
                    PrivilegeLevel::Machine => Exception::MachineEnvCall,
                };
                return Err(Trap::exception(exc, 0));
            }
            DecodedInstruction::Ebreak => {
                return Err(Trap::exception(Exception::Breakpoint, pc));
            }
            DecodedInstruction::Wfi => StepOutcome::Halted,
            DecodedInstruction::Mret
            | DecodedInstruction::Sret
            | DecodedInstruction::SfenceVma { .. } => StepOutcome::Fence,
            DecodedInstruction::Csrrw { rd, rs1, csr } => {
                let result = self
                    .csr
                    .write_op(csr, CsrOp::ReadWrite, self.read_x(rs1), self.privilege)
                    .map_err(|_| Trap::exception(Exception::IllegalInstruction, csr as u64))?;
                if csr_write_requires_mmu_flush(csr, result.old_value, result.new_value) {
                    self.mmu.flush();
                }
                self.write_x(rd, result.old_value);
                StepOutcome::Continue
            }
            DecodedInstruction::Csrrs { rd, rs1, csr } => {
                let old = if rs1 == 0 {
                    self.csr
                        .read(csr, self.privilege)
                        .map_err(|_| Trap::exception(Exception::IllegalInstruction, csr as u64))?
                } else {
                    let result = self
                        .csr
                        .write_op(csr, CsrOp::ReadSet, self.read_x(rs1), self.privilege)
                        .map_err(|_| Trap::exception(Exception::IllegalInstruction, csr as u64))?;
                    if csr_write_requires_mmu_flush(csr, result.old_value, result.new_value) {
                        self.mmu.flush();
                    }
                    result.old_value
                };
                self.write_x(rd, old);
                StepOutcome::Continue
            }
            DecodedInstruction::Csrrc { rd, rs1, csr } => {
                let old = if rs1 == 0 {
                    self.csr
                        .read(csr, self.privilege)
                        .map_err(|_| Trap::exception(Exception::IllegalInstruction, csr as u64))?
                } else {
                    let result = self
                        .csr
                        .write_op(csr, CsrOp::ReadClear, self.read_x(rs1), self.privilege)
                        .map_err(|_| Trap::exception(Exception::IllegalInstruction, csr as u64))?;
                    if csr_write_requires_mmu_flush(csr, result.old_value, result.new_value) {
                        self.mmu.flush();
                    }
                    result.old_value
                };
                self.write_x(rd, old);
                StepOutcome::Continue
            }
            DecodedInstruction::Csrrwi { rd, zimm, csr } => {
                let result = self
                    .csr
                    .write_op(csr, CsrOp::ReadWrite, zimm as u64, self.privilege)
                    .map_err(|_| Trap::exception(Exception::IllegalInstruction, csr as u64))?;
                if csr_write_requires_mmu_flush(csr, result.old_value, result.new_value) {
                    self.mmu.flush();
                }
                self.write_x(rd, result.old_value);
                StepOutcome::Continue
            }
            DecodedInstruction::Csrrsi { rd, zimm, csr } => {
                let old = if zimm == 0 {
                    self.csr
                        .read(csr, self.privilege)
                        .map_err(|_| Trap::exception(Exception::IllegalInstruction, csr as u64))?
                } else {
                    let result = self
                        .csr
                        .write_op(csr, CsrOp::ReadSet, zimm as u64, self.privilege)
                        .map_err(|_| Trap::exception(Exception::IllegalInstruction, csr as u64))?;
                    if csr_write_requires_mmu_flush(csr, result.old_value, result.new_value) {
                        self.mmu.flush();
                    }
                    result.old_value
                };
                self.write_x(rd, old);
                StepOutcome::Continue
            }
            DecodedInstruction::Csrrci { rd, zimm, csr } => {
                let old = if zimm == 0 {
                    self.csr
                        .read(csr, self.privilege)
                        .map_err(|_| Trap::exception(Exception::IllegalInstruction, csr as u64))?
                } else {
                    let result = self
                        .csr
                        .write_op(csr, CsrOp::ReadClear, zimm as u64, self.privilege)
                        .map_err(|_| Trap::exception(Exception::IllegalInstruction, csr as u64))?;
                    if csr_write_requires_mmu_flush(csr, result.old_value, result.new_value) {
                        self.mmu.flush();
                    }
                    result.old_value
                };
                self.write_x(rd, old);
                StepOutcome::Continue
            }
            DecodedInstruction::Atomic {
                op,
                rd,
                rs1,
                rs2,
                aq,
                rl,
            } => {
                let addr = self.read_x(rs1);
                let value = atomic(bus, self, op, addr, self.read_x(rs2), aq, rl)?;
                self.write_x(rd, value);
                StepOutcome::Continue
            }
        };
        self.pc = next_pc;
        self.csr.instret = self.csr.instret.wrapping_add(1);
        Ok(outcome)
    }
}

fn csr_write_requires_mmu_flush(csr: u16, old_value: u64, new_value: u64) -> bool {
    match csr {
        CSR_SATP => old_value != new_value,
        _ => false,
    }
}

fn branch_taken(kind: BranchKind, lhs: u64, rhs: u64) -> bool {
    match kind {
        BranchKind::Eq => lhs == rhs,
        BranchKind::Ne => lhs != rhs,
        BranchKind::Lt => (lhs as i64) < (rhs as i64),
        BranchKind::Ge => (lhs as i64) >= (rhs as i64),
        BranchKind::Ltu => lhs < rhs,
        BranchKind::Geu => lhs >= rhs,
    }
}

fn load(cpu: &mut Cpu, bus: &Bus, kind: LoadKind, addr: u64) -> std::result::Result<u64, Trap> {
    Ok(match kind {
        LoadKind::Byte => sext8(cpu.load_u8(bus, addr)?),
        LoadKind::Half => sext16(cpu.load_u16(bus, addr)?),
        LoadKind::Word => sext32(cpu.load_u32(bus, addr)?),
        LoadKind::Double => cpu.load_u64(bus, addr)?,
        LoadKind::ByteUnsigned => cpu.load_u8(bus, addr)? as u64,
        LoadKind::HalfUnsigned => cpu.load_u16(bus, addr)? as u64,
        LoadKind::WordUnsigned => cpu.load_u32(bus, addr)? as u64,
    })
}

fn store(
    cpu: &mut Cpu,
    bus: &Bus,
    kind: StoreKind,
    addr: u64,
    value: u64,
) -> std::result::Result<(), Trap> {
    match kind {
        StoreKind::Byte => cpu.store_u8(bus, addr, value as u8)?,
        StoreKind::Half => cpu.store_u16(bus, addr, value as u16)?,
        StoreKind::Word => cpu.store_u32(bus, addr, value as u32)?,
        StoreKind::Double => cpu.store_u64(bus, addr, value)?,
    }
    Ok(())
}

fn float_load(
    cpu: &mut Cpu,
    bus: &Bus,
    kind: FloatLoadKind,
    addr: u64,
) -> std::result::Result<u64, Trap> {
    Ok(match kind {
        FloatLoadKind::Word => 0xffff_ffff_0000_0000 | cpu.load_u32(bus, addr)? as u64,
        FloatLoadKind::Double => cpu.load_u64(bus, addr)?,
    })
}

fn float_store(
    cpu: &mut Cpu,
    bus: &Bus,
    kind: FloatStoreKind,
    addr: u64,
    value: u64,
) -> std::result::Result<(), Trap> {
    match kind {
        FloatStoreKind::Word => cpu.store_u32(bus, addr, value as u32)?,
        FloatStoreKind::Double => cpu.store_u64(bus, addr, value)?,
    }
    Ok(())
}

fn atomic(
    bus: &Bus,
    cpu: &mut Cpu,
    op: AtomicOp,
    addr: u64,
    value: u64,
    aq: bool,
    rl: bool,
) -> std::result::Result<u64, Trap> {
    if rl {
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
    let result = match op {
        AtomicOp::LrW => {
            let phys = cpu.translate_data(bus, addr, AccessType::Read)?;
            let atomic = bus.atomic_lock();
            let old = bus
                .read_u32_locked(phys, &atomic)
                .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?
                as u64;
            cpu.reservation = Some(addr);
            cpu.reservation_value = old;
            cpu.reservation_epoch = bus.reservation_epoch();
            sext32(old as u32)
        }
        AtomicOp::ScW => {
            let phys = cpu.translate_data(bus, addr, AccessType::Write)?;
            let atomic = bus.atomic_lock();
            if cpu.reservation == Some(addr) && cpu.reservation_epoch == bus.reservation_epoch() {
                bus.write_u32_locked(phys, value as u32, &atomic)
                    .map_err(|_| Trap::exception(Exception::StoreAccessFault, addr))?;
                cpu.reservation = None;
                0
            } else {
                1
            }
        }
        AtomicOp::LrD => {
            let phys = cpu.translate_data(bus, addr, AccessType::Read)?;
            let atomic = bus.atomic_lock();
            let old = bus
                .read_u64_locked(phys, &atomic)
                .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?;
            cpu.reservation = Some(addr);
            cpu.reservation_value = old;
            cpu.reservation_epoch = bus.reservation_epoch();
            old
        }
        AtomicOp::ScD => {
            let phys = cpu.translate_data(bus, addr, AccessType::Write)?;
            let atomic = bus.atomic_lock();
            if cpu.reservation == Some(addr) && cpu.reservation_epoch == bus.reservation_epoch() {
                bus.write_u64_locked(phys, value, &atomic)
                    .map_err(|_| Trap::exception(Exception::StoreAccessFault, addr))?;
                cpu.reservation = None;
                0
            } else {
                1
            }
        }
        _ => {
            let phys = cpu.translate_data(bus, addr, AccessType::Write)?;
            let atomic = bus.atomic_lock();
            let (old, new) = match op {
                AtomicOp::AmoswapW => {
                    let old = bus
                        .read_u32_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?
                        as u64;
                    (sext32(old as u32), value as u32 as u64)
                }
                AtomicOp::AmoaddW => {
                    let old = bus
                        .read_u32_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?
                        as u64;
                    (
                        sext32(old as u32),
                        (old as u32).wrapping_add(value as u32) as u64,
                    )
                }
                AtomicOp::AmoxorW => {
                    let old = bus
                        .read_u32_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?
                        as u64;
                    (sext32(old as u32), ((old as u32) ^ value as u32) as u64)
                }
                AtomicOp::AmoandW => {
                    let old = bus
                        .read_u32_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?
                        as u64;
                    (sext32(old as u32), ((old as u32) & value as u32) as u64)
                }
                AtomicOp::AmoorW => {
                    let old = bus
                        .read_u32_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?
                        as u64;
                    (sext32(old as u32), ((old as u32) | value as u32) as u64)
                }
                AtomicOp::AmominW => {
                    let old = bus
                        .read_u32_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?
                        as i32;
                    (
                        sext32(old as u32),
                        std::cmp::min(old, value as i32) as u32 as u64,
                    )
                }
                AtomicOp::AmomaxW => {
                    let old = bus
                        .read_u32_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?
                        as i32;
                    (
                        sext32(old as u32),
                        std::cmp::max(old, value as i32) as u32 as u64,
                    )
                }
                AtomicOp::AmominuW => {
                    let old = bus
                        .read_u32_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?;
                    (sext32(old), old.min(value as u32) as u64)
                }
                AtomicOp::AmomaxuW => {
                    let old = bus
                        .read_u32_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?;
                    (sext32(old), old.max(value as u32) as u64)
                }
                AtomicOp::AmoswapD => {
                    let old = bus
                        .read_u64_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?;
                    (old, value)
                }
                AtomicOp::AmoaddD => {
                    let old = bus
                        .read_u64_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?;
                    (old, old.wrapping_add(value))
                }
                AtomicOp::AmoxorD => {
                    let old = bus
                        .read_u64_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?;
                    (old, old ^ value)
                }
                AtomicOp::AmoandD => {
                    let old = bus
                        .read_u64_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?;
                    (old, old & value)
                }
                AtomicOp::AmoorD => {
                    let old = bus
                        .read_u64_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?;
                    (old, old | value)
                }
                AtomicOp::AmominD => {
                    let old = bus
                        .read_u64_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?
                        as i64;
                    (old as u64, std::cmp::min(old, value as i64) as u64)
                }
                AtomicOp::AmomaxD => {
                    let old = bus
                        .read_u64_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?
                        as i64;
                    (old as u64, std::cmp::max(old, value as i64) as u64)
                }
                AtomicOp::AmominuD => {
                    let old = bus
                        .read_u64_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?;
                    (old, old.min(value))
                }
                AtomicOp::AmomaxuD => {
                    let old = bus
                        .read_u64_locked(phys, &atomic)
                        .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))?;
                    (old, old.max(value))
                }
                _ => return Err(Trap::exception(Exception::IllegalInstruction, addr)),
            };
            match op {
                AtomicOp::AmoswapW
                | AtomicOp::AmoaddW
                | AtomicOp::AmoxorW
                | AtomicOp::AmoandW
                | AtomicOp::AmoorW
                | AtomicOp::AmominW
                | AtomicOp::AmomaxW
                | AtomicOp::AmominuW
                | AtomicOp::AmomaxuW => bus
                    .write_u32_locked(phys, new as u32, &atomic)
                    .map_err(|_| Trap::exception(Exception::StoreAccessFault, addr))?,
                _ => bus
                    .write_u64_locked(phys, new, &atomic)
                    .map_err(|_| Trap::exception(Exception::StoreAccessFault, addr))?,
            }
            old
        }
    };
    if aq {
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
    Ok(result)
}

fn sext8(value: u8) -> u64 {
    value as i8 as i64 as u64
}

fn sext16(value: u16) -> u64 {
    value as i16 as i64 as u64
}

fn sext32(value: u32) -> u64 {
    value as i32 as i64 as u64
}

fn div_signed(lhs: u64, rhs: u64) -> u64 {
    let lhs = lhs as i64;
    let rhs = rhs as i64;
    if rhs == 0 {
        u64::MAX
    } else if lhs == i64::MIN && rhs == -1 {
        lhs as u64
    } else {
        lhs.wrapping_div(rhs) as u64
    }
}

fn div_unsigned(lhs: u64, rhs: u64) -> u64 {
    if rhs == 0 { u64::MAX } else { lhs / rhs }
}

fn rem_signed(lhs: u64, rhs: u64) -> u64 {
    let lhs = lhs as i64;
    let rhs = rhs as i64;
    if rhs == 0 {
        lhs as u64
    } else if lhs == i64::MIN && rhs == -1 {
        0
    } else {
        lhs.wrapping_rem(rhs) as u64
    }
}

fn rem_unsigned(lhs: u64, rhs: u64) -> u64 {
    if rhs == 0 { lhs } else { lhs % rhs }
}

fn divw_signed(lhs: u64, rhs: u64) -> i32 {
    let lhs = lhs as i32;
    let rhs = rhs as i32;
    if rhs == 0 {
        -1
    } else if lhs == i32::MIN && rhs == -1 {
        lhs
    } else {
        lhs.wrapping_div(rhs)
    }
}

fn divw_unsigned(lhs: u64, rhs: u64) -> u32 {
    let lhs = lhs as u32;
    let rhs = rhs as u32;
    if rhs == 0 { u32::MAX } else { lhs / rhs }
}

fn remw_signed(lhs: u64, rhs: u64) -> i32 {
    let lhs = lhs as i32;
    let rhs = rhs as i32;
    if rhs == 0 {
        lhs
    } else if lhs == i32::MIN && rhs == -1 {
        0
    } else {
        lhs.wrapping_rem(rhs)
    }
}

fn remw_unsigned(lhs: u64, rhs: u64) -> u32 {
    let lhs = lhs as u32;
    let rhs = rhs as u32;
    if rhs == 0 { lhs } else { lhs % rhs }
}

#[cfg(test)]
mod tests {
    use rvx_core::{Bus, Ram};

    use super::csr_write_requires_mmu_flush;
    use crate::Cpu;
    use crate::csr::{CSR_MSTATUS, CSR_SATP, CSR_SSTATUS, MSTATUS_MXR, MSTATUS_SUM};
    use crate::decode::DecodedInstruction;

    #[test]
    fn execute_add_and_store_load() {
        let mut cpu = Cpu::new();
        let mut bus = Bus::new(Ram::new(0x8000_0000, 0x1000).unwrap());
        cpu.pc = 0x8000_0000;
        cpu.write_x(1, 0x8000_0000);
        cpu.write_x(2, 7);
        cpu.execute(
            DecodedInstruction::Store {
                kind: crate::StoreKind::Word,
                rs1: 1,
                rs2: 2,
                imm: 8,
            },
            4,
            &mut bus,
        )
        .unwrap();
        cpu.execute(
            DecodedInstruction::Load {
                kind: crate::LoadKind::WordUnsigned,
                rd: 3,
                rs1: 1,
                imm: 8,
            },
            4,
            &mut bus,
        )
        .unwrap();
        cpu.execute(
            DecodedInstruction::Addi {
                rd: 4,
                rs1: 3,
                imm: 5,
            },
            4,
            &mut bus,
        )
        .unwrap();
        assert_eq!(cpu.read_x(3), 7);
        assert_eq!(cpu.read_x(4), 12);
    }

    #[test]
    fn csr_mmu_flush_only_triggers_on_satp_change() {
        assert!(!csr_write_requires_mmu_flush(
            CSR_SSTATUS,
            0,
            MSTATUS_SUM | MSTATUS_MXR,
        ));
        assert!(!csr_write_requires_mmu_flush(
            CSR_MSTATUS,
            0,
            MSTATUS_SUM | MSTATUS_MXR,
        ));
        assert!(!csr_write_requires_mmu_flush(CSR_SATP, 0x1234, 0x1234));
        assert!(csr_write_requires_mmu_flush(CSR_SATP, 0x1234, 0x5678));
    }
}
