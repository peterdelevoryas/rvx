use anyhow::{Result, bail};

use crate::cpu::PrivilegeLevel;

pub const CSR_FFLAGS: u16 = 0x001;
pub const CSR_FRM: u16 = 0x002;
pub const CSR_FCSR: u16 = 0x003;
pub const CSR_SSTATUS: u16 = 0x100;
pub const CSR_SIE: u16 = 0x104;
pub const CSR_STVEC: u16 = 0x105;
pub const CSR_SCOUNTEREN: u16 = 0x106;
pub const CSR_SSCRATCH: u16 = 0x140;
pub const CSR_SEPC: u16 = 0x141;
pub const CSR_SCAUSE: u16 = 0x142;
pub const CSR_STVAL: u16 = 0x143;
pub const CSR_SIP: u16 = 0x144;
pub const CSR_SATP: u16 = 0x180;
pub const CSR_MSTATUS: u16 = 0x300;
pub const CSR_MISA: u16 = 0x301;
pub const CSR_MEDELEG: u16 = 0x302;
pub const CSR_MIDELEG: u16 = 0x303;
pub const CSR_MIE: u16 = 0x304;
pub const CSR_MTVEC: u16 = 0x305;
pub const CSR_MCOUNTEREN: u16 = 0x306;
pub const CSR_MSCRATCH: u16 = 0x340;
pub const CSR_MEPC: u16 = 0x341;
pub const CSR_MCAUSE: u16 = 0x342;
pub const CSR_MTVAL: u16 = 0x343;
pub const CSR_MIP: u16 = 0x344;
pub const CSR_MHARTID: u16 = 0xf14;
pub const CSR_CYCLE: u16 = 0xc00;
pub const CSR_TIME: u16 = 0xc01;
pub const CSR_INSTRET: u16 = 0xc02;

pub const MSTATUS_SIE: u64 = 1 << 1;
pub const MSTATUS_MIE: u64 = 1 << 3;
pub const MSTATUS_SPIE: u64 = 1 << 5;
pub const MSTATUS_MPIE: u64 = 1 << 7;
pub const MSTATUS_SPP: u64 = 1 << 8;
pub const MSTATUS_MPP_MASK: u64 = 0x3 << 11;
pub const MSTATUS_FS_MASK: u64 = 0x3 << 13;
pub const MSTATUS_MPRV: u64 = 1 << 17;
pub const MSTATUS_SUM: u64 = 1 << 18;
pub const MSTATUS_MXR: u64 = 1 << 19;
const MSTATUS_UXL_64: u64 = 2 << 32;
const MSTATUS_SXL_64: u64 = 2 << 34;
const MSTATUS_SD: u64 = 1 << 63;
const MSTATUS_WRITE_MASK: u64 = MSTATUS_SIE
    | MSTATUS_MIE
    | MSTATUS_SPIE
    | MSTATUS_MPIE
    | MSTATUS_SPP
    | MSTATUS_MPP_MASK
    | MSTATUS_FS_MASK
    | MSTATUS_MPRV
    | MSTATUS_SUM
    | MSTATUS_MXR;
const SSTATUS_MASK: u64 = MSTATUS_SIE
    | MSTATUS_SPIE
    | MSTATUS_SPP
    | MSTATUS_FS_MASK
    | MSTATUS_SUM
    | MSTATUS_MXR
    | MSTATUS_UXL_64
    | MSTATUS_SD;
const SSTATUS_WRITE_MASK: u64 =
    MSTATUS_SIE | MSTATUS_SPIE | MSTATUS_SPP | MSTATUS_FS_MASK | MSTATUS_SUM | MSTATUS_MXR;
const SIP_MASK: u64 = (1 << 1) | (1 << 5) | (1 << 9);
const SATP_MODE_BARE: u64 = 0;
const SATP_MODE_SV39: u64 = 8;
const SATP_MODE_SV48: u64 = 9;
const SATP_MODE_SV57: u64 = 10;
const SATP_MODE_SHIFT: u32 = 60;
const FFLAGS_MASK: u64 = 0x1f;
const FRM_MASK: u64 = 0x7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsrOp {
    ReadWrite,
    ReadSet,
    ReadClear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsrResult {
    pub old_value: u64,
    pub new_value: u64,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct CsrFile {
    pub mstatus: u64,
    pub misa: u64,
    pub medeleg: u64,
    pub mideleg: u64,
    pub mie: u64,
    pub mtvec: u64,
    pub mcounteren: u64,
    pub mscratch: u64,
    pub mepc: u64,
    pub mcause: u64,
    pub mtval: u64,
    pub mip: u64,
    pub sscratch: u64,
    pub sepc: u64,
    pub scause: u64,
    pub stval: u64,
    pub stvec: u64,
    pub scounteren: u64,
    pub satp: u64,
    pub cycle: u64,
    pub time: u64,
    pub instret: u64,
    pub fflags: u64,
    pub frm: u64,
    pub mhartid: u64,
}

impl Default for CsrFile {
    fn default() -> Self {
        Self {
            mstatus: MSTATUS_UXL_64 | MSTATUS_SXL_64,
            misa: misa_rv64imafdsu(),
            medeleg: (1 << 8) | (1 << 9) | (1 << 12) | (1 << 13) | (1 << 15),
            mideleg: (1 << 1) | (1 << 5) | (1 << 9),
            mie: 0,
            mtvec: 0,
            mcounteren: 0,
            mscratch: 0,
            mepc: 0,
            mcause: 0,
            mtval: 0,
            mip: 0,
            sscratch: 0,
            sepc: 0,
            scause: 0,
            stval: 0,
            stvec: 0,
            scounteren: 0,
            satp: 0,
            cycle: 0,
            time: 0,
            instret: 0,
            fflags: 0,
            frm: 0,
            mhartid: 0,
        }
    }
}

impl CsrFile {
    pub fn read(&self, csr: u16, privilege: PrivilegeLevel) -> Result<u64> {
        ensure_privilege(csr, privilege)?;
        let value = match csr {
            CSR_FFLAGS => self.fflags & FFLAGS_MASK,
            CSR_FRM => self.frm & FRM_MASK,
            CSR_FCSR => ((self.frm & FRM_MASK) << 5) | (self.fflags & FFLAGS_MASK),
            CSR_SSTATUS => (self.mstatus | sd_bit(self.mstatus) | MSTATUS_UXL_64) & SSTATUS_MASK,
            CSR_SIE => self.mie & SIP_MASK,
            CSR_STVEC => self.stvec,
            CSR_SCOUNTEREN => self.scounteren,
            CSR_SSCRATCH => self.sscratch,
            CSR_SEPC => self.sepc,
            CSR_SCAUSE => self.scause,
            CSR_STVAL => self.stval,
            CSR_SIP => self.mip & SIP_MASK,
            CSR_SATP => self.satp,
            CSR_MSTATUS => self.mstatus | sd_bit(self.mstatus) | MSTATUS_UXL_64 | MSTATUS_SXL_64,
            CSR_MISA => self.misa,
            CSR_MEDELEG => self.medeleg,
            CSR_MIDELEG => self.mideleg,
            CSR_MIE => self.mie,
            CSR_MTVEC => self.mtvec,
            CSR_MCOUNTEREN => self.mcounteren,
            CSR_MSCRATCH => self.mscratch,
            CSR_MEPC => self.mepc,
            CSR_MCAUSE => self.mcause,
            CSR_MTVAL => self.mtval,
            CSR_MIP => self.mip,
            CSR_MHARTID => self.mhartid,
            CSR_CYCLE => self.cycle,
            CSR_TIME => self.time,
            CSR_INSTRET => self.instret,
            _ => bail!("unsupported csr 0x{csr:03x}"),
        };
        Ok(value)
    }

    pub fn write_op(
        &mut self,
        csr: u16,
        op: CsrOp,
        value: u64,
        privilege: PrivilegeLevel,
    ) -> Result<CsrResult> {
        ensure_privilege(csr, privilege)?;
        let old = self.read(csr, privilege)?;
        let new = match op {
            CsrOp::ReadWrite => value,
            CsrOp::ReadSet => old | value,
            CsrOp::ReadClear => old & !value,
        };
        match csr {
            CSR_FFLAGS => self.fflags = new & FFLAGS_MASK,
            CSR_FRM => self.frm = new & FRM_MASK,
            CSR_FCSR => {
                self.fflags = new & FFLAGS_MASK;
                self.frm = (new >> 5) & FRM_MASK;
            }
            CSR_SSTATUS => {
                self.mstatus = (self.mstatus & !SSTATUS_WRITE_MASK) | (new & SSTATUS_WRITE_MASK);
            }
            CSR_SIE => self.mie = (self.mie & !SIP_MASK) | (new & SIP_MASK),
            CSR_STVEC => self.stvec = new,
            CSR_SCOUNTEREN => self.scounteren = new,
            CSR_SSCRATCH => self.sscratch = new,
            CSR_SEPC => self.sepc = new & !1,
            CSR_SCAUSE => self.scause = new,
            CSR_STVAL => self.stval = new,
            CSR_SIP => self.mip = (self.mip & !SIP_MASK) | (new & SIP_MASK),
            CSR_SATP => {
                let mode = (new >> SATP_MODE_SHIFT) & 0xf;
                if matches!(
                    mode,
                    SATP_MODE_BARE | SATP_MODE_SV39 | SATP_MODE_SV48 | SATP_MODE_SV57
                ) {
                    self.satp = new;
                } else {
                    bail!("unsupported satp mode {mode}");
                }
            }
            CSR_MSTATUS => {
                self.mstatus = (self.mstatus & !MSTATUS_WRITE_MASK) | (new & MSTATUS_WRITE_MASK);
                self.mstatus |= MSTATUS_UXL_64 | MSTATUS_SXL_64;
            }
            CSR_MEDELEG => self.medeleg = new,
            CSR_MIDELEG => self.mideleg = new,
            CSR_MIE => self.mie = new,
            CSR_MTVEC => self.mtvec = new,
            CSR_MCOUNTEREN => self.mcounteren = new,
            CSR_MSCRATCH => self.mscratch = new,
            CSR_MEPC => self.mepc = new & !1,
            CSR_MCAUSE => self.mcause = new,
            CSR_MTVAL => self.mtval = new,
            CSR_MIP => self.mip = new,
            _ => bail!("unsupported writable csr 0x{csr:03x}"),
        }
        Ok(CsrResult {
            old_value: old,
            new_value: self.read(csr, privilege)?,
        })
    }
}

fn ensure_privilege(csr: u16, privilege: PrivilegeLevel) -> Result<()> {
    let required = match (csr >> 8) & 0x3 {
        0 => PrivilegeLevel::User,
        1 => PrivilegeLevel::Supervisor,
        _ => PrivilegeLevel::Machine,
    };
    if privilege < required {
        bail!("insufficient privilege for csr 0x{csr:03x}");
    }
    Ok(())
}

fn sd_bit(mstatus: u64) -> u64 {
    if (mstatus & MSTATUS_FS_MASK) == MSTATUS_FS_MASK {
        MSTATUS_SD
    } else {
        0
    }
}

fn misa_rv64imafdsu() -> u64 {
    (2u64 << 62) | (1 << 0) | (1 << 3) | (1 << 5) | (1 << 8) | (1 << 12) | (1 << 18) | (1 << 20)
}
