use rvx_core::Bus;

use crate::csr::{
    CsrFile, MSTATUS_MIE, MSTATUS_MPIE, MSTATUS_MPP_MASK, MSTATUS_MPRV, MSTATUS_MXR, MSTATUS_SIE,
    MSTATUS_SPIE, MSTATUS_SPP, MSTATUS_SUM,
};
use crate::mmu::{AccessType, InstructionFetchProbe, Mmu, PAGE_SIZE};
use crate::trap::{Exception, Interrupt, Trap};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrivilegeLevel {
    User = 0,
    Supervisor = 1,
    Machine = 3,
}

impl PrivilegeLevel {
    pub(crate) fn from_mstatus_mpp(mstatus: u64) -> Self {
        match (mstatus & MSTATUS_MPP_MASK) >> 11 {
            0 => Self::User,
            1 => Self::Supervisor,
            3 => Self::Machine,
            _ => Self::Machine,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Cpu {
    pub x: [u64; 32],
    pub f: [u64; 32],
    pub pc: u64,
    pub privilege: PrivilegeLevel,
    pub csr: CsrFile,
    pub reservation: Option<u64>,
    pub reservation_value: u64,
    pub reservation_epoch: u64,
    pub pending_mip: u64,
    pub mmu: Mmu,
}

#[derive(Debug, Clone)]
pub struct BlockBuilderCpu {
    pub pc: u64,
    privilege: PrivilegeLevel,
    satp: u64,
    mstatus: u64,
    mmu: InstructionFetchProbe,
}

impl Cpu {
    pub fn new() -> Self {
        Self {
            x: [0; 32],
            f: [0; 32],
            pc: 0,
            privilege: PrivilegeLevel::Machine,
            csr: CsrFile::default(),
            reservation: None,
            reservation_value: 0,
            reservation_epoch: 0,
            pending_mip: 0,
            mmu: Mmu::default(),
        }
    }

    pub fn read_x(&self, index: u8) -> u64 {
        if index == 0 {
            0
        } else {
            self.x[index as usize]
        }
    }

    pub fn write_x(&mut self, index: u8, value: u64) {
        if index != 0 {
            self.x[index as usize] = value;
        }
    }

    pub fn read_f(&self, index: u8) -> u64 {
        self.f[index as usize]
    }

    pub fn write_f(&mut self, index: u8, value: u64) {
        self.f[index as usize] = value;
    }

    pub fn sync_mip(&mut self) {
        self.csr.mip = self.pending_mip;
    }

    pub fn update_time(&mut self, time_ticks: u64) {
        self.csr.time = time_ticks;
        self.csr.cycle = self.csr.cycle.wrapping_add(1);
    }

    pub(crate) fn effective_data_privilege(&self) -> PrivilegeLevel {
        if self.privilege == PrivilegeLevel::Machine && (self.csr.mstatus & MSTATUS_MPRV) != 0 {
            PrivilegeLevel::from_mstatus_mpp(self.csr.mstatus)
        } else {
            self.privilege
        }
    }

    pub(crate) fn data_mstatus_vm(&self) -> u64 {
        self.csr.mstatus & (MSTATUS_SUM | MSTATUS_MXR)
    }

    pub fn block_builder(&self) -> BlockBuilderCpu {
        BlockBuilderCpu {
            pc: self.pc,
            privilege: self.privilege,
            satp: self.csr.satp,
            mstatus: self.csr.mstatus,
            mmu: self.mmu.instruction_fetch_probe(),
        }
    }

    pub fn fetch_u16(&mut self, bus: &Bus) -> Result<u16, Trap> {
        let addr = self.mmu.translate(
            bus,
            self.csr.satp,
            self.csr.mstatus,
            self.privilege,
            self.pc,
            AccessType::Execute,
        )?;
        bus.read_u16(addr)
            .map_err(|_| Trap::exception(Exception::InstructionAccessFault, self.pc))
    }

    pub fn fetch_u32(&mut self, bus: &Bus) -> Result<u32, Trap> {
        let addr = self.mmu.translate(
            bus,
            self.csr.satp,
            self.csr.mstatus,
            self.privilege,
            self.pc,
            AccessType::Execute,
        )?;
        bus.read_u32(addr)
            .map_err(|_| Trap::exception(Exception::InstructionAccessFault, self.pc))
    }

    pub fn load_u8(&mut self, bus: &Bus, addr: u64) -> Result<u8, Trap> {
        let phys = self.translate_data(bus, addr, AccessType::Read)?;
        bus.read_u8(phys)
            .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))
    }

    pub fn load_u16(&mut self, bus: &Bus, addr: u64) -> Result<u16, Trap> {
        if crosses_guest_page(addr, 2) {
            return self.load_split_u16(bus, addr);
        }
        let phys = self.translate_data(bus, addr, AccessType::Read)?;
        bus.read_u16(phys)
            .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))
    }

    pub fn load_u32(&mut self, bus: &Bus, addr: u64) -> Result<u32, Trap> {
        if crosses_guest_page(addr, 4) {
            return self.load_split_u32(bus, addr);
        }
        let phys = self.translate_data(bus, addr, AccessType::Read)?;
        bus.read_u32(phys)
            .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))
    }

    pub fn load_u64(&mut self, bus: &Bus, addr: u64) -> Result<u64, Trap> {
        if crosses_guest_page(addr, 8) {
            return self.load_split_u64(bus, addr);
        }
        let phys = self.translate_data(bus, addr, AccessType::Read)?;
        bus.read_u64(phys)
            .map_err(|_| Trap::exception(Exception::LoadAccessFault, addr))
    }

    pub fn store_u8(&mut self, bus: &Bus, addr: u64, value: u8) -> Result<(), Trap> {
        let phys = self.translate_data(bus, addr, AccessType::Write)?;
        bus.write_u8(phys, value)
            .map_err(|_| Trap::exception(Exception::StoreAccessFault, addr))
    }

    pub fn store_u16(&mut self, bus: &Bus, addr: u64, value: u16) -> Result<(), Trap> {
        if crosses_guest_page(addr, 2) {
            return self.store_split_bytes(bus, addr, &value.to_le_bytes());
        }
        let phys = self.translate_data(bus, addr, AccessType::Write)?;
        bus.write_u16(phys, value)
            .map_err(|_| Trap::exception(Exception::StoreAccessFault, addr))
    }

    pub fn store_u32(&mut self, bus: &Bus, addr: u64, value: u32) -> Result<(), Trap> {
        if crosses_guest_page(addr, 4) {
            return self.store_split_bytes(bus, addr, &value.to_le_bytes());
        }
        let phys = self.translate_data(bus, addr, AccessType::Write)?;
        bus.write_u32(phys, value)
            .map_err(|_| Trap::exception(Exception::StoreAccessFault, addr))
    }

    pub fn store_u64(&mut self, bus: &Bus, addr: u64, value: u64) -> Result<(), Trap> {
        if crosses_guest_page(addr, 8) {
            return self.store_split_bytes(bus, addr, &value.to_le_bytes());
        }
        let phys = self.translate_data(bus, addr, AccessType::Write)?;
        bus.write_u64(phys, value)
            .map_err(|_| Trap::exception(Exception::StoreAccessFault, addr))
    }

    pub(crate) fn translate_data(
        &mut self,
        bus: &Bus,
        addr: u64,
        access: AccessType,
    ) -> Result<u64, Trap> {
        self.mmu.translate(
            bus,
            self.csr.satp,
            self.csr.mstatus,
            self.effective_data_privilege(),
            addr,
            access,
        )
    }

    pub fn take_trap(&mut self, trap: Trap) {
        let cause = trap.cause();
        let delegated = if trap.interrupt.is_some() {
            if trap.interrupt == Some(Interrupt::SupervisorSoft)
                || trap.interrupt == Some(Interrupt::SupervisorTimer)
                || trap.interrupt == Some(Interrupt::SupervisorExternal)
            {
                true
            } else {
                self.privilege != PrivilegeLevel::Machine
                    && ((self.csr.mideleg >> (cause & !(1u64 << 63))) & 1) != 0
            }
        } else {
            self.privilege != PrivilegeLevel::Machine && ((self.csr.medeleg >> cause) & 1) != 0
        };

        if delegated || self.privilege != PrivilegeLevel::Machine {
            self.csr.sepc = self.pc;
            self.csr.scause = cause;
            self.csr.stval = trap.tval;
            if self.privilege == PrivilegeLevel::Supervisor {
                self.csr.mstatus |= MSTATUS_SPP;
            } else {
                self.csr.mstatus &= !MSTATUS_SPP;
            }
            if (self.csr.mstatus & MSTATUS_SIE) != 0 {
                self.csr.mstatus |= MSTATUS_SPIE;
            } else {
                self.csr.mstatus &= !MSTATUS_SPIE;
            }
            self.csr.mstatus &= !MSTATUS_SIE;
            self.privilege = PrivilegeLevel::Supervisor;
            self.pc = trap_vector(self.csr.stvec, cause);
        } else {
            self.csr.mepc = self.pc;
            self.csr.mcause = cause;
            self.csr.mtval = trap.tval;
            if (self.csr.mstatus & MSTATUS_MIE) != 0 {
                self.csr.mstatus |= MSTATUS_MPIE;
            } else {
                self.csr.mstatus &= !MSTATUS_MPIE;
            }
            self.csr.mstatus &= !MSTATUS_MIE;
            self.privilege = PrivilegeLevel::Machine;
            self.pc = trap_vector(self.csr.mtvec, cause);
        }
    }

    pub fn pending_interrupt(&mut self) -> Option<Trap> {
        self.sync_mip();
        let pending = self.csr.mip & self.csr.mie;
        for interrupt in [
            Interrupt::MachineExternal,
            Interrupt::MachineSoft,
            Interrupt::MachineTimer,
            Interrupt::SupervisorExternal,
            Interrupt::SupervisorSoft,
            Interrupt::SupervisorTimer,
        ] {
            let bit = 1u64 << interrupt as u64;
            if (pending & bit) == 0 {
                continue;
            }
            let delegated = matches!(
                interrupt,
                Interrupt::SupervisorExternal
                    | Interrupt::SupervisorSoft
                    | Interrupt::SupervisorTimer
            ) || ((self.csr.mideleg >> interrupt as u64) & 1) != 0;
            let enabled = if delegated {
                self.privilege < PrivilegeLevel::Supervisor
                    || (self.privilege == PrivilegeLevel::Supervisor
                        && (self.csr.mstatus & MSTATUS_SIE) != 0)
            } else {
                self.privilege < PrivilegeLevel::Machine
                    || (self.privilege == PrivilegeLevel::Machine
                        && (self.csr.mstatus & MSTATUS_MIE) != 0)
            };
            if enabled {
                return Some(Trap::interrupt(interrupt));
            }
        }
        None
    }

    pub fn execute_sret(&mut self) -> Result<(), Trap> {
        if self.privilege < PrivilegeLevel::Supervisor {
            return Err(Trap::exception(Exception::IllegalInstruction, self.pc));
        }
        let new_privilege = if (self.csr.mstatus & MSTATUS_SPP) != 0 {
            PrivilegeLevel::Supervisor
        } else {
            PrivilegeLevel::User
        };
        self.privilege = new_privilege;
        if (self.csr.mstatus & MSTATUS_SPIE) != 0 {
            self.csr.mstatus |= MSTATUS_SIE;
        } else {
            self.csr.mstatus &= !MSTATUS_SIE;
        }
        self.csr.mstatus |= MSTATUS_SPIE;
        self.csr.mstatus &= !MSTATUS_SPP;
        self.csr.mstatus &= !MSTATUS_MPRV;
        self.pc = self.csr.sepc;
        Ok(())
    }

    pub fn execute_mret(&mut self) -> Result<(), Trap> {
        if self.privilege != PrivilegeLevel::Machine {
            return Err(Trap::exception(Exception::IllegalInstruction, self.pc));
        }
        let new_privilege = PrivilegeLevel::from_mstatus_mpp(self.csr.mstatus);
        self.privilege = new_privilege;
        if (self.csr.mstatus & MSTATUS_MPIE) != 0 {
            self.csr.mstatus |= MSTATUS_MIE;
        } else {
            self.csr.mstatus &= !MSTATUS_MIE;
        }
        self.csr.mstatus |= MSTATUS_MPIE;
        self.csr.mstatus &= !MSTATUS_MPP_MASK;
        if new_privilege != PrivilegeLevel::Machine {
            self.csr.mstatus &= !MSTATUS_MPRV;
        }
        self.pc = self.csr.mepc;
        Ok(())
    }

    fn load_split_u16(&mut self, bus: &Bus, addr: u64) -> Result<u16, Trap> {
        Ok(u16::from_le_bytes(self.load_split_bytes(bus, addr)?))
    }

    fn load_split_u32(&mut self, bus: &Bus, addr: u64) -> Result<u32, Trap> {
        Ok(u32::from_le_bytes(self.load_split_bytes(bus, addr)?))
    }

    fn load_split_u64(&mut self, bus: &Bus, addr: u64) -> Result<u64, Trap> {
        Ok(u64::from_le_bytes(self.load_split_bytes(bus, addr)?))
    }

    fn load_split_bytes<const N: usize>(&mut self, bus: &Bus, addr: u64) -> Result<[u8; N], Trap> {
        let mut bytes = [0; N];
        for (offset, byte) in bytes.iter_mut().enumerate() {
            let guest_addr = addr.wrapping_add(offset as u64);
            let phys = self.translate_data(bus, guest_addr, AccessType::Read)?;
            *byte = bus
                .read_u8(phys)
                .map_err(|_| Trap::exception(Exception::LoadAccessFault, guest_addr))?;
        }
        Ok(bytes)
    }

    fn store_split_bytes(&mut self, bus: &Bus, addr: u64, bytes: &[u8]) -> Result<(), Trap> {
        for (offset, byte) in bytes.iter().enumerate() {
            let guest_addr = addr.wrapping_add(offset as u64);
            let phys = self.translate_data(bus, guest_addr, AccessType::Write)?;
            bus.write_u8(phys, *byte)
                .map_err(|_| Trap::exception(Exception::StoreAccessFault, guest_addr))?;
        }
        Ok(())
    }
}

impl BlockBuilderCpu {
    pub fn fetch_u16(&mut self, bus: &Bus) -> Result<u16, Trap> {
        let addr = self
            .mmu
            .translate(bus, self.satp, self.mstatus, self.privilege, self.pc)?;
        bus.read_u16(addr)
            .map_err(|_| Trap::exception(Exception::InstructionAccessFault, self.pc))
    }

    pub fn fetch_u32(&mut self, bus: &Bus) -> Result<u32, Trap> {
        let addr = self
            .mmu
            .translate(bus, self.satp, self.mstatus, self.privilege, self.pc)?;
        bus.read_u32(addr)
            .map_err(|_| Trap::exception(Exception::InstructionAccessFault, self.pc))
    }
}

fn trap_vector(tvec: u64, cause: u64) -> u64 {
    let base = tvec & !0x3;
    let vectored = (tvec & 0x3) == 1 && ((cause >> 63) != 0);
    if vectored {
        base.wrapping_add((cause & !(1u64 << 63)) * 4)
    } else {
        base
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

fn crosses_guest_page(addr: u64, size: usize) -> bool {
    let page_offset = (addr & (PAGE_SIZE - 1)) as usize;
    page_offset + size > PAGE_SIZE as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmu::{AccessType, PTE_R, PTE_U, PTE_V, PTE_W, SATP_MODE_SHIFT};
    use rvx_core::Ram;

    const RAM_BASE: u64 = 0x8000_0000;
    const RAM_SIZE: u64 = 0x20_000;
    const ROOT_PT: u64 = RAM_BASE + 0x1_000;
    const MID_PT: u64 = RAM_BASE + 0x2_000;
    const LEAF_PT: u64 = RAM_BASE + 0x3_000;
    const DATA_PAGE_0: u64 = RAM_BASE + 0x4_000;
    const DATA_PAGE_1: u64 = RAM_BASE + 0x5_000;
    const TEST_VA: u64 = 0x0000_0000_4000_1000;
    const SV39_MODE: u64 = 8;

    #[test]
    fn mret_restores_mpp_and_preserves_mmu_cache() {
        let mut cpu = Cpu::new();
        cpu.pc = 0x1000;
        cpu.csr.mepc = 0x2000;
        cpu.csr.mstatus = MSTATUS_MPIE | MSTATUS_MPRV | mpp_bits(PrivilegeLevel::Supervisor);
        cpu.mmu.tlb[0].valid = true;

        cpu.execute_mret().unwrap();

        assert_eq!(cpu.privilege, PrivilegeLevel::Supervisor);
        assert_eq!(cpu.pc, 0x2000);
        assert_ne!(cpu.csr.mstatus & MSTATUS_MIE, 0);
        assert_eq!(cpu.csr.mstatus & MSTATUS_MPP_MASK, 0);
        assert_eq!(cpu.csr.mstatus & MSTATUS_MPRV, 0);
        assert!(cpu.mmu.tlb[0].valid);
    }

    #[test]
    fn sret_clears_mprv_without_flushing_mmu() {
        let mut cpu = Cpu::new();
        cpu.privilege = PrivilegeLevel::Supervisor;
        cpu.pc = 0x1000;
        cpu.csr.sepc = 0x3000;
        cpu.csr.mstatus = MSTATUS_SPIE | MSTATUS_MPRV;
        cpu.mmu.tlb[0].valid = true;

        cpu.execute_sret().unwrap();

        assert_eq!(cpu.privilege, PrivilegeLevel::User);
        assert_eq!(cpu.pc, 0x3000);
        assert_ne!(cpu.csr.mstatus & MSTATUS_SIE, 0);
        assert_eq!(cpu.csr.mstatus & MSTATUS_MPRV, 0);
        assert!(cpu.mmu.tlb[0].valid);
    }

    #[test]
    fn translate_data_honors_mprv_effective_privilege() {
        let bus = setup_sv39_bus(PTE_R | PTE_W);
        let mut cpu = Cpu::new();
        cpu.privilege = PrivilegeLevel::Machine;
        cpu.csr.satp = (SV39_MODE << SATP_MODE_SHIFT as u64) | ppn(ROOT_PT);
        cpu.csr.mstatus = MSTATUS_MPRV | mpp_bits(PrivilegeLevel::Supervisor);

        let phys = cpu.translate_data(&bus, TEST_VA, AccessType::Read).unwrap();
        assert_eq!(phys, DATA_PAGE_0);

        cpu.csr.mstatus = MSTATUS_MPRV | mpp_bits(PrivilegeLevel::User);
        let trap = cpu
            .translate_data(&bus, TEST_VA, AccessType::Read)
            .unwrap_err();
        assert_eq!(trap.exception, Some(Exception::LoadPageFault));
        assert_eq!(trap.tval, TEST_VA);
    }

    #[test]
    fn cross_page_load_rechecks_translation_on_second_page() {
        let bus = setup_cross_page_bus(false);
        let mut cpu = Cpu::new();
        cpu.privilege = PrivilegeLevel::User;
        cpu.csr.satp = (SV39_MODE << SATP_MODE_SHIFT as u64) | ppn(ROOT_PT);

        let trap = cpu.load_u32(&bus, TEST_VA + PAGE_SIZE - 2).unwrap_err();
        assert_eq!(trap.exception, Some(Exception::LoadPageFault));
        assert_eq!(trap.tval, TEST_VA + PAGE_SIZE);
    }

    fn setup_cross_page_bus(map_second_page: bool) -> Bus {
        let bus = setup_sv39_bus(PTE_R | PTE_W | PTE_U);
        bus.write_u16(DATA_PAGE_0 + PAGE_SIZE - 2, 0xbbaa).unwrap();
        if map_second_page {
            map_sv39_page(
                &bus,
                TEST_VA + PAGE_SIZE,
                DATA_PAGE_1,
                PTE_R | PTE_W | PTE_U,
            );
        }
        bus
    }

    fn setup_sv39_bus(perm: u64) -> Bus {
        let bus = Bus::new(Ram::new(RAM_BASE, RAM_SIZE).unwrap());
        map_sv39_page(&bus, TEST_VA, DATA_PAGE_0, perm);
        bus
    }

    fn map_sv39_page(bus: &Bus, va: u64, pa: u64, perm: u64) {
        let root_index = (va >> 30) & 0x1ff;
        let mid_index = (va >> 21) & 0x1ff;
        let leaf_index = (va >> 12) & 0x1ff;
        bus.write_u64(ROOT_PT + root_index * 8, pte_table(MID_PT))
            .unwrap();
        bus.write_u64(MID_PT + mid_index * 8, pte_table(LEAF_PT))
            .unwrap();
        bus.write_u64(LEAF_PT + leaf_index * 8, pte_leaf(pa, perm))
            .unwrap();
    }

    fn pte_table(addr: u64) -> u64 {
        (ppn(addr) << 10) | PTE_V
    }

    fn pte_leaf(addr: u64, perm: u64) -> u64 {
        (ppn(addr) << 10) | perm | PTE_V
    }

    fn ppn(addr: u64) -> u64 {
        addr >> 12
    }

    fn mpp_bits(privilege: PrivilegeLevel) -> u64 {
        (privilege as u64) << 11
    }
}
