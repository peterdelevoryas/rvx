use std::mem::offset_of;

use rvx_core::Bus;

use crate::cpu::PrivilegeLevel;
use crate::trap::{Exception, Trap};

pub(crate) const PAGE_SIZE: u64 = 4096;
const PTE_SIZE: u64 = 8;
const VPN_BITS: u32 = 9;
const VPN_MASK: u64 = (1 << VPN_BITS) - 1;
pub(crate) const SATP_MODE_SHIFT: u32 = 60;
pub(crate) const SATP_MODE_BARE: u64 = 0;
const SATP_MODE_SV39: u64 = 8;
const SATP_MODE_SV48: u64 = 9;
const SATP_MODE_SV57: u64 = 10;
pub(crate) const SATP_ASID_SHIFT: u32 = 44;
pub(crate) const SATP_ASID_MASK: u64 = 0xffff;
const SATP_PPN_MASK: u64 = (1u64 << 44) - 1;
pub(crate) const PTE_V: u64 = 1 << 0;
pub(crate) const PTE_R: u64 = 1 << 1;
pub(crate) const PTE_W: u64 = 1 << 2;
pub(crate) const PTE_X: u64 = 1 << 3;
pub(crate) const PTE_U: u64 = 1 << 4;
const PTE_A: u64 = 1 << 6;
const PTE_D: u64 = 1 << 7;
pub(crate) const MSTATUS_SUM: u64 = 1 << 18;
pub(crate) const MSTATUS_MXR: u64 = 1 << 19;
pub(crate) const TLB_SIZE: usize = 256;
pub(crate) const JIT_CACHE_SIZE: usize = 4096;
pub(crate) const JIT_CACHE_SIZE_MASK: u64 = (JIT_CACHE_SIZE - 1) as u64;
const INVALID_GUEST_PAGE: u64 = u64::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    Read,
    Write,
    Execute,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MmuStats {
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct TlbEntry {
    pub(crate) virt_base: u64,
    pub(crate) phys_base: u64,
    pub(crate) page_size: u64,
    pub(crate) perm: u8,
    pub(crate) asid: u16,
    pub(crate) valid: bool,
}

impl Default for TlbEntry {
    fn default() -> Self {
        Self {
            virt_base: 0,
            phys_base: 0,
            page_size: PAGE_SIZE,
            perm: 0,
            asid: 0,
            valid: false,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct JitPageCacheEntry {
    pub(crate) guest_page: u64,
    pub(crate) host_addend: u64,
    pub(crate) context_tag: u64,
}

impl Default for JitPageCacheEntry {
    fn default() -> Self {
        Self {
            guest_page: INVALID_GUEST_PAGE,
            host_addend: 0,
            context_tag: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Mmu {
    pub(crate) tlb: Box<[TlbEntry; TLB_SIZE]>,
    pub(crate) jit_user_read: Box<[JitPageCacheEntry; JIT_CACHE_SIZE]>,
    pub(crate) jit_user_write: Box<[JitPageCacheEntry; JIT_CACHE_SIZE]>,
    pub(crate) jit_supervisor_read: Box<[JitPageCacheEntry; JIT_CACHE_SIZE]>,
    pub(crate) jit_supervisor_write: Box<[JitPageCacheEntry; JIT_CACHE_SIZE]>,
    pub(crate) jit_supervisor_sum_read: Box<[JitPageCacheEntry; JIT_CACHE_SIZE]>,
    pub(crate) jit_supervisor_sum_write: Box<[JitPageCacheEntry; JIT_CACHE_SIZE]>,
    pub(crate) jit_supervisor_mxr_read: Box<[JitPageCacheEntry; JIT_CACHE_SIZE]>,
    pub(crate) jit_supervisor_sum_mxr_read: Box<[JitPageCacheEntry; JIT_CACHE_SIZE]>,
    pub(crate) stats: MmuStats,
    pub(crate) stats_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct InstructionFetchProbe {
    tlb: Box<[TlbEntry; TLB_SIZE]>,
}

pub(crate) const MMU_JIT_USER_READ_OFFSET: i32 = offset_of!(Mmu, jit_user_read) as i32;
pub(crate) const MMU_JIT_USER_WRITE_OFFSET: i32 = offset_of!(Mmu, jit_user_write) as i32;
pub(crate) const MMU_JIT_SUPERVISOR_READ_OFFSET: i32 = offset_of!(Mmu, jit_supervisor_read) as i32;
pub(crate) const MMU_JIT_SUPERVISOR_WRITE_OFFSET: i32 =
    offset_of!(Mmu, jit_supervisor_write) as i32;
pub(crate) const MMU_JIT_SUPERVISOR_SUM_READ_OFFSET: i32 =
    offset_of!(Mmu, jit_supervisor_sum_read) as i32;
pub(crate) const MMU_JIT_SUPERVISOR_SUM_WRITE_OFFSET: i32 =
    offset_of!(Mmu, jit_supervisor_sum_write) as i32;
pub(crate) const MMU_JIT_SUPERVISOR_MXR_READ_OFFSET: i32 =
    offset_of!(Mmu, jit_supervisor_mxr_read) as i32;
pub(crate) const MMU_JIT_SUPERVISOR_SUM_MXR_READ_OFFSET: i32 =
    offset_of!(Mmu, jit_supervisor_sum_mxr_read) as i32;
pub(crate) const JIT_CACHE_ENTRY_SIZE: i32 = std::mem::size_of::<JitPageCacheEntry>() as i32;
pub(crate) const JIT_CACHE_ENTRY_GUEST_PAGE_OFFSET: i32 =
    offset_of!(JitPageCacheEntry, guest_page) as i32;
pub(crate) const JIT_CACHE_ENTRY_HOST_ADDEND_OFFSET: i32 =
    offset_of!(JitPageCacheEntry, host_addend) as i32;
pub(crate) const JIT_CACHE_ENTRY_CONTEXT_TAG_OFFSET: i32 =
    offset_of!(JitPageCacheEntry, context_tag) as i32;
pub(crate) const MMU_STATS_TRANSLATED_LOAD_HITS_OFFSET: i32 =
    offset_of!(Mmu, stats) as i32 + offset_of!(MmuStats, translated_load_hits) as i32;
pub(crate) const MMU_STATS_TRANSLATED_LOAD_MISSES_OFFSET: i32 =
    offset_of!(Mmu, stats) as i32 + offset_of!(MmuStats, translated_load_misses) as i32;
pub(crate) const MMU_STATS_TRANSLATED_STORE_HITS_OFFSET: i32 =
    offset_of!(Mmu, stats) as i32 + offset_of!(MmuStats, translated_store_hits) as i32;
pub(crate) const MMU_STATS_TRANSLATED_STORE_MISSES_OFFSET: i32 =
    offset_of!(Mmu, stats) as i32 + offset_of!(MmuStats, translated_store_misses) as i32;

impl Mmu {
    pub fn new() -> Self {
        Self {
            tlb: Box::new([TlbEntry::default(); TLB_SIZE]),
            jit_user_read: Box::new([JitPageCacheEntry::default(); JIT_CACHE_SIZE]),
            jit_user_write: Box::new([JitPageCacheEntry::default(); JIT_CACHE_SIZE]),
            jit_supervisor_read: Box::new([JitPageCacheEntry::default(); JIT_CACHE_SIZE]),
            jit_supervisor_write: Box::new([JitPageCacheEntry::default(); JIT_CACHE_SIZE]),
            jit_supervisor_sum_read: Box::new([JitPageCacheEntry::default(); JIT_CACHE_SIZE]),
            jit_supervisor_sum_write: Box::new([JitPageCacheEntry::default(); JIT_CACHE_SIZE]),
            jit_supervisor_mxr_read: Box::new([JitPageCacheEntry::default(); JIT_CACHE_SIZE]),
            jit_supervisor_sum_mxr_read: Box::new([JitPageCacheEntry::default(); JIT_CACHE_SIZE]),
            stats: MmuStats::default(),
            stats_enabled: std::env::var_os("RVX_MMU_STATS").is_some(),
        }
    }

    pub fn flush(&mut self) {
        self.record_flush();
        self.flush_all_internal();
    }

    pub fn flush_vma(&mut self, addr: Option<u64>, asid: Option<u16>) {
        self.record_flush();
        match (addr, asid) {
            (None, None) => self.flush_all_internal(),
            (None, Some(asid)) => self.flush_asid_internal(asid),
            (Some(addr), asid) => self.invalidate_page_internal(addr, asid),
        }
    }

    pub fn flush_range(&mut self, start: u64, size: u64, asid: Option<u16>) {
        self.record_flush();
        if (start == 0 && size == 0) || size == u64::MAX {
            match asid {
                Some(asid) => self.flush_asid_internal(asid),
                None => self.flush_all_internal(),
            }
            return;
        }
        if size == 0 {
            return;
        }

        let mut offset = 0u64;
        loop {
            self.invalidate_page_internal(start.wrapping_add(offset), asid);
            if offset > size.saturating_sub(PAGE_SIZE) {
                break;
            }
            offset = offset.wrapping_add(PAGE_SIZE);
        }
    }

    pub fn stats(&self) -> MmuStats {
        self.stats
    }

    pub fn instruction_fetch_probe(&self) -> InstructionFetchProbe {
        InstructionFetchProbe {
            tlb: self.tlb.clone(),
        }
    }

    pub fn translate(
        &mut self,
        bus: &Bus,
        satp: u64,
        mstatus: u64,
        privilege: PrivilegeLevel,
        addr: u64,
        access: AccessType,
    ) -> Result<u64, Trap> {
        let mode = (satp >> SATP_MODE_SHIFT) & 0xf;
        if privilege == PrivilegeLevel::Machine || mode == SATP_MODE_BARE {
            return Ok(addr);
        }
        let levels = match mode {
            SATP_MODE_SV39 => 3,
            SATP_MODE_SV48 => 4,
            SATP_MODE_SV57 => 5,
            _ => {
                return Err(Trap::exception(page_fault(access), addr));
            }
        };
        if !is_canonical_va(addr, levels) {
            return Err(Trap::exception(page_fault(access), addr));
        }

        let asid = satp_asid(satp);
        let idx = tlb_index(addr);
        let entry = self.tlb[idx];
        if entry.valid
            && entry.asid == asid
            && addr >= entry.virt_base
            && addr < entry.virt_base + entry.page_size
        {
            if self.stats_enabled {
                self.stats.tlb_hits = self.stats.tlb_hits.wrapping_add(1);
            }
            check_perm(entry.perm as u64, access, privilege, mstatus, addr)?;
            let phys = entry.phys_base + (addr - entry.virt_base);
            self.update_jit_cache(bus, addr, privilege, access, mstatus, satp, phys);
            return Ok(phys);
        }
        if self.stats_enabled {
            self.stats.tlb_misses = self.stats.tlb_misses.wrapping_add(1);
            self.stats.page_walks = self.stats.page_walks.wrapping_add(1);
        }

        let root_ppn = satp & SATP_PPN_MASK;
        let mut table = root_ppn * PAGE_SIZE;
        for level in (0..levels).rev() {
            let index = vpn_index(addr, level);
            let pte_addr = table + index * PTE_SIZE;
            let mut pte = bus
                .read_u64(pte_addr)
                .map_err(|_| Trap::exception(access_fault(access), pte_addr))?;
            let perm = pte & 0xff;
            let r = (perm & PTE_R) != 0;
            let w = (perm & PTE_W) != 0;
            let x = (perm & PTE_X) != 0;
            if (perm & PTE_V) == 0 || (!r && w) {
                return Err(Trap::exception(page_fault(access), addr));
            }
            if r || x {
                if level > 0 {
                    let align_mask = (1u64 << (level as u32 * VPN_BITS)) - 1;
                    if ((pte >> 10) & align_mask) != 0 {
                        return Err(Trap::exception(page_fault(access), addr));
                    }
                }
                check_perm(perm, access, privilege, mstatus, addr)?;

                let mut updated = pte | PTE_A;
                if access == AccessType::Write {
                    updated |= PTE_D;
                }
                if updated != pte {
                    bus.write_u64(pte_addr, updated)
                        .map_err(|_| Trap::exception(access_fault(access), pte_addr))?;
                    pte = updated;
                }

                let page_size = level_page_size(level);
                let page_offset = addr & (page_size - 1);
                let phys_base = pte_phys_base(pte, page_size);
                let virt_base = addr & !(page_size - 1);
                self.tlb[idx] = TlbEntry {
                    virt_base,
                    phys_base,
                    page_size,
                    perm: (pte & 0xff) as u8,
                    asid,
                    valid: true,
                };
                let phys = phys_base + page_offset;
                self.update_jit_cache(bus, addr, privilege, access, mstatus, satp, phys);
                return Ok(phys);
            }
            table = (pte >> 10) * PAGE_SIZE;
        }

        Err(Trap::exception(page_fault(access), addr))
    }
}

impl InstructionFetchProbe {
    pub fn translate(
        &mut self,
        bus: &Bus,
        satp: u64,
        mstatus: u64,
        privilege: PrivilegeLevel,
        addr: u64,
    ) -> Result<u64, Trap> {
        let mode = (satp >> SATP_MODE_SHIFT) & 0xf;
        if privilege == PrivilegeLevel::Machine || mode == SATP_MODE_BARE {
            return Ok(addr);
        }
        let levels = match mode {
            SATP_MODE_SV39 => 3,
            SATP_MODE_SV48 => 4,
            SATP_MODE_SV57 => 5,
            _ => {
                return Err(Trap::exception(page_fault(AccessType::Execute), addr));
            }
        };
        if !is_canonical_va(addr, levels) {
            return Err(Trap::exception(page_fault(AccessType::Execute), addr));
        }

        let asid = satp_asid(satp);
        let idx = tlb_index(addr);
        let entry = self.tlb[idx];
        if entry.valid
            && entry.asid == asid
            && addr >= entry.virt_base
            && addr < entry.virt_base + entry.page_size
        {
            check_perm(
                entry.perm as u64,
                AccessType::Execute,
                privilege,
                mstatus,
                addr,
            )?;
            return Ok(entry.phys_base + (addr - entry.virt_base));
        }

        let root_ppn = satp & SATP_PPN_MASK;
        let mut table = root_ppn * PAGE_SIZE;
        for level in (0..levels).rev() {
            let index = vpn_index(addr, level);
            let pte_addr = table + index * PTE_SIZE;
            let pte = bus
                .read_u64(pte_addr)
                .map_err(|_| Trap::exception(access_fault(AccessType::Execute), pte_addr))?;
            let perm = pte & 0xff;
            let r = (perm & PTE_R) != 0;
            let w = (perm & PTE_W) != 0;
            let x = (perm & PTE_X) != 0;
            if (perm & PTE_V) == 0 || (!r && w) {
                return Err(Trap::exception(page_fault(AccessType::Execute), addr));
            }
            if r || x {
                if level > 0 {
                    let align_mask = (1u64 << (level as u32 * VPN_BITS)) - 1;
                    if ((pte >> 10) & align_mask) != 0 {
                        return Err(Trap::exception(page_fault(AccessType::Execute), addr));
                    }
                }
                check_perm(perm, AccessType::Execute, privilege, mstatus, addr)?;

                let page_size = level_page_size(level);
                let page_offset = addr & (page_size - 1);
                let phys_base = pte_phys_base(pte, page_size);
                let virt_base = addr & !(page_size - 1);
                self.tlb[idx] = TlbEntry {
                    virt_base,
                    phys_base,
                    page_size,
                    perm: (pte & 0xff) as u8,
                    asid,
                    valid: true,
                };
                return Ok(phys_base + page_offset);
            }
            table = (pte >> 10) * PAGE_SIZE;
        }

        Err(Trap::exception(page_fault(AccessType::Execute), addr))
    }
}

impl Default for Mmu {
    fn default() -> Self {
        Self::new()
    }
}

impl Mmu {
    fn record_flush(&mut self) {
        if self.stats_enabled {
            self.stats.flushes = self.stats.flushes.wrapping_add(1);
        }
    }

    fn flush_all_internal(&mut self) {
        for entry in self.tlb.iter_mut() {
            *entry = TlbEntry::default();
        }
        self.flush_all_jit_caches();
    }

    fn flush_asid_internal(&mut self, asid: u16) {
        for entry in self.tlb.iter_mut() {
            if entry.valid && entry.asid == asid {
                *entry = TlbEntry::default();
            }
        }
        self.flush_jit_caches_asid(asid);
    }

    fn invalidate_page_internal(&mut self, addr: u64, asid: Option<u16>) {
        for entry in self.tlb.iter_mut() {
            if !entry.valid {
                continue;
            }
            if asid.is_some_and(|asid| entry.asid != asid) {
                continue;
            }
            if addr >= entry.virt_base && addr < entry.virt_base + entry.page_size {
                *entry = TlbEntry::default();
            }
        }
        match asid {
            Some(asid) => self.invalidate_jit_page_asid(addr, asid),
            None => self.invalidate_jit_page(addr),
        }
    }

    fn flush_all_jit_caches(&mut self) {
        flush_jit_cache(&mut self.jit_user_read);
        flush_jit_cache(&mut self.jit_user_write);
        flush_jit_cache(&mut self.jit_supervisor_read);
        flush_jit_cache(&mut self.jit_supervisor_write);
        flush_jit_cache(&mut self.jit_supervisor_sum_read);
        flush_jit_cache(&mut self.jit_supervisor_sum_write);
        flush_jit_cache(&mut self.jit_supervisor_mxr_read);
        flush_jit_cache(&mut self.jit_supervisor_sum_mxr_read);
    }

    fn invalidate_jit_page(&mut self, addr: u64) {
        let guest_page = addr & !(PAGE_SIZE - 1);
        invalidate_jit_cache_page(&mut self.jit_user_read, guest_page);
        invalidate_jit_cache_page(&mut self.jit_user_write, guest_page);
        invalidate_jit_cache_page(&mut self.jit_supervisor_read, guest_page);
        invalidate_jit_cache_page(&mut self.jit_supervisor_write, guest_page);
        invalidate_jit_cache_page(&mut self.jit_supervisor_sum_read, guest_page);
        invalidate_jit_cache_page(&mut self.jit_supervisor_sum_write, guest_page);
        invalidate_jit_cache_page(&mut self.jit_supervisor_mxr_read, guest_page);
        invalidate_jit_cache_page(&mut self.jit_supervisor_sum_mxr_read, guest_page);
    }

    fn invalidate_jit_page_asid(&mut self, addr: u64, asid: u16) {
        let guest_page = addr & !(PAGE_SIZE - 1);
        invalidate_jit_cache_page_asid(&mut self.jit_user_read, guest_page, asid);
        invalidate_jit_cache_page_asid(&mut self.jit_user_write, guest_page, asid);
        invalidate_jit_cache_page_asid(&mut self.jit_supervisor_read, guest_page, asid);
        invalidate_jit_cache_page_asid(&mut self.jit_supervisor_write, guest_page, asid);
        invalidate_jit_cache_page_asid(&mut self.jit_supervisor_sum_read, guest_page, asid);
        invalidate_jit_cache_page_asid(&mut self.jit_supervisor_sum_write, guest_page, asid);
        invalidate_jit_cache_page_asid(&mut self.jit_supervisor_mxr_read, guest_page, asid);
        invalidate_jit_cache_page_asid(
            &mut self.jit_supervisor_sum_mxr_read,
            guest_page,
            asid,
        );
    }

    fn flush_jit_caches_asid(&mut self, asid: u16) {
        flush_jit_cache_asid(&mut self.jit_user_read, asid);
        flush_jit_cache_asid(&mut self.jit_user_write, asid);
        flush_jit_cache_asid(&mut self.jit_supervisor_read, asid);
        flush_jit_cache_asid(&mut self.jit_supervisor_write, asid);
        flush_jit_cache_asid(&mut self.jit_supervisor_sum_read, asid);
        flush_jit_cache_asid(&mut self.jit_supervisor_sum_write, asid);
        flush_jit_cache_asid(&mut self.jit_supervisor_mxr_read, asid);
        flush_jit_cache_asid(&mut self.jit_supervisor_sum_mxr_read, asid);
    }

    fn update_jit_cache(
        &mut self,
        bus: &Bus,
        addr: u64,
        privilege: PrivilegeLevel,
        access: AccessType,
        mstatus: u64,
        satp: u64,
        phys_addr: u64,
    ) {
        let cache = match (privilege, access) {
            (PrivilegeLevel::User, AccessType::Read) => &mut self.jit_user_read,
            (PrivilegeLevel::User, AccessType::Write) => &mut self.jit_user_write,
            (PrivilegeLevel::Supervisor, AccessType::Read) => {
                match mstatus & (MSTATUS_SUM | MSTATUS_MXR) {
                    0 => &mut self.jit_supervisor_read,
                    MSTATUS_SUM => &mut self.jit_supervisor_sum_read,
                    MSTATUS_MXR => &mut self.jit_supervisor_mxr_read,
                    _ => &mut self.jit_supervisor_sum_mxr_read,
                }
            }
            (PrivilegeLevel::Supervisor, AccessType::Write) => {
                if (mstatus & MSTATUS_SUM) != 0 {
                    &mut self.jit_supervisor_sum_write
                } else {
                    &mut self.jit_supervisor_write
                }
            }
            (PrivilegeLevel::Machine, _) => return,
            (_, AccessType::Execute) => return,
        };
        let phys_page = phys_addr & !(PAGE_SIZE - 1);
        if !bus.ram().contains(phys_page, PAGE_SIZE as usize) {
            return;
        }
        let idx = jit_cache_index(addr);
        let guest_page = addr & !(PAGE_SIZE - 1);
        let host_page = (bus.ram().as_mut_ptr() as usize as u64)
            .wrapping_add(phys_page.wrapping_sub(bus.ram().base()));
        cache[idx] = JitPageCacheEntry {
            guest_page,
            host_addend: host_page.wrapping_sub(guest_page),
            context_tag: satp,
        };
        if self.stats_enabled {
            self.stats.jit_cache_fills = self.stats.jit_cache_fills.wrapping_add(1);
        }
    }
}

fn flush_jit_cache(entries: &mut [JitPageCacheEntry; JIT_CACHE_SIZE]) {
    for entry in entries.iter_mut() {
        *entry = JitPageCacheEntry::default();
    }
}

fn invalidate_jit_cache_page(entries: &mut [JitPageCacheEntry; JIT_CACHE_SIZE], guest_page: u64) {
    let idx = jit_cache_index(guest_page);
    if entries[idx].guest_page == guest_page {
        entries[idx] = JitPageCacheEntry::default();
    }
}

fn invalidate_jit_cache_page_asid(
    entries: &mut [JitPageCacheEntry; JIT_CACHE_SIZE],
    guest_page: u64,
    asid: u16,
) {
    let idx = jit_cache_index(guest_page);
    if entries[idx].guest_page == guest_page && satp_asid(entries[idx].context_tag) == asid {
        entries[idx] = JitPageCacheEntry::default();
    }
}

fn flush_jit_cache_asid(entries: &mut [JitPageCacheEntry; JIT_CACHE_SIZE], asid: u16) {
    for entry in entries.iter_mut() {
        if entry.guest_page != INVALID_GUEST_PAGE && satp_asid(entry.context_tag) == asid {
            *entry = JitPageCacheEntry::default();
        }
    }
}

fn vpn_index(addr: u64, level: usize) -> u64 {
    (addr >> (12 + level as u32 * VPN_BITS)) & VPN_MASK
}

fn tlb_index(addr: u64) -> usize {
    let vpn = addr >> 12;
    ((vpn ^ (vpn >> 8)) as usize) & (TLB_SIZE - 1)
}

fn jit_cache_index(addr: u64) -> usize {
    let vpn = addr >> 12;
    ((vpn ^ (vpn >> 8)) as usize) & (JIT_CACHE_SIZE - 1)
}

fn satp_asid(satp: u64) -> u16 {
    ((satp >> SATP_ASID_SHIFT) & SATP_ASID_MASK) as u16
}

fn level_page_size(level: usize) -> u64 {
    PAGE_SIZE << (level as u32 * VPN_BITS)
}

fn pte_phys_base(pte: u64, page_size: u64) -> u64 {
    ((pte >> 10) << 12) & !(page_size - 1)
}

fn is_canonical_va(addr: u64, levels: usize) -> bool {
    let va_bits = 12 + levels as u32 * VPN_BITS;
    let sign_bit = va_bits - 1;
    let top = addr >> va_bits;
    let top_mask = (1u64 << (64 - va_bits)) - 1;
    if ((addr >> sign_bit) & 1) == 0 {
        top == 0
    } else {
        top == top_mask
    }
}

fn check_perm(
    perm: u64,
    access: AccessType,
    privilege: PrivilegeLevel,
    mstatus: u64,
    fault_addr: u64,
) -> Result<(), Trap> {
    let r = (perm & PTE_R) != 0;
    let w = (perm & PTE_W) != 0;
    let x = (perm & PTE_X) != 0;
    let u = (perm & PTE_U) != 0;
    let sum = (mstatus & MSTATUS_SUM) != 0;
    let mxr = (mstatus & MSTATUS_MXR) != 0;
    let allowed = match access {
        AccessType::Read => r || (mxr && x),
        AccessType::Write => w,
        AccessType::Execute => x,
    };
    if !allowed {
        return Err(Trap::exception(page_fault(access), fault_addr));
    }
    match privilege {
        PrivilegeLevel::User => {
            if !u {
                return Err(Trap::exception(page_fault(access), fault_addr));
            }
        }
        PrivilegeLevel::Supervisor => {
            if u {
                if access == AccessType::Execute || !sum {
                    return Err(Trap::exception(page_fault(access), fault_addr));
                }
            }
        }
        PrivilegeLevel::Machine => {}
    }
    Ok(())
}

fn page_fault(access: AccessType) -> Exception {
    match access {
        AccessType::Read => Exception::LoadPageFault,
        AccessType::Write => Exception::StorePageFault,
        AccessType::Execute => Exception::InstructionPageFault,
    }
}

fn access_fault(access: AccessType) -> Exception {
    match access {
        AccessType::Read => Exception::LoadAccessFault,
        AccessType::Write => Exception::StoreAccessFault,
        AccessType::Execute => Exception::InstructionAccessFault,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_entry(guest_page: u64) -> JitPageCacheEntry {
        JitPageCacheEntry {
            guest_page,
            host_addend: 0x1234_5678,
            context_tag: (7u64) << SATP_ASID_SHIFT,
        }
    }

    #[test]
    fn flush_vma_page_preserves_unrelated_entries() {
        let mut mmu = Mmu::new();
        let page_a = 0x4000;
        let page_b = 0x8000;
        mmu.tlb[0] = TlbEntry {
            virt_base: page_a,
            phys_base: 0x1000,
            page_size: PAGE_SIZE,
            perm: (PTE_V | PTE_R) as u8,
            asid: 7,
            valid: true,
        };
        mmu.tlb[1] = TlbEntry {
            virt_base: page_b,
            phys_base: 0x2000,
            page_size: PAGE_SIZE,
            perm: (PTE_V | PTE_R) as u8,
            asid: 7,
            valid: true,
        };
        mmu.jit_supervisor_read[jit_cache_index(page_a)] = cache_entry(page_a);
        mmu.jit_supervisor_read[jit_cache_index(page_b)] = cache_entry(page_b);

        mmu.flush_vma(Some(page_a), None);

        assert!(!mmu.tlb[0].valid);
        assert!(mmu.tlb[1].valid);
        assert_eq!(
            mmu.jit_supervisor_read[jit_cache_index(page_a)].guest_page,
            INVALID_GUEST_PAGE
        );
        assert_eq!(
            mmu.jit_supervisor_read[jit_cache_index(page_b)].guest_page,
            page_b
        );
    }

    #[test]
    fn flush_vma_asid_preserves_other_asids_in_tlb() {
        let mut mmu = Mmu::new();
        mmu.tlb[0] = TlbEntry {
            virt_base: 0x4000,
            phys_base: 0x1000,
            page_size: PAGE_SIZE,
            perm: (PTE_V | PTE_R) as u8,
            asid: 1,
            valid: true,
        };
        mmu.tlb[1] = TlbEntry {
            virt_base: 0x8000,
            phys_base: 0x2000,
            page_size: PAGE_SIZE,
            perm: (PTE_V | PTE_R) as u8,
            asid: 2,
            valid: true,
        };
        mmu.jit_supervisor_read[jit_cache_index(0x4000)] = JitPageCacheEntry {
            guest_page: 0x4000,
            host_addend: 0x1234_5678,
            context_tag: (1u64) << SATP_ASID_SHIFT,
        };

        mmu.flush_vma(None, Some(1));

        assert!(!mmu.tlb[0].valid);
        assert!(mmu.tlb[1].valid);
        assert_eq!(
            mmu.jit_supervisor_read[jit_cache_index(0x4000)].guest_page,
            INVALID_GUEST_PAGE
        );
    }

    #[test]
    fn flush_vma_asid_preserves_other_asids_in_jit_cache() {
        let mut mmu = Mmu::new();
        let page_a = 0x4000;
        let page_b = 0x8000;
        mmu.jit_supervisor_read[jit_cache_index(page_a)] = JitPageCacheEntry {
            guest_page: page_a,
            host_addend: 0x1234_0000,
            context_tag: (1u64) << SATP_ASID_SHIFT,
        };
        mmu.jit_supervisor_read[jit_cache_index(page_b)] = JitPageCacheEntry {
            guest_page: page_b,
            host_addend: 0x5678_0000,
            context_tag: (2u64) << SATP_ASID_SHIFT,
        };

        mmu.flush_vma(None, Some(1));

        assert_eq!(
            mmu.jit_supervisor_read[jit_cache_index(page_a)].guest_page,
            INVALID_GUEST_PAGE
        );
        assert_eq!(
            mmu.jit_supervisor_read[jit_cache_index(page_b)].guest_page,
            page_b
        );
        assert_eq!(
            mmu.jit_supervisor_read[jit_cache_index(page_b)].context_tag,
            (2u64) << SATP_ASID_SHIFT
        );
    }

    #[test]
    fn flush_range_invalidates_each_page_in_range() {
        let mut mmu = Mmu::new();
        for (index, page) in [0x4000, 0x5000, 0x6000, 0x7000].into_iter().enumerate() {
            mmu.tlb[index] = TlbEntry {
                virt_base: page,
                phys_base: 0x1000 + page,
                page_size: PAGE_SIZE,
                perm: (PTE_V | PTE_R) as u8,
                asid: 3,
                valid: true,
            };
            mmu.jit_supervisor_read[jit_cache_index(page)] = cache_entry(page);
        }

        mmu.flush_range(0x4800, PAGE_SIZE + 1, None);

        assert!(!mmu.tlb[0].valid);
        assert!(!mmu.tlb[1].valid);
        assert!(mmu.tlb[2].valid);
        assert!(mmu.tlb[3].valid);
        assert_eq!(
            mmu.jit_supervisor_read[jit_cache_index(0x4000)].guest_page,
            INVALID_GUEST_PAGE
        );
        assert_eq!(
            mmu.jit_supervisor_read[jit_cache_index(0x5000)].guest_page,
            INVALID_GUEST_PAGE
        );
        assert_eq!(
            mmu.jit_supervisor_read[jit_cache_index(0x6000)].guest_page,
            0x6000
        );
    }
}
