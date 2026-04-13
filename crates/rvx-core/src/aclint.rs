use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::IrqLine;

#[derive(Debug)]
pub struct TimerState {
    pub epoch: Instant,
    pub base_ticks: u64,
}

impl Default for TimerState {
    fn default() -> Self {
        Self {
            epoch: Instant::now(),
            base_ticks: 0,
        }
    }
}

pub struct Aclint {
    hart_count: usize,
    timer_hz: u64,
    timer: TimerState,
    mtimecmp: Vec<u64>,
    msip: Vec<u32>,
    mti: Vec<Option<IrqLine>>,
    msi: Vec<Option<IrqLine>>,
    generation: Arc<Vec<AtomicU64>>,
}

impl Aclint {
    pub fn new(hart_count: usize, timer_hz: u64) -> Self {
        let generation = (0..hart_count).map(|_| AtomicU64::new(0)).collect();
        Self {
            hart_count,
            timer_hz,
            timer: TimerState::default(),
            mtimecmp: vec![u64::MAX; hart_count],
            msip: vec![0; hart_count],
            mti: vec![None; hart_count],
            msi: vec![None; hart_count],
            generation: Arc::new(generation),
        }
    }

    pub fn connect_mti(&mut self, hart: usize, line: IrqLine) {
        self.mti[hart] = Some(line);
    }

    pub fn connect_msi(&mut self, hart: usize, line: IrqLine) {
        self.msi[hart] = Some(line);
    }

    pub fn read(&self, offset: u64, _size: u8) -> u64 {
        match offset {
            0xbff8 => self.read_mtime(),
            offset if offset >= 0x4000 => {
                let hart = ((offset - 0x4000) / 8) as usize;
                self.mtimecmp.get(hart).copied().unwrap_or(0)
            }
            offset => {
                let hart = (offset / 4) as usize;
                self.msip.get(hart).copied().unwrap_or(0) as u64
            }
        }
    }

    pub fn write(&mut self, offset: u64, _size: u8, value: u64) {
        match offset {
            0xbff8 => {
                self.timer.base_ticks = value;
                self.timer.epoch = Instant::now();
                self.recompute_all();
            }
            offset if offset >= 0x4000 => {
                let hart = ((offset - 0x4000) / 8) as usize;
                if hart < self.hart_count {
                    self.generation[hart].fetch_add(1, Ordering::SeqCst);
                    self.mtimecmp[hart] = value;
                    self.update_timer_irq(hart);
                }
            }
            offset => {
                let hart = (offset / 4) as usize;
                if hart < self.hart_count {
                    self.msip[hart] = (value & 1) as u32;
                    if let Some(ref irq) = self.msi[hart] {
                        irq.set((value & 1) != 0);
                    }
                }
            }
        }
    }

    pub fn read_mtime(&self) -> u64 {
        let elapsed = self.timer.epoch.elapsed().as_nanos();
        self.timer.base_ticks + ((elapsed * self.timer_hz as u128) / 1_000_000_000) as u64
    }

    fn recompute_all(&self) {
        for hart in 0..self.hart_count {
            self.update_timer_irq(hart);
        }
    }

    fn update_timer_irq(&self, hart: usize) {
        let pending = self.read_mtime() >= self.mtimecmp[hart];
        if let Some(ref irq) = self.mti[hart] {
            irq.set(pending);
        }
    }
}
