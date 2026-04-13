use std::collections::VecDeque;
use std::io::{self, Write};

use crate::IrqLine;

pub trait UartInput: Send {
    fn read_byte(&mut self) -> Option<u8>;
}

pub struct Uart16550 {
    dll: u8,
    dlm: u8,
    ier: u8,
    lcr: u8,
    mcr: u8,
    lsr: u8,
    msr: u8,
    scr: u8,
    rx_fifo: VecDeque<u8>,
    tx_buf: Vec<u8>,
    irq: Option<IrqLine>,
}

impl Uart16550 {
    const TX_FLUSH_BYTES: usize = 256;

    pub fn new() -> Self {
        Self {
            dll: 0,
            dlm: 0,
            ier: 0,
            lcr: 0,
            mcr: 0,
            lsr: 0x60,
            msr: 0xb0,
            scr: 0,
            rx_fifo: VecDeque::new(),
            tx_buf: Vec::with_capacity(Self::TX_FLUSH_BYTES),
            irq: None,
        }
    }

    pub fn attach_irq(&mut self, irq: IrqLine) {
        self.irq = Some(irq);
    }

    pub fn receive(&mut self, byte: u8) {
        self.rx_fifo.push_back(byte);
        self.lsr |= 0x01;
        self.update_irq();
    }

    pub fn read(&mut self, offset: u64) -> u8 {
        match offset & 0x7 {
            0 if self.dlab() => self.dll,
            1 if self.dlab() => self.dlm,
            0 => {
                let byte = self.rx_fifo.pop_front().unwrap_or(0);
                if self.rx_fifo.is_empty() {
                    self.lsr &= !0x01;
                }
                self.update_irq();
                byte
            }
            1 => self.ier,
            2 => self.interrupt_id(),
            3 => self.lcr,
            4 => self.mcr,
            5 => self.lsr,
            6 => self.msr,
            7 => self.scr,
            _ => 0,
        }
    }

    pub fn write(&mut self, offset: u64, value: u8) {
        match offset & 0x7 {
            0 if self.dlab() => self.dll = value,
            1 if self.dlab() => self.dlm = value,
            0 => {
                self.lsr |= 0x20 | 0x40;
                self.tx_buf.push(value);
                if matches!(value, b'\n' | b'\r') || self.tx_buf.len() >= Self::TX_FLUSH_BYTES {
                    self.flush_output();
                }
            }
            1 => self.ier = value & 0x0f,
            2 => {}
            3 => self.lcr = value,
            4 => self.mcr = value,
            7 => self.scr = value,
            _ => {}
        }
        self.update_irq();
    }

    pub fn flush_output(&mut self) {
        if self.tx_buf.is_empty() {
            return;
        }

        let mut stdout = io::stdout().lock();
        let _ = stdout.write_all(&self.tx_buf);
        let _ = stdout.flush();
        self.tx_buf.clear();
    }

    fn dlab(&self) -> bool {
        (self.lcr & 0x80) != 0
    }

    fn interrupt_id(&self) -> u8 {
        if self.rx_interrupt_pending() {
            0x04
        } else if self.tx_interrupt_pending() {
            0x02
        } else {
            0x01
        }
    }

    fn update_irq(&self) {
        if let Some(ref irq) = self.irq {
            irq.set(self.rx_interrupt_pending() || self.tx_interrupt_pending());
        }
    }

    fn rx_interrupt_pending(&self) -> bool {
        (self.ier & 0x01) != 0 && !self.rx_fifo.is_empty()
    }

    fn tx_interrupt_pending(&self) -> bool {
        (self.ier & 0x02) != 0 && (self.lsr & 0x20) != 0
    }
}

impl Default for Uart16550 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use crate::IrqSink;

    #[derive(Default)]
    struct TestIrqSink {
        level: Mutex<bool>,
    }

    impl TestIrqSink {
        fn level(&self) -> bool {
            *self.level.lock().unwrap()
        }
    }

    impl IrqSink for TestIrqSink {
        fn set_irq(&self, _irq: u32, level: bool) {
            *self.level.lock().unwrap() = level;
        }
    }

    #[test]
    fn enables_thre_interrupt_when_tx_empty_interrupt_is_unmasked() {
        let sink = Arc::new(TestIrqSink::default());
        let mut uart = Uart16550::new();
        uart.attach_irq(IrqLine::new(sink.clone(), 10));

        uart.write(1, 0x02);

        assert!(sink.level());
        assert_eq!(uart.read(2), 0x02);
    }
}
