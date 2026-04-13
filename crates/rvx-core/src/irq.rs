use std::sync::Arc;

pub trait IrqSink: Send + Sync {
    fn set_irq(&self, irq: u32, level: bool);
}

#[derive(Clone)]
pub struct IrqLine {
    sink: Arc<dyn IrqSink>,
    irq: u32,
}

impl IrqLine {
    pub fn new(sink: Arc<dyn IrqSink>, irq: u32) -> Self {
        Self { sink, irq }
    }

    pub fn set(&self, level: bool) {
        self.sink.set_irq(self.irq, level);
    }
}
