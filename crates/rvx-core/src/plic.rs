pub struct Plic {
    num_sources: u32,
    priority: Vec<u32>,
    level: Vec<bool>,
    pending: Vec<bool>,
    in_service: Vec<bool>,
    enable: Vec<Vec<u32>>,
    threshold: Vec<u32>,
}

impl Plic {
    pub fn new(contexts: usize, num_sources: u32) -> Self {
        let words = source_words(num_sources);
        Self {
            num_sources,
            priority: vec![0; num_sources as usize + 1],
            level: vec![false; num_sources as usize + 1],
            pending: vec![false; num_sources as usize + 1],
            in_service: vec![false; num_sources as usize + 1],
            enable: vec![vec![0; words]; contexts],
            threshold: vec![0; contexts],
        }
    }

    pub fn context_pending(&self, context: usize) -> bool {
        self.best_source(context).is_some()
    }

    pub fn set_irq(&mut self, irq: u32, level: bool) {
        if !self.valid_source(irq) {
            return;
        }
        let irq = irq as usize;
        self.level[irq] = level;
        self.pending[irq] = level && !self.in_service[irq];
    }

    pub fn read(&mut self, offset: u64, _size: u8) -> u64 {
        match offset {
            offset if offset < 0x1000 => {
                let irq = (offset / 4) as usize;
                self.priority.get(irq).copied().unwrap_or(0) as u64
            }
            offset if (0x1000..0x2000).contains(&offset) => {
                let word = ((offset - 0x1000) / 4) as usize;
                self.pending_word(word) as u64
            }
            offset if (0x2000..0x20_0000).contains(&offset) => {
                let context = ((offset - 0x2000) / 0x80) as usize;
                let word = (((offset - 0x2000) % 0x80) / 4) as usize;
                self.enable
                    .get(context)
                    .and_then(|words| words.get(word))
                    .copied()
                    .unwrap_or(0) as u64
            }
            offset if offset >= 0x20_0000 => {
                let context = ((offset - 0x20_0000) / 0x1000) as usize;
                match (offset - 0x20_0000) % 0x1000 {
                    0 => self.threshold.get(context).copied().unwrap_or(0) as u64,
                    4 => self.claim(context) as u64,
                    _ => 0,
                }
            }
            _ => 0,
        }
    }

    pub fn write(&mut self, offset: u64, _size: u8, value: u64) {
        match offset {
            offset if offset < 0x1000 => {
                let irq = (offset / 4) as usize;
                if irq != 0 && irq <= self.num_sources as usize {
                    self.priority[irq] = value as u32;
                }
            }
            offset if (0x1000..0x2000).contains(&offset) => {}
            offset if (0x2000..0x20_0000).contains(&offset) => {
                let context = ((offset - 0x2000) / 0x80) as usize;
                let word = (((offset - 0x2000) % 0x80) / 4) as usize;
                if let Some(words) = self.enable.get_mut(context)
                    && let Some(slot) = words.get_mut(word)
                {
                    *slot = value as u32;
                    if word == 0 {
                        *slot &= !1;
                    }
                }
            }
            offset if offset >= 0x20_0000 => {
                let context = ((offset - 0x20_0000) / 0x1000) as usize;
                match (offset - 0x20_0000) % 0x1000 {
                    0 => {
                        if let Some(threshold) = self.threshold.get_mut(context) {
                            *threshold = value as u32;
                        }
                    }
                    4 => self.complete(context, value as u32),
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn valid_source(&self, irq: u32) -> bool {
        irq != 0 && irq <= self.num_sources
    }

    fn pending_word(&self, word: usize) -> u32 {
        let mut bits = 0u32;
        for bit in 0..32usize {
            let irq = word * 32 + bit;
            if irq == 0 || irq > self.num_sources as usize {
                continue;
            }
            if self.pending[irq] {
                bits |= 1 << bit;
            }
        }
        bits
    }

    fn claim(&mut self, context: usize) -> u32 {
        let Some(irq) = self.best_source(context) else {
            return 0;
        };
        self.pending[irq] = false;
        self.in_service[irq] = true;
        irq as u32
    }

    fn complete(&mut self, _context: usize, irq: u32) {
        if !self.valid_source(irq) {
            return;
        }
        let irq = irq as usize;
        self.in_service[irq] = false;
        self.pending[irq] = self.level[irq];
    }

    fn best_source(&self, context: usize) -> Option<usize> {
        let enable = self.enable.get(context)?;
        let threshold = self.threshold[context];
        let mut best_irq = None;
        let mut best_priority = 0;
        for irq in 1..=self.num_sources as usize {
            if !self.pending[irq] {
                continue;
            }
            let word = irq / 32;
            let bit = irq % 32;
            if word >= enable.len() || ((enable[word] >> bit) & 1) == 0 {
                continue;
            }
            let priority = self.priority[irq];
            if priority == 0 || priority <= threshold {
                continue;
            }
            if best_irq.is_none() || priority > best_priority {
                best_irq = Some(irq);
                best_priority = priority;
            }
        }
        best_irq
    }
}

fn source_words(num_sources: u32) -> usize {
    (num_sources as usize + 32) / 32
}
