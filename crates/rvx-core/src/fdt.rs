const FDT_BEGIN_NODE: u32 = 1;
const FDT_END_NODE: u32 = 2;
const FDT_PROP: u32 = 3;
const FDT_END: u32 = 9;

const FDT_MAGIC: u32 = 0xd00d_feed;
const FDT_VERSION: u32 = 17;
const FDT_LAST_COMP_VERSION: u32 = 16;

pub struct FdtBuilder {
    strings: Vec<u8>,
    structure: Vec<u8>,
}

impl FdtBuilder {
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            structure: Vec::new(),
        }
    }

    pub fn begin_node(&mut self, name: &str) {
        self.push_u32(FDT_BEGIN_NODE);
        self.structure.extend_from_slice(name.as_bytes());
        self.structure.push(0);
        self.align_structure();
    }

    pub fn end_node(&mut self) {
        self.push_u32(FDT_END_NODE);
    }

    pub fn property_bytes(&mut self, name: &str, value: &[u8]) {
        let nameoff = self.string_offset(name);
        self.push_u32(FDT_PROP);
        self.push_u32(value.len() as u32);
        self.push_u32(nameoff);
        self.structure.extend_from_slice(value);
        self.align_structure();
    }

    pub fn property_string(&mut self, name: &str, value: &str) {
        let mut bytes = Vec::with_capacity(value.len() + 1);
        bytes.extend_from_slice(value.as_bytes());
        bytes.push(0);
        self.property_bytes(name, &bytes);
    }

    pub fn property_u32(&mut self, name: &str, value: u32) {
        self.property_bytes(name, &value.to_be_bytes());
    }

    pub fn property_u64(&mut self, name: &str, value: u64) {
        self.property_bytes(name, &value.to_be_bytes());
    }

    pub fn property_u32_list(&mut self, name: &str, values: &[u32]) {
        let mut out = Vec::with_capacity(values.len() * 4);
        for value in values {
            out.extend_from_slice(&value.to_be_bytes());
        }
        self.property_bytes(name, &out);
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.push_u32(FDT_END);
        let header_size = 40u32;
        let mem_rsvmap_size = 16u32;
        let off_dt_struct = header_size + mem_rsvmap_size;
        let off_dt_strings = off_dt_struct + self.structure.len() as u32;
        let totalsize = off_dt_strings + self.strings.len() as u32;

        let mut out = Vec::with_capacity(totalsize as usize);
        out.extend_from_slice(&FDT_MAGIC.to_be_bytes());
        out.extend_from_slice(&totalsize.to_be_bytes());
        out.extend_from_slice(&off_dt_struct.to_be_bytes());
        out.extend_from_slice(&off_dt_strings.to_be_bytes());
        out.extend_from_slice(&(header_size).to_be_bytes());
        out.extend_from_slice(&FDT_VERSION.to_be_bytes());
        out.extend_from_slice(&FDT_LAST_COMP_VERSION.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(&(self.strings.len() as u32).to_be_bytes());
        out.extend_from_slice(&(self.structure.len() as u32).to_be_bytes());
        out.extend_from_slice(&0u64.to_be_bytes());
        out.extend_from_slice(&0u64.to_be_bytes());
        out.extend_from_slice(&self.structure);
        out.extend_from_slice(&self.strings);
        out
    }

    fn string_offset(&mut self, name: &str) -> u32 {
        let mut offset = 0usize;
        while offset < self.strings.len() {
            let end = self.strings[offset..]
                .iter()
                .position(|byte| *byte == 0)
                .map(|pos| offset + pos)
                .unwrap();
            if &self.strings[offset..end] == name.as_bytes() {
                return offset as u32;
            }
            offset = end + 1;
        }
        let offset = self.strings.len() as u32;
        self.strings.extend_from_slice(name.as_bytes());
        self.strings.push(0);
        offset
    }

    fn push_u32(&mut self, value: u32) {
        self.structure.extend_from_slice(&value.to_be_bytes());
    }

    fn align_structure(&mut self) {
        while self.structure.len() % 4 != 0 {
            self.structure.push(0);
        }
    }
}

impl Default for FdtBuilder {
    fn default() -> Self {
        Self::new()
    }
}
