pub trait Serializer {
    /// The number of bytes in the serialized string.
    fn size(&self) -> usize;
    fn serialize(&self) -> Vec<u8>;
    fn deserialize(data: Vec<u8>) -> Self;
}

pub struct BytesReader {
    bytes: Vec<u8>,
    pointer: usize,
}

impl BytesReader {
    pub fn new(bytes: Vec<u8>) -> Self {
        BytesReader { bytes, pointer: 0 }
    }
    /// Yields the number of bytes left
    /// starting from the pointer
    pub fn remaining(&self) -> usize {
        self.bytes[self.pointer..].len()
    }
    pub fn iter_remaining(&self) -> impl Iterator<Item = &u8> {
        self.bytes[self.pointer..].iter()
    }
    /// reads exactly one byte from the buffer
    /// and increases the pointer by one
    pub fn u8(&mut self) -> [u8; 1] {
        self.pointer += 1;
        self.bytes[(self.pointer - 1)..self.pointer]
            .try_into()
            .unwrap()
    }
    /// reads exactly two bytes from the buffer
    /// and increases the pointer by two
    pub fn u16(&mut self) -> [u8; 2] {
        self.pointer += 2;
        self.bytes[(self.pointer - 2)..self.pointer]
            .try_into()
            .unwrap()
    }
    /// reads exactly four bytes from the buffer
    /// and increases the pointer by four
    pub fn u32(&mut self) -> [u8; 4] {
        self.pointer += 4;
        self.bytes[(self.pointer - 4)..self.pointer]
            .try_into()
            .unwrap()
    }
    /// reads exactly eight bytes from the buffer
    /// and increases the pointer by eight
    pub fn u64(&mut self) -> [u8; 8] {
        self.pointer += 8;
        self.bytes[(self.pointer - 8)..self.pointer]
            .try_into()
            .unwrap()
    }
    /// reads `n` bytes from the buffer
    /// and increases the pointer by `n`
    pub fn read_bytes(&mut self, n: usize) -> &[u8] {
        self.pointer += n;
        &self.bytes[(self.pointer - n)..self.pointer]
    }
}
