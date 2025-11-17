#[derive(Debug, Clone)]
pub struct ByteVec {
    pub data: Vec<u8>,
}

impl ByteVec {
    pub fn new(data: Vec<u8>) -> ByteVec {
        ByteVec { data }
    }
}
