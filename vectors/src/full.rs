use crate::{VecSer, VecTrait, serializer::Serializer};

const CHUNK_SIZE: usize = 8;

#[derive(Debug, Clone)]
pub struct FullVec {
    pub vals: Vec<f32>,
}

impl FullVec {
    pub fn new(vals: Vec<f32>) -> Self {
        FullVec { vals }
    }
    pub fn iter_vals_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.vals.iter_mut()
    }
    pub fn iter_vals(&mut self) -> impl Iterator<Item = &f32> {
        self.vals.iter()
    }
}

impl VecTrait for FullVec {
    fn distance(&self, other: &impl VecTrait) -> f32 {
        let other = other.get_vals();
        let mut acc = [0.0f32; CHUNK_SIZE];
        let vector_chunks = other.chunks_exact(CHUNK_SIZE);
        let chunks_iter = self.vals.chunks_exact(CHUNK_SIZE);
        let self_rem = chunks_iter.remainder();
        let other_rem = vector_chunks.remainder();

        for (chunkx, chunky) in chunks_iter.zip(vector_chunks) {
            let acc_iter = chunkx.iter().zip(chunky);
            for (idx, (x, y)) in acc_iter.enumerate() {
                acc[idx] += (x - y).powi(2);
            }
        }
        for (x, y) in self_rem.iter().zip(other_rem) {
            acc[0] += (x - y).powi(2);
        }
        acc.iter().sum::<f32>().sqrt()
    }

    fn dist2other(&self, other: &Self) -> f32 {
        let mut acc = [0.0f32; CHUNK_SIZE];
        let chunks_iter = self.vals.chunks_exact(CHUNK_SIZE);
        let vector_chunks = other.vals.chunks_exact(CHUNK_SIZE);
        let self_rem = chunks_iter.remainder();
        let other_rem = vector_chunks.remainder();
        for (chunkx, chunky) in chunks_iter.zip(vector_chunks) {
            let acc_iter = chunkx.iter().zip(chunky);
            for (idx, (x, y)) in acc_iter.enumerate() {
                acc[idx] += (x - y).powi(2);
            }
        }
        for (x, y) in self_rem.iter().zip(other_rem) {
            acc[0] += (x - y).powi(2);
        }
        let dist = acc.iter().sum::<f32>().sqrt();
        dist
    }

    fn iter_vals(&self) -> impl Iterator<Item = f32> {
        self.vals.iter().copied()
    }
    fn dim(&self) -> usize {
        self.vals.len()
    }

    fn center(&mut self, means: &Vec<f32>) {
        if self.dim() != means.len() {
            panic!("Vector dimensions are not equal")
        }
        self.iter_vals_mut()
            .enumerate()
            .for_each(|(idx, x)| *x -= means[idx]);
    }

    fn decenter(&mut self, means: &Vec<f32>) {
        if self.dim() != means.len() {
            panic!("Vector dimensions are not equal")
        }
        self.iter_vals_mut()
            .enumerate()
            .for_each(|(idx, x)| *x += means[idx]);
    }
}

impl Serializer for FullVec {
    fn size(&self) -> usize {
        self.dim() * 4
    }

    /// blocks of 4 bytes
    /// for the floats (big endian)
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for float in self.vals.iter() {
            bytes.extend_from_slice(&float.to_be_bytes());
        }
        bytes
    }

    fn deserialize(data: Vec<u8>) -> Self {
        let bytes = data.chunks_exact(4);
        let mut vals: Vec<f32> = Vec::new();
        for bytes_arr in bytes {
            vals.push(f32::from_be_bytes(bytes_arr.try_into().unwrap()))
        }
        FullVec::new(vals)
    }
}

impl VecSer for FullVec {}
