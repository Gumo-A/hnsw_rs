const BITS: i32 = 8;
const CHUNK_SIZE: usize = 8;

use std::slice::ChunksExact;

use crate::{VecBase, VecTrait, serializer::Serializer};

#[derive(Debug, Clone)]
pub struct LVQVec {
    pub delta: f32,
    pub lower: f32,
    pub quantized_vec: Vec<u8>,
}

impl LVQVec {
    // Have to read this: https://www.reidatcheson.com/hpc/architecture/performance/rust/c++/2019/10/19/measure-cache.html
    pub fn dist2vec(&self, vector: &Vec<f32>) -> f32 {
        let mut acc = [0.0f32; CHUNK_SIZE];
        let vector_chunks = vector.chunks_exact(CHUNK_SIZE);
        let chunks_iter = self.quantized_vec.chunks_exact(CHUNK_SIZE);
        let self_rem = chunks_iter.remainder();
        let other_rem = vector_chunks.remainder();

        for (chunkx, chunky) in chunks_iter.zip(vector_chunks) {
            let acc_iter = chunkx.iter().zip(chunky);
            for (idx, (x, y)) in acc_iter.enumerate() {
                acc[idx] += (((*x as f32) * self.delta + self.lower) - y).powi(2);
            }
        }
        for (x, y) in self_rem.iter().zip(other_rem) {
            acc[0] += (((*x as f32) * self.delta + self.lower) - y).powi(2);
        }
        acc.iter().sum::<f32>().sqrt()
    }

    fn distance_unrolled_full(&self, other_chunks: ChunksExact<f32>) -> f32 {
        let self_chunks = self.quantized_vec.chunks_exact(CHUNK_SIZE);
        let mut acc = [0.0f32; CHUNK_SIZE];

        let self_rem = self_chunks.remainder();
        let other_rem = other_chunks.remainder();

        for (chunkx, chunky) in self_chunks.zip(other_chunks) {
            let acc_iter = chunkx.iter().zip(chunky);
            for (idx, (x, y)) in acc_iter.enumerate() {
                acc[idx] += (((*x as f32) * self.delta + self.lower) - y).powi(2);
            }
        }
        for (x, y) in self_rem.iter().zip(other_rem) {
            acc[0] += (((*x as f32) * self.delta + self.lower) - y).powi(2);
        }
        acc.iter().sum::<f32>().sqrt()
    }

    fn distance_unrolled_quant(&self, other: &LVQVec) -> f32 {
        let mut acc = [0.0f32; CHUNK_SIZE];

        let self_chunks = self.quantized_vec.chunks_exact(CHUNK_SIZE);
        let other_chunks = other.quantized_vec.chunks_exact(CHUNK_SIZE);

        let self_rem = self_chunks.remainder();
        let other_rem = other_chunks.remainder();

        for (chunkx, chunky) in self_chunks.zip(other_chunks) {
            let acc_iter = chunkx.iter().zip(chunky);
            for (idx, (x, y)) in acc_iter.enumerate() {
                let x_f32 = ((*x as f32) * self.delta) + self.lower;
                let y_f32 = ((*y as f32) * other.delta) + other.lower;
                acc[idx] += (x_f32 - y_f32).powi(2);
            }
        }
        for (x, y) in self_rem.iter().zip(other_rem) {
            let x_f32 = (*x as f32) * self.delta + self.lower;
            let y_f32 = (*y as f32) * other.delta + other.lower;
            acc[0] += (x_f32 - y_f32).powi(2);
        }
        acc.iter().sum::<f32>().sqrt()
    }
}

impl VecBase for LVQVec {
    fn new(vector: &Vec<f32>) -> LVQVec {
        let upper_bound: f32 = *vector
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let lower_bound: f32 = *vector
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let delta: f32 = (upper_bound - lower_bound) / (2.0f32.powi(BITS) - 1.0);

        let quantized: Vec<u8> = vector
            .iter()
            .map(|x| {
                let mut buffer: f32 = (x - lower_bound) / delta;
                buffer += 0.5f32;
                buffer.floor() as u8
            })
            .collect();

        LVQVec {
            delta,
            lower: lower_bound,
            quantized_vec: quantized,
        }
    }
    fn distance(&self, other: &impl VecBase) -> f32 {
        self.distance_unrolled_full(other.get_vals().chunks_exact(CHUNK_SIZE))
    }

    fn dist2other(&self, other: &Self) -> f32 {
        self.distance_unrolled_quant(other)
    }

    fn iter_vals(&self) -> impl Iterator<Item = f32> {
        self.quantized_vec
            .iter()
            .map(|x| ((*x as f32) * self.delta) + self.lower)
    }

    fn dim(&self) -> usize {
        self.quantized_vec.len()
    }

    fn quantize(&self) -> LVQVec {
        self.clone()
    }

    fn center(&mut self, means: &Vec<f32>) {
        if self.dim() != means.len() {
            panic!("Vector dimensions are not equal")
        }
        let centered_full = self
            .iter_vals()
            .enumerate()
            .map(|(idx, x)| x - means[idx])
            .collect();
        *self = LVQVec::new(&centered_full);
    }

    fn decenter(&mut self, means: &Vec<f32>) {
        if self.dim() != means.len() {
            panic!("Vector dimensions are not equal")
        }
        let decentered_full = self
            .iter_vals()
            .enumerate()
            .map(|(idx, x)| x + means[idx])
            .collect();
        *self = LVQVec::new(&decentered_full);
    }
}

impl Serializer for LVQVec {
    fn size(&self) -> usize {
        8 + self.dim()
    }

    /// 4 for low, 4 for delta
    /// followed by dim bytes
    ///
    /// Val       Bytes
    /// low       4
    /// delta     4
    /// quant_vec dim
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.lower.to_be_bytes());
        bytes.extend_from_slice(&self.delta.to_be_bytes());
        for byte in self.quantized_vec.iter() {
            bytes.push(*byte);
        }
        bytes
    }

    fn deserialize(data: Vec<u8>) -> Self {
        let lower = f32::from_be_bytes(data[..4].try_into().unwrap());
        let delta = f32::from_be_bytes(data[4..8].try_into().unwrap());
        let mut quantized_vec: Vec<u8> = Vec::new();
        for bytes_arr in data.iter().skip(8) {
            quantized_vec.push(*bytes_arr);
        }
        LVQVec {
            delta,
            lower,
            quantized_vec,
        }
    }
}

impl VecTrait for LVQVec {}

#[cfg(test)]
mod tests {
    use crate::gen_rand_vecs;

    use super::*;

    #[test]
    fn serialization() {
        let a = LVQVec::new(&gen_rand_vecs(128, 1)[0].clone());
        let ser_a = a.serialize();
        let b = LVQVec::deserialize(ser_a);
        for (a_val, b_val) in a.iter_vals().zip(b.iter_vals()) {
            assert_eq!(a_val, b_val);
        }
    }

    #[test]
    fn distance() {
        let mut others = Vec::new();
        for _ in 0..100 {
            let a = LVQVec::new(&gen_rand_vecs(128, 1)[0].clone());
            others.push(a);
        }
        let a = LVQVec::new(&gen_rand_vecs(128, 1)[0].clone());
        a.dist2many(others.iter())
            .for_each(|dist| assert!(dist >= 0.0));

        let a = LVQVec::new(&vec![0.5]);
        let b = LVQVec::new(&vec![0.25]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert_eq!(dist, 0.25);
        assert_eq!(dist, dist2other);

        let a = LVQVec::new(&vec![0.75]);
        let b = LVQVec::new(&vec![0.25]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert_eq!(dist, 0.5);
        assert_eq!(dist, dist2other);

        let a = LVQVec::new(&vec![0.0, 0.0]);
        let b = LVQVec::new(&vec![0.0, 1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert_eq!(dist, 1.0);
        assert_eq!(dist, dist2other);

        let a = LVQVec::new(&vec![1.0, 0.0]);
        let b = LVQVec::new(&vec![0.0, 1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert_eq!(dist, 2.0f32.sqrt());
        assert_eq!(dist, dist2other);

        let a = LVQVec::new(&vec![-1.0, 0.0]);
        let b = LVQVec::new(&vec![0.0, 1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert_eq!(dist, 2.0f32.sqrt());
        assert_eq!(dist, dist2other);

        let a = LVQVec::new(&vec![1.0, 0.0]);
        let b = LVQVec::new(&vec![0.0, -1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert_eq!(dist, 2.0f32.sqrt());
        assert_eq!(dist, dist2other);

        let a = LVQVec::new(&gen_rand_vecs(128, 1)[0].clone());
        let b = a.clone();
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert_eq!(dist, 0.0);
        assert_eq!(dist, dist2other);
    }

    #[test]
    fn center_decenter() {
        let n = 128;
        let means = gen_rand_vecs(n, 1)[0].clone();
        let mut vectors: Vec<LVQVec> = gen_rand_vecs(n, 4).iter().map(|v| LVQVec::new(v)).collect();
        let vectors_clone = vectors.clone();
        for (v, vc) in vectors.iter_mut().zip(vectors_clone.iter()) {
            v.center(&means);
            for (idx, (v_val, vc_val)) in v.iter_vals().zip(vc.iter_vals()).enumerate() {
                let diff = (v_val - (vc_val - means[idx])).abs();
                println!("{diff}");
                assert!(diff < 0.01);
            }
        }

        for (v, vc) in vectors.iter_mut().zip(vectors_clone.iter()) {
            v.decenter(&means);
            for (v_val, vc_val) in v.iter_vals().zip(vc.iter_vals()) {
                let diff = (v_val - vc_val).abs();
                println!("{diff}");
                assert!(diff < 0.01);
            }
        }
    }
}
