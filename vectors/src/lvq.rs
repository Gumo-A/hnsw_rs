const BITS: i32 = 8;
const CHUNK_SIZE: usize = 8;

use crate::VecTrait;

#[derive(Debug, Clone)]
pub struct LVQVec {
    delta: f32,
    lower: f32,
    pub quantized_vec: Vec<u8>,
}

impl LVQVec {
    // Quantizes an already mean-centered vector
    pub fn new(vector: &Vec<f32>) -> LVQVec {
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
}

impl VecTrait for LVQVec {
    fn distance(&self, other: &impl VecTrait) -> f32 {
        let other = other.get_vals();
        let mut acc = [0.0f32; CHUNK_SIZE];
        let vector_chunks = other.chunks_exact(CHUNK_SIZE);
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

    fn dist2other(&self, other: &Self) -> f32 {
        let mut acc = [0.0f32; CHUNK_SIZE];
        let chunks_iter = self.quantized_vec.chunks_exact(CHUNK_SIZE);
        let vector_chunks = other.quantized_vec.chunks_exact(CHUNK_SIZE);
        let self_rem = chunks_iter.remainder();
        let other_rem = vector_chunks.remainder();
        for (chunkx, chunky) in chunks_iter.zip(vector_chunks) {
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
        let dist = acc.iter().sum::<f32>().sqrt();
        dist
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

#[cfg(test)]
mod tests {}
