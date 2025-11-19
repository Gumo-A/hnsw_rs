const BITS: i32 = 8;

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

impl VecTrait for LVQVec {
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
mod tests {

    use rand::Rng;
    use std::time::Instant;

    use super::LVQVec;
    use crate::{FullVec, VecTrait};

    fn gen_rand_vecs(dim: usize, n: usize) -> Vec<LVQVec> {
        let mut rng = rand::thread_rng();
        let mut vecs = vec![];
        for _ in 0..n {
            vecs.push(LVQVec::new(&(0..dim).map(|_| rng.r#gen::<f32>()).collect()))
        }
        vecs
    }
}
