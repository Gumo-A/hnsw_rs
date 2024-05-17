/// TODO:
/// Implement an F8 type for this module.
use crate::helpers::minifloat::F8;

#[derive(Debug, Clone)]
pub struct CompressedVec {
    upper: f32,
    lower: f32,
    vec: Vec<u8>,
    delta: f32,
}

impl CompressedVec {
    pub fn new(vector: &Vec<f32>) -> Self {
        let upper_bound: f32 = *vector
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let lower_bound: f32 = *vector
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let delta = (upper_bound - lower_bound) / 255.0;

        let scaled: Vec<u8> = vector
            .iter()
            .map(|x| (255.0 * (x - lower_bound) / (upper_bound - lower_bound)) as u8)
            .collect();
        CompressedVec {
            upper: upper_bound,
            lower: lower_bound,
            vec: scaled,
            delta,
        }
    }
    pub fn dist2vec(&self, vector: &Vec<f32>) -> f32 {
        let mut result: f32 = 0.0;
        for (x, y) in self.vec.iter().zip(vector) {
            let decompressed = ((*x as f32) * self.delta) + self.lower;
            result += (decompressed - y).powi(2);
        }
        result
    }
}

// Only usable with bits = 8 for now
#[derive(Debug)]
pub struct LVQVec {
    upper: f32,
    lower: f32,
    quantized_vec: Vec<u8>,
}

impl LVQVec {
    pub fn reconstruct(&self) -> Vec<f32> {
        let recontructed: Vec<f32> = self
            .quantized_vec
            .iter()
            .map(|x| self.lower + ((*x as f32) * (self.upper - self.lower) / 255.0f32))
            .collect();
        recontructed
    }
}

// Scalar quantization as defined in the paper
pub fn Q(vector: &Vec<f32>, bits: usize) -> LVQVec {
    let upper_bound: f32 = *vector
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let lower_bound: f32 = *vector
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let delta: f32 = (upper_bound - lower_bound) / (2.0f32.powi(bits as i32) - 1.0);

    let quantized_inter: Vec<f32> = vector
        .iter()
        .map(|x| {
            let mut buffer: f32 = (x - lower_bound) / delta;
            buffer += 0.5f32;
            buffer = buffer.floor() * delta;
            buffer += lower_bound;
            buffer
        })
        .collect();

    let quantized: Vec<u8> = quantized_inter
        .iter()
        .map(|x| (255.0 * (x - lower_bound) / (upper_bound - lower_bound)) as u8)
        .collect();

    LVQVec {
        upper: upper_bound,
        lower: lower_bound,
        quantized_vec: quantized,
    }
}

pub fn minmax_scaler(vector: &Vec<f32>) -> Vec<u8> {
    let upper_bound: f32 = *vector
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let lower_bound: f32 = *vector
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let quantized: Vec<u8> = vector
        .iter()
        .map(|x| (255.0 * (x - lower_bound) / (upper_bound - lower_bound)) as u8)
        .collect();
    quantized
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::time::Instant;

    fn gen_rand_vecs(dim: usize, n: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        let mut vecs = vec![];
        for i in 0..n {
            vecs.push((0..dim).map(|_| rng.gen::<f32>()).collect())
        }
        vecs
    }

    /// Ment to be run with --nocapture
    /// This is to show that distances are
    /// almost the same
    #[test]
    fn compressed_distance() {
        let mut rng = rand::thread_rng();
        let dim = 100;
        let vecs = gen_rand_vecs(dim, 10);
        let compressed = Vec::from_iter(vecs.iter().map(|x| CompressedVec::new(x)));
        let query = (0..dim).map(|_| rng.gen::<f32>()).collect();
        for comp in compressed {
            let dist = comp.dist2vec(&query);
            println!("Compressed: {dist}");
        }
        for vector in vecs {
            let mut dist: f32 = 0.0;
            for (x, y) in vector.iter().zip(&query) {
                dist += (x - y).powi(2);
            }
            println!("Full precision: {dist}");
        }
    }

    /// Ment to be run with --nocapture
    /// This is to show the performance gain
    #[test]
    fn compressed_distance_bench() {
        let mut rng = rand::thread_rng();
        let dim = 1000;
        let n = 400_000;
        println!("Generating random vectors");
        let vecs = gen_rand_vecs(dim, n);
        println!("Compressing vectors");
        let compressed = Vec::from_iter(vecs.iter().map(|x| CompressedVec::new(x)));
        let query = (0..dim).map(|_| rng.gen::<f32>()).collect();

        let start = Instant::now();
        for comp in compressed.iter() {
            let _dist = comp.dist2vec(&query);
        }
        let end = Instant::now();
        println!("Time with compressed: {0}", (end - start).as_secs_f32());

        let start = Instant::now();
        for vector in vecs.iter() {
            let mut _dist: f32 = 0.0;
            for (x, y) in vector.iter().zip(&query) {
                _dist += (x - y).powi(2);
            }
        }
        let end = Instant::now();
        println!("Time with full precision: {0}", (end - start).as_secs_f32());

        println!(
            "Memory footprint of full precision (MB) {0}",
            dim * n * 32 / 8 / 1024
        );
        println!("Memory footprint of compressed (MB) {0}", dim * n / 1024);
    }

    #[test]
    fn quantization() {
        let mut rng = rand::thread_rng();
        let test_vec = (0..100).map(|_| rng.gen::<f32>()).collect();
        let quantized = Q(&test_vec, 8);
        let q = &quantized.quantized_vec;
        let recontructed = quantized.reconstruct();
        let minmaxscale = minmax_scaler(&test_vec);

        println!("{:?}", test_vec);
        println!("{:?}", quantized.quantized_vec);
        println!("{:?}", minmaxscale);
        println!("{:?}", recontructed);
    }
}
