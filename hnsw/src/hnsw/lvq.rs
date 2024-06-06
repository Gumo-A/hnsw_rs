const CHUNK_SIZE: usize = 8;

#[derive(Debug, Clone)]
pub struct LVQVec {
    delta: f32,
    lower: f32,
    quantized_vec: Vec<u8>,
}

impl LVQVec {
    // Quantizes an already mean-centered vector
    pub fn new(vector: &Vec<f32>, bits: usize) -> LVQVec {
        let upper_bound: f32 = *vector
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let lower_bound: f32 = *vector
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let delta: f32 = (upper_bound - lower_bound) / (2.0f32.powi(bits as i32) - 1.0);

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

    pub fn reconstruct(&self) -> Vec<f32> {
        let recontructed: Vec<f32> = self
            .quantized_vec
            .iter()
            .map(|x| ((*x as f32) * self.delta) + self.lower)
            .collect();
        recontructed
    }

    // Have to read this: https://www.reidatcheson.com/hpc/architecture/performance/rust/c++/2019/10/19/measure-cache.html
    pub fn dist2vec(&self, vector: &Vec<f32>) -> f32 {
        let mut acc = [0.0f32; CHUNK_SIZE];
        let chunks_iter = self
            .quantized_vec
            .chunks_exact(CHUNK_SIZE)
            .zip(vector.chunks_exact(CHUNK_SIZE));

        for (chunkx, chunky) in chunks_iter {
            let acc_iter = chunkx.iter().zip(chunky);
            for (idx, (x, y)) in acc_iter.enumerate() {
                acc[idx] += ((*x as f32) * self.delta + self.lower) * y;
            }
        }
        acc.iter().sum()
    }

    pub fn dist2other(&self, other: &Self) -> f32 {
        let mut acc = [0.0f32; CHUNK_SIZE];
        let chunks_iter = self
            .quantized_vec
            .chunks_exact(CHUNK_SIZE)
            .zip(other.quantized_vec.chunks_exact(CHUNK_SIZE));

        for (chunkx, chunky) in chunks_iter {
            let acc_iter = chunkx.iter().zip(chunky);
            for (idx, (x, y)) in acc_iter.enumerate() {
                let x_f32 = (*x as f32) * self.delta + self.lower;
                let y_f32 = (*y as f32) * other.delta + other.lower;
                acc[idx] += x_f32 * y_f32;
            }
        }
        acc.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::time::Instant;

    fn gen_rand_vecs(dim: usize, n: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        let mut vecs = vec![];
        for _ in 0..n {
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
        let compressed = Vec::from_iter(vecs.iter().map(|x| LVQVec::new(x, 8)));

        let query = (0..dim).map(|_| rng.gen::<f32>()).collect();

        for comp in compressed {
            let dist = comp.dist2vec(&query);
            println!("Compressed: {dist}");
        }
        for vector in vecs {
            let mut dist: f32 = 0.0;
            for (x, y) in vector.iter().zip(&query) {
                dist += x * y;
                // dist += (x - y).powi(2);
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
        let n = 40_000;
        println!("Generating random vectors");
        let vecs = gen_rand_vecs(dim, n);
        println!("Compressing vectors");
        let compressed = Vec::from_iter(vecs.iter().map(|x| LVQVec::new(x, 8)));
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
        let test_vec = (0..10).map(|_| rng.gen::<f32>()).collect();
        let quantized = LVQVec::new(&test_vec, 8);
        // let q = &quantized.quantized_vec;
        let recontructed = quantized.reconstruct();

        println!("Original: {:?}", test_vec);
        println!("Quantized: {:?}", quantized.quantized_vec);
        println!("Reconstructed: {:?}", recontructed);
    }
}