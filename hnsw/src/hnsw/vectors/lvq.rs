const CHUNK_SIZE: usize = 8;
const BITS: i32 = 8;

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

    pub fn iter_full(&self) -> impl Iterator<Item = f32> + use<'_> {
        self.quantized_vec
            .iter()
            .map(|x| ((*x as f32) * self.delta) + self.lower)
    }
}

#[cfg(test)]
mod tests {

    use crate::hnsw::vectors::FullVec;

    use super::*;
    use rand::Rng;
    use std::time::Instant;

    fn gen_rand_vecs(dim: usize, n: usize) -> Vec<FullVec> {
        let mut rng = rand::thread_rng();
        let mut vecs = vec![];
        for _ in 0..n {
            vecs.push(FullVec::new((0..dim).map(|_| rng.gen::<f32>()).collect()))
        }
        vecs
    }

    /// Ment to be run with --nocapture
    /// This is to show that distances are almost the same
    #[test]
    fn compressed_distance() {
        let mut rng = rand::thread_rng();
        let dim = 100;

        let vecs = gen_rand_vecs(dim, 10);
        let compressed = Vec::from_iter(vecs.iter().map(|x| x.quantize()));

        let query = FullVec::new((0..dim).map(|_| rng.gen::<f32>()).collect());

        for comp in compressed {
            let dist = comp.distance(&query);
            println!("Compressed: {dist}");
        }
        for vector in vecs {
            let dist = vector.distance(&query);
            println!("Full precision: {dist}");
        }
    }

    /// Ment to be run with --nocapture
    /// This is to show the performance gain
    #[test]
    fn compressed_distance_bench() {
        let mut rng = rand::thread_rng();
        let dim = 1000;
        let n = 4000;
        println!("Generating random vectors");
        let vecs = gen_rand_vecs(dim, n);
        println!("Compressing vectors");
        let compressed = Vec::from_iter(vecs.iter().map(|x| x.quantize()));
        let query = FullVec::new((0..dim).map(|_| rng.gen::<f32>()).collect());

        let start = Instant::now();
        for comp in compressed.iter() {
            comp.distance(&query);
        }
        let end = Instant::now();
        println!("Time with compressed: {0}", (end - start).as_secs_f32());

        let start = Instant::now();
        for vector in vecs.iter() {
            vector.distance(&query);
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
        let quantized = LVQVec::new(&test_vec);
        // let q = &quantized.quantized_vec;
        let recontructed = quantized.get_vals();

        println!("Original: {:?}", test_vec);
        println!("Quantized: {:?}", quantized.quantized_vec);
        println!("Reconstructed: {:?}", recontructed);
    }
}
