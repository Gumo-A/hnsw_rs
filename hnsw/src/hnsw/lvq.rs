use unroll::unroll_for_loops;

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

    #[unroll_for_loops]
    pub fn dist2vec(&self, vector: &Vec<f32>) -> f32 {
        let mut result: f32 = 0.0;
        for (x, y) in self.quantized_vec.iter().zip(vector) {
            let decompressed = ((*x as f32) * self.delta) + self.lower;
            result += (decompressed - y).powi(2);
        }
        result.sqrt()
    }

    #[unroll_for_loops]
    pub fn dist2other(&self, other: &Self) -> f32 {
        let mut result: f32 = 0.0;
        for (x_u8, y_u8) in self.quantized_vec.iter().zip(other.quantized_vec.iter()) {
            let x_f32 = ((*x_u8 as f32) * self.delta) + self.lower;
            let y_f32 = ((*y_u8 as f32) * other.delta) + other.lower;
            result += (x_f32 - y_f32).powi(2);
        }
        result.sqrt()
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
        let compressed = Vec::from_iter(vecs.iter().map(|x| LVQVec::new(x, 8)));

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

/// This comes directly from the ndarray crate
/// TODO: use this function to improve our current dist calculation function
/// Compute the dot product.
///
/// `xs` and `ys` must be the same length
pub fn unrolled_dot<A>(xs: &[A], ys: &[A]) -> A
where
    A: LinalgScalar,
{
    debug_assert_eq!(xs.len(), ys.len());
    // eightfold unrolled so that floating point can be vectorized
    // (even with strict floating point accuracy semantics)
    let len = cmp::min(xs.len(), ys.len());
    let mut xs = &xs[..len];
    let mut ys = &ys[..len];
    let mut sum = A::zero();
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) = (
        A::zero(),
        A::zero(),
        A::zero(),
        A::zero(),
        A::zero(),
        A::zero(),
        A::zero(),
        A::zero(),
    );
    while xs.len() >= 8 {
        p0 = p0 + xs[0] * ys[0];
        p1 = p1 + xs[1] * ys[1];
        p2 = p2 + xs[2] * ys[2];
        p3 = p3 + xs[3] * ys[3];
        p4 = p4 + xs[4] * ys[4];
        p5 = p5 + xs[5] * ys[5];
        p6 = p6 + xs[6] * ys[6];
        p7 = p7 + xs[7] * ys[7];

        xs = &xs[8..];
        ys = &ys[8..];
    }
    sum = sum + (p0 + p4);
    sum = sum + (p1 + p5);
    sum = sum + (p2 + p6);
    sum = sum + (p3 + p7);

    for (i, (&x, &y)) in xs.iter().zip(ys).enumerate() {
        if i >= 7 {
            break;
        }
        sum = sum + x * y;
    }
    sum
}
