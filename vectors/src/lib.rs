mod full;
mod quant;
pub use full::FullVec;
pub use quant::QuantVec;
use rand::Rng;

pub trait VecBase {
    fn new(vector: &Vec<f32>) -> Self;
    fn dim(&self) -> usize;
    fn iter_vals(&self) -> impl Iterator<Item = f32>;

    fn get_vals(&self) -> Vec<f32> {
        self.iter_vals().collect()
    }
    fn dist2other(&self, other: &Self) -> f32 {
        self.iter_vals()
            .zip(other.iter_vals())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    fn distance2vec(&self, other: &Vec<f32>) -> f32 {
        self.iter_vals()
            .zip(other.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

pub fn gen_rand_vecs(dim: usize, n: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut vecs = vec![];
    assert!(n > 0);
    for _ in 0..n {
        vecs.push((0..dim).map(|_| rng.gen_range(-1.0..=1.0)).collect())
        // vecs.push((0..dim).map(|_| rng.r#gen::<f32>()).collect())
    }
    vecs
}
