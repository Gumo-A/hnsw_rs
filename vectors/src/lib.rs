mod full;
mod quant;
pub mod serializer;

use std::sync::Arc;

pub use full::FullVec;
pub use quant::QuantVec;

use rand::Rng;

pub trait VecBase {
    fn new(vector: &Vec<f32>) -> Self;
    fn dim(&self) -> usize;
    fn dist2other(&self, other: &Self) -> f32;
    fn iter_vals(&self) -> impl Iterator<Item = f32>;
    fn get_vals(&self) -> Vec<f32> {
        self.iter_vals().collect()
    }
    fn distance(&self, other: &impl VecBase) -> f32 {
        self.iter_vals()
            .zip(other.iter_vals())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    fn dist2many<I>(&self, others: I) -> impl Iterator<Item = f32>
    where
        I: Iterator<Item = Arc<Self>>,
    {
        others.map(move |other| self.dist2other(&other))
    }
}

pub fn gen_rand_vecs(dim: usize, n: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut vecs = vec![];
    assert!(n > 0);
    for _ in 0..n {
        vecs.push((0..dim).map(|_| rng.r#gen::<f32>()).collect())
    }
    vecs
}
