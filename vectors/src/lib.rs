mod bytes;
mod full;
mod lvq;
pub mod serializer;

pub use bytes::ByteVec;
pub use full::FullVec;
pub use lvq::LVQVec;

use rand::Rng;

use crate::serializer::Serializer;

pub trait VecTrait: VecBase + Serializer + Clone {}

pub trait VecBase {
    fn center(&mut self, means: &Vec<f32>);
    fn decenter(&mut self, means: &Vec<f32>);
    fn dim(&self) -> usize;
    fn iter_vals(&self) -> impl Iterator<Item = f32>;
    fn distance(&self, other: &impl VecBase) -> f32;
    fn dist2other(&self, other: &Self) -> f32;

    fn dist2many<'a, I>(&'a self, others: I) -> impl Iterator<Item = f32> + 'a
    where
        I: Iterator<Item = &'a Self> + 'a,
    {
        others.map(move |other| self.dist2other(other))
    }

    fn get_vals(&self) -> Vec<f32> {
        self.iter_vals().collect()
    }

    fn quantize(&self) -> LVQVec {
        LVQVec::new(&self.get_vals())
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
