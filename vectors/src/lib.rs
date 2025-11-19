mod bytes;
mod full;
mod lvq;

pub use bytes::ByteVec;
pub use full::FullVec;
pub use lvq::LVQVec;

pub trait VecTrait {
    fn iter_vals(&self) -> impl Iterator<Item = f32>;

    fn dim(&self) -> usize;

    fn distance(&self, other: &impl VecTrait) -> f32 {
        self.iter_vals()
            .zip(other.iter_vals())
            .fold(0.0, |acc, e| acc + (e.0 - e.1).powi(2))
            .sqrt()
    }

    fn get_vals(&self) -> Vec<f32> {
        self.iter_vals().collect()
    }

    fn quantize(&self) -> LVQVec {
        LVQVec::new(&self.get_vals())
    }

    fn center(&mut self, means: &Vec<f32>);
    fn decenter(&mut self, means: &Vec<f32>);
}
