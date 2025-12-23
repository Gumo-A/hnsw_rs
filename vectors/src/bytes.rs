use crate::VecBase;

#[derive(Debug, Clone)]
pub struct ByteVec {
    pub vector: Vec<u8>,
}

impl ByteVec {
    pub fn iter_vals_mut(&mut self) -> impl Iterator<Item = &mut u8> {
        self.vector.iter_mut()
    }
}

impl VecBase for ByteVec {
    fn new(vector: &Vec<f32>) -> ByteVec {
        ByteVec {
            vector: vector.iter().map(|x| *x as u8).collect(),
        }
    }
    fn distance(&self, other: &impl VecBase) -> f32 {
        self.iter_vals()
            .zip(other.iter_vals())
            .fold(0.0, |acc, e| acc + (e.0 - e.1).powi(2))
            .sqrt()
    }
    fn dist2other(&self, other: &Self) -> f32 {
        self.distance(other)
    }
    fn dist2many<'a, I>(&'a self, others: I) -> impl Iterator<Item = f32> + 'a
    where
        I: Iterator<Item = &'a Self> + 'a,
    {
        others.map(move |other| self.distance(other))
    }

    fn iter_vals(&self) -> impl Iterator<Item = f32> {
        self.vector.iter().map(|x| *x as f32)
    }

    fn dim(&self) -> usize {
        self.vector.len()
    }
    fn center(&mut self, _: &Vec<f32>) {}
    fn decenter(&mut self, _: &Vec<f32>) {}
}
