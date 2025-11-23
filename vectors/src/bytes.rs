use crate::VecTrait;

#[derive(Debug, Clone)]
pub struct ByteVec {
    pub vals: Vec<u8>,
}

impl ByteVec {
    pub fn new(data: Vec<u8>) -> ByteVec {
        ByteVec { vals: data }
    }
    pub fn iter_vals_mut(&mut self) -> impl Iterator<Item = &mut u8> {
        self.vals.iter_mut()
    }
}

impl VecTrait for ByteVec {
    fn distance(&self, other: &impl VecTrait) -> f32 {
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
        self.vals.iter().map(|x| *x as f32)
    }

    fn dim(&self) -> usize {
        self.vals.len()
    }
    fn center(&mut self, _: &Vec<f32>) {}
    fn decenter(&mut self, _: &Vec<f32>) {}
}
