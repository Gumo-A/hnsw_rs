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
    fn iter_vals(&self) -> impl Iterator<Item = f32> {
        self.vals.iter().map(|x| *x as f32)
    }

    fn dim(&self) -> usize {
        self.vals.len()
    }
    fn center(&mut self, _: &Vec<f32>) {}
    fn decenter(&mut self, _: &Vec<f32>) {}
}
