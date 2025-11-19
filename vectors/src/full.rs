use crate::VecTrait;
const CHUNK_SIZE: usize = 8;

#[derive(Debug, Clone)]
pub struct FullVec {
    pub vals: Vec<f32>,
}

impl FullVec {
    pub fn new(vals: Vec<f32>) -> Self {
        FullVec { vals }
    }
    pub fn iter_vals_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.vals.iter_mut()
    }
    pub fn iter_vals(&mut self) -> impl Iterator<Item = &f32> {
        self.vals.iter()
    }
}

impl VecTrait for FullVec {
    fn iter_vals(&self) -> impl Iterator<Item = f32> {
        self.vals.iter().copied()
    }
    fn dim(&self) -> usize {
        self.vals.len()
    }

    fn center(&mut self, means: &Vec<f32>) {
        if self.dim() != means.len() {
            panic!("Vector dimensions are not equal")
        }
        self.iter_vals_mut()
            .enumerate()
            .for_each(|(idx, x)| *x -= means[idx]);
    }

    fn decenter(&mut self, means: &Vec<f32>) {
        if self.dim() != means.len() {
            panic!("Vector dimensions are not equal")
        }
        self.iter_vals_mut()
            .enumerate()
            .for_each(|(idx, x)| *x += means[idx]);
    }
}
