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
}
