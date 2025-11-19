use crate::{FullVec, LVQVec};

pub trait VecVisitor {
    fn dist_quant(&self, vector: LVQVec) -> f32;
    fn dist_full(&self, vector: FullVec) -> f32;
}

struct LVQVisitor;

impl VecVisitor for LVQVisitor {
    fn dist_quant(&self, vector: LVQVec) {}
    fn dist_full(&self, vector: FullVec) {}
}

struct FullVisitor;

impl VecVisitor for FullVisitor {
    fn dist_quant(&self, vector: LVQVec) {}
    fn dist_full(&self, vector: FullVec) {}
}
