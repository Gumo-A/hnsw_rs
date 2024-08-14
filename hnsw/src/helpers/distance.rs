use crate::hnsw::{dist::Dist, lvq::LVQVec};

pub fn l2_compressed(vector: &LVQVec, compressed: &LVQVec) -> Dist {
    compressed.dist2other(vector)
}

pub fn v2v_dist(a: &Vec<f32>, b: &Vec<f32>) -> usize {
    let mut result: f32 = 0.0;
    for (x, y) in a.iter().zip(b) {
        result += (x - y).powi(2);
    }
    (result * 10_000.0) as usize
}
