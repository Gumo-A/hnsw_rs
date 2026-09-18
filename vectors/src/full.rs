use serde::{Deserialize, Serialize};

use crate::VecBase;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullVec {
    pub vector: Vec<f32>,
}

impl FullVec {
    pub fn iter_vals_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.vector.iter_mut()
    }
    pub fn iter_vals(&mut self) -> impl Iterator<Item = &f32> {
        self.vector.iter()
    }
}

impl VecBase for FullVec {
    fn new(vector: &Vec<f32>) -> Self {
        FullVec {
            vector: vector.clone(),
        }
    }
    fn dist2other(&self, other: &Self) -> f32 {
        self.distance(other)
    }

    fn iter_vals(&self) -> impl Iterator<Item = f32> {
        self.vector.iter().copied()
    }

    fn dim(&self) -> usize {
        self.vector.len()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::gen_rand_vecs;

    use super::*;

    #[test]
    fn serialization() {
        let a = FullVec::new(&gen_rand_vecs(128, 1)[0].clone());
        let ser_a = rmp_serde::encode::to_vec(&a).unwrap();
        let b: FullVec = rmp_serde::decode::from_slice(&ser_a).unwrap();
        for (a_val, b_val) in a.iter_vals().zip(b.iter_vals()) {
            assert_eq!(a_val, b_val);
        }
    }

    #[test]
    fn distance() {
        let mut others = Vec::new();
        for _ in 0..100 {
            let a = FullVec::new(&gen_rand_vecs(128, 1)[0].clone());
            others.push(Arc::new(a));
        }
        let a = FullVec::new(&gen_rand_vecs(128, 1)[0].clone());
        a.dist2many(others.iter().cloned())
            .for_each(|dist| assert!(dist >= 0.0));

        let a = FullVec::new(&vec![0.5]);
        let b = FullVec::new(&vec![0.25]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert!(dist == 0.25);
        assert_eq!(dist, dist2other);

        let a = FullVec::new(&vec![0.75]);
        let b = FullVec::new(&vec![0.25]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert!(dist == 0.5);
        assert_eq!(dist, dist2other);

        let a = FullVec::new(&vec![0.0, 0.0]);
        let b = FullVec::new(&vec![0.0, 1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert!(dist == 1.0);
        assert_eq!(dist, dist2other);

        let a = FullVec::new(&vec![1.0, 0.0]);
        let b = FullVec::new(&vec![0.0, 1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert!(dist == 2.0f32.sqrt());
        assert_eq!(dist, dist2other);

        let a = FullVec::new(&vec![-1.0, 0.0]);
        let b = FullVec::new(&vec![0.0, 1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert!(dist == 2.0f32.sqrt());
        assert_eq!(dist, dist2other);

        let a = FullVec::new(&vec![1.0, 0.0]);
        let b = FullVec::new(&vec![0.0, -1.0]);
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert!(dist == 2.0f32.sqrt());
        assert_eq!(dist, dist2other);

        let a = FullVec::new(&gen_rand_vecs(128, 1)[0].clone());
        let b = a.clone();
        let dist = a.dist2other(&b);
        let dist2other = a.dist2other(&b);
        assert!(dist == 0.0);
        assert_eq!(dist, dist2other);
    }
}
