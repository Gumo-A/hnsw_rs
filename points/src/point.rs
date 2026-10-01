use serde::{Deserialize, Serialize};
use vectors::{QuantVec, VecBase};

type NodeID = u32;
pub type VecType = QuantVec;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub id: NodeID,
    pub level: u8,
    pub vector: VecType,
}

impl Point {
    pub fn with_level_and_id(vector: &Vec<f32>, level: usize, id: NodeID) -> Point {
        let mut point = Self::new(vector);
        point.id = id;
        point.level = level as u8;
        point
    }

    pub fn with_vectype_and_id(vector: &VecType, id: NodeID) -> Point {
        Self {
            id,
            level: 0,
            vector: vector.clone(),
        }
    }
}

impl VecBase for Point {
    /// Builds a point with ID 0 and level 0 by default.
    /// Use new_with to specify these values.
    fn new(vector: &Vec<f32>) -> Point {
        Point {
            id: 0,
            level: 0,
            vector: VecType::new(vector),
        }
    }
    fn dist2other(&self, other: &Self) -> f32 {
        self.vector.dist2other(&other.vector)
    }
    fn iter_vals(&self) -> impl Iterator<Item = f32> {
        self.vector.iter_vals()
    }
    fn dim(&self) -> usize {
        self.vector.dim()
    }
}

#[cfg(test)]
mod test {

    // Cannot test for ID serialization here,
    // it is done at the block level

    use vectors::gen_rand_vecs;

    use super::*;

    #[test]
    fn serialization() {
        let rand_vecs = gen_rand_vecs(128, 1);
        let level = 2;
        let a = Point::with_level_and_id(&rand_vecs[0], level, 0);
        let a_vector = a.get_vals();
        let a_ser = rmp_serde::encode::to_vec(&a).unwrap();
        let a: Point = rmp_serde::decode::from_slice(&a_ser).unwrap();
        assert_eq!(level, a.level as usize);
        assert_eq!(a_vector, a.vector.get_vals());
    }

    #[test]
    fn dist() {
        let rand_vecs = gen_rand_vecs(128, 2);
        let a = Point::with_level_and_id(&rand_vecs[0], 0, 0);
        let a_vector = a.get_vals();

        let b = Point::with_level_and_id(&rand_vecs[0], 0, 0);
        let b_vector = b.get_vals();

        let vec_dist: f32 = a_vector
            .iter()
            .zip(b_vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        let dist = a.dist2other(&b);
        assert_eq!(dist, vec_dist);
        let dist = a.distance(&b);
        assert_eq!(dist, vec_dist);
    }
}
