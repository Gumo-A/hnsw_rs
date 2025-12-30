use graph::nodes::NodeID;
use vectors::{FullVec, VecBase, VecTrait, serializer::Serializer};

#[derive(Debug, Clone)]
pub struct Point<T: VecTrait> {
    pub id: NodeID,
    pub level: u8,
    vector: T,
}

impl<T: VecTrait> Point<T> {
    pub fn new_with(level: usize, vector: &Vec<f32>) -> Point<T> {
        let mut point = Self::new(vector);
        point.id = 0;
        point.level = level as u8;
        point
    }
    // Marks the point as removed,
    // returns true if the toggle was made,
    // false if the point was already marked.
    // pub fn set_removed(&mut self) -> bool {
    //     if self.removed {
    //         false
    //     } else {
    //         self.removed = true;
    //         true
    //     }
    // }
}

impl<T: VecTrait> VecBase for Point<T> {
    /// Builds a point with ID 0 and level 0 by default.
    /// Use new_with to specify these values.
    fn new(vector: &Vec<f32>) -> Point<T> {
        Point {
            id: 0,
            level: 0,
            vector: T::new(vector),
        }
    }

    fn distance(&self, other: &impl VecBase) -> f32 {
        self.vector.distance(other)
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
    fn center(&mut self, means: &Vec<f32>) {
        self.vector.center(means);
    }
    fn decenter(&mut self, means: &Vec<f32>) {
        self.vector.decenter(means);
    }
}

impl Point<FullVec> {
    pub fn iter_vals_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.vector.iter_vals_mut()
    }

    pub fn get_low_vector(&self) -> &Vec<f32> {
        &self.vector.vector
    }
}

impl<T: VecTrait> Serializer for Point<T> {
    /// One byte for level
    /// followed by vector byte string.
    ///
    /// Val        Bytes
    /// level      1
    /// vector     self.vector.size()
    fn size(&self) -> usize {
        1 + self.vector.size()
    }

    fn serialize(&self) -> Vec<u8> {
        let vec_bytes = self.vector.serialize();
        let mut bytes = Vec::with_capacity(6 + vec_bytes.len());
        bytes.extend_from_slice(&self.level.to_be_bytes());
        bytes.extend(vec_bytes);
        bytes
    }

    // When we deserialize, we dont get the ID, it is inferred
    // from its position within its block and the block ID.
    fn deserialize(data: Vec<u8>) -> Self {
        let level = u8::from_be_bytes(data[..1].try_into().unwrap());
        let vector = T::deserialize(data[1..].try_into().unwrap());
        Point {
            id: 0,
            level,
            vector,
        }
    }
}

impl<T: VecTrait> VecTrait for Point<T> {}

#[cfg(test)]
mod test {

    // Cannot test for ID serialization here,
    // it is done at the block level

    use vectors::{LVQVec, gen_rand_vecs};

    use super::*;

    #[test]
    fn full_serialization() {
        let rand_vecs = gen_rand_vecs(128, 2);

        let a_level = 2;
        let a_vector = rand_vecs[0].clone();

        let b_level = 1;
        let b_vector = rand_vecs[1].clone();

        let a: Point<FullVec> = Point::new_with(a_level, &a_vector);
        let b: Point<FullVec> = Point::new_with(b_level, &b_vector);

        let a_ser = a.serialize();
        let b_ser = b.serialize();

        let a: Point<FullVec> = Point::deserialize(a_ser);
        let b: Point<FullVec> = Point::deserialize(b_ser);

        assert_eq!(a_level, a.level as usize);
        assert_eq!(a_vector, a.vector.get_vals());

        assert_eq!(b_level, b.level as usize);
        assert_eq!(b_vector, b.vector.get_vals());
    }

    #[test]
    fn quant_serialization() {
        let rand_vecs = gen_rand_vecs(128, 2);

        let a_level = 2;
        let b_level = 1;

        let a: Point<LVQVec> = Point::new_with(a_level, &rand_vecs[0]);
        let a_vector = a.get_vals();

        let b: Point<LVQVec> = Point::new_with(b_level, &rand_vecs[1]);
        let b_vector = b.get_vals();

        let a_ser = a.serialize();
        let b_ser = b.serialize();

        let a: Point<LVQVec> = Point::deserialize(a_ser);
        let b: Point<LVQVec> = Point::deserialize(b_ser);

        assert_eq!(a_level, a.level as usize);
        assert_eq!(a_vector, a.vector.get_vals());

        assert_eq!(b_level, b.level as usize);
        assert_eq!(b_vector, b.vector.get_vals());
    }

    #[test]
    fn placeholder() {}
}
