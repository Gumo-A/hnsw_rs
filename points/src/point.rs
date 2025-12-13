use core::panic;

use graph::nodes::Node;
use vectors::{FullVec, LVQVec, VecBase, VecTrait, serializer::Serializer};

#[derive(Debug, Clone)]
pub struct Point<T: VecTrait> {
    pub id: Node,
    pub level: u8,
    removed: bool,
    vector: T,
}

impl<T: VecTrait> Point<T> {
    /// Marks the point as removed,
    /// returns true if the toggle was made,
    /// false if the point was already marked.
    pub fn set_removed(&mut self) -> bool {
        if self.removed {
            false
        } else {
            self.removed = true;
            true
        }
    }
}

impl<T: VecTrait> VecBase for Point<T> {
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

impl Point<LVQVec> {
    pub fn new_quant(id: Node, level: u8, vector: &Vec<f32>) -> Point<LVQVec> {
        Point {
            id,
            level,
            removed: false,
            vector: LVQVec::new(vector),
        }
    }
}

impl Point<FullVec> {
    pub fn new_full(id: Node, level: u8, vector: Vec<f32>) -> Point<FullVec> {
        Point {
            id,
            level,
            removed: false,
            vector: FullVec::new(vector),
        }
    }

    pub fn iter_vals_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.vector.iter_vals_mut()
    }

    pub fn get_low_vector(&self) -> &Vec<f32> {
        &self.vector.vals
    }

    pub fn quantized(&self) -> Point<LVQVec> {
        Point::new_quant(self.id, self.level, self.get_low_vector())
    }
}

impl<T: VecTrait> Serializer for Point<T> {
    fn size(&self) -> usize {
        6 + self.vector.size()
    }

    /// 4 bytes for ID, one byte for level,
    /// one for removed flag, followed by vector
    /// byte string.
    /// Val        Bytes
    /// ID         4
    /// level      1
    /// removed    1
    /// vector     variable
    fn serialize(&self) -> Vec<u8> {
        let vec_bytes = self.vector.serialize();
        let mut bytes = Vec::with_capacity(6 + vec_bytes.len());
        bytes.extend_from_slice(&self.id.to_be_bytes());
        bytes.extend_from_slice(&self.level.to_be_bytes());
        bytes.extend_from_slice(&(self.removed as u8).to_be_bytes());
        bytes.extend(vec_bytes);
        bytes
    }

    /// 4 bytes for ID, one byte for level,
    /// one for removed flag, followed by vector
    /// byte string.
    /// Val        Bytes
    /// ID         4
    /// level      1
    /// removed    1
    /// vector     variable
    fn deserialize(data: Vec<u8>) -> Self {
        let id: Node = u32::from_be_bytes(data[..4].try_into().unwrap());
        let level = u8::from_be_bytes(data[4..5].try_into().unwrap());
        let removed = u8::from_be_bytes(data[5..6].try_into().unwrap()) != 0;
        let vector = T::deserialize(data[6..].try_into().unwrap());
        Point {
            id,
            level,
            removed,
            vector,
        }
    }
}

impl<T: VecTrait> VecTrait for Point<T> {}

#[cfg(test)]
mod test {

    use vectors::gen_rand_vecs;

    use super::*;

    #[test]
    fn serialization() {
        let rand_vecs = gen_rand_vecs(128, 2);

        let a_id = 32;
        let a_level = 2;
        let a_vector = rand_vecs[0].clone();

        let b_id = 16;
        let b_level = 1;
        let b_vector = rand_vecs[1].clone();

        let a = Point::new_full(a_id, a_level, a_vector.clone());
        let b = Point::new_full(b_id, b_level, b_vector.clone());

        let a_ser = a.serialize();
        let b_ser = b.serialize();

        let a: Point<FullVec> = Point::deserialize(a_ser);
        let b: Point<FullVec> = Point::deserialize(b_ser);

        assert_eq!(a_id, a.id);
        assert_eq!(a_level, a.level);
        assert_eq!(a_vector, a.vector.get_vals());

        assert_eq!(b_id, b.id);
        assert_eq!(b_level, b.level);
        assert_eq!(b_vector, b.vector.get_vals());
    }

    #[test]
    fn placeholder() {}
}
