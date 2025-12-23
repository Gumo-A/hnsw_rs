use graph::nodes::Node;
use vectors::{FullVec, VecBase, VecTrait, serializer::Serializer};

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

impl<T: VecTrait> Point<T> {
    pub fn new_with(id: Node, level: usize, vector: &Vec<f32>) -> Point<T> {
        let mut point = Self::new(vector);
        point.id = id;
        point.level = level as u8;
        point
    }
}

impl<T: VecTrait> VecBase for Point<T> {
    /// Builds a point with ID 0 and level 0 by default.
    /// Use new_with to specify these values.
    fn new(vector: &Vec<f32>) -> Point<T> {
        Point {
            id: 0,
            level: 0,
            removed: false,
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

    use vectors::{LVQVec, gen_rand_vecs};

    use super::*;

    #[test]
    fn full_serialization() {
        let rand_vecs = gen_rand_vecs(128, 2);

        let a_id = 32;
        let a_level = 2;
        let a_vector = rand_vecs[0].clone();

        let b_id = 16;
        let b_level = 1;
        let b_vector = rand_vecs[1].clone();

        let a: Point<FullVec> = Point::new_with(a_id, a_level, &a_vector);
        let b: Point<FullVec> = Point::new_with(b_id, b_level, &b_vector);

        let a_ser = a.serialize();
        let b_ser = b.serialize();

        let a: Point<FullVec> = Point::deserialize(a_ser);
        let b: Point<FullVec> = Point::deserialize(b_ser);

        assert_eq!(a_id, a.id);
        assert_eq!(a_level, a.level as usize);
        assert_eq!(a_vector, a.vector.get_vals());

        assert_eq!(b_id, b.id);
        assert_eq!(b_level, b.level as usize);
        assert_eq!(b_vector, b.vector.get_vals());
    }

    #[test]
    fn quant_serialization() {
        let rand_vecs = gen_rand_vecs(128, 2);

        let a_id = 32;
        let a_level = 2;
        let a_vector = rand_vecs[0].clone();

        let b_id = 16;
        let b_level = 1;
        let b_vector = rand_vecs[1].clone();

        let a: Point<LVQVec> = Point::new_with(a_id, a_level, &a_vector);
        let b: Point<LVQVec> = Point::new_with(b_id, b_level, &b_vector);

        let a_ser = a.serialize();
        let b_ser = b.serialize();

        let a: Point<LVQVec> = Point::deserialize(a_ser);
        let b: Point<LVQVec> = Point::deserialize(b_ser);

        assert_eq!(a_id, a.id);
        assert_eq!(a_level, a.level as usize);
        let ratio = (a_vector[0] - a.vector.get_vals()[0]).abs() / a_vector[0];
        println!("{ratio}");
        assert!(ratio < 0.02);

        assert_eq!(b_id, b.id);
        assert_eq!(b_level, b.level as usize);
        let ratio = (b_vector[0] - b.vector.get_vals()[0]).abs() / b_vector[0];
        println!("{ratio}");
        assert!(ratio < 0.02);
    }

    #[test]
    fn placeholder() {}
}
