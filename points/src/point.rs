use graph::nodes::NodeID;
use vectors::{QuantVec, VecBase, serializer::Serializer};

type VecType = QuantVec;
#[derive(Debug, Clone)]
pub struct Point {
    pub id: NodeID,
    pub level: u8,
    vector: VecType,
}

impl Point {
    pub fn with_level_and_id(vector: &Vec<f32>, level: usize, id: usize) -> Point {
        let mut point = Self::new(vector);
        point.id = id as NodeID;
        point.level = level as u8;
        point
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
}

impl Serializer for Point {
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
    // from its position.
    fn deserialize(data: Vec<u8>) -> Self {
        let level = u8::from_be_bytes(data[..1].try_into().unwrap());
        let vector = VecType::deserialize(data[1..].try_into().unwrap());
        Point {
            id: 0,
            level,
            vector,
        }
    }
}

#[cfg(test)]
mod test {

    // Cannot test for ID serialization here,
    // it is done at the block level

    use vectors::gen_rand_vecs;

    use super::*;

    #[test]
    fn quant_serialization() {
        let rand_vecs = gen_rand_vecs(128, 2);

        let a_level = 2;
        let b_level = 1;

        let a = Point::with_level_and_id(&rand_vecs[0], a_level, 0);
        let a_vector = a.get_vals();

        let b = Point::with_level_and_id(&rand_vecs[0], b_level, 0);
        let b_vector = b.get_vals();

        let a_ser = a.serialize();
        let b_ser = b.serialize();

        let a = Point::deserialize(a_ser);
        let b = Point::deserialize(b_ser);

        assert_eq!(a_level, a.level as usize);
        assert_eq!(a_vector, a.vector.get_vals());

        assert_eq!(b_level, b.level as usize);
        assert_eq!(b_vector, b.vector.get_vals());
    }

    #[test]
    fn placeholder() {}
}
