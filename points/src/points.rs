pub mod block_points;

use graph::NodeID;
use vectors::VecBase;

use vectors::serializer::Serializer;

use rand::Rng;
use rand::rngs::ThreadRng;

use crate::point::Point;

pub trait Points {
    fn new(vecs: Vec<Vec<f32>>, ml: f32) -> Self;
    fn len(&self) -> usize;
    fn ids(&self) -> impl Iterator<Item = NodeID>;
    fn dim(&self) -> Option<usize>;
    fn push(&mut self, v: &Vec<f32>, ml: f32) -> NodeID;
    fn extend(&mut self, other: Self, ml: f32) -> Vec<NodeID>;
    // fn remove(&mut self, index: Node) -> bool;
    fn get_point(&self, idx: NodeID) -> Option<&Point>;
    fn get_points_iter<I>(&self, indices: I) -> impl Iterator<Item = &Point>
    where
        I: Iterator<Item = NodeID>;
    // extends collection and returns all new ids
    fn distance(&self, a_idx: NodeID, b_idx: NodeID) -> Option<f32>;
    fn distance2point(&self, point: &Point, idx: NodeID) -> Option<f32>;
}

#[derive(Debug, Clone)]
pub struct SimplePoints {
    pub collection: Vec<Point>,
}

impl Points for SimplePoints {
    fn new(vecs: Vec<Vec<f32>>, ml: f32) -> Self {
        let mut rng = ThreadRng::default();
        let i = vecs.iter().enumerate().map(|(idx, v)| {
            let level = new_layer(ml, &mut rng);
            Point::with_level_and_id(v, level, idx)
        });
        Self {
            collection: Vec::from_iter(i),
        }
    }
    fn len(&self) -> usize {
        self.collection.len()
    }

    fn ids(&self) -> impl Iterator<Item = NodeID> {
        self.collection.iter().map(|p| p.id)
    }

    fn dim(&self) -> Option<usize> {
        match self.collection.first() {
            Some(p) => Some(p.dim()),
            None => None,
        }
    }

    fn push(&mut self, v: &Vec<f32>, ml: f32) -> NodeID {
        let id = self.len();
        let point = Point::with_level_and_id(v, new_layer(ml, &mut ThreadRng::default()), id);
        self.collection.push(point);
        id as NodeID
    }

    fn get_point(&self, idx: NodeID) -> Option<&Point> {
        self.collection.get(idx as usize)
    }

    fn get_points_iter<I>(&self, indices: I) -> impl Iterator<Item = &Point>
    where
        I: Iterator<Item = NodeID>,
    {
        indices.map(|idx| self.get_point(idx).unwrap())
    }

    fn distance(&self, a_idx: NodeID, b_idx: NodeID) -> Option<f32> {
        let point_a = self.get_point(a_idx);
        let point_b = self.get_point(b_idx);
        match (point_a, point_b) {
            (Some(a), Some(b)) => Some(a.dist2other(b)),
            _ => None,
        }
    }

    fn distance2point(&self, point: &Point, idx: NodeID) -> Option<f32> {
        let other = self.get_point(idx);
        match other {
            Some(b) => Some(point.dist2other(b)),
            _ => None,
        }
    }

    fn extend(&mut self, other: Self, ml: f32) -> Vec<NodeID> {
        let mut ids = Vec::with_capacity(other.len());
        let mut other = other;
        for point in other.collection.drain(..) {
            ids.push(self.push(&point.get_vals(), ml));
        }
        ids
    }
}

impl Serializer for SimplePoints {
    fn size(&self) -> usize {
        // how many points, size of each point, plus the points themselves
        8 + 8 + (self.len() * self.get_point(0).unwrap().size())
    }
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.collection.len().to_be_bytes());
        bytes.extend_from_slice(&self.get_point(0).unwrap().size().to_be_bytes());
        for point in self.collection.iter() {
            bytes.extend_from_slice(&point.serialize());
        }
        bytes
    }

    fn deserialize(data: Vec<u8>) -> Self {
        let len = usize::from_be_bytes(data[0..8].try_into().unwrap());
        let point_size = usize::from_be_bytes(data[8..16].try_into().unwrap());
        let mut collection = Vec::new();
        let mut idx = 16;
        for _ in 0..len {
            let point = Point::deserialize(data[idx..(idx + point_size)].to_vec());
            collection.push(point);
            idx += point_size
        }
        Self { collection }
    }
}

pub fn new_layer(ml: f32, rng: &mut ThreadRng) -> usize {
    let mut rand_nb = 0.0;
    loop {
        if (rand_nb == 0.0) | (rand_nb == 1.0) {
            rand_nb = rng.r#gen::<f32>();
        } else {
            break;
        }
    }

    (-rand_nb.log(std::f32::consts::E) * ml).floor() as usize
}

impl<'a> SimplePoints {
    pub fn iter_points<'b: 'a>(&'b self) -> impl Iterator<Item = &'a Point> {
        self.collection.iter()
    }
    pub fn iter_points_mut<'b: 'a>(&mut self) -> impl Iterator<Item = &mut Point> {
        self.collection.iter_mut()
    }
}

#[cfg(test)]
mod test {

    use vectors::gen_rand_vecs;

    use super::*;

    fn gen_rand_points(dim: usize, n: usize) -> SimplePoints {
        let vectors = gen_rand_vecs(dim, n);
        SimplePoints::new(vectors, 0.5)
    }

    fn get_collection_data(points: &SimplePoints) -> (f32, f32, u32, usize) {
        let dist = points
            .get_point(32)
            .unwrap()
            .distance(points.get_point(64).unwrap());
        let val = points.get_point(16).unwrap().iter_vals().next().unwrap();
        (dist, val, points.len() as u32, points.size())
    }

    #[test]
    fn build_simple_points() {
        let points = gen_rand_points(4, 100);

        let point_0 = points.get_point(0).unwrap();
        assert_eq!(point_0.id, 0);

        let point_1 = points.get_point(1).unwrap();
        assert_eq!(point_1.id, 1);

        let point_80_000 = points.get_point(80).unwrap();
        assert_eq!(point_80_000.id, 80);
    }

    #[test]
    fn distance_two_points() {
        let points = gen_rand_points(4, 16);

        let dist = points.distance(12, 4);
        assert!(dist.unwrap() > 0.0);
    }

    #[test]
    fn serialization() {
        let points = gen_rand_points(128, 100);
        let (dist, val, len, size) = get_collection_data(&points);

        let points_ser = points.serialize();
        let points_des = SimplePoints::deserialize(points_ser);
        let (dist_ser, val_ser, len_ser, size_ser) = get_collection_data(&points_des);

        assert_eq!(dist, dist_ser);
        assert_eq!(val, val_ser);
        assert_eq!(len, len_ser);
        assert_eq!(size, size_ser);

        for i in 0..100 {
            assert_eq!(
                points.get_point(i).unwrap().get_vals(),
                points_des.get_point(i).unwrap().get_vals()
            );
        }
    }
}
