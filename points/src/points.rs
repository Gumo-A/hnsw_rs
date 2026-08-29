use std::fmt::Debug;

use graph::NodeID;
use log::{debug, trace};
use vectors::VecBase;

use vectors::serializer::{BytesReader, Serializer};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::point::Point;

pub trait Points {
    fn new(vecs: Vec<Vec<f32>>, ml: f32, idx_offset: NodeID) -> Self;
    fn len(&self) -> usize;
    fn ids(&self) -> impl Iterator<Item = NodeID>;
    fn dim(&self) -> Option<usize>;
    // fn remove(&mut self, index: Node) -> bool;
    fn push(&mut self, point: Point) -> NodeID;
    // extends collection and returns all new ids
    fn extend(&mut self, other: Self) -> Vec<NodeID>
    where
        Self: Sized,
    {
        let mut ids = Vec::new();
        for idx in other.ids() {
            ids.push(self.push(other.get_point(idx).unwrap().clone()));
        }
        ids
    }
    fn get_point(&self, idx: NodeID) -> Option<&Point>;
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
}

#[derive(Debug, Clone)]
pub struct SimplePoints {
    pub collection: Vec<Point>,
    idx_offset: NodeID,
}

impl Points for SimplePoints {
    fn new(vecs: Vec<Vec<f32>>, ml: f32, idx_offset: NodeID) -> Self {
        let mut rng = StdRng::seed_from_u64(0);
        let i = vecs.iter().enumerate().map(|(idx, v)| {
            let level = new_layer(ml, &mut rng);
            Point::with_level_and_id(v, level, idx as NodeID + idx_offset)
        });
        Self {
            collection: Vec::from_iter(i),
            idx_offset,
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

    fn push(&mut self, mut point: Point) -> NodeID {
        point.id = self.len() as NodeID + self.idx_offset;
        let id = point.id;
        trace!(
            "Pushing new point to collection with ID {} and level {}",
            point.id, point.level
        );
        self.collection.push(point);
        id
    }

    fn get_point(&self, idx: NodeID) -> Option<&Point> {
        // The smallest valid idx is idx_offset
        if idx < self.idx_offset {
            None
        } else {
            self.collection.get((idx - self.idx_offset) as usize)
        }
    }

    fn extend(&mut self, other: Self) -> Vec<NodeID> {
        debug!("Extending current points struct from another");
        debug!(
            "Current points has length {0}, other has {1}",
            self.len(),
            other.len()
        );
        let mut ids = Vec::with_capacity(other.len());
        let mut other = other;
        for point in other.collection.drain(..) {
            ids.push(self.push(point));
        }
        ids
    }
}

const NB_POINTS_U64: usize = 8;
const POINT_SIZE_U64: usize = 8;
const IDX_OFFSET_U32: usize = 4;
const SIMPLE_POINTS_HEADER_SIZE: usize = NB_POINTS_U64 + POINT_SIZE_U64 + IDX_OFFSET_U32;

impl Serializer for SimplePoints {
    fn size(&self) -> usize {
        // how many points, size of each point, plus the points themselves
        SIMPLE_POINTS_HEADER_SIZE + (self.len() * self.get_point(self.idx_offset).unwrap().size())
    }
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.collection.len().to_be_bytes());
        bytes.extend_from_slice(
            &self
                .get_point(self.idx_offset)
                .unwrap()
                .size()
                .to_be_bytes(),
        );
        bytes.extend_from_slice(&self.idx_offset.to_be_bytes());
        for point in self.collection.iter() {
            bytes.extend_from_slice(&point.serialize());
        }
        bytes
    }

    fn deserialize(data: Vec<u8>) -> Self {
        let mut reader = BytesReader::new(data);
        let nb_points = usize::from_be_bytes(reader.u64());
        let point_size = usize::from_be_bytes(reader.u64());
        let idx_offset = NodeID::from_be_bytes(reader.u32());
        let mut collection = Vec::new();
        for id in 0..nb_points {
            let mut point = Point::deserialize(reader.read_bytes(point_size).to_vec());
            point.id = idx_offset + id as NodeID;
            collection.push(point);
        }
        Self {
            collection,
            idx_offset,
        }
    }
}

pub fn new_layer(ml: f32, rng: &mut StdRng) -> usize {
    let mut rand_nb = 0.0;
    loop {
        if (rand_nb == 0.0) | (rand_nb == 1.0) {
            rand_nb = rng.r#gen::<f32>();
        } else {
            break;
        }
    }

    let level = (-rand_nb.ln() * ml).floor() as usize;
    level
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

    const N: NodeID = 100;
    const OFFSET: NodeID = 16;

    fn gen_rand_points(dim: usize, n: NodeID, offset: NodeID) -> SimplePoints {
        let vectors = gen_rand_vecs(dim, n as usize);
        SimplePoints::new(vectors, 0.5, offset)
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
        let points = gen_rand_points(4, 100, 0);

        let point_0 = points.get_point(0).unwrap();
        assert_eq!(point_0.id, 0);

        let point_1 = points.get_point(1).unwrap();
        assert_eq!(point_1.id, 1);

        let point_80_000 = points.get_point(80).unwrap();
        assert_eq!(point_80_000.id, 80);
    }

    #[test]
    fn distance_two_points() {
        let points = gen_rand_points(4, 16, 0);

        let dist = points.distance(12, 4);
        assert!(dist.unwrap() > 0.0);
    }

    #[test]
    fn serialization() {
        let points = gen_rand_points(128, 100, 0);
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

    #[test]
    fn build_simple_points_with_offset() {
        let points = gen_rand_points(4, N, OFFSET);

        let point_0 = points.get_point(0);
        assert!(point_0.is_none());
        let point = points.get_point(OFFSET - 1);
        assert!(point.is_none());

        let point_min_idx = points.get_point(OFFSET).unwrap();
        assert_eq!(point_min_idx.id, OFFSET);

        let point_last = points.get_point(N + OFFSET - 1).unwrap();
        assert_eq!(point_last.id, N + OFFSET - 1);
    }

    #[test]
    fn distance_two_points_with_offset() {
        let points = gen_rand_points(4, 16, OFFSET);

        // Both exist
        let dist = points.distance(18, 20);
        assert!(dist.unwrap() > 0.0);

        // Neither exist
        let dist = points.distance(12, 4);
        assert!(dist.is_none());

        // One doesn't exist
        let dist = points.distance(18, 4);
        assert!(dist.is_none());
    }

    #[test]
    fn serialization_with_offset() {
        let points = gen_rand_points(128, N, OFFSET);
        let (dist, val, len, size) = get_collection_data(&points);

        let points_ser = points.serialize();
        let points_des = SimplePoints::deserialize(points_ser);
        let (dist_ser, val_ser, len_ser, size_ser) = get_collection_data(&points_des);

        assert_eq!(dist, dist_ser);
        assert_eq!(val, val_ser);
        assert_eq!(len, len_ser);
        assert_eq!(size, size_ser);

        for i in OFFSET..OFFSET + N {
            assert_eq!(
                points.get_point(i).unwrap().get_vals(),
                points_des.get_point(i).unwrap().get_vals()
            );
        }
    }
}
