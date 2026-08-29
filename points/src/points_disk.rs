use std::fmt::Debug;

use graph::NodeID;
use vectors::VecBase;

use vectors::serializer::{BytesReader, Serializer};

use crate::point::Point;
use crate::points::{Points, SimplePoints};

const BLOCK_CAPACITY: usize = 1024;

#[derive(Debug, Clone)]
pub struct PointBlock {
    points: SimplePoints,
}

impl Points for PointBlock {
    fn new(vecs: Vec<Vec<f32>>, ml: f32, idx_offset: NodeID) -> Self {
        if vecs.len() > BLOCK_CAPACITY {
            panic!(
                "Cannot create PointBlock. It can only hold {BLOCK_CAPACITY}, but vecs contains {} vectors",
                vecs.len()
            )
        }
        PointBlock {
            points: SimplePoints::new(vecs, ml, idx_offset),
        }
    }
    fn len(&self) -> usize {
        self.points.len()
    }
    fn distance2point(&self, point: &Point, idx: NodeID) -> Option<f32> {
        self.points.distance2point(point, idx)
    }
    fn ids(&self) -> impl Iterator<Item = NodeID> {
        self.points.ids()
    }
    fn dim(&self) -> Option<usize> {
        self.points.dim()
    }
    fn push(&mut self, point: Point) -> NodeID {
        if self.len() < BLOCK_CAPACITY {
            self.points.push(point)
        } else {
            panic!("Could not add point to PointBlock, it would breach the limit")
        }
    }
    fn extend(&mut self, other: Self) -> Vec<NodeID> {
        if self.len() + other.len() < BLOCK_CAPACITY {
            self.points.extend(other.points)
        } else {
            panic!("Could not extend PointBlock, it would breach the limit")
        }
    }
    fn get_point(&self, idx: NodeID) -> Option<&Point> {
        self.points.get_point(idx)
    }
    fn get_points_iter<I>(&self, indices: I) -> impl Iterator<Item = &Point>
    where
        I: Iterator<Item = NodeID>,
    {
        self.points.get_points_iter(indices)
    }
    fn distance(&self, a_idx: NodeID, b_idx: NodeID) -> Option<f32> {
        self.points.distance(a_idx, b_idx)
    }
}

impl Serializer for PointBlock {
    fn size(&self) -> usize {
        self.points.size()
    }
    fn serialize(&self) -> Vec<u8> {
        self.points.serialize()
    }
    fn deserialize(data: Vec<u8>) -> Self {
        PointBlock {
            points: SimplePoints::deserialize(data),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PointsDisk {
    pub blocks: Vec<PointBlock>,
    ml: f32,
}

impl Points for PointsDisk {
    fn new(vecs: Vec<Vec<f32>>, ml: f32, idx_offset: NodeID) -> Self {
        let nb_blocks = (vecs.len() as f32 / BLOCK_CAPACITY as f32).ceil() as usize;
        let mut vecs_iter = vecs.iter().cloned();
        let mut blocks = Vec::new();
        for block_idx in 0..nb_blocks {
            let block_vecs = vecs_iter.by_ref().take(BLOCK_CAPACITY).collect();
            let block_offset = (block_idx * BLOCK_CAPACITY) + idx_offset as usize;
            let block = PointBlock::new(block_vecs, ml, block_offset as NodeID);
            blocks.push(block)
        }
        Self { blocks, ml }
    }

    fn get_point(&self, idx: NodeID) -> Option<&Point> {
        let block_idx = idx as usize / BLOCK_CAPACITY;
        match self.blocks.get(block_idx) {
            Some(b) => b.get_point(idx),
            None => None,
        }
    }

    fn push(&mut self, point: Point) -> NodeID {
        let last_block = self.blocks.last_mut().unwrap(); // fails if no points in collection
        if last_block.len() == BLOCK_CAPACITY {
            let new_block = PointBlock::new(vec![point.get_vals()], self.ml, self.len() as NodeID);
            self.blocks.push(new_block);
            self.len() as NodeID
        } else {
            last_block.push(point)
        }
    }

    fn len(&self) -> usize {
        // Every block contains BLOCK_CAPACITY points,
        // except the last one
        let last_idx = self.blocks.len() - 1;
        (last_idx * BLOCK_CAPACITY) + self.blocks[last_idx].len()
    }

    fn dim(&self) -> Option<usize> {
        match self.blocks.get(0) {
            Some(b) => b.dim(),
            None => None,
        }
    }

    fn ids(&self) -> impl Iterator<Item = NodeID> {
        self.blocks.iter().map(|b| b.ids()).flatten()
    }
}

const NB_BLOCKS_U32: usize = 4;
const ML_F32: usize = 4;
const POINTS_DISK_HEADER_SIZE: usize = NB_BLOCKS_U32 + ML_F32;
impl Serializer for PointsDisk {
    fn size(&self) -> usize {
        // add 8 bytes to each block size to account for
        // the size of each block, stored right before it
        POINTS_DISK_HEADER_SIZE + self.blocks.iter().map(|b| b.size() + 8).sum::<usize>()
    }

    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.blocks.len().to_be_bytes());
        bytes.extend_from_slice(&self.ml.to_be_bytes());
        for block in self.blocks.iter() {
            bytes.extend_from_slice(&block.size().to_be_bytes());
            bytes.extend_from_slice(&block.serialize());
        }
        bytes
    }

    fn deserialize(data: Vec<u8>) -> Self {
        let mut reader = BytesReader::new(data);
        let nb_blocks = u64::from_be_bytes(reader.u64());
        assert_ne!(nb_blocks, 0);
        let ml = f32::from_be_bytes(reader.u32());
        let mut blocks = Vec::new();
        for _ in 0..nb_blocks {
            let block_size = u64::from_be_bytes(reader.u64());
            blocks.push(PointBlock::deserialize(
                reader.read_bytes(block_size as usize).to_vec(),
            ));
        }
        Self { blocks, ml }
    }
}

#[cfg(test)]
mod test {

    use vectors::gen_rand_vecs;

    use super::*;

    // Current limit per block is 1024,
    // so PointsDisk should contain 3 blocks.
    // Two should be full, one should contain 2
    const N: NodeID = 2050;

    fn gen_rand_points(dim: usize, n: NodeID) -> PointsDisk {
        let vectors = gen_rand_vecs(dim, n as usize);
        PointsDisk::new(vectors, 0.5, 0)
    }

    fn get_collection_data(points: &PointsDisk) -> (f32, f32) {
        let dist = points
            .get_point(32)
            .unwrap()
            .distance(points.get_point(2049).unwrap());
        let val = points.get_point(1089).unwrap().iter_vals().next().unwrap();
        (dist, val)
    }

    #[test]
    fn build_simple_points() {
        let points = gen_rand_points(4, N);

        assert_eq!(points.blocks.len(), 3);
        assert_eq!(points.len(), N as usize);

        let point_0 = points.get_point(0).unwrap();
        assert_eq!(point_0.id, 0);

        let point_1 = points.get_point(1).unwrap();
        assert_eq!(point_1.id, 1);

        let point = points.get_point(1023).unwrap();
        assert_eq!(point.id, 1023);

        let point = points.get_point(1024).unwrap();
        assert_eq!(point.id, 1024);

        let point = points.get_point(1025).unwrap();
        assert_eq!(point.id, 1025);

        let point = points.get_point(1029).unwrap();
        assert_eq!(point.id, 1029);

        let point = points.get_point(2100);
        assert!(point.is_none());
    }

    #[test]
    fn distance_two_points_same_block() {
        let points = gen_rand_points(4, N);

        let dist = points.distance(12, 0);
        assert!(dist.unwrap() > 0.0);
    }

    #[test]
    fn distance_two_points_different_blocks() {
        let points = gen_rand_points(4, N);

        let dist = points.distance(12, 0);
        assert!(dist.unwrap() > 0.0);
    }

    #[test]
    fn serialization() {
        let points = gen_rand_points(128, N);
        let points_len = points.len();
        let points_size = points.size();

        let (dist, val) = get_collection_data(&points);

        let points_ser = points.serialize();
        let points_des = PointsDisk::deserialize(points_ser);

        assert_eq!(points.blocks.len(), points_des.blocks.len());
        assert_eq!(points_len, points_des.len());
        assert_eq!(points_size, points_des.size());

        let (dist_ser, val_ser) = get_collection_data(&points_des);

        assert_eq!(dist, dist_ser);
        assert_eq!(val, val_ser);

        for i in 0..N {
            assert_eq!(
                points.get_point(i).unwrap().get_vals(),
                points_des.get_point(i).unwrap().get_vals()
            );
        }
    }
}
