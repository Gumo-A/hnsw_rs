pub mod block;

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use crate::point::Point;
use crate::points::block::data::BlockData;
use crate::points::block::header::{BLOCK_HEADER_SIZE, BlockHeader};
use crate::points::block::{BlockID, MAX_PER_BLOCK, PointsBlock};

use vectors::VecBase;

use graph::nodes::NodeID;
use vectors::serializer::Serializer;

use rand::rngs::ThreadRng;
use rand::{Rng, thread_rng};

pub trait Points {
    fn new(vecs: Vec<Vec<f32>>, ml: f32) -> Self;
    fn len(&self) -> usize;
    fn nb_blocks(&self) -> usize;
    fn ids(&self) -> impl Iterator<Item = NodeID>;
    fn dim(&self) -> Option<usize>;
    fn push(&mut self, point: Point) -> NodeID;
    fn extend(&mut self, other: Self) -> Vec<NodeID>;
    // fn remove(&mut self, index: Node) -> bool;
    fn get_point(&self, idx: NodeID) -> Option<&Point>;
    fn get_points(&self, indices: &Vec<NodeID>) -> Vec<&Point>;
    fn get_points_iter<I>(&self, indices: I) -> impl Iterator<Item = &Point>
    where
        I: Iterator<Item = NodeID>;
    // extends collection and returns all new ids
    fn distance(&self, a_idx: NodeID, b_idx: NodeID) -> Option<f32>;
    fn distance2point(&self, point: &Point, idx: NodeID) -> Option<f32>;
}

fn new_layer(ml: f32, rng: &mut ThreadRng) -> usize {
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

#[derive(Debug, Clone)]
pub struct BlockPoints {
    ml: f32,
    pub collection: Vec<PointsBlock>,
}

impl BlockPoints {
    fn add_point_new_block(&mut self, point: Point) -> NodeID {
        let new_id = self.collection.len() as BlockID;
        let mut new_block = PointsBlock::new(new_id);
        let node_id = new_block.add_point(&point).unwrap();
        self.collection.push(new_block);
        node_id
    }
}

impl Points for BlockPoints {
    fn new(mut vecs: Vec<Vec<f32>>, ml: f32) -> BlockPoints {
        let mut collection = Vec::new();
        let mut rng = thread_rng();
        let mut block_idx = 0;
        let mut block = PointsBlock::new(block_idx);
        for v in vecs.drain(..) {
            let point = Point::new_with(new_layer(ml, &mut rng), &v);
            if block.is_full() {
                collection.push(block);
                block_idx += 1;
                block = PointsBlock::new(block_idx);
            }
            block.add_point(&point);
        }

        if block.len() > 0 {
            collection.push(block);
        }

        BlockPoints { collection, ml }
    }
    fn len(&self) -> usize {
        self.collection.iter().map(|block| block.len()).sum()
    }

    fn nb_blocks(&self) -> usize {
        self.collection.len()
    }

    fn ids(&self) -> impl Iterator<Item = NodeID> {
        self.collection.iter().flat_map(|p| p.point_ids())
    }

    fn dim(&self) -> Option<usize> {
        match self.collection.first() {
            Some(p) => Some(p.dim()),
            None => None,
        }
    }

    fn push(&mut self, point: Point) -> NodeID {
        if self.collection.len() == 0 {
            self.add_point_new_block(point)
        } else {
            let last_block = self.collection.last_mut().unwrap();
            match last_block.add_point(&point) {
                Some(node_id) => node_id,
                None => self.add_point_new_block(point),
            }
        }
    }

    /// Removes a Point from the collection,
    /// returning true if it was removed,
    /// or false if it was already.
    // pub fn remove(&mut self, index: Node) -> bool {
    //     if index >= self.len() as Node {
    //         false
    //     } else {
    //         self.collection
    //             .get_mut(index as usize)
    //             .unwrap()
    //             .set_removed()
    //     }
    // }

    fn get_point(&self, idx: NodeID) -> Option<&Point> {
        let block_id = (idx as f32 / MAX_PER_BLOCK as f32).floor() as usize;
        let block = match self.collection.get(block_id) {
            Some(b) => b,
            None => return None,
        };
        block.get_point(idx)
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

    fn get_points(&self, indices: &Vec<NodeID>) -> Vec<&Point> {
        indices
            .iter()
            .map(|idx| self.get_point(*idx).unwrap())
            .collect()
    }

    fn get_points_iter<I>(&self, indices: I) -> impl Iterator<Item = &Point>
    where
        I: Iterator<Item = NodeID>,
    {
        indices.map(|idx| self.get_point(idx).unwrap())
    }

    fn extend(&mut self, mut other: BlockPoints) -> Vec<NodeID> {
        let mut ids = Vec::with_capacity(other.len());
        for block in other.collection.drain(..) {
            for point in block.block.data {
                ids.push(self.push(point));
            }
        }
        ids
    }
}

impl<'a> BlockPoints {
    pub fn iter_points<'b: 'a>(&'b self) -> impl Iterator<Item = &'a Point> {
        self.collection
            .iter()
            .flat_map(|block| block.block.data.iter())
    }
    pub fn iter_points_mut<'b: 'a>(&mut self) -> impl Iterator<Item = &mut Point> {
        self.collection
            .iter_mut()
            .flat_map(|block| block.block.data.iter_mut())
    }
}

pub const POINTS_HEADER_SIZE: usize = 7;
#[derive(Debug)]
pub struct PointsHeader {
    pub point_size: BlockID,
    pub nb_blocks: BlockID,
}

impl PointsHeader {
    pub fn new(points: &BlockPoints) -> Self {
        let point_size = points.get_point(0).unwrap().size() as BlockID;
        let nb_blocks = points.nb_blocks() as BlockID;
        PointsHeader {
            point_size,
            nb_blocks,
        }
    }

    pub fn from_path(file_path: PathBuf) -> Self {
        let mut file_handle = File::open(file_path.clone()).expect("Could not open points file");
        let mut data = [0; POINTS_HEADER_SIZE];
        file_handle.read(&mut data).unwrap();
        PointsHeader::deserialize(data.into())
    }
}

impl Serializer for PointsHeader {
    /// Val        Bytes
    /// quantized  1
    /// point_size 2
    /// nb_blocks  2
    fn size(&self) -> usize {
        POINTS_HEADER_SIZE
    }

    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.point_size).to_be_bytes());
        bytes.extend_from_slice(&(self.nb_blocks).to_be_bytes());
        bytes.extend_from_slice(&MAX_PER_BLOCK.to_be_bytes());
        bytes
    }

    fn deserialize(data: Vec<u8>) -> Self {
        let quantized = u8::from_be_bytes(data[..1].try_into().unwrap()) != 0;
        let point_size = BlockID::from_be_bytes(data[1..3].try_into().unwrap());
        let nb_blocks = BlockID::from_be_bytes(data[3..5].try_into().unwrap());
        PointsHeader {
            point_size,
            nb_blocks,
        }
    }
}

impl Serializer for BlockPoints {
    /// Val        Bytes
    /// header     header.size()
    /// blocks     header.nb_blocks * variable
    fn size(&self) -> usize {
        let mut total = 0;
        let header = PointsHeader::new(self);
        total += header.size();
        for block in self.collection.iter() {
            total += block.header.size();
            total += block.header.block_data_size();
        }
        total
    }

    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let header = PointsHeader::new(self);
        bytes.extend(header.serialize());

        for block in self.collection.iter() {
            bytes.extend(block.header.serialize());
            bytes.extend(block.block.serialize_block_data());
        }

        bytes
    }

    fn deserialize(data: Vec<u8>) -> Self {
        let mut i = 0;
        let header = PointsHeader::deserialize(data[i..i + POINTS_HEADER_SIZE].into());
        i += POINTS_HEADER_SIZE;

        let mut collection = Vec::new();
        for _ in 0..header.nb_blocks {
            let block_header = BlockHeader::deserialize(data[i..i + BLOCK_HEADER_SIZE].into());
            i += BLOCK_HEADER_SIZE;
            let block_data = BlockData::deserialize_block_data(
                &block_header,
                data[i..i + block_header.block_data_size()].into(),
            );
            i += block_header.block_data_size();
            let block = PointsBlock {
                header: block_header,
                block: block_data,
            };
            collection.push(block);
        }

        BlockPoints {
            collection,
            ml: 0.0, // TODO
        }
    }
}

#[cfg(test)]
mod test {

    use vectors::gen_rand_vecs;

    use super::*;

    fn gen_rand_points(dim: usize, n: usize) -> BlockPoints {
        let vectors = gen_rand_vecs(dim, n);
        BlockPoints::new(vectors, 0.5)
    }

    fn get_collection_data(points: &BlockPoints) -> (f32, f32, u32, usize) {
        let dist = points
            .get_point(32)
            .unwrap()
            .distance(points.get_point(64).unwrap());
        let val = points.get_point(16).unwrap().iter_vals().next().unwrap();
        (dist, val, points.len() as u32, points.size())
    }

    #[test]
    fn build_points_multi_block() {
        let points = gen_rand_points(4, 100_000);

        let point_0 = points.get_point(0).unwrap();
        assert_eq!(point_0.id, 0);

        let point_1 = points.get_point(1).unwrap();
        assert_eq!(point_1.id, 1);

        let point_256 = points.get_point(256).unwrap();
        assert_eq!(point_256.id, 256);

        let point_80_000 = points.get_point(80_000).unwrap();
        assert_eq!(point_80_000.id, 80_000);
    }

    #[test]
    fn distance_two_points() {
        let points = gen_rand_points(4, 16);

        let dist = points.distance(12, 4);
        assert!(dist.is_some());
        assert!(dist.unwrap() > 0.0);
    }

    #[test]
    fn serialization_multi_block() {
        let points = gen_rand_points(128, 100);
        let (dist, val, len, size) = get_collection_data(&points);

        let points_ser = points.serialize();
        let points_des = BlockPoints::deserialize(points_ser);
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
