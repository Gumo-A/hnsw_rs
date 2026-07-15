pub mod block;
use block::data::BlockData;
use block::header::{BLOCK_HEADER_SIZE, BlockHeader};
use block::{BlockID, MAX_PER_BLOCK, PointsBlock};
use graph::NodeID;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use vectors::VecBase;
use vectors::serializer::Serializer;

use crate::point::Point;
use crate::points::{Points, new_layer};

#[derive(Debug, Clone)]
pub struct BlockPoints {
    ml: f32,
    pub collection: Vec<PointsBlock>,
}

impl BlockPoints {
    fn nb_blocks(&self) -> usize {
        self.collection.len()
    }
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
        let mut rng = StdRng::seed_from_u64(0);
        let mut block_idx = 0;
        let mut block = PointsBlock::new(block_idx);
        for (idx, v) in vecs.drain(..).enumerate() {
            let level = new_layer(ml, &mut rng);
            let point = Point::with_level_and_id(&v, level, idx);
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

    fn ids(&self) -> impl Iterator<Item = NodeID> {
        self.collection.iter().flat_map(|p| p.point_ids())
    }

    fn dim(&self) -> Option<usize> {
        match self.collection.first() {
            Some(p) => Some(p.dim()),
            None => None,
        }
    }

    fn push(&mut self, mut point: Point) -> NodeID {
        point.id = self.len() as NodeID;
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
