pub mod data;
pub mod header;

use crate::{
    point::Point,
    points::block_points::block::{
        data::BlockData,
        header::{BLOCK_HEADER_SIZE, BlockHeader},
    },
};
use graph::NodeID;
use vectors::serializer::Serializer;

pub const MAX_PER_BLOCK: usize = 32;
pub type BlockID = u16;

#[derive(Clone, Debug)]
pub struct PointsBlock {
    pub header: BlockHeader,
    pub block: BlockData,
}

impl PointsBlock {
    pub fn new(id: BlockID) -> Self {
        let header = BlockHeader {
            id,
            nb_points: 0,
            point_size: 0,
        };
        let block = BlockData::new();

        Self { header, block }
    }

    pub fn from_parts(header: BlockHeader, block: BlockData) -> Self {
        Self { header, block }
    }

    pub fn dim(&self) -> usize {
        self.block.dim()
    }

    pub fn len(&self) -> usize {
        self.block.len()
    }

    pub fn point_ids(&self) -> Vec<NodeID> {
        self.block.point_ids()
    }

    pub fn is_full(&self) -> bool {
        self.block.len() == MAX_PER_BLOCK
    }

    pub fn add_point(&mut self, point: &Point) -> Option<NodeID> {
        if self.is_full() {
            return None;
        }

        if self.block.len() == 0 {
            self.header.point_size = point.size() as u16;
        }

        let mut point = point.clone();
        let point_id = (self.header.id as u32 * MAX_PER_BLOCK as u32) + self.len() as u32;
        point.id = point_id;
        self.block.add_point(point);
        self.header.nb_points += 1;

        Some(point_id)
    }

    pub fn get_point(&self, idx: NodeID) -> Option<&Point> {
        self.block.get_point(idx)
    }
}

impl Serializer for PointsBlock {
    fn size(&self) -> usize {
        BLOCK_HEADER_SIZE + (self.header.block_data_size())
    }

    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.header.serialize());
        bytes.extend_from_slice(&self.block.serialize_block_data());
        bytes
    }

    fn deserialize(data: Vec<u8>) -> Self {
        let header = BlockHeader::deserialize(data[..BLOCK_HEADER_SIZE].into());
        let block = BlockData::deserialize_block_data(
            &header,
            data[BLOCK_HEADER_SIZE..].try_into().unwrap(),
        );

        PointsBlock { header, block }
    }
}

#[cfg(test)]
mod test {

    use graph::NodeID;
    use vectors::{VecBase, gen_rand_vecs, serializer::Serializer};

    use crate::{
        point::Point,
        points::block_points::block::{BlockID, MAX_PER_BLOCK, PointsBlock},
    };

    const N: usize = 128;

    fn gen_rand_block(id: usize, n: usize) -> PointsBlock {
        let vectors = gen_rand_vecs(4, n);
        let mut block = PointsBlock::new(id as BlockID);
        for v in vectors.iter() {
            let point = Point::new(v);
            block.add_point(&point);
        }
        block
    }

    #[test]
    fn nb_points() {
        let block = gen_rand_block(0, N * 2);
        assert_eq!(block.len(), MAX_PER_BLOCK as usize);

        let block = gen_rand_block(0, N);
        assert_eq!(block.len(), MAX_PER_BLOCK as usize);

        let block = gen_rand_block(0, 4);
        assert_eq!(block.len(), 4);
    }

    #[test]
    fn point_ids() {
        let n = 16;
        point_ids_from_block(gen_rand_block(0, n));
        point_ids_from_block(gen_rand_block(1, n));
        point_ids_from_block(gen_rand_block(256, n));
    }

    // A point should be given an ID equal to the Block's ID times MAX_PER_BLOCK
    // plus the point's position in the block.
    // The point holds this value when it is in memory,
    // but it doesn't store it on disk, because we can infer it based on
    // its block's ID and its position within the block.
    fn point_ids_from_block(block: PointsBlock) {
        let block_id = block.header.id;
        let n = block.len();

        for idx in 0..n {
            let idx = (block_id as usize * MAX_PER_BLOCK as usize) + idx;
            let point_option = block.get_point(idx as NodeID);

            assert!(point_option.is_some());
            let point = point_option.unwrap();
            assert_eq!(idx, point.id as usize);
        }
    }

    #[test]
    fn serialization() {
        let block = gen_rand_block(1, MAX_PER_BLOCK as usize);
        let ser = block.serialize();
        let block_des = PointsBlock::deserialize(ser);

        assert_eq!(block.dim(), block_des.dim());
        assert_eq!(block.len(), block_des.len());
        assert_eq!(block.is_full(), block_des.is_full());

        let id = MAX_PER_BLOCK as NodeID + 2;
        assert!(block.get_point(id).is_some());
        assert_eq!(
            block.get_point(id).unwrap().get_vals(),
            block_des.get_point(id).unwrap().get_vals()
        );
    }
}
