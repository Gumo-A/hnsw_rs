pub mod data;
pub mod header;

use crate::{
    point::Point,
    points::block::{
        data::BlockData,
        header::{BLOCK_HEADER_SIZE, BlockHeader},
    },
};
use graph::nodes::NodeID;
use vectors::{VecTrait, serializer::Serializer};

pub type BlockID = u16;

#[derive(Clone, Debug)]
pub struct PointsBlock<T: VecTrait> {
    pub header: BlockHeader,
    pub block: BlockData<T>,
}

impl<T: VecTrait> PointsBlock<T> {
    pub fn new(id: BlockID, max_points: BlockID) -> Self {
        let header = BlockHeader {
            id,
            max_points,
            nb_points: 0,
            point_size: 0,
        };
        let block = BlockData::new(max_points);

        Self { header, block }
    }

    pub fn from_parts(header: BlockHeader, block: BlockData<T>) -> Self {
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
        self.block.len() as u16 == self.header.max_points
    }

    pub fn add_point(&mut self, point: &Point<T>) -> Option<NodeID> {
        if self.is_full() {
            return None;
        }

        if self.block.len() == 0 {
            self.header.point_size = point.size() as u16;
        }

        let mut point = point.clone();
        let point_id = (self.header.id as u32 * self.header.max_points as u32) + self.len() as u32;
        point.id = point_id;
        self.block.add_point(point);
        self.header.nb_points += 1;

        Some(point_id)
    }

    pub fn get_point(&self, idx: NodeID) -> Option<&Point<T>> {
        self.block.get_point(idx)
    }
}

impl<T: VecTrait> Serializer for PointsBlock<T> {
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

    use graph::nodes::NodeID;
    use vectors::{FullVec, VecBase, gen_rand_vecs, serializer::Serializer};

    use crate::{
        point::Point,
        points::block::{BlockID, PointsBlock},
    };

    const MAX_POINTS: BlockID = 16;
    const N: usize = 128;

    fn gen_rand_block(id: usize, n: usize) -> PointsBlock<FullVec> {
        let vectors = gen_rand_vecs(4, n);
        let mut block = PointsBlock::new(id as BlockID, MAX_POINTS);
        for v in vectors.iter() {
            let point = Point::new(v);
            block.add_point(&point);
        }
        block
    }

    #[test]
    fn nb_points() {
        let block = gen_rand_block(0, N * 2);
        assert_eq!(block.len(), MAX_POINTS as usize);

        let block = gen_rand_block(0, N);
        assert_eq!(block.len(), MAX_POINTS as usize);

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
    fn point_ids_from_block(block: PointsBlock<FullVec>) {
        let block_id = block.header.id;
        let n = block.len();

        for idx in 0..n {
            let idx = (block_id as usize * block.header.max_points as usize) + idx;
            let point_option = block.get_point(idx as NodeID);

            assert!(point_option.is_some());
            let point = point_option.unwrap();
            assert_eq!(idx, point.id as usize);
        }
    }

    #[test]
    fn serialization() {
        let block = gen_rand_block(1, MAX_POINTS as usize);
        let ser = block.serialize();
        let block_des = PointsBlock::deserialize(ser);

        assert_eq!(block.dim(), block_des.dim());
        assert_eq!(block.len(), block_des.len());
        assert_eq!(block.is_full(), block_des.is_full());

        let id = MAX_POINTS as NodeID + 2;
        assert!(block.get_point(id).is_some());
        assert_eq!(
            block.get_point(id).unwrap().get_low_vector(),
            block_des.get_point(id).unwrap().get_low_vector()
        );
    }
}
