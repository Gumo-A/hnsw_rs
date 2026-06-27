use graph::nodes::NodeID;
use vectors::{VecBase, VecTrait, serializer::Serializer};

use crate::{
    point::Point,
    points::block::{BlockID, MAX_PER_BLOCK, header::BlockHeader},
};

#[derive(Clone, Debug)]
pub struct BlockData {
    pub data: Vec<Point>,
}

impl BlockData {
    pub fn new() -> Self {
        BlockData { data: Vec::new() }
    }

    pub fn dim(&self) -> usize {
        self.data.first().unwrap().dim()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn point_ids(&self) -> Vec<NodeID> {
        self.data.iter().map(|p| p.id).collect()
    }

    pub fn add_point(&mut self, point: Point) {
        if self.len() < MAX_PER_BLOCK.into() {
            self.data.push(point);
        }
    }

    pub fn get_point(&self, idx: NodeID) -> Option<&Point> {
        let pos = idx % MAX_PER_BLOCK as NodeID;
        self.data.get(pos as usize)
    }

    pub fn serialize_block_data(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for point in self.data.iter() {
            bytes.extend(point.serialize());
        }
        bytes
    }

    pub fn deserialize_block_data(header: &BlockHeader, data: Vec<u8>) -> Self {
        let block_id = header.id as usize;
        let nb_points = header.nb_points as usize;
        let point_size = header.point_size as usize;
        let block_adder: usize = block_id * MAX_PER_BLOCK as usize;

        let mut points = Vec::new();
        let mut i = 0;
        for idx in 0..nb_points {
            let point_serialized = data[i..i + point_size].into();
            let mut point = Point::deserialize(point_serialized);
            point.id = (idx + block_adder) as NodeID;
            points.push(point);
            i += point_size;
        }
        Self { data: points }
    }
}

#[cfg(test)]
mod test {

    use graph::nodes::NodeID;
    use vectors::{FullVec, VecBase, gen_rand_vecs, serializer::Serializer};

    use crate::{
        point::Point,
        points::block::{BlockID, MAX_PER_BLOCK, data::BlockData, header::BlockHeader},
    };

    #[test]
    fn max_points() {
        let vectors = gen_rand_vecs(4, (MAX_PER_BLOCK as usize * 2) + 16);
        let mut block: BlockData = BlockData::new();
        for v in vectors {
            let point = Point::new(&v);
            block.add_point(point);
        }
        assert_eq!(block.len(), MAX_PER_BLOCK as usize);
    }

    #[test]
    fn pos_from_ids() {
        let n = 8;
        let vectors = gen_rand_vecs(4, n);
        let mut block = BlockData::new();
        for v in vectors.iter() {
            let point = Point::new_with(0, v);
            block.add_point(point);
        }
        for idx in 0..n {
            let point = block.get_point(idx as NodeID);
            assert!(point.is_some());
            assert_eq!(
                &point.unwrap().get_vals(),
                vectors.get(idx as usize).unwrap()
            );
        }
    }

    #[test]
    fn serialization() {
        let n = 32;
        let vectors = gen_rand_vecs(4, n);
        let mut block = BlockData::new();
        for v in vectors {
            let point = Point::new(&v);

            // block data doesnt care about IDs,
            // so it doesnt change the point IDs
            assert_eq!(point.id, 0);

            block.add_point(point);
        }

        assert_block_with_id(0, &block);
        assert_block_with_id(1, &block);
        assert_block_with_id(6, &block);
        assert_block_with_id(4096, &block);
    }

    fn assert_block_with_id(block_id: usize, block: &BlockData) {
        let header = BlockHeader {
            id: block_id as BlockID,
            nb_points: block.len() as BlockID,
            point_size: block.get_point(0).unwrap().size() as BlockID,
        };

        let ser = block.serialize_block_data();
        let des = BlockData::deserialize_block_data(&header, ser);
        assert_eq!(block.len(), des.len());

        for idx in 0..header.nb_points as usize {
            let point = block.get_point(idx as u32).unwrap();
            let point_des = des.get_point(idx as u32).unwrap();
            assert_eq!(point.get_vals(), point_des.get_vals());

            let correct_id = (block_id * MAX_PER_BLOCK as usize) + idx;
            assert_eq!(point_des.id, correct_id as NodeID);
        }
    }
}
