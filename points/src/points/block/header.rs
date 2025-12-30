use vectors::serializer::Serializer;

use crate::points::block::BlockID;

pub const BLOCK_HEADER_SIZE: usize = 8;
#[derive(Clone, Debug)]
pub struct BlockHeader {
    pub id: BlockID,
    pub max_points: BlockID,
    pub nb_points: BlockID,
    pub point_size: BlockID,
}

impl BlockHeader {
    pub fn block_data_size(&self) -> usize {
        self.nb_points as usize * self.point_size as usize
    }
    pub fn block_size(&self) -> usize {
        self.block_data_size() + BLOCK_HEADER_SIZE
    }
}

impl Serializer for BlockHeader {
    fn size(&self) -> usize {
        BLOCK_HEADER_SIZE
    }

    fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.id.to_be_bytes());
        bytes.extend_from_slice(&self.max_points.to_be_bytes());
        bytes.extend_from_slice(&self.nb_points.to_be_bytes());
        bytes.extend_from_slice(&self.point_size.to_be_bytes());
        bytes
    }

    fn deserialize(data: Vec<u8>) -> Self {
        let id = BlockID::from_be_bytes(data[..2].try_into().unwrap());
        let max_points = u16::from_be_bytes(data[2..4].try_into().unwrap());
        let nb_points = u16::from_be_bytes(data[4..6].try_into().unwrap());
        let point_size = u16::from_be_bytes(data[6..].try_into().unwrap());
        BlockHeader {
            id,
            nb_points,
            max_points,
            point_size,
        }
    }
}

#[cfg(test)]
mod test {
    use vectors::serializer::Serializer;

    use crate::points::block::header::{BLOCK_HEADER_SIZE, BlockHeader};

    #[test]
    fn serialization() {
        let header = BlockHeader {
            id: 0,
            nb_points: 12,
            point_size: 256,
            max_points: 32,
        };

        assert_eq!(header.id, 0);
        assert_eq!(header.max_points, 32);
        assert_eq!(header.nb_points, 12);
        assert_eq!(header.point_size, 256);

        assert_eq!(header.size(), BLOCK_HEADER_SIZE);
        let ser = header.serialize();
        assert_eq!(ser.len(), BLOCK_HEADER_SIZE);
        let des = BlockHeader::deserialize(ser);
        assert_eq!(header.id, des.id);
        assert_eq!(header.nb_points, des.nb_points);
        assert_eq!(header.point_size, des.point_size);
    }
}
