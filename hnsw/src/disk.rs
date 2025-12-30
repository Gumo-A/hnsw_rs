use std::{
    collections::HashMap,
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use graph::{graph::Graph, nodes::NodeID};
use points::{
    point::Point,
    points::{
        block::{
            data::BlockData,
            header::{BlockHeader, BLOCK_HEADER_SIZE},
            BlockID, PointsBlock,
        },
        PointsHeader, POINTS_HEADER_SIZE,
    },
};
use vectors::{serializer::Serializer, VecTrait};

struct HNSWDisk<T: VecTrait> {
    points_handler: PointsDisk<T>,
    layers_handler: LayersDisk,
}

impl<T: VecTrait> HNSWDisk<T> {
    pub fn new(dir: &Path) -> Self {
        HNSWDisk {
            points_handler: PointsDisk::new(dir.join("points")),
            layers_handler: LayersDisk { buffer: Vec::new() },
        }
    }

    pub fn get_point(&mut self, idx: NodeID) -> Option<&Point<T>> {
        self.points_handler.get_point(idx)
    }
}

struct PointsDisk<T: VecTrait> {
    file_handle: File,
    buffer: HashMap<BlockID, PointsBlock<T>>,
    header: PointsHeader,
}

impl<T: VecTrait> PointsDisk<T> {
    fn new(file_path: PathBuf) -> Self {
        let file_handle = File::open(file_path.clone()).expect("Could not open points file");
        PointsDisk {
            file_handle,
            header: PointsHeader::from_path(file_path),
            buffer: HashMap::new(),
        }
    }

    fn get_point(&mut self, idx: NodeID) -> Option<&Point<T>> {
        let block_idx = self.determine_block(idx);
        if !self.buffer.contains_key(&block_idx) {
            self.load_block(block_idx)
        }
        let block = self.buffer.get(&block_idx).unwrap();
        let point = block.get_point(idx);
        return point;
    }

    fn load_block(&mut self, block_id: BlockID) {
        let offset = self.determine_offset(block_id);
        let block = self.read_block_from_offset(offset);
        self.buffer.insert(block.header.id, block);
    }

    fn determine_block(&self, idx: NodeID) -> BlockID {
        (idx as f32 / self.header.max_per_block as f32).floor() as BlockID
    }

    fn determine_offset(&self, block_id: BlockID) -> usize {
        let point_header_size = POINTS_HEADER_SIZE;
        let block_data_size = self.header.max_per_block as usize * self.header.point_size as usize;
        let block_size = BLOCK_HEADER_SIZE + block_data_size;
        point_header_size + (block_id as usize * block_size)
    }

    fn read_block_from_offset(&self, offset: usize) -> PointsBlock<T> {
        let mut header_bytes = [0u8; BLOCK_HEADER_SIZE];
        self.file_handle
            .read_exact_at(&mut header_bytes, offset as u64)
            .expect("Could not read block header");
        let header = BlockHeader::deserialize(header_bytes.into());

        let block_size = header.block_data_size();
        let mut block_bytes = vec![0u8; block_size];
        self.file_handle
            .read_exact_at(&mut block_bytes, (offset + BLOCK_HEADER_SIZE) as u64)
            .expect("Could not read block data");

        let block = BlockData::deserialize_block_data(&header, block_bytes.into());

        PointsBlock::from_parts(header, block)
    }
}

struct LayersDisk {
    buffer: Vec<GraphDisk>,
}

struct GraphDisk {
    file_handle: File,
    buffer: Vec<Graph>,
}

#[cfg(test)]
mod test {
    use std::path::Path;

    use points::points::block::BlockID;
    use vectors::FullVec;

    use crate::{disk::HNSWDisk, template::make_rand_index_full};

    const MAX_PER_BLOCK: BlockID = 32;
    const N: usize = 4;
    const DIM: usize = 2;

    #[test]
    fn read_points_disk() {
        let name = Path::new("read_point_test");
        if name.exists() {
            std::fs::remove_dir_all(name).unwrap();
        }
        let index = make_rand_index_full(N, DIM, MAX_PER_BLOCK);
        index.save(name);
        let mut disk_index: HNSWDisk<FullVec> = HNSWDisk::new(name);

        let mem_point = index.get_point(3).unwrap();
        let disk_point = disk_index.get_point(3).unwrap();

        dbg!(&mem_point);
        dbg!(&disk_point);
        assert_eq!(mem_point.get_low_vector(), disk_point.get_low_vector());
    }
}
