mod blocks;

use std::fmt::Debug;
use std::fs::{File, create_dir_all};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use graph::NodeID;
use vectors::VecBase;

use vectors::serializer::{BytesReader, Serializer};

use crate::point::Point;
use crate::points::Points;
use crate::points_disk::blocks::BLOCK_CAPACITY;
use blocks::PointBlock;

const MAX_LOADED: usize = 8;

#[derive(Debug, Clone)]
pub struct PointsDisk {
    pub blocks: Vec<PointBlock>,
    path: String,
    ml: f32,
}

impl PointsDisk {
    pub fn load_at(path: PathBuf) -> Self {
        if path.exists() & path.is_dir() {
            let nb_blocks = path.read_dir().iter().count();
            let blocks = Vec::from_iter(
                (0..nb_blocks).map(|idx| PointBlock::new_empty(&path.join(format!("{idx}")))),
            );
            let mut ml_string = String::new();
            File::open(path.join("ml.txt"))
                .unwrap()
                .read_to_string(&mut ml_string)
                .unwrap();
            let ml = f32::from_str(ml_string.as_str()).unwrap();
            Self {
                blocks,
                ml,
                path: path.to_string_lossy().to_string(),
            }
        } else {
            panic!("Could not load PointsDisk from {path:?}")
        }
    }

    fn write_all(&self) {
        let path = Path::new(&self.path);
        create_dir_all(path).unwrap();
        let mut ml_file = File::create_new(path.join("ml.txt")).unwrap();
        ml_file.write(&format!("{}", self.ml).into_bytes()).unwrap();
        self.write_all_blocks();
    }

    fn write_all_blocks(&self) {
        for block in self.blocks.iter() {
            block.write();
        }
    }

    fn should_offload(&self) -> bool {
        self.blocks.iter().filter(|b| b.is_loaded()).count() > MAX_LOADED
    }

    fn push_new_block(&mut self, point: Point) -> NodeID {
        let id = self.len() as NodeID;
        let next_block_idx = self.blocks.len();
        let path = Path::new(&self.path).join(format!("{next_block_idx}"));
        let new_block =
            PointBlock::new(vec![point.get_vals()], self.ml, self.len() as NodeID, &path);
        self.blocks.push(new_block);
        id
    }

    fn offload_block(&mut self, block_idx: NodeID) {
        match self.blocks.get_mut(block_idx as usize) {
            Some(block) => block.offload(),
            None => (),
        }
    }

    fn load_block(&mut self, block_idx: NodeID) {
        match self.blocks.get_mut(block_idx as usize) {
            Some(block) => {
                if !block.is_loaded() {
                    block.load();
                }
                self.offload()
            }
            None => panic!("Tried loading a block that doesn't exist ({block_idx})"),
        }
    }

    fn offload_least_requested(&mut self) {
        let idx_to_offload = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(_idx, b)| b.is_loaded())
            .min_by_key(|(_idx, b)| b.get_requests())
            .unwrap()
            .0;
        self.offload_block(idx_to_offload as NodeID);
    }

    fn offload(&mut self) {
        while self.should_offload() {
            self.offload_least_requested()
        }
    }
}

impl Points for PointsDisk {
    fn new(vecs: Vec<Vec<f32>>, ml: f32, idx_offset: NodeID, path: &Path) -> Self {
        if path.exists() & path.is_dir() {
            panic!("Cannot create new PointsDisk at {path:?}, it already exists")
        }
        create_dir_all(&path).unwrap();
        let len = vecs.len();
        let nb_blocks = (len as f32 / BLOCK_CAPACITY as f32).ceil() as usize;
        let mut vecs_iter = vecs.iter().cloned();
        let mut blocks = Vec::new();
        for block_idx in 0..nb_blocks {
            let block_vecs = vecs_iter.by_ref().take(BLOCK_CAPACITY as usize).collect();
            let block_offset = (block_idx * BLOCK_CAPACITY as usize) + idx_offset as usize;
            let block = PointBlock::new(
                block_vecs,
                ml,
                block_offset as NodeID,
                &path.join(format!("{block_idx}")),
            );
            blocks.push(block)
        }
        Self {
            blocks,
            ml,
            path: path.to_string_lossy().to_string(),
        }
    }

    fn get_point(&self, idx: NodeID) -> Option<&Point> {
        let block_idx = idx as usize / BLOCK_CAPACITY as usize;
        match self.blocks.get(block_idx) {
            Some(b) => b.get_point(idx),
            None => None,
        }
    }

    fn push(&mut self, point: Point) -> NodeID {
        let last_block = self.blocks.last_mut();
        match last_block {
            Some(b) => {
                if b.len() == BLOCK_CAPACITY as usize {
                    self.push_new_block(point)
                } else {
                    b.push(point)
                }
            }
            None => self.push_new_block(point),
        }
    }

    fn len(&self) -> usize {
        // Every block contains BLOCK_CAPACITY points,
        // except the last one
        let nb_blocks = self.blocks.len();
        if nb_blocks == 0 {
            0
        } else {
            let last_block_idx = nb_blocks - 1;
            (last_block_idx * BLOCK_CAPACITY as usize) + self.blocks[last_block_idx].len()
        }
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
        Self {
            blocks,
            ml,
            path: String::new(),
        }
    }
}

#[cfg(test)]
mod test {

    use vectors::gen_rand_vecs;

    use super::*;

    // One less block than allowed to load
    const N: NodeID = (BLOCK_CAPACITY * (MAX_LOADED as NodeID - 2)) + 2;

    // Three more blocks than allowed to load
    const N_OFFLOADED: NodeID = (BLOCK_CAPACITY * (MAX_LOADED as NodeID + 2)) + 2;

    fn gen_rand_points(dim: usize, n: NodeID) -> PointsDisk {
        let vectors = gen_rand_vecs(dim, n as usize);
        let id = format!("{}", uuid::Uuid::new_v4());
        let path = Path::new("./test_data").join(id);
        PointsDisk::new(vectors, 0.5, 0, &path)
    }

    fn get_collection_data(points: &PointsDisk) -> (f32, f32) {
        let dist = points
            .get_point(0)
            .unwrap()
            .distance(points.get_point(BLOCK_CAPACITY).unwrap());
        let val = points
            .get_point((BLOCK_CAPACITY * 2) + 1)
            .unwrap()
            .iter_vals()
            .next()
            .unwrap();
        (dist, val)
    }

    fn assert_can_get_point(points: &PointsDisk, idx: NodeID) {
        let p = points.get_point(idx).unwrap();
        assert_eq!(p.id, idx);
    }

    fn assert_can_get_points(points: &PointsDisk) {
        let max_idx = points.len() as NodeID - 1;
        assert_can_get_point(&points, 0);
        assert_can_get_point(&points, 1);
        assert_can_get_point(&points, BLOCK_CAPACITY - 2);
        assert_can_get_point(&points, BLOCK_CAPACITY - 1);
        assert_can_get_point(&points, BLOCK_CAPACITY);
        assert_can_get_point(&points, BLOCK_CAPACITY + 1);
        assert_can_get_point(&points, BLOCK_CAPACITY + 2);
        assert_can_get_point(&points, max_idx - 1);
        assert_can_get_point(&points, max_idx);
    }

    #[test]
    fn build_points_multiblock() {
        let points = gen_rand_points(4, N);

        assert_eq!(points.blocks.len(), MAX_LOADED - 1);
        assert_eq!(points.len(), N as usize);

        assert_can_get_points(&points);

        let idx = BLOCK_CAPACITY * 10;
        let point = points.get_point(idx);
        assert!(point.is_none());
    }

    #[test]
    fn build_points_offloaded() {
        let path = Path::new("./points_disk_test/").to_path_buf();
        if path.exists() {
            std::fs::remove_dir_all(path.clone()).unwrap();
        }

        let vectors = gen_rand_vecs(4, N_OFFLOADED as usize);
        let p_disk = PointsDisk::new(vectors, 0.5, 0, &path);

        // I decided to have the Disk version completely offloaded
        // on creation by default. It should load points from
        // disk when necessary.
        let loaded_blocks = p_disk.blocks.iter().filter(|b| b.is_loaded()).count();
        assert_eq!(loaded_blocks, 0);

        assert_can_get_points(&p_disk);

        let loaded_blocks = p_disk.blocks.iter().filter(|b| b.is_loaded()).count();
        assert!(loaded_blocks > 0);

        if path.exists() {
            std::fs::remove_dir_all(path.clone()).unwrap();
        }
    }

    #[test]
    fn distance_two_points_same_block() {
        let points = gen_rand_points(4, N);
        let dist = points.distance(BLOCK_CAPACITY, BLOCK_CAPACITY + 2);
        assert!(dist.unwrap() > 0.0);
    }

    #[test]
    fn distance_two_points_different_blocks() {
        let points = gen_rand_points(4, N);
        let dist = points.distance(0, BLOCK_CAPACITY + 1);
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
