use std::fmt::Debug;
use std::fs::{File, create_dir_all};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use graph::NodeID;
use vectors::VecBase;

use vectors::serializer::{BytesReader, Serializer};

use crate::point::Point;
use crate::points::{Points, SimplePoints};

const MAX_LOADED: usize = 8;
const BLOCK_CAPACITY: NodeID = 2048;

#[derive(Debug, Clone)]
pub struct PointBlock {
    points: Option<SimplePoints>,
    requests: Arc<Mutex<usize>>,
}

impl PointBlock {
    pub fn new_empty() -> Self {
        PointBlock {
            points: None,
            requests: Arc::new(Mutex::new(0)),
        }
    }

    fn get_points(&self) -> &SimplePoints {
        *self.requests.lock().unwrap() += 1;
        match self.points.as_ref() {
            Some(p) => p,
            None => {
                todo!("load the points from disk")
            }
        }
    }
    fn get_points_mut(&mut self) -> &mut SimplePoints {
        *self.requests.lock().unwrap() += 1;
        match self.points.as_mut() {
            Some(p) => p,
            None => {
                todo!("load the points from disk")
            }
        }
    }
}

impl Points for PointBlock {
    fn new(vecs: Vec<Vec<f32>>, ml: f32, idx_offset: NodeID) -> Self {
        if vecs.len() > BLOCK_CAPACITY as usize {
            panic!(
                "Cannot create PointBlock. It can only hold {BLOCK_CAPACITY}, but vecs contains {} vectors",
                vecs.len()
            )
        }
        PointBlock {
            points: Some(SimplePoints::new(vecs, ml, idx_offset)),
            requests: Arc::new(Mutex::new(0)),
        }
    }
    fn len(&self) -> usize {
        let points = self.get_points();
        points.len()
    }
    fn distance2point(&self, point: &Point, idx: NodeID) -> Option<f32> {
        let points = self.get_points();
        points.distance2point(point, idx)
    }
    fn ids(&self) -> impl Iterator<Item = NodeID> {
        let points = self.get_points();
        points.ids()
    }
    fn dim(&self) -> Option<usize> {
        let points = self.get_points();
        points.dim()
    }
    fn get_point(&self, idx: NodeID) -> Option<&Point> {
        let points = self.get_points();
        points.get_point(idx)
    }
    fn get_points_iter<I>(&self, indices: I) -> impl Iterator<Item = &Point>
    where
        I: Iterator<Item = NodeID>,
    {
        let points = self.get_points();
        points.get_points_iter(indices)
    }
    fn distance(&self, a_idx: NodeID, b_idx: NodeID) -> Option<f32> {
        let points = self.get_points();
        points.distance(a_idx, b_idx)
    }
    fn push(&mut self, point: Point) -> NodeID {
        let points = self.get_points_mut();
        if points.len() < BLOCK_CAPACITY as usize {
            points.push(point)
        } else {
            panic!("Could not add point to PointBlock, it would breach the limit")
        }
    }
    fn extend(&mut self, other: Self) -> Vec<NodeID> {
        let points = self.get_points_mut();
        let other_points = other.get_points().clone();
        if points.len() + other_points.len() < BLOCK_CAPACITY as usize {
            points.extend(other_points)
        } else {
            panic!("Could not extend PointBlock, it would breach the limit")
        }
    }
}

impl Serializer for PointBlock {
    fn size(&self) -> usize {
        self.get_points().size()
    }
    fn serialize(&self) -> Vec<u8> {
        self.get_points().serialize()
    }
    fn deserialize(data: Vec<u8>) -> Self {
        PointBlock {
            points: Some(SimplePoints::deserialize(data)),
            requests: Arc::new(Mutex::new(0)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PointsDisk {
    pub blocks: Vec<PointBlock>,
    path: String,
    ml: f32,
}

impl PointsDisk {
    pub fn new_at(vecs: Vec<Vec<f32>>, ml: f32, idx_offset: NodeID, path: PathBuf) -> Self {
        let mut points = Self::new(vecs, ml, idx_offset);
        if path.exists() & path.is_dir() {
            if path.read_dir().iter().count() > 0 {
                panic!("Cannot create new PointsDisk at {path:?}, it exists and is not empty")
            }
        }
        create_dir_all(path.clone()).unwrap();
        let mut ml_file = File::create_new(path.join("ml.txt")).unwrap();
        ml_file.write(&format!("{ml}").into_bytes()).unwrap();
        points.path = path.to_string_lossy().to_string();
        points
    }

    pub fn load_at(path: PathBuf) -> Self {
        if path.exists() & path.is_dir() {
            let nb_blocks = path.read_dir().iter().count();
            let blocks = Vec::from_iter((0..nb_blocks).map(|_| PointBlock::new_empty()));
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

    fn should_offload(&self) -> bool {
        self.blocks.iter().filter(|b| b.points.is_some()).count() > MAX_LOADED
    }

    fn push_new_block(&mut self, point: Point) -> NodeID {
        let id = self.len() as NodeID;
        let new_block = PointBlock::new(vec![point.get_vals()], self.ml, self.len() as NodeID);
        self.blocks.push(new_block);
        id
    }

    fn offload_block(&mut self, block_idx: NodeID) {
        match self.blocks.get_mut(block_idx as usize) {
            Some(block) => {
                block.points = None;
                block.requests = Arc::new(Mutex::new(0));
            }
            None => (),
        }
    }

    fn load_block(&mut self, block_idx: NodeID) {
        match self.blocks.get_mut(block_idx as usize) {
            Some(block) => {
                if block.points.is_none() {
                    let path = Path::new(&self.path).join(format!("{block_idx}"));
                    let mut file = File::open(path).unwrap();
                    let mut buffer = Vec::new();
                    file.read_to_end(&mut buffer).unwrap();
                    let points = SimplePoints::deserialize(buffer);
                    block.points = Some(points);
                }
                while self.should_offload() {
                    let idx_to_offload = self
                        .blocks
                        .iter()
                        .enumerate()
                        .filter(|(_idx, b)| b.points.is_some())
                        .min_by_key(|(_idx, b)| *b.requests.lock().unwrap())
                        .unwrap()
                        .0;
                    self.offload_block(idx_to_offload as NodeID);
                }
            }
            None => panic!("Tried loading a block that doesn't exist ({block_idx})"),
        }
    }
}

impl Points for PointsDisk {
    fn new(vecs: Vec<Vec<f32>>, ml: f32, idx_offset: NodeID) -> Self {
        let len = vecs.len();
        let nb_blocks = (len as f32 / BLOCK_CAPACITY as f32).ceil() as usize;
        let mut vecs_iter = vecs.iter().cloned();
        let mut blocks = Vec::new();
        for block_idx in 0..nb_blocks {
            let block_vecs = vecs_iter.by_ref().take(BLOCK_CAPACITY as usize).collect();
            let block_offset = (block_idx * BLOCK_CAPACITY as usize) + idx_offset as usize;
            let block = PointBlock::new(block_vecs, ml, block_offset as NodeID);
            blocks.push(block)
        }
        let path = Path::new("").to_string_lossy().to_string();
        Self { blocks, ml, path }
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

    const N: NodeID = (BLOCK_CAPACITY * 2) + 2;

    fn gen_rand_points(dim: usize, n: NodeID) -> PointsDisk {
        let vectors = gen_rand_vecs(dim, n as usize);
        PointsDisk::new(vectors, 0.5, 0)
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

    #[test]
    fn build_simple_points() {
        let points = gen_rand_points(4, N);

        assert_eq!(points.blocks.len(), 3);
        assert_eq!(points.len(), N as usize);

        let point_0 = points.get_point(0).unwrap();
        assert_eq!(point_0.id, 0);

        let point_1 = points.get_point(1).unwrap();
        assert_eq!(point_1.id, 1);

        let idx = BLOCK_CAPACITY - 1;
        let point = points.get_point(idx).unwrap();
        assert_eq!(point.id, idx);

        let idx = BLOCK_CAPACITY;
        let point = points.get_point(idx).unwrap();
        assert_eq!(point.id, idx);

        let idx = BLOCK_CAPACITY + 1;
        let point = points.get_point(idx).unwrap();
        assert_eq!(point.id, idx);

        let idx = BLOCK_CAPACITY + 2;
        let point = points.get_point(idx).unwrap();
        assert_eq!(point.id, idx);

        let idx = BLOCK_CAPACITY * 10;
        let point = points.get_point(idx);
        assert!(point.is_none());
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
