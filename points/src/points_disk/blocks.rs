use std::fmt::Debug;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use graph::NodeID;

use vectors::serializer::Serializer;

use crate::point::Point;
use crate::points::Points;
use crate::simple_points::SimplePoints;

pub const BLOCK_CAPACITY: NodeID = 131072;

fn new_cell<T>(val: T) -> Arc<RwLock<T>> {
    Arc::new(RwLock::new(val))
}

#[derive(Debug, Clone)]
pub struct PointBlock {
    points: Arc<RwLock<Option<SimplePoints>>>,
    path: String,
    requests: Arc<RwLock<usize>>,
}

impl Points for PointBlock {
    /// The points are written to disk and offloaded from memory
    fn new(vecs: Vec<Vec<f32>>, ml: f32, idx_offset: NodeID, path: &Path) -> Self {
        if vecs.len() > BLOCK_CAPACITY as usize {
            panic!(
                "Cannot create PointBlock. It can only hold {BLOCK_CAPACITY}, but vecs contains {} vectors",
                vecs.len()
            )
        }
        let points = new_cell(Some(SimplePoints::new(vecs, ml, idx_offset, &path)));
        let block = PointBlock {
            points: points,
            requests: new_cell(0),
            path: path.to_string_lossy().to_string(),
        };
        block.offload();
        block
    }
    fn len(&self) -> usize {
        self.get_points().as_ref().unwrap().len()
    }
    fn distance2point(&self, point: &Point, idx: NodeID) -> Option<f32> {
        self.get_points()
            .as_ref()
            .unwrap()
            .distance2point(point, idx)
    }
    fn dim(&self) -> Option<usize> {
        self.get_points().as_ref().unwrap().dim()
    }
    fn get_point(&self, idx: NodeID) -> Option<Arc<Point>> {
        self.get_points().as_ref().unwrap().get_point(idx)
    }
    fn distance(&self, a_idx: NodeID, b_idx: NodeID) -> Option<f32> {
        self.get_points().as_ref().unwrap().distance(a_idx, b_idx)
    }
    fn push(&mut self, point: Point) -> NodeID {
        if self.len() < BLOCK_CAPACITY as usize {
            self.get_points_mut().as_mut().unwrap().push(point)
        } else {
            panic!("Could not add point to PointBlock, it would breach the limit")
        }
    }
    fn extend(&mut self, other: Self) -> Vec<NodeID> {
        let other_points = other.get_points().clone().unwrap();
        if self.len() + other.len() < BLOCK_CAPACITY as usize {
            self.get_points_mut().as_mut().unwrap().extend(other_points)
        } else {
            panic!("Could not extend PointBlock, it would breach the limit")
        }
    }
}

impl PointBlock {
    pub fn get_read_lock(&self) -> RwLockReadGuard<'_, Option<SimplePoints>> {
        if !self.is_loaded() {
            self.load();
        }
        self.points.read().unwrap()
    }
    pub fn new_empty(path: &Path) -> Self {
        PointBlock {
            points: new_cell(None),
            requests: new_cell(0),
            path: path.to_string_lossy().to_string(),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.points.read().unwrap().is_some()
    }

    pub fn load(&self) {
        let mut file = File::open(&self.path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        let points = SimplePoints::deserialize(buffer);
        self.points.write().unwrap().replace(points);
    }

    pub fn offload(&self) {
        self.write();
        *self.points.write().unwrap() = None;
        *self.requests.write().unwrap() = 0;
    }

    pub fn write(&self) {
        match self.points.read().unwrap().as_ref() {
            Some(p) => {
                let path = Path::new(&self.path);
                let mut file = File::options().write(true).create(true).open(path).unwrap();
                file.write_all(&p.serialize()).unwrap();
            }
            None => (),
        }
    }

    pub fn get_requests(&self) -> usize {
        *self.requests.read().unwrap()
    }

    fn get_points(&self) -> RwLockReadGuard<'_, Option<SimplePoints>> {
        *self.requests.write().unwrap() += 1;
        let guard = self.points.read().unwrap();
        if guard.is_some() {
            guard
        } else {
            drop(guard);
            self.load();
            self.points.read().unwrap()
        }
    }

    fn get_points_mut(&mut self) -> RwLockWriteGuard<'_, Option<SimplePoints>> {
        *self.requests.write().unwrap() += 1;
        self.points.write().unwrap()
    }
}

impl Serializer for PointBlock {
    fn size(&self) -> usize {
        self.get_points().as_ref().unwrap().size()
    }
    fn serialize(&self) -> Vec<u8> {
        self.get_points().as_ref().unwrap().serialize()
    }
    fn deserialize(data: Vec<u8>) -> Self {
        PointBlock {
            points: new_cell(Some(SimplePoints::deserialize(data))),
            requests: new_cell(0),
            path: String::new(),
        }
    }
}
