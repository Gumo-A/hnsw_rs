use std::fmt::Debug;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

use graph::NodeID;

use vectors::serializer::Serializer;

use crate::point::Point;
use crate::points::Points;
use crate::simple_points::SimplePoints;

pub const BLOCK_CAPACITY: NodeID = 2048;

#[derive(Debug, Clone)]
pub struct PointBlock {
    points: Option<SimplePoints>,
    path: String,
    requests: Arc<Mutex<usize>>,
}

impl Points for PointBlock {
    fn new(vecs: Vec<Vec<f32>>, ml: f32, idx_offset: NodeID, path: &Path) -> Self {
        if vecs.len() > BLOCK_CAPACITY as usize {
            panic!(
                "Cannot create PointBlock. It can only hold {BLOCK_CAPACITY}, but vecs contains {} vectors",
                vecs.len()
            )
        }
        let mut block = PointBlock {
            points: Some(SimplePoints::new(vecs, ml, idx_offset, &path)),
            requests: Arc::new(Mutex::new(0)),
            path: path.to_string_lossy().to_string(),
        };
        block.write();
        block.offload();
        block
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

impl PointBlock {
    pub fn new_empty(path: &Path) -> Self {
        PointBlock {
            points: None,
            requests: Arc::new(Mutex::new(0)),
            path: path.to_string_lossy().to_string(),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.points.is_some()
    }

    pub fn load(&mut self) {
        let mut file = File::open(&self.path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        let points = SimplePoints::deserialize(buffer);
        self.points = Some(points);
    }

    pub fn offload(&mut self) {
        self.points = None;
        self.requests = Arc::new(Mutex::new(0));
    }

    pub fn write(&self) {
        match self.points.as_ref() {
            Some(p) => {
                let path = Path::new(&self.path);
                let mut file = File::options().write(true).create(true).open(path).unwrap();
                file.write_all(&p.serialize()).unwrap();
            }
            None => (),
        }
    }

    pub fn get_requests(&self) -> usize {
        *self.requests.lock().unwrap()
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
            path: String::new(),
        }
    }
}
