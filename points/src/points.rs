use std::{path::Path, sync::Arc};

use crate::point::Point;
use graph::NodeID;
use vectors::VecBase;

pub trait Points {
    fn new(vecs: Vec<Vec<f32>>, ml: f32, idx_offset: NodeID, path: &Path) -> Self;
    fn len(&self) -> usize;
    fn dim(&self) -> Option<usize>;
    fn push(&mut self, point: Point) -> NodeID;
    fn get_point(&self, idx: NodeID) -> Option<Arc<Point>>;
    // fn remove(&mut self, index: Node) -> bool;

    // extends collection and returns all new ids
    fn extend(&mut self, other: Self) -> Vec<NodeID>;

    fn distance(&self, a_idx: NodeID, b_idx: NodeID) -> Option<f32> {
        let point_a = self.get_point(a_idx);
        let point_b = self.get_point(b_idx);
        match (point_a, point_b) {
            (Some(a), Some(b)) => Some(a.dist2other(&b)),
            _ => None,
        }
    }

    fn distance2point(&self, point: &Point, idx: NodeID) -> Option<f32> {
        let other = self.get_point(idx);
        match other {
            Some(b) => Some(point.dist2other(&b)),
            _ => None,
        }
    }
}
