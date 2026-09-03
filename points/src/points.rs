use std::path::Path;

use crate::point::Point;
use graph::NodeID;
use vectors::VecBase;

pub trait Points {
    fn new(vecs: Vec<Vec<f32>>, ml: f32, idx_offset: NodeID, path: &Path) -> Self;
    fn len(&self) -> usize;
    fn ids(&self) -> impl Iterator<Item = NodeID>;
    fn dim(&self) -> Option<usize>;
    fn push(&mut self, point: Point) -> NodeID;
    fn get_point(&self, idx: NodeID) -> Option<&Point>;
    // fn remove(&mut self, index: Node) -> bool;

    // extends collection and returns all new ids
    fn extend(&mut self, other: Self) -> Vec<NodeID>
    where
        Self: Sized,
    {
        let mut ids = Vec::new();
        for idx in other.ids() {
            ids.push(self.push(other.get_point(idx).unwrap().clone()));
        }
        ids
    }

    fn get_points_iter<I>(&self, indices: I) -> impl Iterator<Item = &Point>
    where
        I: Iterator<Item = NodeID>,
    {
        indices.map(|idx| self.get_point(idx).unwrap())
    }

    fn iter_points<I>(&self) -> impl Iterator<Item = &Point> {
        self.ids().map(|id| self.get_point(id).unwrap())
    }

    fn distance(&self, a_idx: NodeID, b_idx: NodeID) -> Option<f32> {
        let point_a = self.get_point(a_idx);
        let point_b = self.get_point(b_idx);
        match (point_a, point_b) {
            (Some(a), Some(b)) => Some(a.dist2other(b)),
            _ => None,
        }
    }

    fn distance2point(&self, point: &Point, idx: NodeID) -> Option<f32> {
        let other = self.get_point(idx);
        match other {
            Some(b) => Some(point.dist2other(b)),
            _ => None,
        }
    }
}
