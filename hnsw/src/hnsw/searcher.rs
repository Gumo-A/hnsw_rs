use std::collections::BTreeMap;

use nohash_hasher::IntSet;

use super::{dist::Dist, points::Point};

pub struct Searcher<'p> {
    ep: usize,
    point: &'p Point,
    seen: IntSet<usize>,
    candidates: BTreeMap<Dist, usize>,
    selected: BTreeMap<Dist, usize>,
}

impl<'p> Searcher<'p> {
    pub fn init(&mut self, point: &Point) {
        self.point = point;
    }

    pub fn sort_candidates(&mut self, point: &Point) {
        self.point = point;
    }
}
