use crate::dist::Dist;
use nohash_hasher::IntSet;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]
pub struct Searcher {
    pub selected: BinaryHeap<Dist>,
    pub candidates: BinaryHeap<Reverse<Dist>>,
    pub visited: IntSet<u32>,
    pub visited_heuristic: BinaryHeap<Reverse<Dist>>,
    pub insertion_results: Vec<(u8, u32, BinaryHeap<Dist>)>,
    pub prune_results: Vec<(u8, u32, BinaryHeap<Dist>)>,
}

impl Searcher {
    pub fn new() -> Self {
        Self {
            selected: BinaryHeap::new(),
            candidates: BinaryHeap::new(),
            visited: IntSet::default(),
            visited_heuristic: BinaryHeap::new(),
            insertion_results: Vec::new(),
            prune_results: Vec::new(),
        }
    }

    fn clear_all(&mut self) {
        self.selected.clear();
        self.candidates.clear();
        self.visited.clear();
        self.visited_heuristic.clear();
        self.insertion_results.clear();
        self.prune_results.clear();
    }

    fn clear_searchers(&mut self) {
        self.selected.clear();
        self.candidates.clear();
        self.visited.clear();
        self.visited_heuristic.clear();
    }
}
