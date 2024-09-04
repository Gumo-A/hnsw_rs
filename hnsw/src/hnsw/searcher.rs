use std::collections::BTreeMap;

use nohash_hasher::IntSet;

use super::{dist::Dist, points::Point};

pub struct Searcher<'p> {
    pub ep: IntSet<usize>,
    pub point: Option<&'p Point>,
    pub search_seen: IntSet<usize>,
    pub search_candidates: BTreeMap<Dist, usize>,
    pub search_selected: BTreeMap<Dist, usize>,
    pub heuristic_candidates: BTreeMap<Dist, usize>,
    pub heuristic_visited: BTreeMap<Dist, usize>,
    pub heuristic_selected: BTreeMap<Dist, usize>,
}

impl<'p> Searcher<'p> {
    pub fn new() -> Self {
        Searcher {
            ep: IntSet::default(),
            point: None,
            search_seen: IntSet::default(),
            search_candidates: BTreeMap::new(),
            search_selected: BTreeMap::new(),
            heuristic_candidates: BTreeMap::new(),
            heuristic_visited: BTreeMap::new(),
            heuristic_selected: BTreeMap::new(),
        }
    }
    pub fn init<'a>(&mut self, point: &'a Point, ep: usize)
    where
        'a: 'p,
    {
        self.clear_all();
        self.point = Some(point);
        self.ep.insert(ep);
        self.search_seen.insert(ep);
    }

    pub fn sort_candidates_selected(&mut self, others: Vec<&Point>) {
        self.search_candidates.clear();
        self.search_selected.clear();
        let point = self.point.unwrap();
        for other in others {
            let dist = point.dist2other(other);
            self.search_candidates.insert(dist, other.id);
            self.search_selected.insert(dist, other.id);
        }
    }

    pub fn init_heuristic(&mut self) {
        std::mem::swap(&mut self.heuristic_candidates, &mut self.search_selected);
        self.heuristic_visited.clear();
        self.heuristic_selected.clear();
    }

    pub fn clear_all(&mut self) {
        self.search_seen.clear();
        self.search_candidates.clear();
        self.search_selected.clear();
    }

    pub fn set_next_ep(&mut self) {
        self.ep.clear();
        for (_, id) in self.search_selected.iter() {
            self.ep.insert(*id);
        }
    }
}
