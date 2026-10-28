use core::panic;
use graph::NodeID;
use graph::dist::Dist;
use graph::errors::GraphError;
use graph::graph::Graph;
use log::{info, trace, warn};
use nohash_hasher::IntSet;
use nohash_hasher::{IntMap, IntSet};
use points::point::Point;
use points::points::Points;
use std::collections::BTreeSet;
use vectors::VecBase;

type OrderedDists = BTreeSet<Dist>;
pub type LayerResult = IntMap<NodeID, IntSet<NodeID>>;
type LayersResults = IntMap<usize, LayerResult>;

/// A wrapper around many helper data structures
/// such as BTreeSet and IntMap.
///
/// The Results struct is used to store the visited nodes
/// during traversal, and it uses BTreeSets to keep an ordered
/// trace of the distances to those nodes.
///
/// It uses these structures to provide an interface for quick
/// retrieval of best and worst candidates during insertion
pub struct Results {
    pub selected: OrderedDists,
    pub candidates: OrderedDists,
    pub visited: IntSet<NodeID>,
    pub visited_h: OrderedDists,
    pub insertion_results: LayersResults,
    pub prune_results: LayersResults,
}

impl Results {
    pub fn new() -> Self {
        Self {
            selected: BTreeSet::new(),
            candidates: BTreeSet::new(),
            visited: IntSet::default(),
            visited_h: BTreeSet::new(),
            insertion_results: IntMap::default(),
            prune_results: IntMap::default(),
        }
    }

    fn get_insertion_result(&mut self, layer_nb: usize) -> &mut LayerResult {
        self.insertion_results
            .entry(layer_nb)
            .or_insert(IntMap::default())
    }

    pub fn get_top_selected(&self, n: usize) -> Vec<Dist> {
        self.selected.iter().take(n).copied().collect()
    }

    pub fn select_simple(&mut self, m: usize) {
        while self.selected.len() > m {
            self.selected.pop_last();
        }
    }

    pub fn get_nearest_from_selected(&self, point: &Point, points: &impl Points) -> Dist {
        let selected_points = self
            .selected
            .iter()
            .map(|n| points.get_point(n.id).unwrap());
        let distances = point.dist2many(selected_points);
        distances
            .zip(self.selected.iter().map(|n| n.id))
            .map(|(dist, id)| Dist::new(id, dist))
            .min()
            .unwrap()
    }

    pub fn save_layer_results(&mut self, layer_nb: usize, point_id: NodeID) {
        trace!("Saving layer results for Layer {layer_nb} and point {point_id}");
        let point_neighbors = IntSet::from_iter(self.selected.iter().map(|dist| dist.id));
        let layer_result = self.get_insertion_result(layer_nb);
        layer_result.insert(point_id, point_neighbors);
    }

    pub fn insert_selected(&mut self, dist: Dist) {
        trace!("Inserting distance to selected: {dist:?}");
        self.selected.insert(dist);
    }

    pub fn insert_visited(&mut self, idx: NodeID) -> bool {
        self.visited.insert(idx)
    }

    pub fn select_setup(&mut self) {
        self.visited_h.clear();
        self.candidates.clear();
        self.candidates
            .extend(self.selected.iter().map(|dist| *dist));
        self.selected.clear();
    }

    pub fn insert_candidate(&mut self, dist: Dist) {
        trace!("Inserting distance to candidates: {dist:?}");
        self.candidates.insert(dist);
    }

    pub fn insert_visited_h(&mut self, visited: Dist) {
        self.visited_h.insert(visited);
    }

    pub fn extend_candidates_with_neighbors(
        &mut self,
        point: &Point,
        points: &impl Points,
        layer: &Graph,
    ) -> Result<(), String> {
        let cands: Vec<NodeID> = self.candidates.iter().map(|dist| dist.id).collect();
        for node in cands {
            let neighs_mutex = match layer.neighbors_lock(node) {
                Ok(n) => n,
                Err(e) => match e {
                    GraphError::NodeNotInGraph(n) => panic!("Node {n} is not in the Graph"),
                    _ => panic!("Error while extending candidates"),
                },
            };
            for neighbor in neighs_mutex.lock().unwrap().iter() {
                let dist = Dist::new(*neighbor, points.distance(point.id, *neighbor).unwrap());
                self.insert_candidate(dist);
            }
        }
        Ok(())
    }

    pub fn extend_candidates_with_selected(&mut self) {
        trace!("Extending candidates from selected");
        trace!("Selected: {:?}", self.selected);
        trace!("Candidates: {:?}", self.candidates);
        for node in self.selected.iter() {
            self.candidates.insert(node.clone());
        }
        trace!("Extended candidates");
        trace!("Candidates: {:?}", self.candidates);
    }

    pub fn extend_visited_with_selected(&mut self) {
        trace!("Extending visited from selected");
        trace!("Selected: {:?}", self.selected);
        trace!("Visited: {:?}", self.visited);
        for node in self.selected.iter() {
            self.visited.insert(node.id);
        }
        trace!("Extended visited");
        trace!("Visited: {:?}", self.visited);
    }

    pub fn clear_candidates(&mut self) {
        self.candidates.clear();
    }

    pub fn clear_visited(&mut self) {
        self.visited.clear();
    }

    pub fn clear_all(&mut self) {
        trace!("Clearing all Results' structures");
        self.selected.clear();
        self.candidates.clear();
        self.visited.clear();
        self.visited_h.clear();
        self.insertion_results.clear();
        self.prune_results.clear();
    }
}

#[cfg(test)]
mod test {
    use graph::dist::Dist;

    use crate::template::results::Results;

    fn get_results() -> Results {
        let mut r = Results::new();
        r.insert_selected(Dist::new(0, 0.5));
        r.insert_selected(Dist::new(1, 0.6));
        r.insert_selected(Dist::new(2, 0.7));
        r.insert_selected(Dist::new(3, 0.8));
        r.insert_selected(Dist::new(4, 0.9));
        r
    }

    #[test]
    fn extend_candidates() {
        let mut r = get_results();
        r.extend_candidates_with_selected();
        assert_eq!(r.selected.len(), r.candidates.len());
        while r.selected.len() > 0 {
            assert_eq!(
                r.candidates.pop_first().unwrap().id,
                r.selected.pop_first().unwrap().id
            );
        }
        assert_eq!(r.selected.len(), r.candidates.len());
    }

    #[test]
    fn equal_dists_diff_ids_are_added() {
        let mut r = Results::new();
        r.insert_selected(Dist::new(0, 0.5));
        r.insert_selected(Dist::new(1, 0.5));
        r.insert_selected(Dist::new(2, 0.0));
        r.insert_selected(Dist::new(4, 0.0));
        assert_eq!(r.selected.len(), 4);
    }
}
