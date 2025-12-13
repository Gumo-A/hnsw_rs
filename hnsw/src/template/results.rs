use core::panic;
use graph::errors::GraphError;
use graph::graph::Graph;
use graph::nodes::{Dist, Node};
use nohash_hasher::{IntMap, IntSet};
use points::point::Point;
use points::point_collection::Points;
use std::collections::BTreeSet;
use vectors::{VecBase, VecTrait};

pub type LayerResult = IntMap<Node, BTreeSet<Dist>>;
type LayersResults = IntMap<u8, LayerResult>;
type Selected = BTreeSet<Dist>;
type Candidates = BTreeSet<Dist>;

pub struct Results {
    selected: Selected,
    candidates: Candidates,
    visited: IntSet<Node>,
    visited_heuristic: Candidates,
    insertion_results: LayersResults,
    prune_results: LayersResults,
}

impl Results {
    pub fn new() -> Self {
        Self {
            selected: BTreeSet::new(),
            candidates: BTreeSet::new(),
            visited: IntSet::default(),
            visited_heuristic: BTreeSet::new(),
            insertion_results: IntMap::default(),
            prune_results: IntMap::default(),
        }
    }

    pub fn get_layer_result(&mut self, layer_nb: u8) -> &mut LayerResult {
        self.insertion_results
            .entry(layer_nb)
            .or_insert(IntMap::default())
    }

    pub fn get_top_selected(&self, n: usize) -> Vec<Dist> {
        self.selected.clone().iter().take(n).copied().collect()
    }

    pub fn get_nearest_from_selected<T: VecTrait>(
        &self,
        point: &Point<T>,
        points: &Points<T>,
    ) -> Dist {
        let point_ids: Vec<Node> = self.selected.iter().map(|n| n.id).collect();
        let selected_points = points.get_points_iter(point_ids.iter().copied());
        let distances = point.dist2many(selected_points);
        distances
            .zip(point_ids.iter())
            .map(|(dist, id)| Dist::new(*id, dist))
            .min()
            .unwrap()
    }

    pub fn iter_insertion_results(&self) -> impl Iterator<Item = (&u8, &LayerResult)> {
        self.insertion_results.iter()
    }

    pub fn insert_layer_results(&mut self, layer_nb: u8, point_id: Node) {
        let point_neighbors = self.get_selected_clone();
        let layer_result = self.get_layer_result(layer_nb);
        layer_result.insert(point_id, point_neighbors);
    }

    pub fn insert_prune_result(&mut self, layer_nb: u8, node_id: Node, nearest: BTreeSet<Dist>) {
        let layer_result = self
            .prune_results
            .entry(layer_nb)
            .or_insert(IntMap::default());
        layer_result.insert(node_id, nearest);
    }

    pub fn iter_prune_results(&self) -> impl Iterator<Item = (&u8, &LayerResult)> {
        self.prune_results.iter()
    }

    pub fn get_clone_insertion_results(&self) -> LayersResults {
        self.insertion_results.clone()
    }

    pub fn push_selected(&mut self, dist: Dist) {
        self.selected.insert(dist);
    }

    pub fn push_visited(&mut self, idx: Node) -> bool {
        self.visited.insert(idx)
    }

    pub fn get_selected_clone(&self) -> Selected {
        self.selected.clone()
    }

    pub fn select_setup(&mut self) {
        self.clear_visited_heuristic();
        self.clear_candidates();
        self.candidates
            .extend(self.selected.iter().map(|dist| *dist));
        self.clear_selected();
    }

    pub fn selected_len(&self) -> usize {
        self.selected.len()
    }

    pub fn push_candidate(&mut self, candidate: Dist) {
        self.candidates.insert(candidate);
    }

    pub fn push_visited_heuristic(&mut self, visited: Dist) {
        self.visited_heuristic.insert(visited);
    }

    pub fn pop_selected(&mut self) -> Option<Dist> {
        self.selected.pop_last()
    }

    pub fn pop_best_candidate(&mut self) -> Option<Dist> {
        self.candidates.pop_first()
    }

    pub fn best_candidate(&self) -> Option<&Dist> {
        self.candidates.first()
    }

    pub fn pop_best_visited_heuristic(&mut self) -> Option<Dist> {
        self.visited_heuristic.pop_first()
    }

    pub fn candidates_is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn visited_heuristic_is_empty(&self) -> bool {
        self.visited_heuristic.is_empty()
    }

    pub fn worst_selected(&self) -> Option<&Dist> {
        self.selected.last()
    }

    pub fn extend_candidates_with_neighbors<T: VecTrait>(
        &mut self,
        point: &Point<T>,
        points: &Points<T>,
        layer: &Graph,
    ) -> Result<(), String> {
        let mut neighbors = Vec::new();
        for node in self.candidates.iter() {
            let node_neighbors = match layer.neighbors(node.id) {
                Ok(n) => n,
                Err(e) => match e {
                    GraphError::NodeNotInGraph(n) => panic!("Node {n} is not in the Graph"),
                    _ => panic!("Error while extending candidates"),
                },
            };
            for neighbor in node_neighbors {
                neighbors.push(neighbor)
            }
        }
        for neighbor in neighbors {
            let dist = Dist::new(neighbor, points.distance(point.id, neighbor).unwrap());
            self.push_candidate(dist);
        }
        Ok(())
    }

    pub fn extend_candidates_with_selected(&mut self) {
        for node in self.selected.iter() {
            self.candidates.insert(node.clone());
        }
    }

    pub fn extend_visited_with_selected(&mut self) {
        for node in self.selected.iter() {
            self.visited.insert(node.id);
        }
    }

    pub fn iter_selected(&self) -> impl Iterator<Item = &Dist> {
        self.selected.iter()
    }

    pub fn clear_candidates(&mut self) {
        self.candidates.clear();
    }
    pub fn clear_prune(&mut self) {
        self.prune_results.clear();
    }
    pub fn clear_visited(&mut self) {
        self.visited.clear();
    }
    pub fn clear_selected(&mut self) {
        self.selected.clear();
    }
    pub fn clear_visited_heuristic(&mut self) {
        self.visited_heuristic.clear();
    }

    pub fn clear_all(&mut self) {
        self.selected.clear();
        self.candidates.clear();
        self.visited.clear();
        self.visited_heuristic.clear();
        self.insertion_results.clear();
        self.prune_results.clear();
    }

    pub fn clear_searchers(&mut self) {
        self.selected.clear();
        self.candidates.clear();
        self.visited.clear();
        self.visited_heuristic.clear();
    }
}
