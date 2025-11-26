use graph::graph::Graph;
use graph::nodes::{Dist, Node};
use nohash_hasher::{IntMap, IntSet};
use points::point::Point;
use points::point_collection::Points;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use vectors::VecTrait;

pub type LayerResult = IntMap<Node, BinaryHeap<Dist>>;
type LayersResults = IntMap<u8, LayerResult>;
type Selected = BinaryHeap<Dist>;
type RevSelected = BinaryHeap<Reverse<Dist>>;

pub struct Results {
    selected: Selected,
    candidates: RevSelected,
    visited: IntSet<Node>,
    visited_heuristic: RevSelected,
    insertion_results: LayersResults,
    prune_results: LayersResults,
}

impl Results {
    pub fn new() -> Self {
        Self {
            selected: BinaryHeap::new(),
            candidates: BinaryHeap::new(),
            visited: IntSet::default(),
            visited_heuristic: BinaryHeap::new(),
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
        self.selected
            .clone()
            .into_sorted_vec()
            .iter()
            .take(n)
            .copied()
            .collect()
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

    pub fn insert_prune_result(&mut self, layer_nb: u8, node_id: Node, nearest: BinaryHeap<Dist>) {
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
        self.selected.push(dist);
    }

    pub fn push_visited(&mut self, idx: Node) -> bool {
        self.visited.insert(idx)
    }

    pub fn get_selected_clone(&self) -> Selected {
        self.selected.clone()
    }

    pub fn heuristic_setup(&mut self) {
        self.clear_visited_heuristic();
        self.clear_candidates();
        self.candidates
            .extend(self.selected.iter().map(|dist| Reverse(*dist)));
        self.clear_selected();
    }

    pub fn selected_len(&self) -> usize {
        self.selected.len()
    }

    pub fn push_candidate(&mut self, candidate: Dist) {
        self.candidates.push(Reverse(candidate));
    }

    pub fn push_visited_heuristic(&mut self, visited: Dist) {
        self.visited_heuristic.push(Reverse(visited));
    }

    pub fn pop_selected(&mut self) -> Option<Dist> {
        self.selected.pop()
    }

    pub fn pop_candidate(&mut self) -> Option<Reverse<Dist>> {
        self.candidates.pop()
    }

    pub fn pop_visited_heuristic(&mut self) -> Option<Reverse<Dist>> {
        self.visited_heuristic.pop()
    }

    pub fn candidates_is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn visited_heuristic_is_empty(&self) -> bool {
        self.visited_heuristic.is_empty()
    }

    pub fn peek_selected(&self) -> Option<&Dist> {
        self.selected.peek()
    }

    pub fn extend_candidates_with_neighbors<T: VecTrait>(
        &mut self,
        point: &Point<T>,
        points: &Points<T>,
        layer: &Graph,
    ) -> Result<(), String> {
        // TODO which version is faster?
        // let old_candidates = self.candidates.clone();
        // for node in old_candidates.iter() {
        //     for neighbor in layer.neighbors(node.0.id)? {
        //         self.push_candidate(Node::new_with_dist(
        //             points.distance(point.id, neighbor.id).unwrap(),
        //             neighbor.id,
        //         ));
        //     }
        // }
        let mut neighbors = Vec::new();
        for node in self.candidates.iter() {
            for neighbor in layer.neighbors(node.0.id)? {
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
            self.candidates.push(Reverse(node.clone()));
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
