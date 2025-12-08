use graph::{graph::Graph, nodes::Dist};
use nohash_hasher::IntSet;
use points::{point::Point, point_collection::Points};
use vectors::{VecBase, VecTrait};

use crate::template::results::Results;

pub struct Searcher {}

impl Searcher {
    pub fn new() -> Self {
        Self {}
    }

    pub fn search_layer<T: VecTrait>(
        &self,
        results: &mut Results,
        layer: &Graph,
        point: &Point<T>,
        points: &Points<T>,
        ef: usize,
    ) -> Result<(), String> {
        results.extend_candidates_with_selected();
        results.extend_visited_with_selected();

        while !results.candidates_is_empty() {
            let cand_dist = results.pop_best_candidate().unwrap();
            let furthest2q_dist = results.worst_selected().unwrap();
            if cand_dist > *furthest2q_dist {
                break;
            }
            let cand_neighbors = match layer.neighbors_vec(cand_dist.id) {
                Ok(neighs) => neighs,
                Err(msg) => return Err(format!("Error in search_layer: {msg}")),
            };

            let q2cand_neighbors_dists: Vec<Dist> = cand_neighbors
                .iter()
                .filter(|node| results.push_visited(**node))
                .copied()
                .map(|node| {
                    let other = points
                        .get_point(node)
                        .expect("Point ID not found in collection.");
                    Dist::new(other.id, point.dist2other(&other))
                })
                .collect();
            for n2q_dist in q2cand_neighbors_dists {
                let f2q_dist = results.worst_selected().unwrap();
                if (n2q_dist < *f2q_dist) | (results.selected_len() < ef as usize) {
                    results.push_selected(n2q_dist);
                    results.push_candidate(n2q_dist);

                    if results.selected_len() > ef as usize {
                        results.pop_selected();
                    }
                }
            }
        }
        results.clear_candidates();
        results.clear_visited();
        Ok(())
    }

    pub fn select_simple(&self, results: &mut Results, m: usize) -> Result<(), String> {
        results.select_setup();
        for _ in 0..m {
            let node = match results.best_candidate() {
                None => break,
                Some(n) => n,
            };
            results.push_selected(node.clone());
        }
        Ok(())
    }

    pub fn select_heuristic<T: VecTrait>(
        &self,
        results: &mut Results,
        layer: &Graph,
        point: &Point<T>,
        points: &Points<T>,
        m: usize,
        extend_cands: bool,
        keep_pruned: bool,
    ) -> Result<(), String> {
        results.select_setup();
        if extend_cands {
            results.extend_candidates_with_neighbors(point, points, layer)?;
        }

        let node_e = results.pop_best_candidate().unwrap();
        results.push_selected(node_e);

        while (!results.candidates_is_empty()) & (results.selected_len() < m as usize) {
            let node_e = results.pop_best_candidate().unwrap();
            let e_point = points.get_point(node_e.id).unwrap();

            let nearest_selected = results.get_nearest_from_selected(e_point, points);

            if node_e < nearest_selected {
                results.push_selected(node_e);
            } else if keep_pruned {
                results.push_visited_heuristic(node_e);
            }
        }

        if keep_pruned {
            while (!results.visited_heuristic_is_empty()) & (results.selected_len() < m as usize) {
                let node_e = results.pop_best_visited_heuristic().unwrap();
                results.push_selected(node_e);
            }
        }

        Ok(())
    }
}
