use core::panic;

use graph::{errors::GraphError, graph::Graph, nodes::Dist};
use points::{point::Point, points::Points};
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

        while !results.candidates.is_empty() {
            let cand_dist = results.candidates.pop_first().unwrap();
            let furthest2q_dist = results.selected.last().unwrap();
            if cand_dist > *furthest2q_dist {
                break;
            }
            let cand_neighbors = match layer.neighbors_vec(cand_dist.id) {
                Ok(neighs) => neighs,
                Err(e) => match e {
                    GraphError::NodeNotInGraph(n) => {
                        return Err(format!("Error in search_layer: {n} not in Graph"))
                    }
                    _ => panic!("Error in search_layer: got an unexpected error"),
                },
            };

            let q2cand_neighbors_dists: Vec<Dist> = cand_neighbors
                .iter()
                .filter(|node| results.insert_visited(**node))
                .copied()
                .map(|node| {
                    let other = points
                        .get_point(node)
                        .expect("Point ID not found in collection.");
                    Dist::new(other.id, point.dist2other(&other))
                })
                .collect();
            for n2q_dist in q2cand_neighbors_dists {
                let f2q_dist = results.selected.last().unwrap();
                if (n2q_dist < *f2q_dist) | (results.selected.len() < ef as usize) {
                    results.insert_selected(n2q_dist);
                    results.insert_candidate(n2q_dist);

                    if results.selected.len() > ef as usize {
                        results.selected.pop_last();
                    }
                }
            }
        }
        results.clear_candidates();
        results.clear_visited();
        Ok(())
    }

    pub fn select_simple(&self, results: &mut Results, m: usize) -> Result<(), String> {
        while results.selected.len() > m {
            results.selected.pop_last();
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

        let node_e = results.candidates.pop_first().unwrap();
        results.insert_selected(node_e);

        while (!results.candidates.is_empty()) & (results.selected.len() < m as usize) {
            let node_e = results.candidates.pop_first().unwrap();
            let e_point = points.get_point(node_e.id).unwrap();

            let nearest_selected = results.get_nearest_from_selected(e_point, points);

            if node_e < nearest_selected {
                results.insert_selected(node_e);
            } else if keep_pruned {
                results.insert_visited_h(node_e);
            }
        }

        if keep_pruned {
            while (!results.visited_h.is_empty()) & (results.selected.len() < m as usize) {
                let node_e = results.visited_h.pop_first().unwrap();
                results.insert_selected(node_e);
            }
        }

        Ok(())
    }
}
