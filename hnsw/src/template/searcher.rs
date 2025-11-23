use graph::{
    graph::Graph,
    nodes::{Dist, Node},
};
use points::{point::Point, point_collection::Points};
use vectors::VecTrait;

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
        ef: u32,
    ) -> Result<(), String> {
        results.extend_candidates_with_selected();
        results.extend_visited_with_selected();

        while !results.candidates_is_empty() {
            let cand_dist = results.pop_candidate().unwrap();
            let furthest2q_dist = results.peek_selected().unwrap();
            if cand_dist.0 > *furthest2q_dist {
                break;
            }
            let cand_neighbors = match layer.neighbors_vec(cand_dist.0.id) {
                Ok(neighs) => neighs,
                Err(msg) => return Err(format!("Error in search_layer: {msg}")),
            };

            // pre-compute distances to candidate neighbors to take advantage of
            // caches and to prevent the re-construction of the query to a full vector
            let not_visited: Vec<Node> = cand_neighbors
                .iter()
                .filter(|node| results.push_visited(**node))
                .copied()
                .collect();
            let q2cand_dists = point.dist2many(not_visited.iter().map(|node| {
                points
                    .get_point(*node)
                    .expect("Point ID not found in collection.")
            }));
            let q2cand_neighbors_dists: Vec<Dist> = q2cand_dists
                .zip(not_visited.iter())
                .map(|(dist, id)| Dist::new(*id, dist))
                .collect();
            for n2q_dist in q2cand_neighbors_dists {
                let f2q_dist = results.peek_selected().unwrap();
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

    pub fn select_heuristic<T: VecTrait>(
        &self,
        results: &mut Results,
        layer: &Graph,
        point: &Point<T>,
        points: &Points<T>,
        m: u8,
        extend_cands: bool,
        keep_pruned: bool,
    ) -> Result<(), String> {
        results.heuristic_setup();
        if extend_cands {
            results.extend_candidates_with_neighbors(point, points, layer)?;
        }

        let node_e = results.pop_candidate().unwrap();
        results.push_selected(node_e.0);

        while (!results.candidates_is_empty()) & (results.selected_len() < m as usize) {
            let node_e = results.pop_candidate().unwrap();
            let e_point = points.get_point(node_e.0.id).unwrap();

            let nearest_selected = results.get_nearest_from_selected(e_point, points);

            if node_e.0 < nearest_selected {
                results.push_selected(node_e.0);
            } else if keep_pruned {
                results.push_visited_heuristic(node_e.0);
            }
        }

        if keep_pruned {
            while (!results.visited_heuristic_is_empty()) & (results.selected_len() < m as usize) {
                let node_e = results.pop_visited_heuristic().unwrap();
                results.push_selected(node_e.0);
            }
        }

        Ok(())
    }
}
