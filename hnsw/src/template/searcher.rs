use core::panic;

use graph::{dist::Dist, errors::GraphError, graph::Graph};
use log::trace;
use points::{
    point::Point,
    points::{Points, SimplePoints},
};
use vectors::VecBase;

use crate::template::{results::Results, PointsType, HNSW};

pub struct Searcher {}

/// The Searcher doesn't contain any data,
/// its only purpose is to implement methods to traverse
/// a Graph struct looking for nearest neighbors of some Point
impl Searcher {
    pub fn new() -> Self {
        Self {}
    }

    pub fn search_layer(
        &self,
        results: &mut Results,
        layer: &Graph,
        point: &Point,
        index: &HNSW,
        ef: usize,
    ) -> Result<(), String> {
        trace!("Searching Layer");
        results.extend_candidates_with_selected();
        results.extend_visited_with_selected();

        while !results.candidates.is_empty() {
            trace!("Begin loop");
            let cand_dist = results.candidates.pop_first().unwrap();
            let furthest2q_dist = results.selected.last().unwrap();
            trace!("Best candidate: {cand_dist:?}");
            trace!("Worst selected: {furthest2q_dist:?}");
            if cand_dist > *furthest2q_dist {
                trace!("Best candidate is further away than worst selected, breaking from loop");
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
            trace!("{0} has {1} neighbors", cand_dist.id, cand_neighbors.len());

            trace!(
                "Getting distances from query {0} to {1}'s neighbors",
                point.id,
                cand_dist.id
            );
            let q2cand_neighbors_dists: Vec<Dist> = cand_neighbors
                .iter()
                .filter(|node| results.insert_visited(**node))
                .copied()
                .map(|node| {
                    let dist = index
                        .distance(point.id, node)
                        .expect("Could not compute distance between points");
                    Dist::new(node, dist)
                })
                .collect();
            trace!("Selecting among the best of the candidate's neighbors");
            for n2q_dist in q2cand_neighbors_dists {
                let f2q_dist = results.selected.last().unwrap();

                if results.selected.len() < ef as usize {
                    trace!("Need to fill selected struct, inserting {n2q_dist:?}");
                    results.insert_selected(n2q_dist);
                    results.insert_candidate(n2q_dist);
                    continue;
                }

                if n2q_dist < *f2q_dist {
                    trace!("Candidate's neighbor {n2q_dist:?} is better than worst selected {f2q_dist:?}, inserting it");
                    results.insert_selected(n2q_dist);
                    results.insert_candidate(n2q_dist);

                    if results.selected.len() > ef as usize {
                        let removed = results.selected.pop_last().unwrap();
                        trace!("Selected was too full, removed {removed:?}");
                    }
                }
            }
        }
        trace!(
            "Finished traversing the Layer. Selected: {:?}",
            results.selected
        );
        results.clear_candidates();
        results.clear_visited();
        Ok(())
    }

    pub fn select_simple(&self, results: &mut Results, m: usize) {
        results.select_simple(m);
    }

    pub fn select_heuristic(
        &self,
        results: &mut Results,
        layer: &Graph,
        point: &Point,
        points: &SimplePoints,
        m: usize,
        extend_cands: bool,
        keep_pruned: bool,
    ) -> Result<(), String> {
        trace!("Begin select_heuristic, selected is {:?}", results.selected);
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

        trace!(
            "Finished select_heuristic, selected is {:?}",
            results.selected
        );
        Ok(())
    }
}
