// use std::{collections::HashSet, sync::Arc, thread::Builder};

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

use crate::hnsw::results::Results;

// use crate::helpers::get_progress_bar;

/// The role of the Traverser is to encapsulate the navigation
/// logic of the HNSW data structure, exposing an `insert` and
/// an `ann_by_vec` methods.
pub struct Traverser {
    results: Results,
}

impl Traverser {
    pub fn new() -> Self {
        Self {
            results: Results::new(),
        }
    }

    pub fn ann_by_vec(
        &self,
        vector: &Vec<f32>,
        n: usize,
        ef: usize,
    ) -> Result<Vec<NodeID>, String> {
        info!("Start ANN search");
        let point = Point::new(vector);
        self.clear_results();
        self.results.insert_selected(Dist::new(
            self.params.ep,
            self.points.distance2point(&point, self.params.ep).unwrap(),
        ));
        let nb_layers = self.layers.len();

        for layer_nb in (1..nb_layers).rev() {
            self.search_layer(self.get_layer(layer_nb), &point, self, 1)?;
        }

        self.search_layer(&self.get_layer(0), &point, &self, ef)?;

        let anns: Vec<NodeID> = self
            .results
            .get_top_selected(n)
            .iter()
            .map(|dist| dist.id)
            .collect();
        info!("Finished ANN search");
        Ok(anns)
    }

    pub fn insert(&mut self, index: &HNSW, point: &Point) {
        self.build_insertion_results(index, point);
        self.make_connections(index);
    }

    fn build_insertion_results(&mut self, index: &HNSW, point: &Point) -> Result<(), String> {
        trace!("Begin insertion results");
        if point.id == index.params.ep {
            warn!(
                "Tried to build insertion results for point {}, but it is the EP, not building results",
                point.id
            );
            return Ok(());
        }
        self.setup_insert(index, point);
        self.traverse_layers_above(index, point)?;
        self.traverse_layers_below(index, point)?;
        trace!("Finish insertion results");
        Ok(())
    }

    fn setup_insert(&mut self, index: &HNSW, point: &Point) {
        trace!("Setting up insertion");
        self.results.clear_all();

        let dist2ep = index
            .distance(index.params.ep, point.id)
            .expect("Could not compute distance between EP and point to insert.");
        trace!(
            "Setting up insertion. Distance between {0} and EP {1} is {dist2ep}",
            point.id, index.params.ep
        );

        self.results
            .insert_selected(Dist::new(index.params.ep, dist2ep));
    }

    fn traverse_layers_above(&mut self, index: &HNSW, point: &Point) -> Result<(), String> {
        trace!("Traversing Layers above query point");
        let layers_len = index.layers.len();

        for layer_nb in (point.level as usize + 1..layers_len).rev() {
            let layer = index.get_layer(layer_nb);
            trace!(
                "Looking for nearest neighbors for point {0} in layer {layer_nb}",
                point.id,
            );
            trace!("The layer has {} nodes", layer.nb_nodes());
            trace!(
                "The entry point to the layer is {:?}",
                self.results.selected
            );
            self.search_layer(layer, point, &index, 1)?;
        }
        Ok(())
    }

    fn traverse_layers_below(&mut self, index: &HNSW, point: &Point) -> Result<(), String> {
        trace!("Traversing Layers below or equal to query point");
        let bound = (point.level as usize).min(index.layers.len() - 1);
        for layer_nb in (0..=bound).rev() {
            let layer = index.get_layer(layer_nb);
            trace!(
                "Looking for nearest neighbors for point {0} in layer {layer_nb}",
                point.id,
            );
            trace!("The layer has {} nodes", layer.nb_nodes());
            trace!(
                "The entry points to the layer are {:?}",
                self.results.selected
            );
            self.search_layer(layer, point, &index, index.params.ef_cons)?;
            // self.select_simple(&mut self.results, index.params.m);
            self.select_heuristic(layer, point, &index.points, index.params.m, true, true)?;
            self.results.save_layer_results(layer_nb, point.id);
        }
        Ok(())
    }

    fn make_connections(&self, index: &HNSW) -> Result<(), String> {
        info!("Making connections using results");
        for (layer_nb, node_data) in self.results.insertion_results.iter() {
            let layer = index.get_layer(*layer_nb);
            for (node, neighbors) in node_data.iter() {
                layer
                    .add_neighbors(*node, neighbors.iter().copied())
                    .unwrap();
                for n in neighbors {
                    if layer.degree(*n).unwrap() > layer.m {
                        let query = index.get_point(*n).unwrap();
                        let neighs = layer.neighbors(*n).unwrap();
                        let candidate_dists = neighs.iter().copied().map(|id| {
                            let point = index.get_point(id).unwrap();
                            Dist::new(id, point.dist2other(&query))
                        });
                        let new = Self::select_simple(candidate_dists, layer.m);
                        layer.replace_neighbors(*n, new.iter().copied()).unwrap();
                    }
                }
            }
        }
        info!("Finished making connections");
        Ok(())
    }

    fn select_simple<I>(candidate_dists: I, m: usize) -> IntSet<NodeID>
    where
        I: Iterator<Item = Dist>,
    {
        let mut cands = Vec::from_iter(candidate_dists);
        cands.sort();
        IntSet::from_iter(cands.iter().take(m).map(|dist| dist.id))
    }

    fn search_layer(
        &mut self,
        layer: &Graph,
        point: &Point,
        index: &HNSW,
        ef: usize,
    ) -> Result<(), String> {
        trace!("Searching Layer");
        self.results.extend_candidates_with_selected();
        self.results.extend_visited_with_selected();

        while !self.results.candidates.is_empty() {
            trace!("Begin loop");
            let cand_dist = self.results.candidates.pop_first().unwrap();
            let furthest2q_dist = self.results.selected.last().unwrap();
            trace!("Best candidate: {cand_dist:?}");
            trace!("Worst selected: {furthest2q_dist:?}");
            if cand_dist > *furthest2q_dist {
                trace!("Best candidate is further away than worst selected, breaking from loop");
                break;
            }
            let cand_neighbors_locked = match layer.neighbors_lock(cand_dist.id) {
                Ok(neighs) => neighs,
                Err(e) => match e {
                    GraphError::NodeNotInGraph(n) => {
                        return Err(format!("Error in search_layer: {n} not in Graph"));
                    }
                    _ => panic!("Error in search_layer: got an unexpected error"),
                },
            };
            let cand_neighbors = cand_neighbors_locked.lock().unwrap();
            trace!("{0} has {1} neighbors", cand_dist.id, cand_neighbors.len());

            trace!(
                "Getting distances from query {0} to {1}'s neighbors",
                point.id, cand_dist.id
            );
            let q2cand_neighbors_dists: Vec<Dist> = cand_neighbors
                .iter()
                .filter(|node| self.results.insert_visited(**node))
                .map(|node| {
                    let dist = index
                        .get_point(*node)
                        .expect(format!("Could get point {node}").as_str())
                        .dist2other(point);
                    Dist::new(*node, dist)
                })
                .collect();
            trace!("Selecting among the best of the candidate's neighbors");
            for n2q_dist in q2cand_neighbors_dists {
                let f2q_dist = self.results.selected.last().unwrap();

                if self.results.selected.len() < ef as usize {
                    trace!("Need to fill selected struct, inserting {n2q_dist:?}");
                    self.results.insert_selected(n2q_dist);
                    self.results.insert_candidate(n2q_dist);
                    continue;
                }

                if n2q_dist < *f2q_dist {
                    trace!(
                        "Candidate's neighbor {n2q_dist:?} is better than worst selected {f2q_dist:?}, inserting it"
                    );
                    self.results.insert_selected(n2q_dist);
                    self.results.insert_candidate(n2q_dist);

                    if self.results.selected.len() > ef as usize {
                        let removed = self.results.selected.pop_last().unwrap();
                        trace!("Selected was too full, removed {removed:?}");
                    }
                }
            }
        }
        trace!(
            "Finished traversing the Layer. Selected: {:?}",
            self.results.selected
        );
        self.results.clear_candidates();
        self.results.clear_visited();
        Ok(())
    }

    fn select_simple(&self, results: &mut Results, m: usize) {
        results.select_simple(m);
    }

    fn select_heuristic(
        &mut self,
        layer: &Graph,
        point: &Point,
        points: &HNSW,
        m: usize,
        extend_cands: bool,
        keep_pruned: bool,
    ) -> Result<(), String> {
        trace!(
            "Begin select_heuristic, selected is {:?}",
            self.results.selected
        );
        self.results.select_setup();
        if extend_cands {
            self.results
                .extend_candidates_with_neighbors(point, points, layer)?;
        }

        let node_e = self.results.candidates.pop_first().unwrap();
        self.results.insert_selected(node_e);

        while (!self.results.candidates.is_empty()) & (self.results.selected.len() < m as usize) {
            let node_e = self.results.candidates.pop_first().unwrap();
            let e_point = points.get_point(node_e.id).unwrap();

            let nearest_selected = self.results.get_nearest_from_selected(&e_point, points);

            if node_e < nearest_selected {
                self.results.insert_selected(node_e);
            } else if keep_pruned {
                self.results.insert_visited_h(node_e);
            }
        }

        if keep_pruned {
            while (!self.results.visited_h.is_empty()) & (self.results.selected.len() < m as usize)
            {
                let node_e = self.results.visited_h.pop_first().unwrap();
                self.results.insert_selected(node_e);
            }
        }

        trace!(
            "Finished select_heuristic, selected is {:?}",
            self.results.selected
        );
        Ok(())
    }

    // pub fn insert_bulk(
    //     mut self,
    //     vectors: Vec<Vec<f32>>,
    //     nb_threads: usize,
    //     verbose: bool,
    // ) -> Result<HNSW, String> {
    //     info!("inserting {0} vectors to index in bulk", vectors.len());
    //     let stored_ids: HashSet<NodeID> =
    //         HashSet::from_iter(self.store_points(vectors).iter().cloned());

    //     let bar = get_progress_bar("Building HNSW index".to_string(), stored_ids.len(), verbose);
    //     let bar_arc = Arc::new(bar);

    //     let index_arc = Arc::new(self);

    //     for layer_nb in (0..index_arc.layers.len()).rev() {
    //         let layer = index_arc.layers.get_layer(layer_nb);
    //         info!("Connecting nodes in layer {layer_nb}, using {nb_threads} threads",);
    //         let chunks = ((layer.nb_nodes() as f64) / (nb_threads as f64)).ceil() as usize;
    //         let mut ids: Vec<Vec<NodeID>> = layer
    //             .iter_nodes()
    //             .filter(|id| {
    //                 stored_ids.contains(id)
    //                     & (index_arc.points.get_point(*id).unwrap().level == layer_nb as u8)
    //             })
    //             .collect::<Vec<NodeID>>()
    //             .chunks(chunks)
    //             .map(|chunk| Vec::from(chunk))
    //             .collect();

    //         let mut handlers = Vec::new();
    //         let mut idx = 0;
    //         while let Some(ids_split) = ids.pop() {
    //             info!("Sending {} node IDs to thread {idx}", ids_split.len());
    //             let index_copy = Arc::clone(&index_arc);
    //             let bar_ref = bar_arc.clone();
    //             let thread = Builder::new()
    //                 .name(format!("worker {idx}"))
    //                 .spawn(move || {
    //                     let mut inserter = Inserter::new();
    //                     for id in ids_split.iter() {
    //                         index_copy.insert(*id, &mut inserter).unwrap();
    //                         bar_ref.inc(1);
    //                     }
    //                 })
    //                 .expect("Could not spawn a thread");
    //             handlers.push(thread);
    //             idx += 1;
    //         }
    //         for handle in handlers {
    //             handle.join().unwrap();
    //         }
    //     }

    //     let index = Arc::into_inner(index_arc).expect("Could not get index out of Arc reference");
    //     Ok(index)
    // }
}
