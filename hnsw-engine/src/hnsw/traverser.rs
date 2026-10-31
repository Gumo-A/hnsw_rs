use graph::NodeID;
use graph::dist::Dist;
use log::{info, trace};
use nohash_hasher::IntSet;
use points::point::Point;
use vectors::VecBase;

use crate::{
    engine::INT,
    hnsw::{Error, results::Results},
    index::Index,
};

enum Query {
    ID(INT),
    Point(Point),
}
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

    pub fn insert(&mut self, index: &mut impl Index, id: INT, level: INT) -> Result<(), Error> {
        self.build_insertion_results(index, id, level)?;
        self.make_connections(index)?;
        Ok(())
    }

    fn build_insertion_results(
        &mut self,
        index: &mut impl Index,
        id: INT,
        level: INT,
    ) -> Result<(), Error> {
        trace!("Begin insertion results");
        // if id == index.get_ep() {
        //     warn!(
        //         "Tried to build insertion results for point {}, but it is the EP, not building results",
        //         id
        //     );
        //     return Ok(());
        // }
        self.setup_insert(index, id)?;
        self.traverse_layers_above(index, id, level)?;
        self.traverse_layers_below(index, id, level)?;
        trace!("Finish insertion results");
        Ok(())
    }

    fn clear_results(&mut self) {
        self.results.clear_all();
    }

    fn setup_insert(&mut self, index: &mut impl Index, id: INT) -> Result<(), Error> {
        trace!("Setting up insertion");
        self.clear_results();

        let dist2ep = index
            .distance(index.get_ep(), id)
            .expect("IO error computing distance")
            .expect("Could not compute distance between EP and point to insert.");
        trace!(
            "Setting up insertion. Distance between {0} and EP {1} is {dist2ep}",
            id,
            index.get_ep()
        );

        self.results
            .insert_selected(Dist::new(index.get_ep(), dist2ep));
        Ok(())
    }

    fn traverse_layers_above(
        &mut self,
        index: &mut impl Index,
        id: INT,
        level: INT,
    ) -> Result<(), Error> {
        trace!("Traversing Layers above query point");
        let layers_len = index
            .get_nb_layers()
            .expect("IO error getting nb of layers");

        for layer_nb in (level as usize + 1..layers_len).rev() {
            trace!(
                "Looking for nearest neighbors for point {0} in layer {layer_nb}",
                id,
            );
            trace!(
                "The entry point to the layer is {:?}",
                self.results.selected
            );
            self.search_layer(layer_nb, &Query::ID(id), index, 1)?;
        }
        Ok(())
    }

    fn traverse_layers_below(
        &mut self,
        index: &mut impl Index,
        id: INT,
        level: INT,
    ) -> Result<(), Error> {
        trace!("Traversing Layers below or equal to query point");
        let bound = (level as usize).min(
            index
                .get_nb_layers()
                .expect("IO error getting number of layers")
                - 1,
        );
        for layer_nb in (0..=bound).rev() {
            trace!(
                "Looking for nearest neighbors for point {0} in layer {layer_nb}",
                id,
            );
            trace!(
                "The entry points to the layer are {:?}",
                self.results.selected
            );
            self.search_layer(layer_nb, &Query::ID(id), index, index.get_ef_cons())?;
            // self.select_simple(&mut self.results, index.get_m());
            self.select_heuristic(index, layer_nb, id, index.get_m(), true, true)?;
            self.results.save_layer_results(layer_nb, id);
        }
        Ok(())
    }

    fn make_connections(&self, index: &mut impl Index) -> Result<(), Error> {
        info!("Making connections using results");
        for (layer_nb, node_data) in self.results.insertion_results.iter() {
            for (node, neighbors) in node_data.iter() {
                index
                    .add_neighbors(*node, neighbors.iter().copied(), *layer_nb)
                    .unwrap();
                for n in neighbors {
                    if index.degree(*n, *layer_nb).unwrap() > index.get_m_at_layer(*layer_nb) {
                        let query = index
                            .get_point(*n)
                            .expect("IO error getting point")
                            .expect("Could not get point");
                        let neighs = index.neighbors(*n, *layer_nb).unwrap();
                        let candidate_dists = neighs
                            .iter()
                            .copied()
                            .map(|id| {
                                let point = index
                                    .get_point(id)
                                    .expect("IO error getting point")
                                    .expect("Could not get point");
                                Dist::new(id, point.dist2other(&query))
                            })
                            .collect();
                        let new =
                            Self::select_simple(candidate_dists, index.get_m_at_layer(*layer_nb));
                        index
                            .replace_neighbors(*n, new.iter().copied(), *layer_nb)
                            .unwrap();
                    }
                }
            }
        }
        info!("Finished making connections");
        Ok(())
    }

    fn select_simple(mut cands: Vec<Dist>, m: usize) -> IntSet<INT> {
        cands.sort();
        IntSet::from_iter(cands.iter().take(m).map(|dist| dist.id))
    }

    fn search_layer(
        &mut self,
        layer: INT,
        query: &Query,
        index: &mut impl Index,
        ef: usize,
    ) -> Result<(), Error> {
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
            let cand_neighbors = match index.neighbors(cand_dist.id, layer) {
                Ok(neighs) => neighs,
                Err(_) => return Err(Error::Other),
            };
            trace!("{0} has {1} neighbors", cand_dist.id, cand_neighbors.len());

            let q2cand_neighbors_dists: Vec<Dist> = cand_neighbors
                .iter()
                .filter(|node| self.results.insert_visited(**node))
                .map(|node| {
                    let dist = match query {
                        Query::ID(id) => index
                            .distance(*node, *id)
                            .expect("IO error computing distance")
                            .expect("Could not compute distance"),
                        Query::Point(p) => index
                            .get_point(*node)
                            .expect("IO error getting point")
                            .expect("Could not get point")
                            .dist2other(p),
                    };
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

    // pub fn select_simple(&self, results: &mut Results, m: usize) {
    //     results.select_simple(m);
    // }

    pub fn select_heuristic(
        &mut self,
        index: &mut impl Index,
        layer: INT,
        id: INT,
        m: usize,
        extend_cands: bool,
        keep_pruned: bool,
    ) -> Result<(), Error> {
        trace!(
            "Begin select_heuristic, selected is {:?}",
            self.results.selected
        );
        self.results.select_setup();
        if extend_cands {
            self.results
                .extend_candidates_with_neighbors(id, index, layer)?;
        }

        let node_e = self.results.candidates.pop_first().unwrap();
        self.results.insert_selected(node_e);

        while (!self.results.candidates.is_empty()) & (self.results.selected.len() < m as usize) {
            let node_e = self.results.candidates.pop_first().unwrap();
            let e_point = index.get_point(node_e.id).unwrap().unwrap();

            let nearest_selected = self.results.get_nearest_from_selected(&e_point, index);

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

    pub fn ann_by_vec(
        &mut self,
        index: &mut impl Index,
        vector: &Vec<f32>,
        n: usize,
        ef: usize,
    ) -> Result<Vec<NodeID>, Error> {
        info!("Start ANN search");
        let point = Point::new(vector);
        self.clear_results();
        self.results.insert_selected(Dist::new(
            index.get_ep(),
            index
                .distance2point(index.get_ep(), &point)
                .unwrap()
                .unwrap(),
        ));
        let nb_layers = index.get_nb_layers().expect("IO error getting nb layers");

        let query = Query::Point(point);
        for layer_nb in (1..nb_layers).rev() {
            self.search_layer(layer_nb, &query, index, 1)?;
        }
        self.search_layer(0, &query, index, ef)?;

        let anns: Vec<NodeID> = self
            .results
            .get_top_selected(n)
            .iter()
            .map(|dist| dist.id)
            .collect();
        info!("Finished ANN search");
        Ok(anns)
    }

    // pub fn insert_bulk(
    //     mut self,
    //     vectors: Vec<Vec<f32>>,
    //     nb_threads: usize,
    //     verbose: bool,
    // ) -> Result<impl Index, String> {
    //     info!("inserting {0} vectors to index in bulk", vectors.len());
    //     let stored_ids: HashSet<NodeID> =
    //         HashSet::from_iter(self.store_points(vectors).iter().cloned());

    //     let bar = get_progress_bar("Building index".to_string(), stored_ids.len(), verbose);
    //     let bar_arc = Arc::new(bar);

    //     let index_arc = Arc::new(self);

    //     for layer_nb in (0..index_arc.get_nb_layers()).rev() {
    //         let layer = index_arc.get_layer(layer_nb);
    //         info!("Connecting nodes in layer {layer_nb}, using {nb_threads} threads",);
    //         let chunks = ((layer.nb_nodes() as f64) / (nb_threads as f64)).ceil() as usize;
    //         let mut ids: Vec<Vec<NodeID>> = layer
    //             .iter_nodes()
    //             .filter(|id| {
    //                 stored_ids.contains(id)
    //                     & (index_arc.get_point(*id).unwrap().level == layer_nb as u8)
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
