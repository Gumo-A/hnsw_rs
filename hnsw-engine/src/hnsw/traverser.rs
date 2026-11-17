use std::{env, sync::Arc, thread::Builder};

use graph::dist::Dist;
use hnsw::helpers::get_progress_bar;
use log::{debug, trace};
use points::point::Point;
use vectors::VecBase;

use crate::{
    engine::INT,
    hnsw::results::Results,
    index::{HNSWErr, Index},
};

const DEFAULT_HNSW_THREADS: usize = 4;

enum Query {
    ID(INT),
    Point(Point),
}
/// The role of the Traverser is to encapsulate the navigation
/// logic of the HNSW data structure, exposing an `insert` and
/// an `ann_by_vec` methods.
pub struct Traverser {
    results: Results,
    heuristic: bool,
    ext_cands: bool,
    keep_pruned: bool,
}

impl Traverser {
    pub fn new() -> Self {
        Self {
            results: Results::new(),
            heuristic: env::var("HNSW_HEURISTIC").is_ok(),
            ext_cands: env::var("HNSW_EXT_CANDS").is_ok(),
            keep_pruned: env::var("HNSW_KEEP_PRUNED").is_ok(),
        }
    }

    pub fn insert_bulk(&self, index: impl Index + 'static, ids: Vec<INT>) -> Result<(), HNSWErr> {
        let nb_threads = env::var("HNSW_THREADS")
            .ok()
            .and_then(|nb_threads| nb_threads.parse::<usize>().ok())
            .unwrap_or(DEFAULT_HNSW_THREADS);
        let index_0 = Arc::new(index);
        let nb_levels = index_0.get_nb_levels()?;
        debug!("Inserting in bulk with nb. of levels {nb_levels} and {nb_threads} threads");
        for level in (0..nb_levels).rev() {
            let index_1 = index_0.clone();
            let level_ids: Vec<INT> = ids
                .iter()
                .filter(|id| index_1.get_level(**id).unwrap() >= level)
                .copied()
                .collect();
            if level_ids.len() == 0 {
                debug!("No points to insert with level {level}, skiping level");
                continue;
            } else {
                debug!("Inserting {} points at level {level}", level_ids.len());
            }
            let bar = get_progress_bar(
                format!("Inserting at level {level}").as_str(),
                level_ids.len(),
            );
            let chunk_size = ((level_ids.len() as f64) / (nb_threads as f64)).ceil() as usize;
            let mut level_ids: Vec<Vec<INT>> = level_ids
                .chunks(chunk_size)
                .map(|chunk| Vec::from(chunk))
                .collect();

            let mut handlers = Vec::new();
            let mut idx = 0;
            while let Some(ids_split) = level_ids.pop() {
                trace!("Sending {} node IDs to thread {idx}", ids_split.len());
                let index_2 = index_1.clone();
                let bar_cp = bar.clone();
                let thread = Builder::new()
                    .name(format!("thread {idx}"))
                    .spawn(move || {
                        let mut inserter = Self::new();
                        for id in ids_split.iter() {
                            inserter.insert(index_2.clone(), *id, idx).unwrap();
                            bar_cp.inc(1);
                        }
                    })
                    .expect("Could not spawn a thread");
                handlers.push(thread);
                idx += 1;
            }
            for handle in handlers {
                handle.join().unwrap();
            }
        }
        index_0.commit_results()
    }

    pub fn insert(
        &mut self,
        index: Arc<impl Index>,
        id: INT,
        thread_id: i32,
    ) -> Result<(), HNSWErr> {
        let is_new_ep = self.build_insertion_results(Arc::clone(&index), id, thread_id)?;
        self.make_connections(Arc::clone(&index), thread_id)?;
        if is_new_ep { index.set_ep(id) } else { Ok(()) }
    }

    fn build_insertion_results(
        &mut self,
        index: Arc<impl Index>,
        id: INT,
        thread_id: i32,
    ) -> Result<bool, HNSWErr> {
        let level = index.get_level(id)?;
        debug!("T{thread_id}: Begin insertion results for {id} at level {level}");
        let nb_levels = self.setup_insert(Arc::clone(&index), id, thread_id)?;
        let is_new_ep =
            self.traverse_layers_above(Arc::clone(&index), id, level, nb_levels, thread_id)?;
        self.traverse_layers_below(index, id, level, nb_levels, thread_id)?;
        debug!("T{thread_id}: Finish insertion results for {id}");
        Ok(is_new_ep)
    }

    fn clear_results(&mut self) {
        self.results.clear_all();
    }

    fn setup_insert(
        &mut self,
        index: Arc<impl Index>,
        id: INT,
        thread_id: i32,
    ) -> Result<INT, HNSWErr> {
        debug!("T{thread_id}: Setting up insertion for {id}");
        self.clear_results();
        let ep = index.get_ep();
        let nb_levels = index.get_entry(ep)?.nb_levels()?;
        let dist2ep = index.distance(ep, id)?;
        trace!("T{thread_id}: Distance between {id} and EP {ep} is {dist2ep}",);
        self.results.insert_selected(Dist::new(ep, dist2ep));
        debug!("T{thread_id}: Finished setting up insertion for {id}");
        // needs to return the nb. of levels for this insertion, because
        // this value comes from the EP, and the EP can change during this
        // insertion
        Ok(nb_levels)
    }

    fn traverse_layers_above(
        &mut self,
        index: Arc<impl Index>,
        id: INT,
        level: INT,
        nb_levels: INT,
        thread_id: i32,
    ) -> Result<bool, HNSWErr> {
        debug!("T{thread_id}: Traversing levels above {level} for {id}");
        let is_new_ep = level > nb_levels - 1;
        if is_new_ep {
            debug!(
                "T{thread_id}: Found new EP: highest level was {}, new is {level}",
                nb_levels - 1
            );
        }
        for level_nb in (level + 1..nb_levels).rev() {
            debug!("T{thread_id}: Searching NNs for {id} at level {level_nb}",);
            self.search_level(level_nb, &Query::ID(id), Arc::clone(&index), 1, thread_id)?;
        }
        debug!("T{thread_id}: Finished traversing levels above {level} for {id}");
        Ok(is_new_ep)
    }

    fn traverse_layers_below(
        &mut self,
        index: Arc<impl Index>,
        id: INT,
        level: INT,
        nb_levels: INT,
        thread_id: i32,
    ) -> Result<(), HNSWErr> {
        debug!("T{thread_id}: Traversing levels at or below level {level} for {id}");
        let index_cp = Arc::clone(&index);
        let bound = (level + 1).min(nb_levels);
        for level_nb in (0..bound).rev() {
            trace!("T{thread_id}: Searching NNs for {id} at level {level_nb}",);
            trace!("T{thread_id}: Entry point is {:?}", self.results.selected);
            self.search_level(
                level_nb,
                &Query::ID(id),
                Arc::clone(&index_cp),
                index_cp.get_ef_cons(),
                thread_id,
            )?;
            if self.heuristic {
                self.select_heuristic(
                    Arc::clone(&index_cp),
                    level_nb,
                    id,
                    index.get_m(),
                    thread_id,
                )?;
            } else {
                self.results.select_simple(index.get_m());
            }
            self.results.save_layer_results(level_nb, id);
        }
        debug!("T{thread_id}: Finished traversing levels at or below level {level} for {id}");
        Ok(())
    }

    fn make_connections(&self, index: Arc<impl Index>, thread_id: i32) -> Result<(), HNSWErr> {
        debug!("T{thread_id}: Making connections");
        let mut tot_connections: isize = 0;
        for (layer_nb, node_data) in self.results.insertion_results.iter() {
            for (node, neighbors) in node_data.iter() {
                tot_connections +=
                    index.add_neighbors(*node, neighbors.iter().copied(), *layer_nb)? as isize;
                for n in neighbors {
                    let degree = index.degree(*n, *layer_nb)?;
                    if degree > index.get_m_at_level(*layer_nb) {
                        trace!("T{thread_id}: Capping neighbors for {n}, currently has {degree}");
                        let m = index.get_m_at_level(*layer_nb);
                        tot_connections += index.cap_neighbors(*n, *layer_nb, m)?;
                    }
                }
            }
        }
        debug!("T{thread_id}: Made {tot_connections} connections");
        Ok(())
    }

    fn search_level(
        &mut self,
        level: INT,
        query: &Query,
        index: Arc<impl Index>,
        ef: usize,
        thread_id: i32,
    ) -> Result<(), HNSWErr> {
        trace!(
            "T{thread_id}: Search level: entry point is {:?}",
            self.results.selected
        );

        self.results.extend_candidates_with_selected();
        self.results.extend_visited_with_selected();

        while !self.results.candidates.is_empty() {
            trace!("T{thread_id}: Begin loop");
            let cand_dist = self.results.candidates.pop_first().unwrap();
            let furthest2q_dist = self.results.selected.last().unwrap();
            trace!("T{thread_id}: Best candidate: {cand_dist:?}");
            trace!("T{thread_id}: Worst selected: {furthest2q_dist:?}");
            if cand_dist > *furthest2q_dist {
                trace!(
                    "T{thread_id}: Best candidate is further away than worst selected, breaking from loop"
                );

                break;
            }
            let cand_neighbors = index.neighbors(cand_dist.id, level)?;
            trace!(
                "T{thread_id}: {0} has {1} neighbors",
                cand_dist.id,
                cand_neighbors.len()
            );

            let mut q2cand_neighbors_dists: Vec<Dist> = Vec::new();
            for node in cand_neighbors
                .iter()
                .filter(|node| self.results.insert_visited(**node))
                .copied()
            {
                let dist = match query {
                    Query::ID(id) => index.distance(node, *id)?,
                    Query::Point(p) => index.distance2point(node, p)?,
                };
                q2cand_neighbors_dists.push(Dist::new(node, dist));
            }

            trace!("T{thread_id}: Selecting among the best of the candidate's neighbors");

            for n2q_dist in q2cand_neighbors_dists {
                let f2q_dist = self.results.selected.last().unwrap();

                if self.results.selected.len() < ef as usize {
                    trace!("T{thread_id}: Need to fill selected struct, inserting {n2q_dist:?}");

                    self.results.insert_selected(n2q_dist);
                    self.results.insert_candidate(n2q_dist);
                    continue;
                }

                if n2q_dist < *f2q_dist {
                    trace!(
                        "T{thread_id}: Candidate's neighbor {n2q_dist:?} is better than worst selected {f2q_dist:?}, inserting it"
                    );
                    self.results.insert_selected(n2q_dist);
                    self.results.insert_candidate(n2q_dist);

                    if self.results.selected.len() > ef as usize {
                        let removed = self.results.selected.pop_last().unwrap();

                        trace!("T{thread_id}: Selected was too full, removed {removed:?}");
                    }
                }
            }
        }
        debug!(
            "T{thread_id}: Finished traversing level | Selected: {:?}",
            self.results.selected
        );
        self.results.clear_candidates();
        self.results.clear_visited();
        Ok(())
    }

    pub fn select_heuristic(
        &mut self,
        index: Arc<impl Index>,
        layer: INT,
        id: INT,
        m: usize,
        thread_id: i32,
    ) -> Result<(), HNSWErr> {
        debug!(
            "T{thread_id}: Begin select_heuristic, selected is {:?}",
            self.results.selected
        );
        self.results.select_setup();
        if self.ext_cands {
            self.results
                .extend_candidates_with_neighbors(id, Arc::clone(&index), layer)?;
        }

        let node_e = self.results.candidates.pop_first().unwrap();
        self.results.insert_selected(node_e);

        while (!self.results.candidates.is_empty()) & (self.results.selected.len() < m as usize) {
            let node_e = self.results.candidates.pop_first().unwrap();

            let nearest_selected = self
                .results
                .get_nearest_from_selected(node_e.id, Arc::clone(&index));

            if node_e < nearest_selected {
                self.results.insert_selected(node_e);
            } else if self.keep_pruned {
                self.results.insert_visited_h(node_e);
            }
        }

        if self.keep_pruned {
            while (!self.results.visited_h.is_empty()) & (self.results.selected.len() < m as usize)
            {
                let node_e = self.results.visited_h.pop_first().unwrap();
                self.results.insert_selected(node_e);
            }
        }

        debug!(
            "T{thread_id}: Finished select_heuristic, selected is {:?}",
            self.results.selected
        );
        Ok(())
    }

    pub fn ann_by_vec(
        &mut self,
        index: Arc<impl Index>,
        vector: &Vec<f32>,
        n: usize,
        ef: usize,
    ) -> Result<Vec<INT>, HNSWErr> {
        let point = Point::new(vector);
        self.ann_by_point(index, point, n, ef)
    }

    pub fn ann_by_point(
        &mut self,
        index: Arc<impl Index>,
        point: Point,
        n: usize,
        ef: usize,
    ) -> Result<Vec<INT>, HNSWErr> {
        debug!("Start ANN search with n {} and ef {}", n, ef);
        self.clear_results();
        self.results.insert_selected(Dist::new(
            index.get_ep(),
            index.distance2point(index.get_ep(), &point)?,
        ));
        let nb_levels = index.get_nb_levels()?;

        let query = Query::Point(point);
        for level in (1..nb_levels).rev() {
            self.search_level(level, &query, Arc::clone(&index), 1, 0)?;
        }
        self.search_level(0, &query, index, ef, 0)?;

        let anns: Vec<INT> = self
            .results
            .get_top_selected(n)
            .iter()
            .map(|dist| dist.id)
            .collect();
        debug!("Finished ANN search, found {}/{} neighbors", anns.len(), n);
        Ok(anns)
    }
}
