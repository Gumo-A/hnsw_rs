use graph::dist::Dist;
use log::{trace, warn};
use points::point::Point;

use crate::template::{results::Results, HNSW};

use crate::template::searcher::Searcher;

/// The Inserter is a wrapper around two other structs:
/// - Results
/// - Searcher
///
/// The role of the Inserter is to provide a clean interface to
/// the insertion algorithm through three methods:
/// - `build_insertion_results(index, point)`: finds the connections to make
/// at each layer of the `index` for `point`
/// - `get_results()`: exposes insertion results
/// - `get_results_mut()`: idem, but can mutate the results
pub struct Inserter {
    results: Results,
    searcher: Searcher,
}

impl Inserter {
    pub fn new() -> Self {
        Inserter {
            results: Results::new(),
            searcher: Searcher::new(),
        }
    }

    pub fn get_results(&self) -> &Results {
        &self.results
    }

    pub fn get_results_mut(&mut self) -> &mut Results {
        &mut self.results
    }

    pub fn build_insertion_results(&mut self, index: &HNSW, point: &Point) -> Result<(), String> {
        trace!("Begin insertion results");
        if point.id == index.params.ep {
            warn!("Tried to build insertion results for point {}, but it is the EP, not building results", point.id);
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
            point.id,
            index.params.ep
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
            self.searcher
                .search_layer(&mut self.results, layer, point, &index, 1)?;
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
            self.searcher.search_layer(
                &mut self.results,
                layer,
                point,
                &index,
                index.params.ef_cons,
            )?;
            // self.searcher
            //     .select_simple(&mut self.results, index.params.m);
            self.searcher.select_heuristic(
                &mut self.results,
                layer,
                point,
                &index.points,
                index.params.m,
                true,
                true,
            )?;
            self.results.save_layer_results(layer_nb, point.id);
        }
        Ok(())
    }
}
