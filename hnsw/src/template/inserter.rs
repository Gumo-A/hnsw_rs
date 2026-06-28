use graph::nodes::Dist;
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
        if point.id == index.params.ep {
            return Ok(());
        }
        self.setup_insert(index, point);
        self.traverse_layers_above(index, point)?;
        self.traverse_layers_below(index, point)?;
        Ok(())
    }

    fn setup_insert(&mut self, index: &HNSW, point: &Point) {
        self.results.clear_all();

        let dist2ep = index
            .distance(index.params.ep, point.id)
            .expect("Could not compute distance between EP and point to insert.");

        self.results
            .insert_selected(Dist::new(index.params.ep, dist2ep));
    }

    fn traverse_layers_above(&mut self, index: &HNSW, point: &Point) -> Result<(), String> {
        let layers_len = index.layers.len();

        for layer_nb in (point.level as usize + 1..layers_len).rev() {
            let layer = index.get_layer(layer_nb);
            self.searcher
                .search_layer(&mut self.results, layer, point, &index.points, 1)?;
        }
        Ok(())
    }

    fn traverse_layers_below(&mut self, index: &HNSW, point: &Point) -> Result<(), String> {
        let bound = (point.level as usize).min(index.layers.len() - 1);
        for layer_nb in (0..=bound).rev() {
            let layer = index.get_layer(layer_nb);
            self.searcher.search_layer(
                &mut self.results,
                layer,
                point,
                &index.points,
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
