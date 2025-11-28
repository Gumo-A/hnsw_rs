use graph::nodes::Dist;
use points::point::Point;

use crate::template::{results::Results, VecSer, HNSW};

use crate::template::searcher::Searcher;

pub struct Inserter {
    results: Results,
    searcher: Searcher,
    new_layers: bool,
}

impl Inserter {
    pub fn new() -> Self {
        Inserter {
            results: Results::new(),
            searcher: Searcher::new(),
            new_layers: false,
        }
    }

    pub fn get_results(&self) -> &Results {
        &self.results
    }

    pub fn get_results_mut(&mut self) -> &mut Results {
        &mut self.results
    }

    pub fn build_insertion_results<T: VecSer>(
        &mut self,
        index: &HNSW<T>,
        point: &Point<T>,
    ) -> Result<(), String> {
        self.setup_insert(index, point);
        self.traverse_layers_above(index, point)?;
        self.traverse_layers_below(index, point)?;
        Ok(())
    }

    pub fn setup_insert<T: VecSer>(&mut self, index: &HNSW<T>, point: &Point<T>) {
        self.results.clear_all();

        let dist2ep = index
            .distance(index.ep, point.id)
            .expect("Could not compute distance between EP and point to insert.");

        self.results.push_selected(Dist::new(index.ep, dist2ep));
    }

    pub fn traverse_layers_above<T: VecSer>(
        &mut self,
        index: &HNSW<T>,
        point: &Point<T>,
    ) -> Result<(), String> {
        let layers_len = index.layers.len() as u8;

        for layer_nb in (point.level + 1..layers_len).rev() {
            let layer = index.get_layer(&layer_nb);
            self.searcher
                .search_layer(&mut self.results, layer, point, &index.points, 1)?;
            if layer_nb == 0 {
                break;
            }
        }
        Ok(())
    }
    pub fn traverse_layers_below<T: VecSer>(
        &mut self,
        index: &HNSW<T>,
        point: &Point<T>,
    ) -> Result<(), String> {
        let bound = (point.level).min((index.layers.len() - 1) as u8);
        for layer_nb in (0..=bound).rev().map(|x| x as u8) {
            let layer = index.get_layer(&layer_nb);
            self.searcher.search_layer(
                &mut self.results,
                layer,
                point,
                &index.points,
                index.params.ef_cons,
            )?;
            self.searcher.select_heuristic(
                &mut self.results,
                layer,
                point,
                &index.points,
                index.params.m,
                false,
                true,
            )?;
            self.results.insert_layer_results(layer_nb, point.id);
        }
        Ok(())
    }
}
