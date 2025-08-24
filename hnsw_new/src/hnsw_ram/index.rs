use crate::config::Config;
use crate::hnsw_ram::layers::Layers;
use crate::hnsw_ram::points::{Point, Points};
use crate::searcher::Searcher;

pub struct HNSWRAM {
    points: Points,
    config: Config,
    layers: Layers,
}

impl HNSWRAM {
    pub fn new(m: u8, dim: u32) -> Self {
        Self {
            points: Points::Empty,
            config: Config::new(m, dim),
            layers: Layers::new(),
        }
    }

    pub fn ann_by_vector(
        &self,
        vector: &Vec<f32>,
        n: usize,
        ef: usize,
    ) -> Result<Vec<u32>, String> {
        let mut searcher = Searcher::new();

        let point = Point::new(0, 0, &self.center_vector(vector)?);

        searcher
            .selected
            .push(point.dist2other(self.points.get_point(0).unwrap()));
        let nb_layer = self.layers.len();

        for layer_nb in (1..nb_layer).rev().map(|x| x as u8) {
            self.search_layer(
                &mut searcher,
                self.layers.get(&(layer_nb)).unwrap(),
                &point,
                1,
            )?;
        }

        let layer_0 = &self.layers.get(&0).unwrap();
        self.search_layer(&mut searcher, layer_0, &point, ef)?;

        let anns: Vec<Dist> = searcher.selected.into_sorted_vec();
        Ok(anns.iter().take(n).map(|x| x.id).collect())
    }
}
