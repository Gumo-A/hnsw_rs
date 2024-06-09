pub mod distid;
pub mod graph;
pub mod index;
pub mod lvq;
pub mod params;
pub mod points;

#[cfg(test)]
mod tests {

    use crate::hnsw::{index::HNSW, points::Point};
    use rand::Rng;

    use super::params::Params;

    #[test]
    fn hnsw_construction() {
        let params = Params::from_m(12, 100);
        let _index: HNSW = HNSW::new(params.m, None, params.dim);
        let _index: HNSW = HNSW::from_params(params);
    }

    #[test]
    fn hnsw_insert() {
        let mut rng = rand::thread_rng();
        let dim = 100;
        let mut index: HNSW = HNSW::new(12, None, dim);
        let n: usize = 100;

        for i in 0..n {
            let vector = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let point = Point::new(i, vector, false);
            let mut level = 0;
            if rng.gen::<f32>() > 0.5 {
                level = 1;
            }
            index.insert(point, level);
        }

        let already_in_index = 0;
        let vector = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let point = Point::new(already_in_index, vector, false);
        index.insert(point, 0);
        assert_eq!(index.points.len(), n);
    }

    #[test]
    fn hnsw_serialize() -> std::io::Result<()> {
        let mut rng = rand::thread_rng();
        let dim = 100;
        let mut index: HNSW = HNSW::new(12, None, dim);
        let n: usize = 100;

        for i in 0..n {
            let vector = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let point = Point::new(i, vector, true);
            let mut level = 0;
            if rng.gen::<f32>() > 0.5 {
                level = 1;
            }
            index.insert(point, level);
        }
        let index_path = "./hnsw_index.json";
        index.save(index_path)?;
        let loaded_index = HNSW::from_path(index_path)?;
        assert_eq!(n, loaded_index.points.len());
        std::fs::remove_file(index_path)?;
        Ok(())
    }
}
