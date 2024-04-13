pub mod filter;
pub mod index;

#[cfg(test)]
mod tests {

    use crate::{hnsw::index::HNSW, points::Point};
    use ndarray::{Array1, Array2};
    use rand::Rng;

    #[test]
    fn hnsw_construction() {
        let _index: HNSW = HNSW::new(12, None, 100);
        let _index: HNSW = HNSW::from_params(12, Some(9), None, None, None, 100);
        let _index: HNSW = HNSW::from_params(12, None, Some(18), None, None, 100);
        let _index: HNSW = HNSW::from_params(12, None, None, Some(0.25), None, 100);
        let _index: HNSW = HNSW::from_params(12, None, None, None, Some(100), 100);
        let _index: HNSW = HNSW::from_params(12, Some(9), Some(18), Some(0.25), Some(100), 100);
    }

    #[test]
    fn hnsw_insert() {
        let mut rng = rand::thread_rng();
        let dim = 100;
        let mut index: HNSW = HNSW::new(12, None, dim);
        let n: usize = 100;

        for i in 0..n {
            let vector = Array1::from_vec((0..dim).map(|_| rng.gen::<f32>()).collect());
            let point = Point::new(i, vector.view(), None, None);
            index.insert(&point, Some(0));
        }

        let already_in_index = 0;
        let vector = Array1::from_vec((0..dim).map(|_| rng.gen::<f32>()).collect());
        let point = Point::new(already_in_index, vector.view(), None, None);
        index.insert(&point, Some(0));
        // index.insert(already_in_index, &vector.view(), None);
        assert_eq!(index.node_ids.len(), n);
    }

    #[test]
    fn ann() {
        let mut rng = rand::thread_rng();
        let dim = 100;
        let mut index: HNSW = HNSW::new(12, None, dim);
        let n: usize = 100;

        for i in 0..n {
            let vector = Array1::from_vec((0..dim).map(|_| rng.gen::<f32>()).collect());
            let point = Point::new(i, vector.view(), None, None);
            index.insert(&point, Some(0));
        }

        let n = 10;
        let vector = index.layers.get(&0).unwrap().node(n).vector.to_owned();
        let anns = index.ann_by_vector(&vector.view(), 10, 16);
        println!("ANNs of {:?}", n);
        for e in anns {
            println!("{:?}", e);
        }
    }

    #[test]
    fn build_multithreaded() {
        let mut index = HNSW::new(12, None, 100);
        index
            .build_index(&Array2::zeros((10, 100)), false)
            .expect("Error");
    }
}
