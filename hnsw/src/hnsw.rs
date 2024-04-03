pub mod filter;
pub mod index;

#[cfg(test)]
mod tests {

    use std::collections::HashMap;

    use crate::hnsw::filter::FilterVectorHolder;
    use crate::hnsw::index::HNSW;
    use ndarray::{Array1, Array2};
    use rand::Rng;

    #[test]
    fn hnsw_construction() {
        let _index: HNSW = HNSW::new(3, 12, None, 100);
        let _index: HNSW = HNSW::from_params(3, 12, Some(9), None, None, None, 100);
        let _index: HNSW = HNSW::from_params(3, 12, None, Some(18), None, None, 100);
        let _index: HNSW = HNSW::from_params(3, 12, None, None, Some(0.25), None, 100);
        let _index: HNSW = HNSW::from_params(3, 12, None, None, None, Some(100), 100);
        let _index: HNSW = HNSW::from_params(3, 12, Some(9), Some(18), Some(0.25), Some(100), 100);
    }

    #[test]
    fn hnsw_insert() {
        let mut rng = rand::thread_rng();
        let dim = 100;
        let mut index: HNSW = HNSW::new(3, 12, None, dim);
        let n: usize = 100;
        let mut filters = FilterVectorHolder::new(n);
        let mut cache: HashMap<(usize, usize), f32> = HashMap::new();

        for i in 0..n {
            let vector = Array1::from_vec((0..dim).map(|_| rng.gen::<f32>()).collect());
            index.insert(i.try_into().unwrap(), &vector, &mut filters, &mut cache);
        }

        let already_in_index = 0;
        let vector = Array1::from_vec((0..dim).map(|_| rng.gen::<f32>()).collect());
        index.insert(already_in_index, &vector, &mut filters, &mut cache);
        assert_eq!(index.node_ids.len(), n);
    }

    #[test]
    fn ann() {
        let mut rng = rand::thread_rng();
        let dim = 100;
        let mut index: HNSW = HNSW::new(3, 12, None, dim);
        let n: usize = 100;
        let mut filters = FilterVectorHolder::new(n);
        let mut cache: HashMap<(usize, usize), f32> = HashMap::new();

        for i in 0..n {
            let vector = Array1::from_vec((0..dim).map(|_| rng.gen::<f32>()).collect());
            index.insert(i.try_into().unwrap(), &vector, &mut filters, &mut cache);
        }

        let n = 10;
        let vector = index.layers.get(&0).unwrap().node(n).1.to_owned();
        let anns = index.ann_by_vector(&vector, 10, 16);
        println!("ANNs of {:?}", n);
        for e in anns {
            println!("{:?}", e);
        }
    }

    #[test]
    fn build_multithreaded() {
        let mut index = HNSW::new(12, 12, None, 100);
        index.build_index(Vec::from_iter(0..10), &Array2::zeros((10, 100)));
    }
}
