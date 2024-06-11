use crate::hnsw::{dist::Dist, lvq::LVQVec};

pub fn l2_compressed(vector: &LVQVec, compressed: &LVQVec) -> Dist {
    compressed.dist2other(vector)
}

pub fn v2v_dist(a: &Vec<f32>, b: &Vec<f32>) -> usize {
    let mut result: f32 = 0.0;
    for (x, y) in a.iter().zip(b) {
        result += (x - y).powi(2);
    }
    (result * 10_000.0) as usize
}

// TODO: repair brute_force_nn
// pub fn v2m_dist(
//     a: &ArrayView<f32, Dim<[usize; 1]>>,
//     b: &ArrayView<f32, Dim<[usize; 2]>>,
// ) -> Array1<f32> {
//     1.0 - a.dot(b)
// }

// pub fn get_nn_bf(a: &Vec<f32>, b: &Vec<Vec<f32>>, n: usize) -> Vec<(usize, f32)> {
//     let distances: Vec<f32> = v2m_dist(&a, &b.t()).to_vec();
//     let mut indices_distances: Vec<(usize, f32)> = distances
//         .iter()
//         .enumerate()
//         .map(|x| (x.0, x.1.to_owned()))
//         .collect();
//     indices_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
//     indices_distances[1..(n + 1).min(b.dim().0)].to_vec()
// }

// pub fn norm_vector(vector: &ArrayView<f32, Dim<[usize; 1]>>) -> Array<f32, Dim<[usize; 1]>> {
//     let norm: f32 = vector.map(|x| x.powf(2.0)).sum().powf(0.5);
//     vector / norm
// }
