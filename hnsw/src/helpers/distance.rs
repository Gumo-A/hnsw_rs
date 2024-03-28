use ndarray::{Array, Array1, ArrayView, Dim};

pub fn v2v_dist(a: &Array<f32, Dim<[usize; 1]>>, b: &Array<f32, Dim<[usize; 1]>>) -> f32 {
    1.0 - a.dot(b)
}

pub fn v2m_dist(
    a: &ArrayView<f32, Dim<[usize; 1]>>,
    b: &ArrayView<f32, Dim<[usize; 2]>>,
) -> Array1<f32> {
    1.0 - a.dot(b)
}

pub fn get_nn_bf(
    a: &ArrayView<f32, Dim<[usize; 1]>>,
    b: &ArrayView<f32, Dim<[usize; 2]>>,
    n: usize,
) -> Vec<(usize, f32)> {
    let distances: Vec<f32> = v2m_dist(&a, &b.t()).to_vec();
    let mut indices_distances: Vec<(usize, f32)> = distances
        .iter()
        .enumerate()
        .map(|x| (x.0, x.1.to_owned()))
        .collect();
    indices_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    indices_distances[1..(n + 1).min(b.dim().0)].to_vec()
}

pub fn norm_vector(vector: Array<f32, Dim<[usize; 1]>>) -> Array<f32, Dim<[usize; 1]>> {
    let norm: f32 = vector.map(|x| x.powf(2.0)).sum().powf(0.5);
    vector / norm
}
