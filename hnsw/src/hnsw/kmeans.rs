use std::collections::HashMap;

use crate::hnsw::points::{Point, Points};
use nohash_hasher::BuildNoHashHasher;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub fn kmeans(nb_centroids: u8, iterations: usize, points: &Points) -> Points {
    let mut centroids = init_centroids(nb_centroids, &points);
    for _ in 0..iterations {
        centroids = update_centroids(&centroids, &points);
    }
    centroids
}

fn update_centroids(centroids: &Points, points: &Points) -> Points {
    let clusters = get_clusters(centroids, points);
    let means = get_means(&clusters, points);
    means
}

fn init_centroids(nb_centroids: u8, points: &Points) -> Points {
    let mut points_idx: Vec<usize> = points.iterate().map(|(idx, _)| *idx).collect();
    points_idx.shuffle(&mut thread_rng());

    let mut collection = HashMap::with_hasher(BuildNoHashHasher::default());
    collection.extend(
        points_idx
            .iter()
            .enumerate()
            .take(nb_centroids as usize)
            .map(|(centroid_id, point_id)| (centroid_id, points.get_point(*point_id).clone())),
    );
    Points::Collection(collection)
}

/// Asigns clusters to each point in 'points', given 'centroids'
fn get_clusters(centroids: &Points, points: &Points) -> HashMap<u8, Vec<usize>> {
    let mut clusters: HashMap<u8, Vec<usize>> =
        HashMap::from_iter((0..centroids.len()).map(|centroid_id| (centroid_id as u8, Vec::new())));

    for (point_id, point) in points.iterate() {
        let mut distances = Vec::new();
        for (centroid_id, centroid) in centroids.iterate() {
            let dist = centroid.dist2vec(&point.vector);
            distances.push((centroid_id, dist));
        }
        distances.sort_by(|a, b| a.1.cmp(&b.1));
        let cluster = *distances
            .get(0)
            .expect("KMeans Error: there was no 0th element in distances vector.")
            .0 as u8;
        clusters.get_mut(&cluster).unwrap().push(*point_id);
    }
    clusters
}

fn get_means(clusters: &HashMap<u8, Vec<usize>>, points: &Points) -> Points {
    let mut means = HashMap::with_hasher(BuildNoHashHasher::default());

    for (cluster_id, point_ids) in clusters.iter() {
        let points_in_cluster = point_ids.len() as f32;
        let mut new_centroid = Vec::from_iter((0..points.get_point(0).dim()).map(|_| 0.0));
        for point in point_ids.iter().map(|id| points.get_point(*id)) {
            for (idx, x) in point.vector.get_vec().iter().enumerate() {
                new_centroid[idx] += x / points_in_cluster;
            }
        }
        means.insert(
            *cluster_id as usize,
            Point::new(*cluster_id as usize, new_centroid, false),
        );
    }
    Points::Collection(means)
}

// Not even necessary
// fn get_full_prrecision_points(points: &Points) -> Points {
//     let mut points = points.clone();
//     points
//         .iterate_mut()
//         .for_each(|(_, point)| point.to_full_precision());
//     points
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{helpers::glove::load_glove_array, hnsw::points::Vector};

    // TODO: compare results with sklearn's
    #[test]
    fn find_kmeans() -> std::io::Result<()> {
        let (_, embeddings) = load_glove_array(100, 10_000, false, 0)?;
        let mut collection = HashMap::with_hasher(BuildNoHashHasher::default());
        collection.extend(
            embeddings
                .iter()
                .enumerate()
                .map(|(id, x)| (id, Point::new(id, x.clone(), false))),
        );

        let points = Points::Collection(collection);

        let centroids = kmeans(8, 500, &points);

        println!("centroids");
        for (_, c) in centroids.iterate() {
            match &c.vector {
                Vector::Full(full) => println!("{:?}", &full[0..5]),
                Vector::Compressed(quant) => println!("{:?}", &quant.reconstruct()[0..5]),
            }
        }

        Ok(())
    }
}
