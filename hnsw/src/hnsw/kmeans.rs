// TODO: improve performance. It seems to work, but its too slow.
use std::collections::HashMap;

use crate::hnsw::points::{Point, Points};
use nohash_hasher::BuildNoHashHasher;
use rand::seq::SliceRandom;
use rand::thread_rng;

use super::dist::Dist;
use indicatif::{ProgressBar, ProgressStyle};

const EARLY_STOP_TOL: f32 = 0.0001;

pub fn kmeans(nb_centroids: u8, iterations: usize, points: &Points) -> Points {
    let mut centroids = init_centroids(nb_centroids, &points);
    let bar = setup_progress_bar(iterations);
    for _ in 0..iterations {
        let new_centroids = update_centroids(&centroids, &points);
        if check_early_stop(&new_centroids, &centroids) {
            println!("Converged before reaching max iterations.");
            break;
        }
        centroids = new_centroids;
        bar.inc(1);
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
    // TODO: apply logic to avoid distance computations using the triangle inequality as in
    //       Charles Elkan (2003) "Using the Triangle Inequality to Accelerate K-Means"
    let centers_distances: HashMap<(usize, usize), Dist> = get_dists_among_centers(centroids);

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

fn get_dists_among_centers(centroids: &Points) -> HashMap<(usize, usize), Dist> {
    let mut centers_distances = HashMap::new();
    for idx in 0..centroids.len() {
        for jdx in 0..centroids.len() {
            if idx == jdx {
                continue;
            }
            centers_distances.insert(
                (idx.min(jdx), idx.max(jdx)),
                centroids
                    .get_point(idx)
                    .dist2vec(&centroids.get_point(jdx).vector),
            );
        }
    }
    centers_distances
}

fn setup_progress_bar(iterations: usize) -> ProgressBar {
    let bar = ProgressBar::new(iterations.try_into().unwrap());
    bar.set_style(
        ProgressStyle::with_template(
            "{msg} {wide_bar} {human_pos}/{human_len} {percent}% [ ETA: {eta} : Elapsed: {elapsed} ] {per_sec}",
        )
        .unwrap(),
    );
    bar.set_message(format!("Finding Clusters"));
    bar
}

fn check_early_stop(new_centers: &Points, old_centers: &Points) -> bool {
    for ((_, nc), (_, oc)) in new_centers.iterate().zip(old_centers.iterate()) {
        for (nv, ov) in nc
            .get_full_precision()
            .iter()
            .zip(oc.get_full_precision().iter())
        {
            if (nv - ov).abs() > EARLY_STOP_TOL {
                return false;
            }
        }
    }
    true
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
    use std::time::Instant;

    #[test]
    fn find_kmeans() -> std::io::Result<()> {
        let (_, embeddings) = load_glove_array(100, 400_000, false, 0)?;
        let mut collection = HashMap::with_hasher(BuildNoHashHasher::default());
        collection.extend(
            embeddings
                .iter()
                .enumerate()
                .map(|(id, x)| (id, Point::new(id, x.clone(), false))),
        );

        let points = Points::Collection(collection);

        let start = Instant::now();
        let centroids = kmeans(8, 500, &points);
        let end = Instant::now();
        println!(
            "Elapsed time: {}s",
            start.elapsed().as_secs() - end.elapsed().as_secs()
        );

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
