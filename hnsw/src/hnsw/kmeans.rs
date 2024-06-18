// TODO: improve performance. It seems to work, but its too slow.
use std::collections::{HashMap, HashSet};

use crate::hnsw::points::{Point, Points};
use rand::seq::SliceRandom;
use rand::thread_rng;

use super::dist::Dist;
use indicatif::{ProgressBar, ProgressStyle};

const EARLY_STOP_TOL: f32 = 0.0001;

pub fn partition_space(
    nb_centroids: usize,
    iterations: usize,
    points: &Vec<Point>,
) -> (Vec<Point>, Vec<Vec<usize>>) {
    let target_size = points.len() / nb_centroids;
    let mut centroids = init_centroids(nb_centroids, &points);
    let mut partitions = get_clusters(&centroids, points);
    let mut sum = 0.0;
    for (idx, cluster) in partitions.iter().enumerate() {
        let frac = (cluster.len() as f32) / (points.len() as f32);
        sum += frac;
        println!("Fraction of total, cluster {idx}: {}", frac);
    }
    println!("{sum}");
    let bar = setup_progress_bar(iterations);
    for _ in 0..iterations {
        let (new_centroids, new_clusters) = update_partitions(&centroids, &partitions, &points);
        // if check_early_stop_partitions(&new_clusters, target_size) {
        //     break;
        // }
        centroids = new_centroids;
        partitions = new_clusters;
        bar.inc(1);
    }
    let mut sum = 0.0;
    for (idx, cluster) in partitions.iter().enumerate() {
        let frac = (cluster.len() as f32) / (points.len() as f32);
        sum += frac;
        println!("Fraction of total, cluster {idx}: {}", frac);
    }
    println!("{sum}");
    (centroids, partitions)
}

pub fn kmeans(
    nb_centroids: usize,
    iterations: usize,
    points: &Vec<Point>,
) -> (Vec<Point>, Vec<Vec<usize>>) {
    let mut centroids = init_centroids(nb_centroids, &points);
    let bar = setup_progress_bar(iterations);
    for _ in 0..iterations {
        let new_centroids = update_centroids(&centroids, &points);
        if check_early_stop(&new_centroids, &centroids) {
            break;
        }
        centroids = new_centroids;
        bar.inc(1);
    }
    let clusters = get_clusters(&centroids, points);
    (centroids, clusters)
}

fn update_partitions(
    centroids: &Vec<Point>,
    clusters: &Vec<Vec<usize>>,
    points: &Vec<Point>,
) -> (Vec<Point>, Vec<Vec<usize>>) {
    let target = points.len() / centroids.len();
    let new_clusters = reduce_borders(centroids, clusters, points, 0.1, target);
    let new_centroids = get_means(&clusters, points);
    (new_centroids, new_clusters)
}

fn update_centroids(centroids: &Vec<Point>, points: &Vec<Point>) -> Vec<Point> {
    let clusters = get_clusters(centroids, points);
    let means = get_means(&clusters, points);
    means
}

fn init_centroids(nb_centroids: usize, points: &Vec<Point>) -> Vec<Point> {
    let mut points_idx: Vec<usize> = (0..points.len()).collect();
    points_idx.shuffle(&mut thread_rng());

    Vec::from_iter(
        points_idx
            .iter()
            .take(nb_centroids as usize)
            .map(|x| points.get(*x).unwrap().clone()),
    )
}

/// Asigns clusters to each point in 'points', given 'centroids'
/// Returns a HashMaps with keys being the cluster number and the values being a vector
/// of the indices in the vector 'points' that belong to that cluster.
fn get_clusters(centroids: &Vec<Point>, points: &Vec<Point>) -> Vec<Vec<usize>> {
    let mut clusters: Vec<Vec<usize>> = Vec::from_iter(
        (0..centroids.len()).map(|_| Vec::with_capacity(points.len() / centroids.len())),
    );

    // TODO: apply logic to avoid distance computations using the triangle inequality as in
    //       Charles Elkan (2003) "Using the Triangle Inequality to Accelerate K-Means"
    // let centers_distances: HashMap<(usize, usize), Dist> = get_dists_among_centers(centroids);

    for (point_idx, point) in points.iter().enumerate() {
        let mut distances = Vec::new();
        for (centroid_idx, centroid) in centroids.iter().enumerate() {
            let dist = centroid.dist2vec(&point.vector);
            distances.push((centroid_idx, dist));
        }
        distances.sort_by(|a, b| a.1.cmp(&b.1));
        let cluster = distances
            .get(0)
            .expect("KMeans Error: there was no 0th element in distances vector.")
            .0;
        clusters.get_mut(cluster).unwrap().push(point_idx);
    }
    clusters
}

fn get_means(clusters: &Vec<Vec<usize>>, points: &Vec<Point>) -> Vec<Point> {
    let mut means = Vec::new();
    for (cluster_id, point_indices) in clusters.iter().enumerate() {
        let points_in_cluster = point_indices.len() as f32;
        let mut new_centroid = Vec::from_iter((0..points[0].dim()).map(|_| 0.0));
        for point in point_indices.iter().map(|id| points.get(*id).unwrap()) {
            for (idx, x) in point.vector.get_vec().iter().enumerate() {
                new_centroid[idx] += x / points_in_cluster;
            }
        }
        means.push(Point::new(cluster_id as usize, new_centroid, false));
    }
    means
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

fn check_early_stop(new_centers: &Vec<Point>, old_centers: &Vec<Point>) -> bool {
    for (nc, oc) in new_centers.iter().zip(old_centers.iter()) {
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

fn check_early_stop_partitions(clusters: &Vec<Vec<usize>>, target_size: usize) -> bool {
    let tolerance = 1000;
    for cluster in clusters {
        if ((cluster.len() as isize) - (target_size as isize)).abs() > tolerance {
            return false;
        }
    }
    true
}

pub fn get_frontier_points(
    centroids: &Vec<Point>,
    clusters: &Vec<Vec<usize>>,
    points: &Vec<Point>,
    threshold: f32,
) -> Vec<usize> {
    let mut on_frontier = Vec::new();
    for (cluster_idx, cluster) in clusters.iter().enumerate() {
        for point_idx in cluster.iter() {
            let point = points.get(*point_idx).unwrap();
            let dist2own = centroids.get(cluster_idx).unwrap().dist2vec(&point.vector);
            let mut dist2centers = Vec::new();
            for (c_idx, center) in centroids.iter().enumerate() {
                if c_idx == cluster_idx {
                    continue;
                }
                dist2centers.push(point.dist2vec(&center.vector));
            }
            dist2centers.sort();
            for distance in dist2centers {
                if (distance.dist - dist2own.dist).abs() < threshold {
                    on_frontier.push(point.id);
                    break;
                }
            }
        }
    }
    on_frontier
}
pub fn reduce_borders(
    centroids: &Vec<Point>,
    current_clusters: &Vec<Vec<usize>>,
    points: &Vec<Point>,
    threshold: f32,
    target_size: usize,
) -> Vec<Vec<usize>> {
    let mut new_clusters: Vec<HashSet<&usize>> = Vec::from_iter(
        current_clusters
            .iter()
            .map(|x| HashSet::from_iter(x.iter())),
    );
    let mut target_points: HashSet<(usize, usize)> = HashSet::new();
    for (idx, cluster) in current_clusters.iter().enumerate() {
        if cluster.len() > target_size {
            target_points.extend(cluster.iter().map(|x| (idx, *x)));
        }
    }
    for (big_idx, point_id) in target_points.iter() {
        if new_clusters.get(*big_idx).unwrap().len() <= target_size {
            continue;
        }
        let point = points
            .get(*point_id)
            .expect(format!("point {point_id} not found in points").as_str());
        let mut dist2centers = Vec::new();
        let mut dist2own = Dist::new(0.0);
        for (idx, center) in centroids.iter().enumerate() {
            if idx == *big_idx {
                dist2own = point.dist2vec(&center.vector);
                continue;
            }
            dist2centers.push((idx, point.dist2vec(&center.vector)));
        }
        dist2centers.sort();
        for (other_idx, dist) in dist2centers {
            if new_clusters.get(other_idx).unwrap().len() > target_size {
                continue;
            }
            if (dist.dist - dist2own.dist).abs() < threshold {
                new_clusters.get_mut(other_idx).unwrap().insert(point_id);
                new_clusters.get_mut(*big_idx).unwrap().remove(point_id);
                break;
            }
        }
    }
    let new_clusters = Vec::from_iter(
        new_clusters
            .iter()
            .map(|x| Vec::from_iter(x.iter().map(|x| **x))),
    );
    new_clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{helpers::glove::load_glove_array, hnsw::points::Vector};
    use std::time::Instant;

    // #[test]
    fn find_kmeans() -> std::io::Result<()> {
        let (_, embeddings) = load_glove_array(100, 400_000, false, 0)?;
        let mut points = Vec::new();
        points.extend(
            embeddings
                .iter()
                .enumerate()
                .map(|(id, x)| Point::new(id, x.clone(), false)),
        );

        let start = Instant::now();
        let (centroids, clusters) = kmeans(8, 500, &points);
        let end = Instant::now();
        println!(
            "Elapsed time: {}s",
            start.elapsed().as_secs() - end.elapsed().as_secs()
        );

        let mut sum = 0.0;
        for (idx, cluster) in clusters.iter().enumerate() {
            let frac = (cluster.len() as f32) / (embeddings.len() as f32);
            sum += frac;
            println!("Fraction of total, cluster {idx}: {}", frac);
        }
        println!("{sum}");

        println!("centroids");
        for c in centroids.iter() {
            match &c.vector {
                Vector::Full(full) => println!("{:?}", &full[0..5]),
                Vector::Compressed(quant) => println!("{:?}", &quant.reconstruct()[0..5]),
            }
        }

        let mut on_frontier = get_frontier_points(&centroids, &clusters, &points, 0.05);
        println!("Points on the frontier: {}", on_frontier.len());
        on_frontier.sort();
        let on_frontier: Vec<&usize> = on_frontier.iter().take(5).collect();
        println!("First 5 points of the frontier : {:?}", on_frontier);

        Ok(())
    }

    #[test]
    fn find_partitions() -> std::io::Result<()> {
        let (_, embeddings) = load_glove_array(100, 400_000, false, 0)?;
        let mut points = Vec::new();
        points.extend(
            embeddings
                .iter()
                .enumerate()
                .map(|(id, x)| Point::new(id, x.clone(), false)),
        );

        let start = Instant::now();
        let (centroids, clusters) = partition_space(8, 10, &points);
        let end = Instant::now();
        println!(
            "Elapsed time: {}s",
            start.elapsed().as_secs() - end.elapsed().as_secs()
        );

        let mut sum = 0.0;
        for (idx, cluster) in clusters.iter().enumerate() {
            let frac = (cluster.len() as f32) / (embeddings.len() as f32);
            sum += frac;
            println!("Fraction of total, cluster {idx}: {}", frac);
        }
        println!("{sum}");

        println!("centroids");
        for c in centroids.iter() {
            match &c.vector {
                Vector::Full(full) => println!("{:?}", &full[0..5]),
                Vector::Compressed(quant) => println!("{:?}", &quant.reconstruct()[0..5]),
            }
        }

        let mut on_frontier = get_frontier_points(&centroids, &clusters, &points, 0.05);
        println!("Points on the frontier: {}", on_frontier.len());
        on_frontier.sort();
        let on_frontier: Vec<&usize> = on_frontier.iter().take(5).collect();
        println!("First 5 points of the frontier : {:?}", on_frontier);

        Ok(())
    }
}
