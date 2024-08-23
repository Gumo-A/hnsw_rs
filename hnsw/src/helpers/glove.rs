use crate::hnsw::dist::Dist;
use crate::hnsw::points::{Point, PointsV2};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Result};
use std::sync::Arc;

pub fn load_glove_array(
    dim: usize,
    lim: usize,
    normalize: bool,
    verbose: bool,
) -> Result<(Vec<String>, Vec<Vec<f32>>)> {
    let file = File::open(format!("/home/gamal/glove_dataset/glove.6B.{dim}d.txt"))?;
    let reader = BufReader::new(file);

    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let mut words: Vec<String> = Vec::new();

    let bar = if verbose {
        let bar = ProgressBar::new(lim.try_into().unwrap());
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} {wide_bar} {human_pos}/{human_len} {percent}% [ ETA: {eta} : Elapsed: {elapsed} ] {per_sec}",
            )
            .unwrap(),
        );
        bar.set_message("Loading Embeddings".to_string());
        bar
    } else {
        
        ProgressBar::hidden()
    };

    for (idx, line_result) in reader.lines().enumerate() {
        bar.inc(1);
        if idx >= lim.try_into().unwrap() {
            break;
        }
        let line = line_result?;
        let mut parts = line.split_whitespace();

        let word = parts.next().expect("Empty line");

        let mut values: Vec<f32> = parts
            .map(|s| s.parse::<f32>().expect("Could not parse float"))
            .collect();
        let norm: f32 = if normalize {
            values
                .clone()
                .iter()
                .map(|x| x.powf(2.0))
                .sum::<f32>()
                .powf(0.5)
        } else {
            1.0
        };
        values = values.iter().map(|x| x / norm).collect();
        embeddings.push(values);
        // for (jdx, val) in values.enumerate() {
        //     let entry = embeddings.get_mut((idx, jdx)).unwrap();
        //     *entry = val / norm
        // }

        words.push(word.to_string());
    }
    Ok((words, embeddings))
}

pub fn brute_force_nns(
    nb_nns: usize,
    train_set: Arc<PointsV2>,
    test_set: Arc<PointsV2>,
    ids: Vec<usize>,
    verbose: bool,
) -> HashMap<usize, Vec<usize>> {
    let bar = if verbose {
        let bar = ProgressBar::new(ids.len() as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} {wide_bar} {human_pos}/{human_len} {percent}% [ ETA: {eta_precise} : Elapsed: {elapsed} ] {per_sec}",
            )
            .unwrap(),
        );
        bar.set_message("Finding NNs");
        bar
    } else {
        
        ProgressBar::hidden()
    };

    let mut brute_force_results: HashMap<usize, Vec<usize>> = HashMap::new();
    for idx in ids.iter() {
        let query = test_set.get_point(*idx).unwrap();
        let nns: Vec<usize> = get_nn_bf(query, &train_set, nb_nns);
        assert_eq!(nb_nns, nns.len());
        brute_force_results.insert(*idx, nns);
        bar.inc(1);
    }

    brute_force_results
}

fn get_nn_bf(point: &Point, others: &Arc<PointsV2>, nb_nns: usize) -> Vec<usize> {
    let sorted = sort_by_distance(point, others);
    sorted.values().copied().take(nb_nns).collect()
}

fn sort_by_distance(point: &Point, others: &Arc<PointsV2>) -> BTreeMap<Dist, usize> {
    let result = others.iterate().map(|(idx, p)| {
        let dist = p.dist2other(point);
        (dist, idx)
    });
    BTreeMap::from_iter(result)
}
