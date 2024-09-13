use crate::hnsw::dist::Dist;
use crate::hnsw::points::{Point, Points};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Result};
use std::sync::Arc;

pub fn load_glove_array(
    lim: usize,
    file_name: String,
    verbose: bool,
) -> Result<(Vec<String>, Vec<Vec<f32>>)> {
    let file = File::open(format!("/home/gamal/glove_dataset/{file_name}.txt"))?;
    let reader = BufReader::new(file);

    let lim = if lim == 0 {
        reader.lines().count()
    } else {
        lim
    };

    let file = File::open(format!("/home/gamal/glove_dataset/{file_name}.txt"))?;
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

        let values: Vec<f32> = parts
            .map(|s| s.parse::<f32>().expect("Could not parse float"))
            .collect();
        embeddings.push(values);
        words.push(word.to_string());
    }
    Ok((words, embeddings))
}

pub fn load_sift_array(lim: usize, verbose: bool) -> Result<Vec<Vec<f32>>> {
    let file = File::open(format!("/home/gamal/sift/SIFT10M/sift10m.txt"))?;
    let reader = BufReader::new(file);
    let limit = if lim == 0 {
        reader.lines().count()
    } else {
        lim
    };
    let file = File::open(format!("/home/gamal/sift/SIFT10M/sift10m.txt"))?;
    let reader = BufReader::new(file);

    let mut embeddings: Vec<Vec<f32>> = Vec::new();

    let bar = if verbose {
        let bar = ProgressBar::new(limit as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} {wide_bar} {human_pos}/{human_len} {percent}% [ ETA: {eta} : Elapsed: {elapsed} ] {per_sec}",
            )
            .unwrap(),
        );
        bar.set_message("Loading SIFT features".to_string());
        bar
    } else {
        ProgressBar::hidden()
    };

    for (idx, line_result) in reader.lines().enumerate() {
        if (lim > 0) & (idx >= limit) {
            break;
        }
        let line = line_result?;
        let parts = line.split_whitespace();

        let values: Vec<f32> = parts
            .map(|s| s.parse::<f32>().expect("Could not parse float"))
            .collect();
        embeddings.push(values);

        bar.inc(1);
    }
    Ok(embeddings)
}

pub fn brute_force_nns(
    nb_nns: usize,
    train_set: Arc<Points>,
    test_set: Arc<Points>,
    ids: Vec<u32>,
    verbose: bool,
) -> HashMap<u32, Vec<u32>> {
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

    let mut brute_force_results: HashMap<u32, Vec<u32>> = HashMap::new();
    for idx in ids.iter() {
        let query = test_set.get_point(*idx as u32).unwrap();
        let nns: Vec<u32> = get_nn_bf(query, &train_set, nb_nns);
        assert_eq!(nb_nns, nns.len());
        brute_force_results.insert(*idx, nns);
        bar.inc(1);
    }

    brute_force_results
}

fn get_nn_bf(point: &Point, others: &Arc<Points>, nb_nns: usize) -> Vec<u32> {
    let sorted = sort_by_distance(point, others);
    sorted
        .values()
        .copied()
        .take(nb_nns)
        .map(|x| x as u32)
        .collect()
}

fn sort_by_distance(point: &Point, others: &Arc<Points>) -> BTreeMap<Dist, usize> {
    let result = others.iterate().map(|(idx, p)| {
        let dist = p.dist2other(point);
        (dist, idx as usize)
    });
    BTreeMap::from_iter(result)
}
