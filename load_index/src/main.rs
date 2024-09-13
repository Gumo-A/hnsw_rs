#[allow(unused_imports)]
use std::collections::HashMap;
use std::time::Instant;

use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::{load_glove_array, load_sift_array};
use hnsw::hnsw::index::HNSW;

use indicatif::{ProgressBar, ProgressStyle};

fn main() -> std::io::Result<()> {
    let file_name = format!("glove.6B.100d");

    let (bf_data, train_ids, test_ids) = match load_bf_data(0, file_name.clone()) {
        Ok(data) => data,
        Err(err) => {
            println!("Error loading bf data: {err}");
            return Ok(());
        }
    };

    let (words, embeddings) = load_glove_array(0, file_name.clone(), true).unwrap();

    let train_set: Vec<Vec<f32>> = embeddings
        .iter()
        .enumerate()
        .filter(|(id, _)| train_ids.contains(id))
        .map(|(_, v)| v.clone())
        .collect();
    let test_set: Vec<Vec<f32>> = embeddings
        .iter()
        .enumerate()
        .filter(|(id, _)| test_ids.contains(id))
        .map(|(_, v)| v.clone())
        .collect();

    let index = HNSW::from_path("./index.ann")?;
    index.print_index();
    estimate_recall(&index, &test_set, &bf_data);
    Ok(())
}

fn estimate_recall(index: &HNSW, test_set: &Vec<Vec<f32>>, bf_data: &HashMap<usize, Vec<usize>>) {
    for ef in (80..=100).step_by(10) {
        println!("Finding ANNs ef={ef}");
        let bar = ProgressBar::new(test_set.len() as u64);
        bar.set_style(
            ProgressStyle::with_template("{msg} {bar:60} {per_sec}")
                .unwrap()
                .progress_chars(">>-"),
        );

        let mut recall_10 = Vec::new();
        let start = Instant::now();
        for (idx, query) in test_set.iter().enumerate() {
            let anns = index.ann_by_vector(query, 10, ef).unwrap();
            let true_nns: &Vec<usize> = bf_data.get(&idx).unwrap();
            let mut hits = 0;
            for true_nn in true_nns.iter().take(10) {
                if anns.contains(&(*true_nn as u32)) {
                    hits += 1;
                }
            }
            recall_10.push((hits as f32) / 10.0);
            bar.inc(1);
        }
        let mut query_time = (start.elapsed().as_nanos() as f32) / (test_set.len() as f32);
        query_time = query_time.trunc();
        query_time /= 1_000_000.0;
        let mut avg_recall = 0.0;
        for recall in recall_10.iter() {
            avg_recall += recall;
        }
        avg_recall /= recall_10.len() as f32;
        println!("Recall@10 {avg_recall}, Query Time: {query_time} ms");
    }
}
