use std::collections::HashMap;
use std::time::Instant;

use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::{load_glove_array, load_sift_array};
use hnsw::hnsw::index::HNSW;

use rand::Rng;

use indicatif::{ProgressBar, ProgressStyle};

fn main() -> std::io::Result<()> {
    let (dim, lim, m) = match parse_args_eval() {
        Ok(args) => args,
        Err(err) => {
            println!("Help: eval_glove");
            println!("{}", err);
            println!("dim[int] lim[int] m[int]");
            return Ok(());
        }
    };

    let (words, embeddings) = load_glove_array(dim, lim, true).unwrap();
    // let embs = load_sift_array(lim, true).unwrap();

    let (bf_data, train_ids, test_ids) = match load_bf_data(dim, lim) {
        Ok(data) => data,
        Err(err) => {
            println!("Error loading bf data: {err}");
            return Ok(());
        }
    };

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

    let mut embs: Vec<Vec<f32>> = train_set.clone();

    let embs = train_set.clone();
    let start = Instant::now();
    let index = HNSW::build_index(m, None, embs, true).unwrap();
    let end = Instant::now();
    index.print_index();
    println!(
        "Multi-thread (v3) elapsed time: {}ms",
        start.elapsed().as_millis() - end.elapsed().as_millis()
    );
    estimate_recall(&index, &test_set, &bf_data);
    // index.assert_param_compliance();

    let train_words: Vec<String> = words
        .iter()
        .enumerate()
        .filter(|(id, _)| train_ids.contains(id))
        .map(|(_, w)| w.clone())
        .collect();
    let test_words: Vec<String> = words
        .iter()
        .enumerate()
        .filter(|(id, _)| test_ids.contains(id))
        .map(|(_, w)| w.clone())
        .collect();

    for (i, idx) in bf_data.keys().enumerate() {
        if i > 3 {
            break;
        }
        let point = test_set.get(*idx).unwrap();
        let anns = index.ann_by_vector(point, 10, 16).unwrap();
        println!("ANNs of {}", test_words[*idx]);
        let anns_words: Vec<String> = anns
            .iter()
            .map(|x| train_words[*x as usize].clone())
            .collect();
        println!("{:?}", anns_words);
        println!("True NN of {}", test_words[*idx]);
        let true_nns: Vec<String> = bf_data
            .get(&idx)
            .unwrap()
            .iter()
            .map(|x| train_words[*x].clone())
            .take(10)
            .collect();
        println!("{:?}", true_nns);
    }
    println!("test 100 {}", test_words[100]);
    println!("test 100 {:?}", bf_data[&100]);
    println!(
        "test 0 {:?}",
        test_set[0].iter().take(10).collect::<Vec<&f32>>()
    );
    Ok(())
}

fn estimate_recall(index: &HNSW, test_set: &Vec<Vec<f32>>, bf_data: &HashMap<usize, Vec<usize>>) {
    for ef in (10..=100).step_by(10) {
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
                if anns.contains(true_nn) {
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
