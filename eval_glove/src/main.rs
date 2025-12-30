#[allow(unused_imports)]
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use std::{collections::HashSet, fs::remove_dir_all};

use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::load_glove_array;

use hnsw::template::HNSW;
use indicatif::{ProgressBar, ProgressStyle};
use points::point::Point;
use points::points::block::BlockID;
use vectors::{LVQVec, VecBase, VecTrait};

use rand::seq::SliceRandom;
use rand::thread_rng;

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

    let file_name = format!("glove.840B.{dim}d");

    let (words, embeddings) = load_glove_array(lim, file_name.clone(), true).unwrap();
    // let embs = load_sift_array(lim, true).unwrap();

    // let (bf_data, train_ids, test_ids) = match load_bf_data(lim, file_name.clone()) {
    //     Ok(data) => data,
    //     Err(err) => {
    //         println!("Error loading bf data: {err}");
    //         return Ok(());
    //     }
    // };

    // let train_set: Vec<Vec<f32>> = embeddings
    //     .iter()
    //     .enumerate()
    //     .filter(|(id, _)| train_ids.contains(id))
    //     .map(|(_, v)| v.clone())
    //     .collect();
    // let test_set: Vec<Vec<f32>> = embeddings
    //     .iter()
    //     .enumerate()
    //     .filter(|(id, _)| test_ids.contains(id))
    //     .map(|(_, v)| v.clone())
    //     .collect();

    let s = Instant::now();
    let mut store: HNSW<LVQVec> = HNSW::new(m, None, embeddings[0].len(), BlockID::MAX);
    store = store.insert_bulk(embeddings, 8, true).unwrap();
    let e = s.elapsed().as_millis();
    println!(
        "took {0} ms to build index with {1} points and M {2}",
        e,
        store.len(),
        store.params.m
    );

    let path = Path::new("./index_eval_test");
    if path.exists() {
        remove_dir_all(path).unwrap();
    }

    let s = Instant::now();
    store.save(path);
    let e = s.elapsed().as_millis();
    println!(
        "took {0} ms to save index with {1} points and M {2}",
        e,
        store.len(),
        store.params.m
    );

    let s = Instant::now();
    let store: HNSW<LVQVec> = HNSW::load(path).unwrap();
    let e = s.elapsed().as_millis();
    println!(
        "took {0} ms to load index with {1} points and M {2}",
        e,
        store.len(),
        store.params.m
    );

    // estimate_recall(&store, &test_set, &bf_data);
    show_nn_words(&words, &store, 10);
    // {
    //     use text_io::read;

    //     let ef = 1000;
    //     loop {
    //         let words_map: HashMap<String, usize> = HashMap::from_iter(
    //             train_words
    //                 .iter()
    //                 .enumerate()
    //                 .map(|(idx, w)| (w.clone(), idx)),
    //         );
    //         println!("Look for NNs of a word ('_quit' to exit):");
    //         let query: String = read!();

    //         if query == "_quit".to_string() {
    //             break;
    //         }

    //         if !words_map.contains_key(&query) {
    //             println!("'{query}' is not in the index, try another word.");
    //             continue;
    //         }

    //         let word_idx = words_map.get(&query).unwrap();
    //         let vector = index
    //             .points
    //             .get_point(*word_idx as u32)
    //             .unwrap()
    //             .get_full_precision();

    //         let anns = index.ann_by_vector(&vector, 10, ef).unwrap();
    //         let anns_words: Vec<String> = anns
    //             .iter()
    //             .map(|x| train_words[*x as usize].clone())
    //             .collect();
    //         println!("ANNs of {query} (ef={ef})");
    //         println!("{:?}", anns_words);
    //     }
    // }
    Ok(())
}

fn show_nn_words<T: VecTrait>(words: &Vec<String>, store: &HNSW<T>, max: usize) {
    let ef = 1000;
    let mut words: Vec<(usize, &String)> = words.iter().enumerate().collect();
    words.shuffle(&mut thread_rng());
    let mut c = 0;
    for (idx, word) in words.iter() {
        if c > max {
            break;
        }
        c += 1;
        let point = store.get_point(*idx as u32).unwrap();
        let anns = store.ann_by_vector(&point, 10, ef).unwrap();
        let anns_words: Vec<String> = anns.iter().map(|x| words[*x as usize].1.clone()).collect();
        println!();
        println!("ANNs of {} (ef={ef})", word);
        for w in anns_words.iter() {
            println!("  - {w}",);
        }
    }
}

fn estimate_recall(
    index: &HNSW<LVQVec>,
    test_set: &Vec<Vec<f32>>,
    bf_data: &HashMap<usize, Vec<usize>>,
) {
    let n = 100;
    for ef in (100..=1000).step_by(100) {
        let bar = ProgressBar::new(test_set.len() as u64);
        bar.set_message(format!("Finding ANNs ef={ef}"));
        bar.set_style(
            ProgressStyle::with_template("{msg} {bar:60} {per_sec}")
                .unwrap()
                .progress_chars(">>-"),
        );

        let start = Instant::now();
        let mut total_neighbors = 0;
        let mut total_hits = 0;
        for (idx, query) in test_set.iter().enumerate() {
            bar.inc(1);
            let anns: Vec<usize> = index
                .ann_by_vector(&Point::new_with(0, &query), n, ef)
                .unwrap()
                .iter()
                .map(|x| *x as usize)
                .collect();
            let true_nns: &Vec<usize> = match bf_data.get(&idx) {
                None => continue,
                Some(t) => t,
            };
            let mut hits = 0;
            for true_nn in true_nns.iter().take(n) {
                if anns.contains(true_nn) {
                    hits += 1;
                }
            }
            total_neighbors += n;
            total_hits += hits;
        }
        let mut query_time = (start.elapsed().as_nanos() as f32) / (test_set.len() as f32);
        query_time = query_time.trunc();
        query_time /= 1_000_000.0;
        let avg_recall = total_hits as f32 / total_neighbors as f32;
        println!("ef={ef} Recall@10 {avg_recall} ({total_hits}/{total_neighbors}), Query Time: {query_time} ms");
    }
}
