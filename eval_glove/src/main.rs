#[allow(unused_imports)]
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::load_glove_array;

use hnsw::params::get_default_ml;
use hnsw::template::HNSW;
use indicatif::{ProgressBar, ProgressStyle};
use points::{point::Point, point_collection::Points};
use vectors::{FullVec, LVQVec};

use std::fs;
use std::io::Write;

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

    let (bf_data, train_ids, test_ids) = match load_bf_data(lim, file_name.clone()) {
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

    let ef_cons = 100;

    let embs = Points::new_quant(train_set.clone(), get_default_ml(m));
    let s = Instant::now();
    let mut store = HNSW::new(m, Some(ef_cons), embs.dim().unwrap());
    store = store.insert_bulk(embs, 8).unwrap();
    let e = s.elapsed().as_millis();
    // println!(
    //     "took {0} ms to build index with {1} points and M {2}",
    //     e,
    //     store.len(),
    //     store.params.m
    // );

    let s = Instant::now();
    store.save(Path::new("./index"));
    let e = s.elapsed().as_millis();
    // println!(
    //     "took {0} ms to save index with {1} points and M {2}",
    //     e,
    //     store.len(),
    //     store.params.m
    // );

    let s = Instant::now();
    let store: HNSW<LVQVec> = HNSW::load(Path::new("./index")).unwrap();
    let e = s.elapsed().as_millis();
    // println!(
    //     "took {0} ms to load index with {1} points and M {2}",
    //     e,
    //     store.len(),
    //     store.params.m
    // );

    // store.layer_degrees(&0);

    // index.print_index();
    // println!(
    //     "Multi-thread (v3) elapsed time: {}ms",
    //     start.elapsed().as_millis() - end.elapsed().as_millis()
    // );

    estimate_recall(&store, &test_set, &bf_data);

    // index.assert_param_compliance();

    // let train_words: Vec<String> = words
    //     .iter()
    //     .enumerate()
    //     .filter(|(id, _)| train_ids.contains(id))
    //     .map(|(_, w)| w.clone())
    //     .collect();
    // let test_words: Vec<String> = words
    //     .iter()
    //     .enumerate()
    //     .filter(|(id, _)| test_ids.contains(id))
    //     .map(|(_, w)| w.clone())
    //     .collect();

    // let ef = 1000;
    // for (i, idx) in bf_data.keys().enumerate() {
    //     if i > 5 {
    //         break;
    //     }
    //     let vector = test_set.get(*idx).unwrap();
    //     let anns = store
    //         .ann_by_vector(&Point::new_quant(0, 0, &vector.clone()), 10, ef)
    //         .unwrap();
    //     let anns_words: Vec<String> = anns
    //         .iter()
    //         .map(|x| train_words[*x as usize].clone())
    //         .collect();
    //     println!("ANNs of {} (ef={ef})", test_words[*idx]);
    //     println!("{:?}", anns_words);

    //     let true_nns: Vec<String> = bf_data
    //         .get(&idx)
    //         .unwrap()
    //         .iter()
    //         .map(|x| train_words[*x].clone())
    //         .take(10)
    //         .collect();
    //     println!("True NN of {}", test_words[*idx]);
    //     println!("{:?}", true_nns);
    //     println!("");
    // }
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

        let mut recall_100 = Vec::new();
        let start = Instant::now();
        for (idx, query) in test_set.iter().enumerate() {
            bar.inc(1);
            let anns: Vec<usize> = index
                .ann_by_vector(&Point::new_quant(0, 0, &query.clone()), n, ef)
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
            recall_100.push((hits as f32) / (n as f32));
        }
        let mut query_time = (start.elapsed().as_nanos() as f32) / (test_set.len() as f32);
        query_time = query_time.trunc();
        query_time /= 1_000_000.0;
        let mut avg_recall = 0.0;
        for recall in recall_100.iter() {
            avg_recall += recall;
        }
        avg_recall /= recall_100.len() as f32;
        println!("ef={ef} Recall@10 {avg_recall}, Query Time: {query_time} ms");
    }
}
