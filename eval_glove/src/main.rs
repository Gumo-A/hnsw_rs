#[allow(unused_imports)]
use std::collections::HashMap;
use std::time::Instant;

use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::{load_glove_array, load_sift_array};
use hnsw::hnsw::index::HNSW;

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

    // let file_name = format!("glove.twitter.27B.{dim}d");
    let file_name = format!("glove.6B.{dim}d");

    let (words, embeddings) = load_glove_array(lim, file_name.clone(), true).unwrap();
    // let embs = load_sift_array(lim, true).unwrap();

    let (bf_data, train_ids, test_ids) = match load_bf_data(lim, file_name) {
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

    let embs = train_set.clone();
    let start = Instant::now();
    let index = HNSW::build_index_par(m, None, embs, true).unwrap();
    let end = Instant::now();

    println!("Saving index to current dir...");
    index.save("./index.ann")?;
    let index = HNSW::from_path("./index.ann")?;

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

    let ef = 1000;
    for (i, idx) in bf_data.keys().enumerate() {
        if i > 3 {
            break;
        }
        let point = test_set.get(*idx).unwrap();
        let anns = index.ann_by_vector(point, 10, ef).unwrap();
        let anns_words: Vec<String> = anns
            .iter()
            .map(|x| train_words[*x as usize].clone())
            .collect();
        println!("ANNs of {} (ef={ef})", test_words[*idx]);
        println!("{:?}", anns_words);

        let true_nns: Vec<String> = bf_data
            .get(&idx)
            .unwrap()
            .iter()
            .map(|x| train_words[*x].clone())
            .take(10)
            .collect();
        println!("True NN of {}", test_words[*idx]);
        println!("{:?}", true_nns);
    }
    {
        use text_io::read;

        let ef = 1000;
        loop {
            let words_map: HashMap<String, usize> = HashMap::from_iter(
                train_words
                    .iter()
                    .enumerate()
                    .map(|(idx, w)| (w.clone(), idx)),
            );
            println!("Look for NNs of a word ('_quit' to exit):");
            let query: String = read!();

            if query == "_quit".to_string() {
                break;
            }

            if !words_map.contains_key(&query) {
                println!("'{query}' is not in the index, try another word.");
                continue;
            }

            let word_idx = words_map.get(&query).unwrap();
            let vector = index
                .points
                .get_point(*word_idx)
                .unwrap()
                .get_full_precision();

            let anns = index.ann_by_vector(&vector, 10, ef).unwrap();
            let anns_words: Vec<String> = anns
                .iter()
                .map(|x| train_words[*x as usize].clone())
                .collect();
            println!("ANNs of {query} (ef={ef})");
            println!("{:?}", anns_words);
        }
    }
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
