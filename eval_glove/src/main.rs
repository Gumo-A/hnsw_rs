use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;

use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::load_glove_array;
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

    let (_, embeddings) = load_glove_array(dim, lim, true, 1).unwrap();

    let bf_data = match load_bf_data(dim, lim) {
        Ok(data) => data,
        Err(err) => {
            println!("Error loading bf data: {err}");
            return Ok(());
        }
    };

    let start = Instant::now();
    let index = HNSW::build_index(m, None, embeddings, true).unwrap();
    let end = Instant::now();
    println!(
        "Single-thread elapsed time: {}ms",
        start.elapsed().as_millis() - end.elapsed().as_millis()
    );
    estimate_recall(&index, &bf_data);

    // let (_, embeddings) = load_glove_array(dim, lim, true, 1).unwrap();
    // let start = Instant::now();
    // let index = HNSW::build_index_par_v2(m, None, embeddings, true).unwrap();
    // let end = Instant::now();
    // index.print_index();
    // println!(
    //     "Multi-thread elapsed time: {}ms",
    //     start.elapsed().as_millis() - end.elapsed().as_millis()
    // );
    // estimate_recall(&index, &bf_data);

    // for (i, idx) in bf_data.keys().enumerate() {
    //     if i > 3 {
    //         break;
    //     }
    //     let point = index.points.get_point(*idx);
    //     let vector = match point {
    //         Some(p) => p.vector.get_full(),
    //         None => continue,
    //     };
    //     let anns = index.ann_by_vector(&vector, 10, 16).unwrap();
    //     println!("ANNs of {}", words[*idx]);
    //     let anns_words: Vec<String> = anns.iter().map(|x| words[*x as usize].clone()).collect();
    //     println!("{:?}", anns_words);
    //     println!("True NN of {}", words[*idx]);
    //     let true_nns: Vec<String> = bf_data
    //         .get(&idx)
    //         .unwrap()
    //         .iter()
    //         .map(|x| words[*x].clone())
    //         .take(10)
    //         .collect();
    //     println!("{:?}", true_nns);
    // }
    Ok(())
}

fn estimate_recall(
    index: &HNSW,
    // TODO: vectors dont need to be passed if "index" stores the vectors
    bf_data: &HashMap<usize, Vec<usize>>,
) {
    let mut rng = rand::thread_rng();
    let max_id = index.points.ids().max().unwrap();
    for ef in (12..100).step_by(12) {
        println!("Finding ANNs ef={ef}");

        let sample_size: usize = 1000;
        let points_ids: Vec<usize> = index.points.ids().collect();
        let mut recall_10: HashMap<usize, f32> = HashMap::new();
        for _ in (0..sample_size).enumerate() {
            let idx = rng.gen_range(0..(index.points.len()));
            let idx = points_ids.get(idx).unwrap();
            let vector = &index.points.get_point(*idx).unwrap().get_full_precision();
            let anns = index.ann_by_vector(&vector, 10, ef).unwrap();
            let true_nns: Vec<usize> = bf_data
                .get(&idx)
                .unwrap()
                .clone()
                .iter()
                .filter(|x| **x <= max_id)
                .map(|x| *x)
                .collect();
            let length = if (true_nns.len() < 10) | (anns.len() < 10) {
                true_nns.len().min(anns.len())
            } else {
                10
            };
            let mut hits = 0;
            for ann in anns[..length].iter() {
                if true_nns[..length].contains(ann) {
                    hits += 1;
                }
            }
            recall_10.insert(*idx, (hits as f32) / 10.0);
        }
        let mut avg_recall = 0.0;
        for (_, recall) in recall_10.iter() {
            avg_recall += recall;
        }
        avg_recall /= recall_10.keys().count() as f32;
        println!("Recall@10 {avg_recall}");
    }
}
