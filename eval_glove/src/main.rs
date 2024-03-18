use std::collections::HashMap;

use hnsw::helpers::args::parse_args;
use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::HNSW;

use ndarray::s;

use indicatif::{ProgressBar, ProgressStyle};

fn main() {
    let (dim, lim) = parse_args();
    let mut index = HNSW::new(16);
    let (words, embeddings) = load_glove_array(dim as i32, lim as i32, true, 0).unwrap();

    let bar = ProgressBar::new(lim.try_into().unwrap());
    bar.set_style(
        ProgressStyle::with_template(
            "{msg} {wide_bar} {human_pos}/{human_len} {percent}% [ ETA: {eta} : Elapsed: {elapsed} ] {per_sec}",
        )
        .unwrap(),
    );
    bar.set_message(format!("Inserting Embeddings"));

    for idx in 0..lim {
        bar.inc(1);
        index.insert(idx as i32, embeddings.slice(s![idx, ..]).to_owned())
    }

    let bf_data = load_bf_data(dim, lim).unwrap();

    let bar = ProgressBar::new(lim.try_into().unwrap());
    bar.set_style(
        ProgressStyle::with_template(
            "{msg} {wide_bar} {human_pos}/{human_len} {percent}% [ ETA: {eta} : Elapsed: {elapsed} ] {per_sec}",
        )
        .unwrap(),
    );
    bar.set_message(format!("Finding ANNs"));

    let mut recall_10: HashMap<i32, f32> = HashMap::new();
    for idx in bf_data.keys() {
        bar.inc(1);
        let vector = embeddings.slice(s![*idx, ..]);
        let anns = index.ann_by_vector(&vector.to_owned(), 10, 16);
        let true_nns: Vec<i32> = bf_data
            .get(idx)
            .unwrap()
            .iter()
            .map(|x| x.0 as i32)
            .collect();
        let mut hits = 0.0;
        for ann in anns.iter() {
            if true_nns.contains(ann) {
                hits += 1.0;
            }
        }
        recall_10.insert(*idx as i32, hits / 10.0);
    }
    let mut avg_recall = 0.0;
    for (idx, recall) in recall_10.iter() {
        avg_recall += recall;
    }
    avg_recall /= lim as f32;
    println!("Average recall was {avg_recall}");
}
