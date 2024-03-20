use std::collections::HashMap;

use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::HNSW;

use ndarray::s;

use indicatif::{ProgressBar, ProgressStyle};

fn main() {
    let (dim, lim, m, ef_cons) = parse_args_eval();
    let (words, embeddings) = load_glove_array(dim as i32, lim as i32, true, 0).unwrap();

    for _ in 0..1 {
        let bar = ProgressBar::new(lim.try_into().unwrap());
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta_precise} : Elapsed: {elapsed} ] {per_sec} {wide_bar}",
            )
            .unwrap(),
        );
        bar.set_message(format!("Inserting Embeddings"));
        let mut index = HNSW::new(20, m, Some(ef_cons));
        index.print_params();

        for idx in 0..lim {
            bar.inc(1);
            let vector = embeddings.slice(s![idx, ..]).clone().to_owned();
            index.insert(idx as i32, vector);
        }
        index.remove_unused();

        index.print_params();

        let bf_data = load_bf_data(dim, lim).unwrap();

        for (i, idx) in bf_data.keys().enumerate() {
            if i > 3 {
                break;
            }
            let vector = embeddings.slice(s![*idx, ..]);
            let anns = index.ann_by_vector(&vector.to_owned(), 10, 16);
            println!("ANNs of {}", words[*idx]);
            let anns_words: Vec<String> = anns.iter().map(|x| words[*x as usize].clone()).collect();
            println!("{:?}", anns_words);
        }

        for ef in [11, 18, 24, 36].iter() {
            let sample_size: usize = 1000;
            let bar = ProgressBar::new(sample_size as u64);
            bar.set_style(
                ProgressStyle::with_template(
                    "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta} : Elapsed: {elapsed} ] {per_sec} {wide_bar}",
                )
                .unwrap(),
            );
            bar.set_message(format!("Finding ANNs ef={ef}"));

            let mut recall_10: HashMap<i32, f32> = HashMap::new();
            for (i, idx) in bf_data.keys().enumerate() {
                if i > sample_size {
                    break;
                }
                bar.inc(1);
                let vector = embeddings.slice(s![*idx, ..]);
                let anns = index.ann_by_vector(&vector.to_owned(), 10, *ef);
                let true_nns: Vec<i32> = bf_data
                    .get(idx)
                    .unwrap()
                    .iter()
                    .map(|x| x.0 as i32)
                    .collect();
                let mut hits = 0;
                for ann in anns.iter() {
                    if true_nns.contains(ann) {
                        hits += 1;
                    }
                }
                recall_10.insert(*idx as i32, (hits as f32) / 10.0);
            }
            let mut avg_recall = 0.0;
            for (_, recall) in recall_10.iter() {
                avg_recall += recall;
            }
            avg_recall /= sample_size as f32;
            println!("Recall@10 {avg_recall}");
        }
    }
}
