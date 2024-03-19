use std::collections::HashMap;

use hnsw::helpers::args::parse_args;
use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::HNSW;

use ndarray::s;

use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

fn main() {
    let (dim, lim) = parse_args();
    let (words, embeddings) = load_glove_array(dim as i32, lim as i32, true, 0).unwrap();

    for _ in 0..1 {
        let bar = ProgressBar::new(lim.try_into().unwrap());
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} {wide_bar} {human_pos}/{human_len} {percent}% [ ETA: {eta} : Elapsed: {elapsed} ] {per_sec}",
            )
            .unwrap(),
        );
        bar.set_message(format!("Inserting Embeddings"));
        let mut index = HNSW::new(36);

        for idx in 0..lim {
            bar.inc(1);
            let vector = embeddings.slice(s![idx, ..]).clone().to_owned();
            index.insert(idx as i32, vector);
        }

        index.print_params();

        let bf_data = load_bf_data(dim, lim).unwrap();

        // let n = 900;
        // let vector = embeddings.slice(s![n, ..]);
        // let example_bf: Vec<String> = bf_data
        //     .get(&n)
        //     .unwrap()
        //     .iter()
        //     .map(|x| words[x.0].clone())
        //     .collect();
        // let anns = index.ann_by_vector(&vector.to_owned(), 10, 36);
        // let example_ann: Vec<String> = anns.iter().map(|x| words[*x as usize].clone()).collect();
        // let trues_ids: Vec<usize> = bf_data.get(&n).unwrap().iter().map(|x| x.0).collect();
        // println!("{:?}", trues_ids);
        // println!("{:?}", anns);
        // println!("{:?}", example_bf);
        // println!("{:?}", example_ann);

        // for (i, idx) in bf_data.keys().enumerate() {
        //     if i > 10 {
        //         break;
        //     }
        //     let vector = embeddings.slice(s![*idx, ..]);
        //     let anns = index.ann_by_vector(&vector.to_owned(), 10, 16);
        //     println!("ANNs of {}", words[*idx]);
        //     for ann in anns.iter() {
        //         println!("{}", words[*ann as usize]);
        //     }
        // }

        for ef in [11, 18, 24, 36].iter() {
            let bar = ProgressBar::new(lim.try_into().unwrap());
            bar.set_style(
                ProgressStyle::with_template(
                    "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta} : Elapsed: {elapsed} ] {per_sec} {wide_bar}",
                )
                .unwrap(),
            );
            bar.set_message(format!("Finding ANNs ef={ef}"));

            let mut recall_10: HashMap<i32, f32> = HashMap::new();
            for idx in bf_data.keys() {
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
            assert_eq!(recall_10.len(), lim);
            let mut avg_recall = 0.0;
            for (_, recall) in recall_10.iter() {
                avg_recall += recall;
            }
            avg_recall /= lim as f32;
            println!("Average recall was {avg_recall}");
        }
    }
}
