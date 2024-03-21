use std::collections::HashMap;
use std::path::Path;

use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::HNSW;

use ndarray::s;

use indicatif::{ProgressBar, ProgressStyle};

fn main() -> std::io::Result<()> {
    let (dim, lim, m, _ef_cons) = parse_args_eval();
    let (words, embeddings) = load_glove_array(dim as i32, lim as i32, true, 0).unwrap();
    let bf_data = load_bf_data(dim, lim).unwrap();

    let checkpoint_path = format!("/home/gamal/indices/checkpoint_dim{dim}_lim{lim}");
    let mut copy_path = checkpoint_path.clone();
    copy_path.push_str("_copy");

    let bar = ProgressBar::new(lim.try_into().unwrap());
    bar.set_style(
        ProgressStyle::with_template(
            "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta_precise} : Elapsed: {elapsed} ] {per_sec} {wide_bar}",
        )
        .unwrap(),
    );
    bar.set_message(format!("Inserting Embeddings"));

    let mut index = if Path::new(&checkpoint_path).exists() {
        HNSW::load(&checkpoint_path)?
    } else {
        HNSW::new(20, m, None, dim as u32)
    };

    index.print_params();

    for idx in 0..lim {
        bar.inc(1);
        index.insert(idx as i32, &embeddings.slice(s![idx, ..]).to_owned());
        if idx % 10_000 == 0 {
            println!("Checkpointing in {checkpoint_path}");
            // index.print_params();
            index.save(&checkpoint_path)?;
            index.save(&copy_path)?;
        }
    }
    index.remove_unused();
    index.save(format!("/home/gamal/indices/eval_glove_dim{dim}_lim{lim}").as_str())?;

    index.print_params();

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
    Ok(())
}
