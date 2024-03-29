use std::collections::HashMap;
use std::path::Path;

use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::HNSW;

use ndarray::{s, Array2};

use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;

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
    let (words, embeddings) = load_glove_array(dim as i32, lim as i32, true, 0).unwrap();
    let bf_data = load_bf_data(dim, lim).unwrap();

    // let checkpoint_path = format!("/home/gamal/indices/checkpoint_dim{dim}_lim{lim}_m{m}");
    // let mut copy_path = checkpoint_path.clone();
    // copy_path.push_str("_copy");

    // let mut index = if Path::new(&checkpoint_path).exists() {
    //     HNSW::load(&checkpoint_path)?
    // } else {
    //     HNSW::from_params(20, m, None, None, None, None, dim as u32)
    // };
    // let nb_nodes = index.node_ids.len();
    // println!("Loaded index with {} inserted nodes.", nb_nodes);
    // if nb_nodes > 0 {
    //     estimate_recall(&mut index, &embeddings, &bf_data);
    // }

    // if nb_nodes != lim {
    //     let bar = ProgressBar::new((lim as usize - nb_nodes).try_into().unwrap());
    //     bar.set_style(
    //         ProgressStyle::with_template(
    //             "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta_precise} : Elapsed: {elapsed} ] {per_sec} {wide_bar}",
    //         )
    //         .unwrap());
    //     bar.set_message(format!("Inserting Embeddings"));
    //     for idx in 0..lim {
    //         let inserted = index.insert(idx as i32, &embeddings.slice(s![idx, ..]).to_owned());
    //         if inserted {
    //             bar.inc(1);
    //         }
    //         if ((idx % 1_000 == 0) & (inserted)) | (idx == lim - 1) {
    //             println!("Checkpointing in {checkpoint_path}");
    //             estimate_recall(&mut index, &embeddings, &bf_data);
    //             // index.print_params();
    //             index.save(&checkpoint_path)?;
    //             index.save(&copy_path)?;
    //         }
    //     }
    //     index.save(format!("/home/gamal/indices/eval_glove_dim{dim}_lim{lim}_m{m}").as_str())?;
    // }
    // index.remove_unused();

    let mut index = HNSW::from_params(20, m, None, None, None, None, dim as u32);
    index.print_params();
    let bar = ProgressBar::new(lim.try_into().unwrap());
    bar.set_style(
        ProgressStyle::with_template(
            "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta_precise} : Elapsed: {elapsed} ] {per_sec} {wide_bar}",
        )
        .unwrap());
    bar.set_message(format!("Inserting Embeddings"));
    for idx in 0..lim {
        bar.inc(1);
        index.insert(idx as i32, &embeddings.slice(s![idx, ..]).to_owned());
    }
    index.remove_unused();
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
    // return Ok(());

    estimate_recall(&mut index, &embeddings, &bf_data);
    Ok(())
}

fn estimate_recall(
    index: &mut HNSW,
    embeddings: &Array2<f32>,
    bf_data: &HashMap<usize, Vec<usize>>,
) {
    let mut rng = rand::thread_rng();
    let max_idx = index.node_ids.iter().max().unwrap_or(&0).clone();
    for ef in (12..37).step_by(12) {
        let sample_size: i32 = 1000;
        let bar = ProgressBar::new(sample_size as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta} : Elapsed: {elapsed} ] {per_sec} {wide_bar}",
            )
            .unwrap(),
        );
        bar.set_message(format!("Finding ANNs ef={ef}"));

        let mut recall_10: HashMap<i32, f32> = HashMap::new();
        for _ in (0..sample_size).enumerate() {
            bar.inc(1);

            let idx = rng.gen_range(0..(index.node_ids.len()));
            let vector = embeddings.slice(s![idx, ..]);
            let anns = index.ann_by_vector(&vector.to_owned(), 10, ef);
            let true_nns: Vec<i32> = bf_data
                .get(&idx)
                .unwrap()
                .iter()
                .map(|x| *x as i32)
                // .filter(|x| x <= &max_idx)
                .collect();
            let mut hits = 0;
            for ann in anns.iter() {
                if true_nns[..10].contains(ann) {
                    hits += 1;
                }
            }
            recall_10.insert(idx as i32, (hits as f32) / 10.0);
        }
        let mut avg_recall = 0.0;
        for (_, recall) in recall_10.iter() {
            avg_recall += recall;
        }
        avg_recall /= sample_size as f32;
        println!("Recall@10 {avg_recall}");
    }
}
