use std::collections::HashMap;
use std::path::Path;

use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::index::HNSW;

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
    let (words, embeddings) = load_glove_array(dim, lim, true, 0).unwrap();
    let bf_data = load_bf_data(dim, lim).unwrap();

    let checkpoint_path = format!("/home/gamal/indices/checkpoint_dim{dim}_lim{lim}_m{m}");
    let mut copy_path = checkpoint_path.clone();
    copy_path.push_str("_copy");

    let mut index = if Path::new(&checkpoint_path).exists() {
        HNSW::load(&checkpoint_path)?
    } else {
        HNSW::from_params(20, m, None, None, None, None, dim)
    };
    let nb_nodes = index.node_ids.len();
    println!("Loaded index with {} inserted nodes.", nb_nodes);
    if nb_nodes > 1 {
        // estimate_recall(&mut index, &embeddings, &bf_data);
    }

    if nb_nodes != lim {
        let bar = ProgressBar::new((lim - nb_nodes).try_into().unwrap());
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta_precise} : Elapsed: {elapsed} ] {per_sec} {wide_bar}",
            )
            .unwrap());
        bar.set_message(format!("Inserting Embeddings"));
        for idx in 0..lim {
            let inserted = index.insert(idx, &embeddings.slice(s![idx, ..]).to_owned());
            if inserted {
                bar.inc(1);
            }
            if ((idx % 10_000 == 0) & (inserted)) | (idx == lim - 1) {
                println!("Checkpointing in {checkpoint_path}");
                // estimate_recall(&mut index, &embeddings, &bf_data);
                index.save(&checkpoint_path)?;
                index.save(&copy_path)?;
                bar.reset_eta();
            }
        }
        index.save(format!("/home/gamal/indices/eval_glove_dim{dim}_lim{lim}_m{m}").as_str())?;
    }
    index.remove_unused();

    // let mut index = HNSW::from_params(20, m, None, None, None, None, dim);
    // let node_ids: Vec<usize> = (0..lim).map(|x| x as usize).collect();
    // index.print_params();
    // index.build_index(node_ids, &embeddings);
    // index.print_params();
    // estimate_recall(&mut index, &embeddings, &bf_data);

    // let function = "insert";
    // let fracs = index.bencher.borrow().get_frac_of(function, vec![]);
    // let mut total = 0.0;
    // for (key, frac) in fracs.iter() {
    //     println!("{key} was {frac} of {function}");
    //     total += frac;
    // }
    // println!("Sum of these fractions is {total}");
    // let mean = *index.bencher.borrow().get_means().get(function).unwrap();
    // println!("Mean execution time of {function} is {mean}");
    // let tot_time_function = *index.bencher.borrow().get_sums().get(function).unwrap();
    // println!("Total execution time of {function} is {tot_time_function}");
    // let tot_time_insert = *index.bencher.borrow().get_sums().get("insert").unwrap();
    // println!("Total execution time of insert is {tot_time_insert}");
    // println!(
    //     "{function} was then {} of insert",
    //     tot_time_function / tot_time_insert
    // );

    for (i, idx) in (123..126).enumerate() {
        if i > 3 {
            break;
        }
        let vector = embeddings.slice(s![idx, ..]);
        let anns = index.ann_by_vector(&vector.to_owned(), 10, 16);
        println!("ANNs of {}", words[idx]);
        let anns_words: Vec<String> = anns.iter().map(|x| words[*x as usize].clone()).collect();
        println!("{:?}", anns_words);
    }

    Ok(())
}

fn estimate_recall(
    index: &mut HNSW,
    embeddings: &Array2<f32>,
    bf_data: &HashMap<usize, Vec<usize>>,
) {
    let mut rng = rand::thread_rng();
    let max_id = index.node_ids.iter().max().unwrap_or(&usize::MAX);
    for ef in (12..113).step_by(12) {
        let sample_size: usize = 1000;
        let bar = ProgressBar::new(sample_size as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta} : Elapsed: {elapsed} ] {per_sec} {wide_bar}",
            )
            .unwrap(),
        );
        bar.set_message(format!("Finding ANNs ef={ef}"));

        let mut recall_10: HashMap<usize, f32> = HashMap::new();
        for _ in (0..sample_size).enumerate() {
            bar.inc(1);

            let idx = rng.gen_range(0..(index.node_ids.len()));
            let vector = embeddings.slice(s![idx, ..]);
            let anns = index.ann_by_vector(&vector.to_owned(), 10, ef);
            let true_nns: Vec<usize> = bf_data
                .get(&idx)
                .unwrap()
                .clone()
                .iter()
                .filter(|x| x <= &max_id)
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
            recall_10.insert(idx, (hits as f32) / 10.0);
        }
        let mut avg_recall = 0.0;
        for (_, recall) in recall_10.iter() {
            avg_recall += recall;
        }
        avg_recall /= recall_10.keys().count() as f32;
        println!("Recall@10 {avg_recall}");
    }
}
