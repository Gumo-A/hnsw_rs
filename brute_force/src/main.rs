use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{BTreeSet, HashMap};
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::thread;

use graph::nodes::Node;
use rand::seq::IteratorRandom;

use hnsw::helpers::args::parse_args_bf;
use hnsw::helpers::glove::{brute_force_nns, load_glove_array};
use points::point_collection::Points;

const NB_NNS: usize = 100;

fn main() -> std::io::Result<()> {
    let (dim, lim) = match parse_args_bf() {
        Ok(args) => args,
        Err(err) => {
            println!("Help: brute_force");
            println!("dim[int] limit[int]");
            println!("{}", err);
            return Ok(());
        }
    };

    let file_name = format!("glove.{dim}d");

    let _ = create_dir_all(format!(
        "/home/gamal/glove_dataset/test_data/{file_name}_lim{lim}/"
    ));

    let (_, embeddings): (Vec<String>, Vec<Vec<f32>>) =
        load_glove_array(lim, file_name.clone(), true).unwrap();

    let test_frac = 0.01;
    let (train_set, test_set, train_idx, test_idx) = split_glove(embeddings, test_frac);

    let train_set = Points::new_full(train_set, 0.0);
    let test_set = Points::new_full(test_set, 0.0);

    let train_arc = Arc::new(train_set);
    let test_arc = Arc::new(test_set);

    let nb_threads = std::thread::available_parallelism().unwrap().get();
    let test_ids: Vec<Node> = test_arc.ids().collect();
    let mut indices_split: Vec<Vec<Node>> = test_ids
        .chunks(((test_ids.len() as f32) / (nb_threads as f32)) as usize)
        .map(|chunk| Vec::from(chunk))
        .collect();
    let mut nb = 0;
    for split in indices_split.iter() {
        nb += split.len()
    }

    assert_eq!(nb, test_ids.len());

    let bar = ProgressBar::new(test_ids.len() as u64);
    bar.set_style(
            ProgressStyle::with_template(
                "{msg} {wide_bar} {human_pos}/{human_len} {percent}% [ ETA: {eta_precise} : Elapsed: {elapsed} ] {per_sec}",
            )
            .unwrap(),
        );
    bar.set_message("Finding NNs");

    let mut handles = Vec::new();
    for _ in 0..nb_threads {
        let train_ref = train_arc.clone();
        let test_ref = test_arc.clone();
        let ids_to_compute = indices_split.pop().unwrap();

        let bar_clone = bar.clone();
        handles.push(thread::spawn(move || {
            let bf_data = brute_force_nns(NB_NNS, train_ref, test_ref, ids_to_compute, bar_clone);
            bf_data
        }));
    }

    let mut bf_data = HashMap::new();
    for thread in handles {
        let bf_result = thread.join().unwrap();
        for (key, val) in bf_result {
            assert!(bf_data.insert(key, val).is_none());
        }
    }

    let data_file_name =
        format!("/home/gamal/glove_dataset/test_data/{file_name}_lim{lim}/bf_data.json");
    let file = File::create(&data_file_name)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &bf_data)?;
    writer.flush()?;

    let data_file_name =
        format!("/home/gamal/glove_dataset/test_data/{file_name}_lim{lim}/test_ids.json");
    let file = File::create(&data_file_name)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &test_idx)?;
    writer.flush()?;

    let data_file_name =
        format!("/home/gamal/glove_dataset/test_data/{file_name}_lim{lim}/train_ids.json");
    let file = File::create(&data_file_name)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &train_idx)?;
    writer.flush()?;

    Ok(())
}

fn split_glove(
    embs: Vec<Vec<f32>>,
    test_frac: f32,
) -> (
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    BTreeSet<usize>,
    BTreeSet<usize>,
) {
    assert!((test_frac > 0.0) & (test_frac < 1.0));
    let test_size = ((embs.len() as f32) * test_frac).round() as usize;
    println!("We will find neighbors for {test_size} vectors by brute-force");

    let ids: BTreeSet<usize> = (0..embs.len()).collect();
    let mut rng = &mut rand::thread_rng();
    let test_ids = BTreeSet::from_iter(
        ids.iter()
            .choose_multiple(&mut rng, test_size)
            .iter()
            .map(|id| **id),
    );
    let train_ids: BTreeSet<usize> = ids.difference(&test_ids).cloned().collect();

    assert_eq!(train_ids.len() + test_ids.len(), embs.len());

    let train = train_ids.iter().map(|id| embs[*id].clone()).collect();
    let test = test_ids.iter().map(|id| embs[*id].clone()).collect();

    (train, test, train_ids, test_ids)
}
