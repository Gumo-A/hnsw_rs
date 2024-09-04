use std::collections::{HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::thread;

use rand::seq::IteratorRandom;

use hnsw::helpers::args::parse_args_bf;
use hnsw::helpers::data::split_ids;
use hnsw::helpers::glove::{brute_force_nns, load_glove_array};
use hnsw::hnsw::points::Points;

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

    let _ = create_dir_all(format!(
        "/home/gamal/glove_dataset/test_data/dim{dim}_lim{lim}/"
    ));

    let (_, embeddings): (Vec<String>, Vec<Vec<f32>>) =
        load_glove_array(dim, lim, true, true).unwrap();

    let test_frac = 0.01;
    let (train_set, test_set, train_idx, test_idx) = split_glove(embeddings, test_frac);

    let train_set = Points::from_vecs(train_set, 0.0);
    let test_set = Points::from_vecs(test_set, 0.0);

    let train_arc = Arc::new(train_set);
    let test_arc = Arc::new(test_set);

    let nb_threads = std::thread::available_parallelism().unwrap().get();
    let test_ids: Vec<usize> = test_arc.ids().collect();
    let mut indices_split = split_ids(test_ids, nb_threads);
    let mut handles = Vec::new();
    for i in 0..nb_threads {
        let train_ref = train_arc.clone();
        let test_ref = test_arc.clone();
        let ids_to_compute = indices_split.pop().unwrap();

        handles.push(thread::spawn(move || {
            let bf_data = brute_force_nns(NB_NNS, train_ref, test_ref, ids_to_compute, i == 0);
            bf_data
        }));
    }

    let mut bf_data = HashMap::new();
    for thread in handles {
        let bf_result = thread.join().unwrap();
        for (key, val) in bf_result {
            bf_data.insert(key, val);
        }
    }

    let file_name = format!("/home/gamal/glove_dataset/test_data/dim{dim}_lim{lim}/bf_data.json");
    let file = File::create(&file_name)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &bf_data)?;
    writer.flush()?;

    let file_name = format!("/home/gamal/glove_dataset/test_data/dim{dim}_lim{lim}/test_ids.json");
    let file = File::create(&file_name)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &test_idx)?;
    writer.flush()?;

    let file_name = format!("/home/gamal/glove_dataset/test_data/dim{dim}_lim{lim}/train_ids.json");
    let file = File::create(&file_name)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &train_idx)?;
    writer.flush()?;

    Ok(())
}

fn split_glove(
    embs: Vec<Vec<f32>>,
    test_frac: f32,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, HashSet<usize>, HashSet<usize>) {
    assert!((test_frac > 0.0) & (test_frac < 1.0));
    let test_size = ((embs.len() as f32) * test_frac).round() as usize;

    let ids: HashSet<usize> = (0..embs.len()).collect();
    let mut rng = &mut rand::thread_rng();
    let test_ids = HashSet::from_iter(
        ids.iter()
            .choose_multiple(&mut rng, test_size)
            .iter()
            .map(|id| **id),
    );
    let train_ids: HashSet<usize> = ids.difference(&test_ids).cloned().collect();

    let train = (0..embs.len())
        .filter(|id| train_ids.contains(id))
        .map(|id| embs[id].clone())
        .collect();
    let test = (0..embs.len())
        .filter(|id| test_ids.contains(id))
        .map(|id| embs[id].clone())
        .collect();

    (train, test, train_ids, test_ids)
}
