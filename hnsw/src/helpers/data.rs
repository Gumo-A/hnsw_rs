use std::collections::HashMap;
use std::fs::{self, ReadDir};
use std::fs::{DirEntry, File};
use std::io::{BufReader, Result};

pub fn split_ids(ids: Vec<usize>, nb_splits: usize, split_to_compute: usize) -> Vec<usize> {
    let mut split_vector: Vec<Vec<usize>> = Vec::new();

    let per_split = ids.len() / nb_splits;

    let mut buffer = 0;
    for idx in 0..nb_splits {
        if idx == nb_splits - 1 {
            split_vector.push(ids[buffer..].to_vec());
            continue;
        }
        split_vector.push(ids[buffer..(buffer + per_split)].to_vec());
        buffer += per_split;
    }

    let mut sum_lens = 0;
    for i in split_vector.iter() {
        sum_lens += i.len();
    }

    assert!(sum_lens == ids.len(), "sum: {sum_lens}");

    split_vector[split_to_compute].to_owned()
}

pub fn load_bf_data(dim: usize, lim: usize) -> Result<HashMap<usize, Vec<usize>>> {
    let mut bf_data: HashMap<usize, Vec<usize>> = HashMap::new();

    let paths: ReadDir = fs::read_dir(format!(
        "/home/gamal/glove_dataset/bf_rust/dim{dim}_lim{lim}"
    ))
    .unwrap();

    for path in paths {
        let file_name: DirEntry = path?;
        let file = File::open(file_name.path())?;
        let reader = BufReader::new(file);
        let split_data: HashMap<usize, Vec<usize>> = serde_json::from_reader(reader)?;
        for key in split_data.keys().into_iter() {
            bf_data.insert(*key, split_data.get(key).unwrap().clone());
        }
    }

    Ok(bf_data)
}
