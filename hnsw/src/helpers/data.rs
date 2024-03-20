use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{BufReader, Result};

pub fn split_vector(vector: Vec<i32>, nb_splits: u8, split_to_compute: u8) -> Vec<i32> {
    let mut split_vector: Vec<Vec<i32>> = Vec::new();

    let per_split = vector.len() / nb_splits as usize;

    let mut buffer = 0;
    for idx in (0..nb_splits).into_iter() {
        if idx == nb_splits - 1 {
            split_vector.push(vector[buffer..].to_vec());
            continue;
        }
        split_vector.push(vector[buffer..(buffer + per_split)].to_vec());
        buffer += per_split;
    }

    let mut sum_lens = 0;
    for i in split_vector.iter() {
        sum_lens += i.len();
    }

    assert!(sum_lens == vector.len(), "sum: {sum_lens}");

    split_vector[split_to_compute as usize].to_owned()
}

pub fn load_bf_data(dim: usize, lim: usize) -> Result<HashMap<usize, Vec<(usize, f32)>>> {
    let mut bf_data: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();

    let paths = fs::read_dir(format!(
        "/home/gamal/glove_dataset/bf_rust/dim{dim}_lim{lim}"
    ))
    .unwrap();

    for path in paths {
        let file_name = path.unwrap().path();
        let file = File::open(file_name)?;
        let reader = BufReader::new(file);
        let split_data: HashMap<usize, Vec<(usize, f32)>> = serde_json::from_reader(reader)?;
        for key in split_data.keys().into_iter() {
            bf_data.insert(*key, split_data.get(key).unwrap().clone());
        }
    }

    Ok(bf_data)
}
