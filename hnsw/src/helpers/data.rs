use std::collections::{HashMap, HashSet};
use std::fs::{self, ReadDir};
use std::fs::{DirEntry, File};
use std::io::{BufReader, Result};

use crate::hnsw::points::Point;

pub fn split_ids(ids: Vec<u32>, nb_splits: u8) -> Vec<Vec<u32>> {
    let mut split_vector = Vec::new();

    let per_split = ids.len() / (nb_splits as usize);

    let mut buffer = 0;
    for idx in 0..nb_splits {
        if idx == nb_splits - 1 {
            split_vector.push(ids[buffer..].to_vec());
        } else {
            split_vector.push(ids[buffer..(buffer + per_split)].to_vec());
            buffer += per_split;
        }
    }

    let mut sum_lens = 0;
    for i in split_vector.iter() {
        sum_lens += i.len();
    }

    assert!(sum_lens == ids.len(), "sum: {sum_lens}");

    split_vector
}

pub fn split(nb_elements: usize, nb_splits: usize) -> Vec<Vec<usize>> {
    let mut split_vector: Vec<Vec<usize>> = Vec::new();

    let per_split: usize = nb_elements / nb_splits;

    let mut buffer = 0;
    for idx in 0..nb_splits {
        if idx == nb_splits - 1 {
            split_vector.push((buffer..nb_elements).collect());
            continue;
        }
        split_vector.push((buffer..(buffer + per_split)).collect());
        buffer += per_split;
    }

    let mut sum_lens = 0;
    for i in split_vector.iter() {
        sum_lens += i.len();
    }

    assert!(
        sum_lens == nb_elements,
        "Total elements: {nb_elements}, sum of splits: {sum_lens}"
    );

    split_vector
}

pub fn split_eps(
    points: HashMap<usize, Point>,
    eps: HashMap<usize, HashSet<usize>>,
    nb_splits: usize,
) -> Vec<Vec<(usize, Point)>> {
    let mut to_split = Vec::new();
    for (ep, points_ids) in eps.iter() {
        for id in points_ids {
            to_split.push((*ep, points.get(id).unwrap().clone()));
        }
    }

    assert_eq!(to_split.len(), points.len());

    let mut split_vector: Vec<Vec<(usize, Point)>> = Vec::new();

    let per_split = points.len() / nb_splits;

    let mut buffer = 0;
    for idx in 0..nb_splits {
        if idx == nb_splits - 1 {
            split_vector.push(to_split[buffer..].to_vec());
            continue;
        }
        split_vector.push(to_split[buffer..(buffer + per_split)].to_vec());
        buffer += per_split;
    }

    let mut sum_lens = 0;
    for i in split_vector.iter() {
        sum_lens += i.len();
    }

    assert!(sum_lens == points.len(), "sum: {sum_lens}");

    split_vector
}

pub fn load_bf_data(
    lim: usize,
    file_name: String,
) -> Result<(HashMap<usize, Vec<usize>>, HashSet<usize>, HashSet<usize>)> {
    let mut bf_data: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut test_ids: HashSet<usize> = HashSet::new();
    let mut train_ids: HashSet<usize> = HashSet::new();

    let paths: ReadDir = fs::read_dir(format!(
        "/home/gamal/glove_dataset/test_data/{file_name}_lim{lim}"
    ))
    .unwrap();

    for path in paths {
        let file_name: DirEntry = path?;
        let file = File::open(file_name.path())?;
        let reader = BufReader::new(file);
        let name_string = file_name.file_name().into_string().unwrap();
        if name_string == *"bf_data.json" {
            bf_data = serde_json::from_reader(reader)?;
        } else if name_string == "test_ids.json" {
            test_ids = serde_json::from_reader(reader)?;
        } else if name_string == "train_ids.json" {
            train_ids = serde_json::from_reader(reader)?;
        }
    }

    Ok((bf_data, train_ids, test_ids))
}
