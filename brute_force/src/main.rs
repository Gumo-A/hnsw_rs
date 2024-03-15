use hnsw::helpers::args::parse_args_bf;
use hnsw::helpers::data::split_vector;
use hnsw::helpers::glove::{brute_force_nns, load_glove_array};

use std::fs::{create_dir, File};
use std::io::{BufWriter, Write};

// use indicatif::{FormattedDuration, ProgressBar, ProgressStyle};
// use std::collections::HashMap;
// use hnsw::helpers::distance::get_nn_bf;

use ndarray::Array2;

fn main() -> std::io::Result<()> {
    let (dim, lim, splits, split_to_compute) = parse_args_bf();
    let nb_nns = 10;

    let (_words, embeddings): (Vec<String>, Array2<f32>) =
        load_glove_array(dim, lim, true, split_to_compute).unwrap();

    let indices: Vec<i32> = (0..lim).collect();
    let indices_split = split_vector(indices, splits, split_to_compute);
    let bf_data = brute_force_nns(nb_nns, embeddings, indices_split, split_to_compute);

    let _ = create_dir(format!(
        "/home/gamal/glove_dataset/bf_rust/dim{dim}_lim{lim}"
    ));

    let file_name = format!(
        "/home/gamal/glove_dataset/bf_rust/dim{dim}_lim{lim}/split_{split_to_compute}.json"
    );

    let file = File::create(&file_name)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &bf_data)?;
    writer.flush()?;

    Ok(())
}
