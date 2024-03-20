use std::fs::{create_dir, File};
use std::io::{BufWriter, Write};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use ndarray::Array2;

use hnsw::helpers::args::parse_args_bf;
use hnsw::helpers::data::split_vector;
use hnsw::helpers::glove::{brute_force_nns, load_glove_array};

fn main() -> std::io::Result<()> {
    let (dim, lim, nb_threads) = parse_args_bf();

    // TODO: delete files in dir if dir exists.
    let _ = create_dir(format!(
        "/home/gamal/glove_dataset/bf_rust/dim{dim}_lim{lim}"
    ));

    let (_words, embeddings): (Vec<String>, Array2<f32>) =
        load_glove_array(dim, lim, true, 0).unwrap();

    let embeddings = Arc::new(embeddings);

    let (tx, rx) = mpsc::channel();

    for i in 0..nb_threads {
        let tx = tx.clone();
        let nb_nns = 10;

        let embs = embeddings.clone();
        let indices: Vec<i32> = (0..lim).collect();

        let indices_split = split_vector(indices, nb_threads, i);

        thread::spawn(move || -> std::io::Result<()> {
            let bf_data = brute_force_nns(nb_nns, &embs, indices_split, i);
            let file_name =
                format!("/home/gamal/glove_dataset/bf_rust/dim{dim}_lim{lim}/split_{i}.json");
            let file = File::create(&file_name)?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer(&mut writer, &bf_data)?;
            writer.flush()?;

            tx.send(()).unwrap();

            Ok(())
        });
    }

    for _ in 0..nb_threads {
        rx.recv().unwrap();
    }

    Ok(())
}
