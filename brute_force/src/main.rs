use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::thread;

use ndarray::Array2;

use hnsw::helpers::args::parse_args_bf;
use hnsw::helpers::data::split_vector;
use hnsw::helpers::glove::{brute_force_nns, load_glove_array};

fn main() -> std::io::Result<()> {
    let (dim, lim, nb_threads) = match parse_args_bf() {
        Ok(args) => args,
        Err(err) => {
            println!("Help: brute_force");
            println!("dim[int] limit[int] number_of_threads[int]");
            println!("{}", err);
            return Ok(());
        }
    };

    // TODO: delete files in dir if dir exists.
    //
    let _ = create_dir_all(format!(
        "/home/gamal/glove_dataset/bf_rust/dim{dim}_lim{lim}"
    ));

    let (_words, embeddings): (Vec<String>, Array2<f32>) =
        load_glove_array(dim, lim, true, 0).unwrap();

    let embeddings = Arc::new(embeddings);
    let mut children = vec![];

    for i in 0..nb_threads {
        let nb_nns = 1000;

        let embs = embeddings.clone();
        let indices: Vec<usize> = (0..lim).collect();

        let indices_split = split_vector(indices, nb_threads, i);

        children.push(thread::spawn(move || -> std::io::Result<()> {
            let bf_data = brute_force_nns(nb_nns, &embs, indices_split, i);
            let file_name =
                format!("/home/gamal/glove_dataset/bf_rust/dim{dim}_lim{lim}/split_{i}.json");
            let file = File::create(&file_name)?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer(&mut writer, &bf_data)?;
            writer.flush()?;
            Ok(())
        }));
    }

    for child in children {
        let _ = child.join();
    }

    Ok(())
}
