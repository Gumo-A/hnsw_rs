use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::thread;

use hnsw::helpers::args::parse_args_bf;
use hnsw::helpers::data::split_ids;
use hnsw::helpers::glove::{brute_force_nns, load_glove_array};

// TODO: make this binary compute a proper dataset:
//         - A set of points to build the index.
//         - A second set of points for which we will find ANNs.
//         - This binary should store the true NNs of the second set, in the first.
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
    let _ = create_dir_all(format!(
        "/home/gamal/glove_dataset/bf_rust/dim{dim}_lim{lim}"
    ));

    let (words, embeddings): (Vec<String>, Vec<Vec<f32>>) =
        load_glove_array(dim, lim, false, 0).unwrap();

    let embeddings = Arc::new(embeddings);
    let mut children = vec![];

    for i in 0..nb_threads {
        let nb_nns = 10_000;

        let embs = embeddings.clone();
        let indices: Vec<usize> = (0..lim).collect();

        let indices_split = split_ids(indices, nb_threads, i);

        let words_clone = words.clone();

        children.push(thread::spawn(move || -> std::io::Result<()> {
            let bf_data = brute_force_nns(nb_nns, &embs, indices_split, i, words_clone);
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
