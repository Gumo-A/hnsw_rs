use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::index::HNSW;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use std::collections::HashMap;

fn main() -> std::io::Result<()> {
    let (dim, lim, m) = match parse_args_eval() {
        Ok(args) => args,
        Err(err) => {
            println!("Help: load_index");
            println!("{}", err);
            println!("dim[int] lim[int] m[int]");
            return Ok(());
        }
    };

    let index =
        HNSW::from_path(format!("/home/gamal/indices/eval_dim{dim}_lim{lim}_m{m}.json").as_str())?;
    index.print_index();

    // TODO: change to adapt to new bf method
    // let bf_data = match load_bf_data(dim, lim) {
    //     Ok(data) => data,
    //     Err(err) => {
    //         println!("Error loading bf data: {err}");
    //         return Ok(());
    //     }
    // };
    // let (_, embeddings) = load_glove_array(dim, lim, true, true).unwrap();
    // estimate_recall(&mut index, &bf_data).unwrap();
    Ok(())
}
