use hnsw::helpers::args::parse_args;
use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::HNSW;

use ndarray::s;

use indicatif::{ProgressBar, ProgressStyle};

fn main() {
    let (dim, lim) = parse_args();
    let mut index = HNSW::new(16);
    let (words, embeddings) = load_glove_array(dim as i32, lim as i32, true, 0).unwrap();

    let bar = ProgressBar::new(lim.try_into().unwrap());
    bar.set_style(
        ProgressStyle::with_template(
            "{msg} {wide_bar} {human_pos}/{human_len} {percent}% [ ETA: {eta} : Elapsed: {elapsed} ] {per_sec}",
        )
        .unwrap(),
    );
    bar.set_message(format!("Inserting Embeddings"));

    for idx in 0..lim {
        bar.inc(1);
        index.insert(idx as i32, embeddings.slice(s![idx, ..]).to_owned())
    }
}
