use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::index::HNSW;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::Result;

fn main() -> Result<()> {
    let (dim, lim, m) = match parse_args_eval() {
        Ok(args) => args,
        Err(err) => {
            println!("Help: load_index");
            println!("{}", err);
            println!("dim[int] lim[int] m[int]");
            return Ok(());
        }
    };
    let (_, embeddings) = load_glove_array(dim, lim, true, 0).unwrap();
    let index =
        HNSW::from_path(format!("/home/gamal/indices/eval_dim{dim}_lim{lim}_m{m}.json").as_str())?;
    bench_ann(&index, &embeddings);
    Ok(())
}

fn bench_ann(index: &HNSW, embeddings: &Vec<Vec<f32>>) {
    for ef in (12..100).step_by(12) {
        let sample_size: usize = 4_000;
        let bar = ProgressBar::new(sample_size as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta} : Elapsed: {elapsed} ] {per_sec} {wide_bar}",
            )
            .unwrap(),
        );
        bar.set_message(format!("Finding ANNs ef={ef}"));

        for vector in embeddings.iter().cycle().take(sample_size) {
            bar.inc(1);
            let _ = index.ann_by_vector(&vector, 10, ef);
        }
        println!("{}ms", bar.elapsed().as_millis());
    }
}
