use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::index::HNSW;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use std::collections::HashMap;
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
    let bf_data = match load_bf_data(dim, lim) {
        Ok(data) => data,
        Err(err) => {
            println!("Error loading bf data: {err}");
            return Ok(());
        }
    };
    let (_, embeddings) = load_glove_array(dim, lim, true, 0).unwrap();
    let mut index =
        HNSW::from_path(format!("/home/gamal/indices/eval_dim{dim}_lim{lim}_m{m}.json").as_str())?;
    // bench_ann(&index, &embeddings);
    index.print_params();
    estimate_recall(&mut index, &embeddings, &bf_data);
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

fn estimate_recall(
    index: &mut HNSW,
    // TODO: this doesnt need to be passed if "index" stores the vectors
    embeddings: &Vec<Vec<f32>>,
    bf_data: &HashMap<usize, Vec<usize>>,
) {
    let mut rng = rand::thread_rng();
    let max_id = index.points.ids().max().unwrap_or(&usize::MAX);
    for ef in (12..100).step_by(12) {
        let sample_size: usize = 1000;
        let bar = ProgressBar::new(sample_size as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} {human_pos}/{human_len} {percent}% [ ETA: {eta} : Elapsed: {elapsed} ] {per_sec} {wide_bar}",
            )
            .unwrap(),
        );
        bar.set_message(format!("Finding ANNs ef={ef}"));

        // let mut bencher = Bencher::new();
        let mut recall_10: HashMap<usize, f32> = HashMap::new();
        for _ in (0..sample_size).enumerate() {
            bar.inc(1);

            let idx = rng.gen_range(0..(index.points.len()));
            let vector = &embeddings[idx];
            let anns = index.ann_by_vector(
                &vector, 10, ef,
                // &mut bencher
            );
            let true_nns: Vec<usize> = bf_data
                .get(&idx)
                .unwrap()
                .clone()
                .iter()
                .filter(|x| x <= &max_id)
                .map(|x| *x)
                .collect();
            let length = if (true_nns.len() < 10) | (anns.len() < 10) {
                true_nns.len().min(anns.len())
            } else {
                10
            };
            let mut hits = 0;
            for ann in anns[..length].iter() {
                if true_nns[..length].contains(ann) {
                    hits += 1;
                }
            }
            recall_10.insert(idx, (hits as f32) / 10.0);
        }
        let mut avg_recall = 0.0;
        for (_, recall) in recall_10.iter() {
            avg_recall += recall;
        }
        avg_recall /= recall_10.keys().count() as f32;
        println!("Recall@10 {avg_recall}");
    }
}
