use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;

use hnsw::helpers::data::load_bf_data;
use hnsw::helpers::glove::load_glove_array;
use hnsw::hnsw::index::HNSW;
use hnsw::{
    helpers::args::parse_args_eval,
    hnsw::points::{Payload, PayloadType},
};

use ndarray::{s, Array2};

use indicatif::{ProgressBar, ProgressStyle};

fn main() -> std::io::Result<()> {
    let start = Instant::now();
    let (dim, lim, m) = match parse_args_eval() {
        Ok(args) => args,
        Err(err) => {
            println!("Help: eval_glove");
            println!("{}", err);
            println!("dim[int] lim[int] m[int]");
            return Ok(());
        }
    };

    let (words, embeddings) = load_glove_array(dim, lim, true, 0).unwrap();
    let bf_data = match load_bf_data(dim, lim) {
        Ok(data) => data,
        Err(err) => {
            println!("Error loading bf data: {err}");
            return Ok(());
        }
    };

    let payloads = Vec::from_iter(words.iter().map(|x| Payload {
        data: HashMap::from([
            ("word".to_string(), PayloadType::StringPayload(x.clone())),
            (
                "starts_with_e".to_string(),
                PayloadType::BoolPayload(x.starts_with('e')),
            ),
        ]),
    }));

    let mut index = HNSW::build_index_par(m, &embeddings, &Some(payloads));
    // let mut index = HNSW::new(m, None, dim);
    // index.build_index(&embeddings, false, Some(payloads))?;
    index.print_params();
    let filters = Some(Payload {
        data: HashMap::from([("starts_with_e".to_string(), PayloadType::BoolPayload(true))]),
    });
    estimate_recall(&mut index, &embeddings, &bf_data, &None);

    for (i, idx) in bf_data.keys().enumerate() {
        // for (i, idx) in (110..115).enumerate() {
        if i > 3 {
            break;
        }
        let vector = embeddings.slice(s![*idx, ..]);
        let anns = index.ann_by_vector(&vector, 10, 16, &filters);
        println!("ANNs of {}", words[*idx]);
        let anns_words: Vec<String> = anns.iter().map(|x| words[*x as usize].clone()).collect();
        println!("{:?}", anns_words);
        println!("True NN of {}", words[*idx]);
        let true_nns: Vec<String> = bf_data
            .get(&idx)
            .unwrap()
            .iter()
            .map(|x| words[*x].clone())
            .take(10)
            .collect();
        println!("{:?}", true_nns);
    }
    let end = Instant::now();
    println!(
        "Elapsed time: {}s",
        start.elapsed().as_secs() - end.elapsed().as_secs()
    );
    Ok(())
}

fn estimate_recall(
    index: &mut HNSW,
    embeddings: &Array2<f32>,
    bf_data: &HashMap<usize, Vec<usize>>,
    filters: &Option<Payload>,
) {
    let mut rng = rand::thread_rng();
    let max_id = index.node_ids.iter().max().unwrap_or(&usize::MAX);
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

        let mut recall_10: HashMap<usize, f32> = HashMap::new();
        for _ in (0..sample_size).enumerate() {
            bar.inc(1);

            let idx = rng.gen_range(0..(index.node_ids.len()));
            let vector = embeddings.slice(s![idx, ..]);
            let anns = index.ann_by_vector(&vector, 10, ef, &filters);
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
