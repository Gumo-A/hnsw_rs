use indicatif::{ProgressBar, ProgressStyle};
// use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Result};

// use crate::helpers::distance::get_nn_bf;

pub fn load_glove_array(
    dim: usize,
    lim: usize,
    normalize: bool,
    progress_bar: usize,
) -> Result<(Vec<String>, Vec<Vec<f32>>)> {
    let file = File::open(format!("/home/gamal/glove_dataset/glove.6B.{dim}d.txt"))?;
    let reader = BufReader::new(file);

    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let mut words: Vec<String> = Vec::new();

    let bar = if progress_bar != 0 {
        let bar = ProgressBar::hidden();
        bar
    } else {
        let bar = ProgressBar::new(lim.try_into().unwrap());
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} {wide_bar} {human_pos}/{human_len} {percent}% [ ETA: {eta} : Elapsed: {elapsed} ] {per_sec}",
            )
            .unwrap(),
        );
        bar.set_message(format!("Loading Embeddings"));
        bar
    };

    for (idx, line_result) in reader.lines().enumerate() {
        bar.inc(1);
        if idx >= lim.try_into().unwrap() {
            break;
        }
        let line = line_result?;
        let mut parts = line.split_whitespace();

        let word = parts.next().expect("Empty line");

        let mut values: Vec<f32> = parts
            .map(|s| s.parse::<f32>().expect("Could not parse float"))
            .collect();
        let norm: f32 = if normalize {
            values
                .clone()
                .iter()
                .map(|x| x.powf(2.0))
                .sum::<f32>()
                .powf(0.5)
        } else {
            1.0
        };
        values = values.iter().map(|x| x / norm).collect();
        embeddings.push(values);
        // for (jdx, val) in values.enumerate() {
        //     let entry = embeddings.get_mut((idx, jdx)).unwrap();
        //     *entry = val / norm
        // }

        words.push(word.to_string());
    }
    Ok((words, embeddings))
}

// pub fn brute_force_nns(
//     nb_of_nn: usize,
//     embeddings: &Vec<Vec<f32>>,
//     indices: Vec<usize>,
//     pros_nb: usize,
//     words: Vec<String>,
// ) -> HashMap<usize, Vec<usize>> {
//     let bar = if pros_nb != 0 {
//         let bar = ProgressBar::hidden();
//         bar
//     } else {
//         let bar = ProgressBar::new(indices.len().try_into().unwrap());
//         bar.set_style(
//             ProgressStyle::with_template(
//                 "{msg} {wide_bar} {human_pos}/{human_len} {percent}% [ ETA: {eta_precise} : Elapsed: {elapsed} ] {per_sec}",
//             )
//             .unwrap(),
//         );
//         bar.set_message("Finding NNs");
//         bar
//     };

//     let mut brute_force_results: HashMap<usize, Vec<usize>> = HashMap::new();
//     for idx in indices.iter() {
//         let nns = get_nn_bf(&embeddings[*idx], &embeddings, nb_of_nn.try_into().unwrap());
//         let index = *idx as usize;
//         brute_force_results.insert(
//             index,
//             nns.iter()
//                 .map(|x| x.0)
//                 .filter(|x| words[*x].starts_with('e'))
//                 .collect(),
//         );
//         bar.inc(1);
//     }

//     brute_force_results
// }
