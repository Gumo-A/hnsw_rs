#[allow(unused_imports)]
use std::collections::HashMap;
use std::env;
use std::fs::remove_dir_all;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::glove::load_glove_array;
use hnsw_engine::engine::Engine;
use hnsw_engine::index::IndexFacade;
use points::point::Point;
use vectors::QuantVec;
use vectors::VecBase;

fn main() -> std::io::Result<()> {
    env_logger::init();
    let (lim, m) = match parse_args_eval() {
        Ok(args) => args,
        Err(err) => {
            println!("Help: eval_glove");
            println!("{}", err);
            println!("lim[int] m[int]");
            return Ok(());
        }
    };

    let file = File::open(format!("/home/gamal/glove/300d.txt"))?;
    // let file = File::open(format!(
    //     "/home/gamal/repos/vector-store/test-data/store.txt"
    // ))?;

    let (words, embeddings) = load_glove_array(lim, file, true).unwrap();

    let tmp_path = Path::new("./persistance");
    if tmp_path.exists() {
        remove_dir_all(tmp_path).unwrap();
    }
    let mut store = Engine::new(Path::new("./persistance").to_path_buf(), m).unwrap();
    let s = Instant::now();
    let embeddings: Vec<Point> = embeddings
        .iter()
        .enumerate()
        .map(|(idx, v)| Point::with_vec_and_id(&QuantVec::new(v), idx))
        .collect();
    store.insert(&embeddings).unwrap();
    let e = s.elapsed().as_secs_f32();
    println!(
        "took {0}s to build index with {1} points and M {2}",
        e,
        store.len().unwrap(),
        store.get_m()
    );
    // {
    //     use text_io::read;

    //     let ef = 100;
    //     let words_map: HashMap<String, usize> =
    //         HashMap::from_iter(words.iter().enumerate().map(|(idx, w)| (w.clone(), idx)));
    //     loop {
    //         println!("Look for NNs of a word (Ctrl+C to exit):");
    //         let query: String = read!();

    //         if !words_map.contains_key(&query) {
    //             println!("'{query}' is not in the index, try another word.");
    //             continue;
    //         }

    //         let word_idx = *words_map.get(&query).unwrap() as u32;
    //         let vector = store.get_point(word_idx).unwrap().get_vals();

    //         let anns = store.ann_by_vector(&vector, 10, ef).unwrap();
    //         let anns_words: Vec<String> = anns.iter().map(|x| words[*x as usize].clone()).collect();
    //         println!("ANNs of {query} (ef={ef})");
    //         println!("{:?}", anns_words);
    //     }
    // }
    Ok(())
}
