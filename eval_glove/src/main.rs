use std::fs::remove_dir_all;
use std::fs::File;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::glove::load_glove_array;
use hnsw_engine::engine::Engine;
use hnsw_engine::index::IndexFacade;
use log::info;

macro_rules! timeit {
    ($f:expr) => {
        let s = Instant::now();
        $f;
        let e = s.elapsed().as_millis();
        info!("Executed {} in {e:?} ms", stringify!($f));
    };
    (let $x:ident = $f:expr) => {
        let s = Instant::now();
        let $x = $f;
        let e = s.elapsed().as_millis();
        info!("Executed {} in {e:?} ms", stringify!($f));
    };
}

fn main() {
    env_logger::init();
    let (lim, m) = match parse_args_eval() {
        Ok(args) => args,
        Err(err) => {
            println!("Help: eval_glove");
            println!("{}", err);
            println!("lim[int] m[int]");
            return ();
        }
    };

    let file = File::open(format!("/home/gamal/glove/300d.txt")).unwrap();
    let (words, embeddings) = load_glove_array(lim, file).unwrap();

    let tmp_path = Path::new("./persistance");
    if tmp_path.exists() {
        remove_dir_all(tmp_path).unwrap();
    }
    let _ = Command::new("cp")
        .arg("-r")
        .arg("./persistance-cp")
        .arg("./persistance")
        .output()
        .unwrap();
    timeit!(let store = Engine::new(tmp_path.to_path_buf(), m).unwrap());
    dbg!(&store.get_m());
    timeit!(store.insert(&embeddings[100_000..110_000]).unwrap());
    // timeit!(store.insert(&embeddings[..100_000]).unwrap());
    timeit!(store.optimize().unwrap());
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

    //         let word_idx = *words_map.get(&query).unwrap();
    //         let vector = store.get_point(word_idx).unwrap().get_vals();

    //         let anns = store.ann(&vector, 10, ef).unwrap();
    //         let anns_words: Vec<String> = anns.iter().map(|x| words[*x as usize].clone()).collect();
    //         println!("ANNs of {query} (ef={ef})");
    //         println!("{:?}", anns_words);
    //     }
    // }
}
