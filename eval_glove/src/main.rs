#[allow(unused_imports)]
use std::collections::HashMap;
use std::fs::remove_dir_all;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use hnsw::helpers::args::parse_args_eval;
use hnsw::helpers::glove::load_glove_array;

use hnsw::template::HNSW;

use rand::seq::SliceRandom;
use rand::thread_rng;

fn main() -> std::io::Result<()> {
    let (lim, m) = match parse_args_eval() {
        Ok(args) => args,
        Err(err) => {
            println!("Help: eval_glove");
            println!("{}", err);
            println!("lim[int] m[int]");
            return Ok(());
        }
    };

    let file = File::open(format!(
        "/home/gamal/repos/vector-store/test-data/store.txt"
    ))?;

    let (words, embeddings) = load_glove_array(lim, file, true).unwrap();

    let s = Instant::now();
    let mut store = HNSW::new(m, None, embeddings[0].len());
    store = store.insert_bulk(embeddings, 1, true).unwrap();
    let e = s.elapsed().as_millis();
    println!(
        "took {0} ms to build index with {1} points and M {2}",
        e,
        store.len(),
        store.params.m
    );

    let path = Path::new("./index_eval_test");
    if path.exists() {
        remove_dir_all(path).unwrap();
    }

    let s = Instant::now();
    store.save(path);
    let e = s.elapsed().as_millis();
    println!(
        "took {0} ms to save index with {1} points and M {2}",
        e,
        store.len(),
        store.params.m
    );

    let s = Instant::now();
    let store = HNSW::load(path).unwrap();
    let e = s.elapsed().as_millis();
    println!(
        "took {0} ms to load index with {1} points and M {2}",
        e,
        store.len(),
        store.params.m
    );
    dbg!(&store);

    show_nn_words(&words, &store, 10);
    {
        use text_io::read;

        let ef = 1000;
        loop {
            let words_map: HashMap<String, usize> =
                HashMap::from_iter(words.iter().enumerate().map(|(idx, w)| (w.clone(), idx)));
            println!("Look for NNs of a word ('_quit' to exit):");
            let query: String = read!();

            if query == "_quit".to_string() {
                break;
            }

            if !words_map.contains_key(&query) {
                println!("'{query}' is not in the index, try another word.");
                continue;
            }

            let word_idx = words_map.get(&query).unwrap();
            let vector = store.get_point(*word_idx as u32).unwrap();

            let anns = store.ann_by_vector(&vector, 10, ef).unwrap();
            let anns_words: Vec<String> = anns.iter().map(|x| words[*x as usize].clone()).collect();
            println!("ANNs of {query} (ef={ef})");
            println!("{:?}", anns_words);
        }
    }
    Ok(())
}

fn show_nn_words(words: &Vec<String>, store: &HNSW, max: usize) {
    let ef = 1000;
    let mut words: Vec<(usize, &String)> = words.iter().enumerate().collect();
    words.shuffle(&mut thread_rng());
    let mut c = 0;
    for (idx, word) in words.iter() {
        if c > max {
            break;
        }
        c += 1;
        let point = store.get_point(*idx as u32).unwrap();
        let anns = store.ann_by_vector(&point, 10, ef).unwrap();
        let anns_words: Vec<String> = anns.iter().map(|x| words[*x as usize].1.clone()).collect();
        println!();
        println!("ANNs of {} (ef={ef})", word);
        for w in anns_words.iter() {
            println!("  - {w}",);
        }
    }
}
