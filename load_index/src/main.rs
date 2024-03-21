use hnsw::hnsw::HNSW;
use std::io::Result;
use std::thread;
use std::time::Duration;

fn main() -> Result<()> {
    let index = HNSW::load("/home/gamal/indices/eval_glove_dim100_lim400000");
    thread::sleep(Duration::from_secs(10));
    Ok(())
}
