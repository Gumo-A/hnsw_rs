use hnsw::helpers::args::parse_args_eval;
use hnsw::hnsw::index::HNSW;
use std::io::Result;
use std::thread;
use std::time::Duration;

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
    let _index =
        HNSW::from_path(format!("/home/gamal/indices/eval_glove_dim{dim}_lim{lim}_m{m}").as_str());
    thread::sleep(Duration::from_secs(10));
    Ok(())
}
