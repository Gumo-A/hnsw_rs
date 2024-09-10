#[allow(unused_imports)]
use hnsw::hnsw::index::HNSW;

fn main() -> std::io::Result<()> {
    let index = HNSW::from_path("./index.ann")?;
    index.print_index();
    Ok(())
}
