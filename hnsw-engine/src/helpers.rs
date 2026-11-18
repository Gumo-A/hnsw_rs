use std::env;

use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};

pub mod args;
pub mod data;
pub mod glove;

pub fn get_progress_bar(message: &str, remaining: usize) -> ProgressBar {
    let bar = if env::var("HNSW_DEBUG").is_ok() | env::var("HNSW_PB").is_ok() {
        ProgressBar::new(remaining as u64).with_finish(ProgressFinish::AndLeave)
    } else {
        return ProgressBar::hidden();
    };
    bar.set_message(message.to_string());
    bar.set_style(
        ProgressStyle::with_template(
            "{msg} {bar:40} {percent}% of {pos}/{len} Elapsed: {elapsed} | ETA: {eta} | {per_sec}\n",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    bar
}
