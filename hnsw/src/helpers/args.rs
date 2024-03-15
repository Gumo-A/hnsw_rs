use std::env;

pub fn parse_args_bf() -> (i32, i32, usize, usize) {
    let args: Vec<String> = env::args().collect();
    let dim = args[1].parse::<i32>().expect("Could not parse dimention");
    let lim = args[2].parse::<i32>().expect("Could not parse limit");
    let splits = args[3]
        .parse::<usize>()
        .expect("Could not parse number of splits");
    let split_to_compute = args[4]
        .parse::<usize>()
        .expect("Could not parse split to compute");
    (dim, lim, splits, split_to_compute)
}

pub fn parse_args() -> (usize, usize) {
    let args: Vec<String> = env::args().collect();
    let dim = args[1].parse::<usize>().expect("Could not parse dimention");
    let lim = args[2].parse::<usize>().expect("Could not parse limit");
    (dim, lim)
}
