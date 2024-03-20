use std::env;

pub fn parse_args_bf() -> (i32, i32, u8) {
    let args: Vec<String> = env::args().collect();
    let dim = args[1].parse::<i32>().expect("Could not parse dimention");
    let lim = args[2].parse::<i32>().expect("Could not parse limit");
    let splits = args[3]
        .parse::<u8>()
        .expect("Could not parse number of splits");
    (dim, lim, splits)
}

pub fn parse_args() -> (usize, usize) {
    let args: Vec<String> = env::args().collect();
    let dim = args[1].parse::<usize>().expect("Could not parse dimention");
    let lim = args[2].parse::<usize>().expect("Could not parse limit");
    (dim, lim)
}

pub fn parse_args_eval() -> (usize, usize, i32, i32) {
    let args: Vec<String> = env::args().collect();
    let dim = args[1].parse::<usize>().expect("Could not parse dimention");
    let lim = args[2].parse::<usize>().expect("Could not parse limit");
    let m = args[3].parse::<i32>().expect("Could not parse M");
    let ef_cons = args[4]
        .parse::<i32>()
        .expect("Could not parse efConstruciton");
    (dim, lim, m, ef_cons)
}
