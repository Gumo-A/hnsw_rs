use std::env;

pub fn parse_args_bf() -> Result<(usize, usize), &'static str> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        return Err("Expected exactly 2 positional arguments.");
    }
    let dim = args[1].parse::<usize>().expect("Could not parse dimention");
    let lim = args[2].parse::<usize>().expect("Could not parse limit");
    Ok((dim, lim))
}

pub fn parse_args() -> (usize, usize) {
    let args: Vec<String> = env::args().collect();
    let dim = args[1].parse::<usize>().expect("Could not parse dimention");
    let lim = args[2].parse::<usize>().expect("Could not parse limit");
    (dim, lim)
}

pub fn parse_args_eval() -> Result<(usize, usize, usize), &'static str> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        return Err("Expected exactly 3 positional arguments.");
    }

    let dim = args[1].parse::<usize>().expect("Could not parse dimention");
    let lim = args[2].parse::<usize>().expect("Could not parse limit");
    let m = args[3].parse::<usize>().expect("Could not parse M");
    Ok((dim, lim, m))
}
