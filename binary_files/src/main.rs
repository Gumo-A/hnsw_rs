use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

fn main() -> std::io::Result<()> {
    save("./temp_file")?;
    modify("./temp_file")?;
    let vector = load("./temp_file")?;
    println!("{vector:?}");
    Ok(())
}

fn save(path: &str) -> std::io::Result<()> {
    let path = std::path::Path::new(path);
    if !path.parent().unwrap().exists() {
        std::fs::create_dir_all(path.parent().unwrap())?;
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let vector: Vec<u8> = Vec::from_iter(0..10);
    for value in vector {
        writer.write(&[value])?;
    }
    writer.flush()?;
    Ok(())
}

fn load(path: &str) -> std::io::Result<Vec<u8>> {
    let path = std::path::Path::new(path);
    let file = File::open(path)?;
    // let reader = BufReader::new(file);

    let mut vector: Vec<u8> = Vec::new();
    let bytes = file.bytes();
    for byte in bytes {
        vector.push(byte?);
    }
    Ok(vector)
}

fn modify(path: &str) -> std::io::Result<()> {
    let path = std::path::Path::new(path);
    let mut file = File::options().write(true).read(true).open(path)?;

    println!("Started modif at {}", file.stream_position()?);
    file.seek(SeekFrom::Start(1))?;
    println!("Moved to {}", file.stream_position()?);

    let float = -1.5f32;

    file.write(&float.to_be_bytes())?;

    println!("Wrote an f32, now at {}", file.stream_position()?);

    for byte in file.try_clone().unwrap().bytes() {
        println!("{}", byte?);
    }

    file.seek(SeekFrom::Start(9))?;

    println!("Jumpted to {}", file.stream_position()?);

    for byte in file.try_clone().unwrap().bytes() {
        println!("{}", byte?);
    }

    file.rewind()?;

    println!("Back to {}", file.stream_position()?);

    for byte in file.try_clone().unwrap().bytes() {
        println!("{}", byte?);
    }

    file.seek(SeekFrom::End(100))?;

    println!("Now at {}, is this undefined?", file.stream_position()?);

    for byte in file.bytes() {
        println!("{}", byte?);
    }

    Ok(())
}
