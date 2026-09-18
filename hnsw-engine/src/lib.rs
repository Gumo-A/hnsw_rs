use std::{
    fs::{self, File, create_dir, create_dir_all, remove_file},
    io::{self, Write},
    path::PathBuf,
    str::FromStr,
    thread::sleep,
    time::Duration,
};

use flate2::{Compression, write::GzEncoder};
use points::point::Point;

type ID = u64;

pub struct Engine {
    loc: PathBuf,
}

impl Engine {
    pub fn new(loc: PathBuf) -> io::Result<Self> {
        let engine = Self { loc: loc.clone() };
        if loc.exists() {
            if loc.is_dir() {
                check_can_use_dir(loc.clone())?;
            } else {
                return Err(io::ErrorKind::NotADirectory.into());
            }
        } else {
            create_dir_all(loc.clone()).unwrap();
        }
        check_or_create_dir_structure(loc.clone()).unwrap();
        Ok(engine)
    }

    fn wait(&self, micros: u64) {
        sleep(Duration::from_micros(micros));
    }

    fn lock(&self, req_len: usize) {
        let lock_file = self.loc.join("metadata-layer").join("lock-file");
        while lock_file.exists() {
            self.wait(100);
        }
        self.create_lock_file(lock_file, req_len);
    }

    fn create_lock_file(&self, lock_file: PathBuf, req_len: usize) {
        match File::create_new(lock_file) {
            Ok(_) => (),
            Err(e) => match e.kind() {
                io::ErrorKind::AlreadyExists => self.lock(req_len),
                _ => panic!("Could not create lock file: {e:?}"),
            },
        }
    }

    fn unlock(&self) {
        let lock_file = self.loc.join("metadata-layer").join("lock-file");
        if lock_file.exists() {
            remove_file(lock_file).unwrap();
        }
    }

    pub fn insert(&self, payloads: Vec<Point>) -> io::Result<Vec<ID>> {
        let p_ids: Vec<ID> = self.add_payloads(payloads);
        // self.add_one_id_to_index(p_id);
        Ok(p_ids)
    }

    fn add_payloads(&self, mut payloads: Vec<Point>) -> Vec<ID> {
        let len = payloads.len();
        self.lock(len);
        let last_pl_id = self.get_last_payload_id();
        payloads
            .iter_mut()
            .enumerate()
            .for_each(|(idx, p)| p.id = 1 + idx as u32 + last_pl_id as u32);
        let ids = (last_pl_id + 1..last_pl_id + len as ID + 1).collect();
        let ids = self.save_payloads(ids, payloads);
        self.unlock();
        ids
    }

    fn save_payloads(&self, ids: Vec<ID>, payloads: Vec<Point>) -> Vec<ID> {
        assert_eq!(ids.len(), payloads.len());
        let first_id = ids.get(0).unwrap();
        let last_id = ids.last().unwrap();
        let f_name = format!("{first_id}-{last_id}-payload");
        let path = self.loc.join("data-layer").join(f_name);
        File::create_new(path)
            .unwrap()
            .write(&self.serialize_payloads(&payloads))
            .unwrap();
        self.overwrite_last_id_file(*last_id);
        ids
    }

    fn serialize_payloads(&self, payloads: &Vec<Point>) -> Vec<u8> {
        let mut buf = GzEncoder::new(vec![], Compression::default());
        // buf.write(&serde_json::to_vec(payloads).unwrap())
        buf.write(&rmp_serde::encode::to_vec(payloads).unwrap())
            .unwrap();
        buf.finish().unwrap()
    }

    fn overwrite_last_id_file(&self, id: ID) {
        let path = self.loc.join("metadata-layer").join("last_id.txt");
        let mut buf = vec![];
        write!(&mut buf, "{id}").unwrap();
        File::create(path).unwrap().write(&buf).unwrap();
    }

    fn get_last_payload_id(&self) -> ID {
        let last_id_file = self.loc.join("metadata-layer").join("last_id.txt");
        let last_id_string = match fs::read_to_string(last_id_file.clone()) {
            Ok(s) => s,
            Err(e) => match e.kind() {
                io::ErrorKind::NotFound => {
                    let mut buf = vec![];
                    write!(&mut buf, "0").unwrap();
                    File::create_new(last_id_file).unwrap().write(&buf).unwrap();
                    String::from_str("0").unwrap()
                }
                _ => panic!("Error while reading {last_id_file:?}: {e:?}"),
            },
        };
        last_id_string.parse::<ID>().unwrap()
    }
}

fn check_can_use_dir(loc: PathBuf) -> io::Result<()> {
    let entries = loc.read_dir().unwrap().count();
    if entries > 0 {
        if entries == 3 {
            let contains_meta_layer = loc.join("metadata-layer").exists();
            let contains_data_layer = loc.join("data-layer").exists();
            let contains_index_layer = loc.join("index-layer").exists();
            if contains_data_layer & contains_meta_layer & contains_index_layer {
                Ok(())
            } else {
                Err(io::ErrorKind::DirectoryNotEmpty.into())
            }
        } else {
            Err(io::ErrorKind::DirectoryNotEmpty.into())
        }
    } else {
        Ok(())
    }
}

fn try_create_dir(path: PathBuf) -> io::Result<()> {
    match create_dir(path.clone()) {
        Ok(()) => Ok(()),
        Err(e) => match e.kind() {
            io::ErrorKind::AlreadyExists => Ok(()),
            _ => panic!("Could not create dir {path:?}: {e:?}"),
        },
    }
}

fn check_or_create_dir_structure(loc: PathBuf) -> io::Result<()> {
    try_create_dir(loc.join("metadata-layer"))?;
    try_create_dir(loc.join("data-layer"))?;
    try_create_dir(loc.join("index-layer"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use points::point::Point;
    use vectors::gen_rand_vecs;

    use crate::Engine;

    fn gen_rand_points(n: usize) -> Vec<Point> {
        let vs = gen_rand_vecs(1024, n);
        let mut points = vec![];
        for v in vs {
            points.push(Point::with_level_and_id(&v, 0, 0));
        }
        points
    }

    #[test]
    fn add_payloads() {
        let e = Engine::new(Path::new("./engine-test-index").to_path_buf()).unwrap();

        for _ in 0..16 {
            let p = gen_rand_points(128);
            e.add_payloads(p);
        }
    }

    #[test]
    fn add_one_payload() {
        let e = Engine::new(Path::new("./engine-test-index").to_path_buf()).unwrap();

        for _ in 0..8 {
            let p = gen_rand_points(1);
            e.add_payloads(p);
        }
    }
}
