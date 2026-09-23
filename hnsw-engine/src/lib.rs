use std::{
    fs::{File, create_dir, create_dir_all, remove_file},
    io::{self, Write},
    path::PathBuf,
    thread::sleep,
    time::Duration,
};

use flate2::{Compression, write::GzEncoder};
use points::point::Point;

type ID = u64;
type SaveInfo = (String, ID, ID);

pub struct Engine {
    loc: PathBuf,
}

impl Engine {
    pub fn new(loc: PathBuf) -> io::Result<Self> {
        let engine = Self { loc: loc.clone() };
        if loc.exists() {
            if loc.is_dir() {
                check_can_use_dir(&loc)?;
            } else {
                return Err(io::ErrorKind::NotADirectory.into());
            }
        } else {
            create_dir_all(&loc).unwrap();
        }
        check_or_create_dir_structure(&loc).unwrap();
        Ok(engine)
    }

    fn wait(&self, micros: u64) {
        sleep(Duration::from_micros(micros));
    }

    fn write_lock(&self) {
        let lock_file = self.loc.join("metadata-layer").join("lock-file");
        while lock_file.exists() {
            self.wait(100);
        }
        self.create_lock_file(&lock_file);
    }

    fn create_lock_file(&self, lock_file: &PathBuf) {
        match File::create_new(lock_file) {
            Ok(_) => (),
            Err(e) => match e.kind() {
                io::ErrorKind::AlreadyExists => {
                    self.write_lock();
                }
                _ => panic!("Could not create {lock_file:?}: {e:?}"),
            },
        }
    }

    fn write_unlock(&self) {
        let lock_file = self.loc.join("metadata-layer").join("write-lock-file");
        if lock_file.exists() {
            match remove_file(&lock_file) {
                Ok(()) => (),
                Err(e) => match e.kind() {
                    io::ErrorKind::NotFound => (),
                    _ => panic!("Could not delete {lock_file:?}: {e:?}"),
                },
            };
        }
    }

    pub fn insert(&self, payloads: Vec<Point>) -> io::Result<SaveInfo> {
        self.write_lock();
        let manifest = self.get_current_manifest();
        let save_info = self.add_payloads(&manifest, payloads);
        self.index_id_range(&save_info);
        self.write_unlock();
        Ok(save_info)
    }

    fn index_id_range(&self, _added_id_range: &SaveInfo) {
        // payloads were added to the data layer
        // their IDs are all IDs in the given range
        // insert these IDs in the index
        todo!("insert IDs in the index")
    }

    fn get_current_manifest(&self) -> Manifest {
        Manifest::new(&self.loc)
    }

    fn add_payloads(&self, manifest: &Manifest, mut payloads: Vec<Point>) -> SaveInfo {
        let save_info = self.save_payloads(manifest, &mut payloads);
        manifest.update(&save_info);
        save_info
    }

    fn save_payloads(&self, manifest: &Manifest, payloads: &Vec<Point>) -> SaveInfo {
        let last_id = manifest.get_last_id();
        let first_id = last_id + 1;
        let last_id = first_id + payloads.len() as ID - 1;
        let f_name = format!("{first_id}-{last_id}-payload");
        let path = self.loc.join("data-layer").join(f_name.clone());
        self.save_serialized_payloads(&path, payloads);
        (f_name, first_id, last_id)
    }

    fn save_serialized_payloads(&self, path: &PathBuf, payloads: &Vec<Point>) {
        let mut file = match File::create_new(path) {
            Ok(f) => f,
            Err(e) => panic!("Could not create payload file at {path:?}: {e:?}"),
        };
        let payloads_ser = self.serialize_payloads(&payloads);
        match file.write(&payloads_ser) {
            Ok(_bytes_written) => {
                todo!("log how many bytes were written")
            }
            Err(e) => panic!("Could not write payload to file {path:?}: {e:?}"),
        };
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

    /// Gets the highest ID in the index from the current manifest
    fn get_last_payload_id(&self) -> ID {
        let manifest = self.get_current_manifest();
        manifest.get_last_id()
    }
}

fn check_can_use_dir(loc: &PathBuf) -> io::Result<()> {
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
    match create_dir(&path) {
        Ok(()) => Ok(()),
        Err(e) => match e.kind() {
            io::ErrorKind::AlreadyExists => Ok(()),
            _ => panic!("Could not create dir {path:?}: {e:?}"),
        },
    }
}

fn check_or_create_dir_structure(loc: &PathBuf) -> io::Result<()> {
    try_create_dir(loc.join("metadata-layer"))?;
    try_create_dir(loc.join("data-layer"))?;
    try_create_dir(loc.join("index-layer"))?;

    Ok(())
}

struct Manifest {}

impl Manifest {
    pub fn new(_loc: &PathBuf) -> Self {
        Manifest {}
    }

    pub fn update(&self, _save_info: &SaveInfo) {
        todo!("update manifest after adding payloads")
    }

    pub fn get_last_id(&self) -> ID {
        todo!("returns last ID")
    }
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
