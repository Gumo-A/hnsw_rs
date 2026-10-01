use std::{
    env,
    fs::{File, create_dir, create_dir_all, remove_file},
    io::{self, Read, Write},
    path::PathBuf,
    thread::sleep,
    time::Duration,
};

use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use points::point::{Point, VecType};

use crate::{
    AddedPoints, ID,
    cache::PointCache,
    get_rand_file_name,
    manifest::{Manifest, ManifestList},
};

pub struct Engine {
    loc: PathBuf,
    point_cache: PointCache,
    debug: bool,
}

impl Engine {
    pub fn new(loc: PathBuf) -> io::Result<Self> {
        let engine = Self {
            loc: loc.clone(),
            point_cache: PointCache::new(),
            debug: env::var("HNSW_DEBUG").is_ok(),
        };
        if loc.exists() {
            if loc.is_dir() {
                Self::check_can_use_dir(&loc)?;
            } else {
                return Err(io::ErrorKind::NotADirectory.into());
            }
        } else {
            match create_dir_all(&loc) {
                Ok(()) => (),
                Err(e) => match e.kind() {
                    io::ErrorKind::AlreadyExists => (),
                    _ => panic!("Could not create {loc:?}: {e}"),
                },
            };
        }
        Self::check_or_create_dir_structure(&loc);
        Ok(engine)
    }

    fn check_can_use_dir(loc: &PathBuf) -> io::Result<()> {
        let check = |p| loc.join(p).exists();
        let empty = loc.read_dir()?.count() == 0;
        if empty {
            Ok(())
        } else if check("metadata-layer") & check("data-layer") & check("index-layer") {
            Ok(())
        } else {
            Err(io::ErrorKind::DirectoryNotEmpty.into())
        }
    }

    fn try_create_dir(path: PathBuf) {
        match create_dir(&path) {
            Ok(()) => (),
            Err(e) => match e.kind() {
                io::ErrorKind::AlreadyExists => (),
                _ => panic!("Could not create dir {path:?}: {e}"),
            },
        }
    }

    fn check_or_create_dir_structure(loc: &PathBuf) {
        Self::try_create_dir(loc.join("metadata-layer"));
        Self::try_create_dir(loc.join("data-layer"));
        Self::try_create_dir(loc.join("index-layer"));
    }

    fn wait(&self, micros: u64) {
        sleep(Duration::from_micros(micros));
    }

    fn write_lock(&self) {
        let lock_file = self.loc.join("metadata-layer").join("write-lock-file");
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
                _ => panic!("Could not create {lock_file:?}: {e}"),
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
                    _ => panic!("Could not delete {lock_file:?}: {e}"),
                },
            };
        }
    }

    pub fn get_point(&mut self, id: ID) -> Option<&Point> {
        if self.point_cache.contains_key(&id) {
            self.point_cache.get(&id)
        } else {
            self.load_data_file_with_id(id)
        }
    }

    fn load_data_file_with_id(&mut self, id: ID) -> Option<&Point> {
        let manifest = self.get_current_manifest_list();
        if manifest.get_nb_points() <= id {
            None
        } else {
            if let Some((file_path, offset)) = manifest.get_file_and_offset_from_id(id) {
                let points = self.load_payload_file(&file_path, offset as usize);
                let mut idx = 0;
                for p in points {
                    self.point_cache.insert(idx + offset, p);
                    idx += 1;
                }
                self.point_cache.get(&id)
            } else {
                None
            }
        }
    }

    pub fn insert(&self, payloads: Vec<Point>) -> io::Result<Manifest> {
        self.write_lock();
        let mut manifest_list = self.get_current_manifest_list();
        let manifest = self.add_payloads(&mut manifest_list, payloads);
        // self.index_id_range(&manifest);
        self.write_unlock();
        Ok(manifest)
    }

    fn index_id_range(&self, _manifest: &Manifest) {
        // payloads were added to the data layer
        // their IDs are all IDs in the given range
        // insert these IDs in the index
        todo!("insert IDs in the index")
    }

    fn load_payload_file(&self, loc: &PathBuf, offset: usize) -> Vec<Point> {
        let mut bytes = vec![];
        File::open(loc)
            .expect("Could not open payload file")
            .read_to_end(&mut bytes)
            .expect("Could not read payload file");
        self.deserialize_payloads(&bytes, offset)
    }

    pub fn get_current_manifest_list(&self) -> ManifestList {
        ManifestList::new(&self.loc)
    }

    fn add_payloads(&self, manifest: &mut ManifestList, mut payloads: Vec<Point>) -> Manifest {
        let new_manifest = self.save_payloads(manifest, &mut payloads);
        manifest.update(&new_manifest);
        new_manifest
    }

    fn save_payloads(&self, manifest: &ManifestList, payloads: &Vec<Point>) -> Manifest {
        let id_offset = manifest.get_nb_points();
        let len = payloads.len() as AddedPoints;
        let path = self.loc.join("data-layer").join(get_rand_file_name());
        let _written_bytes = self.save_serialized_payloads(&path, payloads);
        Manifest::new(&path, id_offset, len)
    }

    fn save_serialized_payloads(&self, path: &PathBuf, payloads: &Vec<Point>) -> usize {
        let mut file = match File::create_new(path) {
            Ok(f) => f,
            Err(e) => panic!("Could not create payload file at {path:?}: {e}"),
        };
        let payloads_ser = self.serialize_payloads(&payloads);
        match file.write(&payloads_ser) {
            Ok(bytes_written) => bytes_written,
            Err(e) => panic!("Could not write payload to file {path:?}: {e}"),
        }
    }

    fn serialize_payloads(&self, payloads: &Vec<Point>) -> Vec<u8> {
        if self.debug {
            self.serialize_payloads_debug(payloads)
        } else {
            self.serialize_payloads_compressed(payloads)
        }
    }

    fn serialize_payloads_compressed(&self, payloads: &Vec<Point>) -> Vec<u8> {
        let p: Vec<&VecType> = payloads.iter().map(|p| &p.vector).collect();
        let ser = rmp_serde::encode::to_vec(&p).expect("Could not serialize payload");
        let mut buf = GzEncoder::new(vec![], Compression::default());
        buf.write_all(&ser).unwrap();
        buf.finish().unwrap()
    }

    fn deserialize_payloads(&self, bytes: &Vec<u8>, offset: usize) -> Vec<Point> {
        if self.debug {
            self.deserialize_payloads_debug(bytes, offset)
        } else {
            self.deserialize_payloads_compressed(bytes, offset)
        }
    }

    fn deserialize_payloads_compressed(&self, bytes: &Vec<u8>, offset: usize) -> Vec<Point> {
        let mut bytes_uncompressed = vec![];
        GzDecoder::new(&bytes[..])
            .read_to_end(&mut bytes_uncompressed)
            .expect("Could not decompress payload contents");
        let vecs: Vec<VecType> =
            rmp_serde::decode::from_slice(&bytes_uncompressed).expect("Could not decode payload");
        vecs.iter()
            .enumerate()
            .map(|(idx, v)| Point::with_vectype_and_id(v, (idx + offset) as u32))
            .collect()
    }

    fn serialize_payloads_debug(&self, payloads: &Vec<Point>) -> Vec<u8> {
        let p: Vec<&VecType> = payloads.iter().map(|p| &p.vector).collect();
        serde_json::to_vec_pretty(&p).expect("Could not encode payload")
    }

    fn deserialize_payloads_debug(&self, bytes: &Vec<u8>, offset: usize) -> Vec<Point> {
        let vecs: Vec<VecType> = serde_json::from_slice(bytes).expect("Could not decode payload");
        vecs.iter()
            .enumerate()
            .map(|(idx, v)| Point::with_vectype_and_id(v, (idx + offset) as u32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::remove_dir_all, path::Path, u64};

    use points::point::Point;
    use vectors::{VecBase, gen_rand_vecs};

    use crate::{ID, engine::Engine};

    const N: u64 = 8;
    const DIM: usize = 4;

    fn init_engine(path: &str) -> Engine {
        let path = Path::new(path).to_path_buf();
        if path.exists() {
            remove_dir_all(&path).unwrap();
        }
        Engine::new(path).unwrap()
    }

    fn gen_rand_points(n: usize) -> Vec<Point> {
        let vs = gen_rand_vecs(DIM, n);
        let mut points = vec![];
        for (idx, v) in vs.iter().enumerate() {
            points.push(Point::with_level_and_id(v, 0, idx as u32));
        }
        points
    }

    #[test]
    fn add_payloads() {
        let path = "./add-payloads-test";
        let e = init_engine(path);
        for _ in 0..4 {
            let p = gen_rand_points(N as usize);
            let r = e.insert(p);
            assert!(r.is_ok());
            assert_eq!(r.unwrap().get_nb_points(), N);
        }
        remove_dir_all(path).unwrap();
    }

    #[test]
    fn add_one_payload() {
        let path = "./add-one-payload-test";
        let e = init_engine(path);
        for _ in 0..4 {
            let p = gen_rand_points(1);
            let r = e.insert(p);
            assert!(r.is_ok());
        }
        remove_dir_all(path).unwrap();
    }

    #[test]
    fn get_points_simple() {
        let path = "./get-points-test";
        let mut e = init_engine(path);

        let points = gen_rand_points(8);
        let _ = e.insert(points[..4].to_vec()).unwrap();
        let _ = e.insert(points[4..].to_vec()).unwrap();

        assert_eq!(e.get_current_manifest_list().get_nb_points(), 8);

        for i in 0..8 {
            println!("i {i}");
            let a = e.get_point(i).unwrap();
            let b = points[i as usize].clone();
            assert_eq!(a.get_vals(), b.get_vals());
            assert_eq!(a.id, b.id);
            assert_eq!(a.level, b.level);
        }

        assert!(e.get_point(8).is_none());
        assert!(e.get_point(16).is_none());
        assert!(e.get_point(1024).is_none());
        assert!(e.get_point(ID::MAX).is_none());

        remove_dir_all(path).unwrap();
    }

    #[test]
    fn get_points_uses_cache() {
        let path = "./get-points-cache-test";
        let mut e = init_engine(path);

        let points = gen_rand_points(8);
        let _ = e.insert(points[..4].to_vec()).unwrap();
        let _ = e.insert(points[4..].to_vec()).unwrap();

        assert_eq!(e.get_current_manifest_list().get_nb_points(), 8);

        let mut check_cache_updates = |idx| {
            assert_eq!(e.point_cache.contains_key(&idx), false);
            assert_eq!(e.point_cache.contains_key(&(idx + 1)), false);
            assert_eq!(e.point_cache.contains_key(&(idx + 2)), false);
            assert_eq!(e.point_cache.contains_key(&(idx + 3)), false);
            e.get_point(idx);
            assert_eq!(e.point_cache.contains_key(&idx), true);
            assert_eq!(e.point_cache.contains_key(&(idx + 1)), true);
            assert_eq!(e.point_cache.contains_key(&(idx + 2)), true);
            assert_eq!(e.point_cache.contains_key(&(idx + 3)), true);
        };

        check_cache_updates(0);
        check_cache_updates(4);

        remove_dir_all(path).unwrap();
    }

    #[test]
    fn get_points_big() {
        let path = "./get-points-big-test";
        let mut e = init_engine(path);

        let mut n = 0;

        for _ in 0..8 {
            let _n = 2_usize.pow(12);
            let points = gen_rand_points(_n);
            let _ = e.insert(points.clone()).unwrap();
            n += _n;
        }

        assert_eq!(e.get_current_manifest_list().get_nb_points(), n as ID);

        for i in 0..n {
            let a = e.get_point(i as ID).unwrap();
            assert_eq!(a.id, i as u32);
        }

        remove_dir_all(path).unwrap();
    }
}
