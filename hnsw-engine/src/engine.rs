use std::{
    env,
    fs::{File, create_dir, create_dir_all, remove_file},
    io::{self, Read, Write},
    path::PathBuf,
    sync::Arc,
    thread::sleep,
    time::Duration,
};

use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use itertools::Itertools;
use moka::{policy::EvictionPolicy, sync::Cache};
use points::point::{Point, VecType};

use crate::{
    AddedPoints, ID, get_rand_file_name,
    manifest::{Manifest, ManifestList},
};

const DEFAULT_CACHE_CAPACITY: u64 = 16384;
const LAYERS: [&str; 3] = ["metadata-layer", "data-layer", "index-layer"];

pub struct Engine {
    loc: PathBuf,
    cache: Cache<ID, Arc<Point>>,
    debug: bool,
    cache_cap: u64,
}

impl Engine {
    pub fn new(loc: PathBuf) -> io::Result<Self> {
        let cache = Cache::builder()
            .max_capacity(DEFAULT_CACHE_CAPACITY)
            .eviction_policy(EvictionPolicy::lru())
            .build();
        let engine = Self {
            loc: loc.clone(),
            cache,
            debug: env::var("HNSW_DEBUG").is_ok(),
            cache_cap: DEFAULT_CACHE_CAPACITY,
        };
        Self::init_loc(&loc)?;
        Ok(engine)
    }

    fn init_loc(loc: &PathBuf) -> io::Result<()> {
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
        Ok(())
    }

    pub fn with_cache_capacity(loc: PathBuf, cap: u64) -> io::Result<Self> {
        let cache = Cache::builder()
            .max_capacity(cap)
            .eviction_policy(EvictionPolicy::lru())
            .build();
        let engine = Self {
            loc: loc.clone(),
            cache,
            debug: env::var("HNSW_DEBUG").is_ok(),
            cache_cap: cap,
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
        for layer in LAYERS {
            Self::try_create_dir(loc.join(layer));
        }
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

    pub fn get_point(&mut self, id: ID) -> Option<Arc<Point>> {
        // let s = Instant::now();
        let result = if let Some(p) = self.cache.get(&id) {
            Some(p)
        } else {
            self.load_point(id)
        };
        // let e = s.elapsed();
        // println!("Got {id} in {e:?}");
        result
    }

    /// Also caches points stored in the same data file
    fn load_point(&mut self, id: ID) -> Option<Arc<Point>> {
        let manifest_list = self.get_current_manifest_list();
        if manifest_list.get_nb_points() <= id {
            None
        } else {
            if let Some(manifest) = manifest_list.get_manifest_from_id(id) {
                let points = self.load_points_from_manifest(manifest);
                for p in points {
                    if !self.cache.contains_key(&(p.id as ID)) {
                        self.cache.insert(p.id as ID, Arc::new(p));
                    }
                }
                self.cache.get(&id)
            } else {
                None
            }
        }
    }

    pub fn insert(&self, payloads: &[Point]) -> io::Result<Manifest> {
        if payloads.len() > self.cache_cap as usize {
            self.divide_and_insert(payloads)
        } else {
            self.write_lock();
            let mut manifest_list = self.get_current_manifest_list();
            let manifest = self.add_payloads(&mut manifest_list, payloads);
            // self.index_id_range(&manifest);
            self.write_unlock();
            Ok(manifest)
        }
    }

    fn divide_and_insert(&self, payloads: &[Point]) -> io::Result<Manifest> {
        let mid = payloads.len() / 2;
        let p1 = &payloads[..mid];
        let p2 = &payloads[mid..];
        self.insert(p1)?;
        self.insert(p2)
    }

    fn index_id_range(&self, _manifest: &Manifest) {
        // payloads were added to the data layer
        // their IDs are all IDs in the given range
        // insert these IDs in the index
        todo!("insert IDs in the index")
    }

    fn load_points_from_manifest(&self, manifest: &Manifest) -> Vec<Point> {
        let mut bytes = vec![];
        File::open(&manifest.loc)
            .expect("Could not open payload file")
            .read_to_end(&mut bytes)
            .expect("Could not read payload file");
        self.deserialize_payloads(&bytes, manifest)
    }

    pub fn get_current_manifest_list(&self) -> ManifestList {
        let r = ManifestList::new(&self.loc);
        r
    }

    fn add_payloads(&self, manifest_list: &mut ManifestList, mut payloads: &[Point]) -> Manifest {
        let new_manifest = self.save_payloads(manifest_list, &mut payloads);
        manifest_list.update(&new_manifest);
        new_manifest
    }

    fn save_payloads(&self, manifest_list: &ManifestList, payloads: &[Point]) -> Manifest {
        let first = manifest_list.get_nb_points();
        let last = first + payloads.len() as ID - 1;
        let path = self.loc.join("data-layer").join(get_rand_file_name());
        let _written_bytes = self.save_serialized_payloads(&path, payloads);
        Manifest::new(&path, first, last)
    }

    fn save_serialized_payloads(&self, path: &PathBuf, payloads: &[Point]) -> usize {
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

    fn serialize_payloads(&self, payloads: &[Point]) -> Vec<u8> {
        if self.debug {
            self.serialize_payloads_debug(payloads)
        } else {
            self.serialize_payloads_compressed(payloads)
        }
    }

    fn serialize_payloads_compressed(&self, payloads: &[Point]) -> Vec<u8> {
        let p: Vec<&VecType> = payloads.iter().map(|p| &p.vector).collect();
        let ser = rmp_serde::encode::to_vec(&p).expect("Could not serialize payload");
        let mut buf = GzEncoder::new(vec![], Compression::default());
        buf.write_all(&ser).unwrap();
        buf.finish().unwrap()
    }

    fn deserialize_payloads(&self, bytes: &Vec<u8>, manifest: &Manifest) -> Vec<Point> {
        if self.debug {
            self.deserialize_payloads_debug(bytes, manifest)
        } else {
            self.deserialize_payloads_compressed(bytes, manifest)
        }
    }

    fn deserialize_payloads_compressed(&self, bytes: &Vec<u8>, manifest: &Manifest) -> Vec<Point> {
        let mut bytes_uncompressed = vec![];
        GzDecoder::new(&bytes[..])
            .read_to_end(&mut bytes_uncompressed)
            .expect("Could not decompress payload contents");
        let vecs: Vec<VecType> =
            rmp_serde::decode::from_slice(&bytes_uncompressed).expect("Could not decode payload");
        let ids = manifest.ids();
        let mut points = vec![];
        for (v, id) in vecs.iter().zip_eq(ids) {
            points.push(Point::with_vectype_and_id(v, id as u32));
        }
        points
    }

    fn serialize_payloads_debug(&self, payloads: &[Point]) -> Vec<u8> {
        let p: Vec<&VecType> = payloads.iter().map(|p| &p.vector).collect();
        serde_json::to_vec_pretty(&p).expect("Could not encode payload")
    }

    fn deserialize_payloads_debug(&self, bytes: &Vec<u8>, manifest: &Manifest) -> Vec<Point> {
        let vecs: Vec<VecType> = serde_json::from_slice(bytes).expect("Could not decode payload");
        let ids = manifest.ids();
        let mut points = vec![];
        for (v, id) in vecs.iter().zip_eq(ids) {
            points.push(Point::with_vectype_and_id(v, id as u32));
        }
        points
    }

    pub fn optimize(&self) -> io::Result<ManifestList> {
        let old_manifest_list = self.get_current_manifest_list();
        let op_plan = old_manifest_list.get_optimization_plan(self.cache_cap);

        let mut optimized_manifest_list = Vec::new();
        for to_combine in op_plan.iter() {
            optimized_manifest_list.push(self.combine_manifests(to_combine));
        }
        let new_manifest_list = ManifestList::from_manifests(&self.loc, optimized_manifest_list);
        remove_file(&old_manifest_list.loc);
        self.remove_old_manifest_lists(&old_manifest_list);
        self.purge_old_manifest_list_data(&old_manifest_list);
        Ok(new_manifest_list)
    }

    fn remove_old_manifest_lists(&self, manifest_list: &ManifestList) -> io::Result<()> {
        for old_manifest_list in manifest_list.get_prev_manifest_list_paths().iter() {
            remove_file(&old_manifest_list)?;
        }
        Ok(())
    }

    fn purge_old_manifest_list_data(&self, manifest_list: &ManifestList) -> io::Result<()> {
        for manifest in manifest_list.manifests.iter() {
            remove_file(&manifest.loc)?;
        }
        Ok(())
    }

    fn combine_manifests(&self, to_combine: &Vec<Manifest>) -> Manifest {
        let mut combined_points = Vec::new();
        let mut id_ranges = Vec::new();
        for manifest in to_combine.iter() {
            combined_points.extend(self.load_points_from_manifest(manifest));
            id_ranges.extend(manifest.id_ranges.clone());
        }
        let path = self.loc.join("data-layer").join(get_rand_file_name());
        let _written_bytes = self.save_serialized_payloads(&path, &combined_points);
        Manifest::with_ranges(&path, id_ranges)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{read_dir, remove_dir_all},
        path::Path,
        u64,
    };

    use moka::sync::Cache;
    use points::point::Point;
    use rand::Rng;
    use vectors::{VecBase, gen_rand_vecs};

    use crate::{
        ID,
        engine::{DEFAULT_CACHE_CAPACITY, Engine},
    };

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
            let r = e.insert(&p);
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
            let r = e.insert(&p);
            assert!(r.is_ok());
        }
        remove_dir_all(path).unwrap();
    }

    #[test]
    fn get_points_simple() {
        let path = "./get-points-test";
        let mut e = init_engine(path);

        let points = gen_rand_points(8);
        let _ = e.insert(&points[..4]).unwrap();
        let _ = e.insert(&points[4..]).unwrap();

        assert_eq!(e.get_current_manifest_list().get_nb_points(), 8);

        for i in 0..8 {
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
        let _ = e.insert(&points[..4]).unwrap();
        let _ = e.insert(&points[4..]).unwrap();

        assert_eq!(e.get_current_manifest_list().get_nb_points(), 8);

        let mut check_cache_updates = |idx| {
            assert_eq!(e.cache.contains_key(&idx), false);
            assert_eq!(e.cache.contains_key(&(idx + 1)), false);
            assert_eq!(e.cache.contains_key(&(idx + 2)), false);
            assert_eq!(e.cache.contains_key(&(idx + 3)), false);
            e.get_point(idx);
            assert_eq!(e.cache.contains_key(&idx), true);
            assert_eq!(e.cache.contains_key(&(idx + 1)), true);
            assert_eq!(e.cache.contains_key(&(idx + 2)), true);
            assert_eq!(e.cache.contains_key(&(idx + 3)), true);
        };

        check_cache_updates(0);
        check_cache_updates(4);

        remove_dir_all(path).unwrap();
    }

    #[test]
    fn get_points_big() {
        let path = "./get-points-big-test";
        let mut e = init_engine(path);

        let nb_inserts = 8;
        let n = 2_usize.pow(12);
        let points = gen_rand_points(n * nb_inserts);
        for i in 0..nb_inserts {
            let _ = e.insert(&points[(i * n)..((i * n) + n)]).unwrap();
        }

        assert_eq!(
            e.get_current_manifest_list().get_nb_points(),
            points.len() as ID
        );

        for (idx, p) in points.iter().enumerate() {
            let a = e.get_point(idx as ID).unwrap();
            assert_eq!(idx, p.id as usize);
            assert_eq!(a.id, p.id);
            assert_eq!(a.get_vals(), p.get_vals());
        }

        remove_dir_all(path).unwrap();
    }

    #[test]
    fn optimize() {
        let path = "./optimize-test";
        let mut e = init_engine(path);

        let nb_inserts = 4;
        let n = DEFAULT_CACHE_CAPACITY as usize / 2;
        let points = gen_rand_points(n * nb_inserts);
        for i in 0..nb_inserts {
            let _ = e.insert(&points[(i * n)..((i * n) + n)]).unwrap();
        }

        assert_eq!(
            e.get_current_manifest_list().get_nb_points(),
            (n * nb_inserts) as ID
        );

        for (idx, p) in points.iter().enumerate() {
            let a = e.get_point(idx as ID).unwrap();
            assert_eq!(idx, p.id as usize);
            assert_eq!(a.id, p.id);
            assert_eq!(a.get_vals(), p.get_vals());
        }

        assert_eq!(e.get_current_manifest_list().manifests.len(), nb_inserts);
        assert_ne!(e.get_current_manifest_list().manifests.len(), 2);
        e.optimize();
        assert_ne!(e.get_current_manifest_list().manifests.len(), nb_inserts);
        assert_eq!(e.get_current_manifest_list().manifests.len(), 2);

        for (idx, p) in points.iter().enumerate() {
            let a = e.get_point(idx as ID).unwrap();
            assert_eq!(idx, p.id as usize);
            assert_eq!(a.id, p.id);
            assert_eq!(a.get_vals(), p.get_vals());
        }

        remove_dir_all(path).unwrap();
    }

    #[test]
    fn optimize_rand_additions() {
        let path = "./optimize-rand-test";
        let mut e = init_engine(path);

        let mut rng = rand::thread_rng();
        let n = DEFAULT_CACHE_CAPACITY as usize * 8;
        let points = gen_rand_points(n);
        let mut cursor = 0;
        loop {
            let jump = rng.gen_range(128..DEFAULT_CACHE_CAPACITY as usize / 2);
            let to_insert = points.get(cursor..cursor + jump);
            match to_insert {
                Some(p) => {
                    let _ = e.insert(p).unwrap();
                }
                None => {
                    let p = &points[cursor..];
                    let _ = e.insert(p).unwrap();
                    break;
                }
            }
            cursor += jump;
            cursor = cursor.min(points.len() - 1);
        }

        for (idx, p) in points.iter().enumerate() {
            let a = e.get_point(idx as ID).unwrap();
            assert_eq!(idx, p.id as usize);
            assert_eq!(a.id, p.id);
            assert_eq!(a.get_vals(), p.get_vals());
        }

        let n = e.get_current_manifest_list().manifests.len();
        assert!(n > 0);
        e.optimize();
        assert!(e.get_current_manifest_list().manifests.len() < n);

        for (idx, p) in points.iter().enumerate() {
            let a = e.get_point(idx as ID).unwrap();
            assert_eq!(idx, p.id as usize);
            assert_eq!(a.id, p.id);
            assert_eq!(a.get_vals(), p.get_vals());
        }

        remove_dir_all(path).unwrap();
    }
}
