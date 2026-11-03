use std::{
    env,
    fs::{File, create_dir, create_dir_all, remove_file},
    io::{self, ErrorKind, Read, Write},
    path::PathBuf,
    sync::Arc,
    thread::sleep,
    time::{Duration, Instant},
};

use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use graph::graph::Graph;
use itertools::Itertools;
use moka::{policy::EvictionPolicy, sync::Cache};
use points::point::{Point, VecType};
use rand::{Rng, rngs::ThreadRng, thread_rng};
use vectors::VecBase;

use crate::{
    get_rand_file_name,
    hnsw::{HNSW, traverser::Traverser},
    index::{Index, IndexEntry},
    manifest::{Manifest, ManifestList},
};

macro_rules! timeit {
    ($f:expr) => {
        let s = Instant::now();
        $f;
        let e = s.elapsed();
        println!("Executed {} in {e:?}", stringify!($f));
    };
    (let $x:ident = $f:expr) => {
        let s = Instant::now();
        let $x = $f;
        let e = s.elapsed();
        println!("Executed {} in {e:?}", stringify!($f));
    };
}

pub type INT = usize;
pub const METADATA_LAYER: &str = "metadata-layer";
pub const DATA_LAYER: &str = "data-layer";
pub const INDEX_LAYER: &str = "index-layer";
const LAYERS: [&str; 3] = [METADATA_LAYER, DATA_LAYER, INDEX_LAYER];

pub struct Engine {
    loc: PathBuf,
    cache: Cache<INT, Arc<Point>>,
    params: HNSW,
}

impl Engine {
    pub fn new(loc: PathBuf, m: INT) -> io::Result<Self> {
        let params = Self::init_at_loc(&loc, m)?;
        let cache = Cache::builder()
            .max_capacity(params.cache_cap as u64)
            .eviction_policy(EvictionPolicy::lru())
            .build();
        let engine = Self {
            loc: loc.clone(),
            cache,
            params,
        };
        Ok(engine)
    }

    fn init_at_loc(loc: &PathBuf, dim: INT) -> io::Result<HNSW> {
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
                    _ => return Err(e),
                },
            };
        }
        Self::check_or_create_dir_structure(&loc, dim)
    }

    fn check_can_use_dir(loc: &PathBuf) -> io::Result<()> {
        let check = |p| loc.join(p).exists();
        let empty = loc.read_dir()?.count() == 0;
        if empty {
            Ok(())
        } else if check(METADATA_LAYER) & check(DATA_LAYER) & check(INDEX_LAYER) {
            Ok(())
        } else {
            Err(io::ErrorKind::DirectoryNotEmpty.into())
        }
    }

    fn try_create_dir(path: PathBuf) -> io::Result<()> {
        match create_dir(&path) {
            Ok(()) => Ok(()),
            Err(e) => match e.kind() {
                io::ErrorKind::AlreadyExists => Ok(()),
                _ => Err(e),
            },
        }
    }

    fn check_or_create_dir_structure(loc: &PathBuf, m: INT) -> io::Result<HNSW> {
        for layer in LAYERS {
            Self::try_create_dir(loc.join(layer))?;
        }
        Self::load_or_create_param_file(loc.join(METADATA_LAYER), m)
    }

    fn load_or_create_param_file(metadata_dir: PathBuf, m: INT) -> io::Result<HNSW> {
        let param_file = metadata_dir.join("params.json");
        match File::create_new(&param_file) {
            Ok(mut f) => {
                let params = HNSW::new(m);
                let params_ser = if env::var("HNSW_DEBUG").is_ok() {
                    serde_json::to_vec_pretty(&params)?
                } else {
                    serde_json::to_vec(&params)?
                };
                f.write_all(&params_ser);
                Ok(params)
            }
            Err(e) => match e.kind() {
                io::ErrorKind::AlreadyExists => Self::read_params_file(&param_file),
                _ => Err(e),
            },
        }
    }

    fn read_params_file(path: &PathBuf) -> io::Result<HNSW> {
        match File::open(path) {
            Ok(mut f) => {
                let mut buf = Vec::new();
                f.read_to_end(&mut buf)?;
                let params = serde_json::from_slice(&buf)?;
                Ok(params)
            }
            Err(e) => Err(e),
        }
    }

    fn wait(&self, micros: u64) {
        sleep(Duration::from_micros(micros));
    }

    fn write_lock(&self) {
        let lock_file = self.loc.join(METADATA_LAYER).join("write-lock-file");
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
        let lock_file = self.loc.join(METADATA_LAYER).join("write-lock-file");
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

    /// Also caches points stored in the same data file
    fn load_point(&mut self, id: INT) -> io::Result<Option<Arc<Point>>> {
        let manifest_list = self.get_current_manifest_list()?;
        if manifest_list.get_nb_points() <= id {
            Ok(None)
        } else {
            if let Some(manifest) = manifest_list.get_manifest_from_id(id) {
                let points = manifest.read_data();
                for p in points {
                    if !self.cache.contains_key(&(p.id as INT)) {
                        self.cache.insert(p.id as INT, Arc::new(p));
                    }
                }
                Ok(self.cache.get(&id))
            } else {
                Ok(None)
            }
        }
    }

    pub fn insert(&mut self, points: &[Point]) -> io::Result<Manifest> {
        self.check_points(points)?;
        if points.len() > self.params.cache_cap as INT {
            self.divide_and_insert(points)
        } else {
            self.write_lock();
            let manifest = self
                .get_current_manifest_list()?
                .add_data(points, self.params.ml)?;
            // self.add_points_to_index(&manifest);
            self.write_unlock();
            Ok(manifest)
        }
    }

    pub fn optimize(&mut self) -> io::Result<()> {
        self.get_current_manifest_list()?
            .optimize(self.params.cache_cap)
    }

    fn check_points(&mut self, points: &[Point]) -> io::Result<()> {
        if points.len() == 0 {
            return Err(ErrorKind::InvalidData.into());
        } else if self.params.dim.is_none() {
            self.update_dim(points)?;
        }

        let dim = self.params.dim.unwrap();
        let nb_valid = points.iter().filter(|p| p.dim() == dim).count();
        if nb_valid != points.len() {
            Err(ErrorKind::InvalidData.into())
        } else {
            Ok(())
        }
    }

    fn update_dim(&mut self, points: &[Point]) -> io::Result<()> {
        self.params.set_dim(points[0].dim());
        let param_file = self.loc.join(METADATA_LAYER).join("params.json");
        let params_ser = if env::var("HNSW_DEBUG").is_ok() {
            serde_json::to_vec_pretty(&self.params)?
        } else {
            serde_json::to_vec(&self.params)?
        };
        let mut f = File::create(&param_file)?;
        f.write_all(&params_ser);
        Ok(())
    }

    fn divide_and_insert(&mut self, points: &[Point]) -> io::Result<Manifest> {
        let mid = points.len() / 2;
        let p1 = &points[..mid];
        let p2 = &points[mid..];
        self.insert(p1)?;
        self.insert(p2)
    }

    fn add_points_to_index(&mut self, manifest: &Manifest) {
        let mut inserter = Traverser::new();
        let mut rng = thread_rng();
        for id in manifest.ids() {
            inserter.insert(self, id);
        }
    }

    pub fn get_current_manifest_list(&self) -> io::Result<ManifestList> {
        ManifestList::new(&self.loc)
    }

    pub fn get_number_of_manifests(&self) -> io::Result<INT> {
        Ok(self.get_current_manifest_list()?.manifests.len())
    }

    pub fn get_nb_points(&self) -> io::Result<INT> {
        Ok(self.get_current_manifest_list()?.get_nb_points())
    }
}

impl Index for Engine {
    fn get_point(&mut self, id: INT) -> io::Result<Option<Arc<Point>>> {
        let result = if let Some(p) = self.cache.get(&id) {
            Some(p)
        } else {
            self.load_point(id)?
        };
        Ok(result)
    }

    fn distance(&mut self, a: INT, b: INT) -> io::Result<Option<f32>> {
        match (self.get_point(a)?, self.get_point(b)?) {
            (Some(a), Some(b)) => Ok(Some(a.dist2other(&b))),
            _ => Ok(None),
        }
    }

    fn distance2point(&mut self, id: INT, point: &Point) -> io::Result<Option<f32>> {
        match self.get_point(id)? {
            Some(p) => Ok(Some(p.dist2other(point))),
            None => Ok(None),
        }
    }

    fn get_ep(&self) -> INT {
        self.params.ep
    }

    fn get_m(&self) -> INT {
        self.params.m
    }

    fn get_m_at_layer(&self, layer: INT) -> INT {
        if layer == 0 {
            self.params.mmax0
        } else {
            self.params.mmax
        }
    }

    fn get_ef_cons(&self) -> INT {
        self.params.ef_cons
    }

    fn degree(&self, id: INT, layer: INT) -> io::Result<INT> {
        todo!("fetch nb of neighbors of id at layer")
    }

    fn add_node(&self, id: INT, layer_idx: INT) -> io::Result<()> {
        todo!(
            "makes sure the IndexEntry for id contains at least layer_idx + 1 entries in 'levels'"
        )
    }

    fn add_edge(&self, a: INT, b: INT, layer: INT) -> io::Result<()> {
        // this would trigger disk io on every edge addition, there should be some
        // kind of build up logic of many edge addtions before committing to a write
        todo!(
            "get the file(s) containing the entries for a and b, add the neighbors, write new file, create and write new manifest"
        )
    }

    fn remove_edge(&self, a: INT, b: INT, layer: INT) -> io::Result<()> {
        todo!("similar as add_edge")
    }

    fn neighbors(&self, id: INT, layer: INT) -> io::Result<std::collections::HashSet<INT>> {
        todo!("get neighbors either from non-stale cache or from disk")
    }

    fn get_nb_layers(&self) -> io::Result<INT> {
        // let entry = self.get_index_entry(self.get_ep());
        todo!("count how many layers there are in EP's IndexEntry")
    }
}

// impl Engine {
//     fn get_index_entry(&self, id: INT) -> IndexEntry {
//         self.get_current_manifest_list()
//             .unwrap()
//             .get_manifest_from_id(id)
//             .unwrap()
//             .load_index_entry(id)
//     }
// }

pub fn get_level(ml: f32, rng: &mut ThreadRng) -> INT {
    let mut rand_nb = 0.0;
    loop {
        if (rand_nb == 0.0) | (rand_nb == 1.0) {
            rand_nb = rng.r#gen::<f32>();
        } else {
            break;
        }
    }

    let level = (-rand_nb.ln() * ml).floor() as usize;
    level
}

#[cfg(test)]
mod tests {
    use std::{
        fs::remove_dir_all,
        path::{Path, PathBuf},
        time::Instant,
        u64,
    };

    use env_logger::init;
    use log::trace;
    use points::point::Point;
    use rand::{
        Rng,
        distributions::{Distribution, Uniform, uniform::UniformSampler},
    };
    use vectors::{VecBase, gen_rand_vecs};

    use crate::{
        engine::{Engine, INT},
        index::Index,
    };

    const DIM: INT = 128;
    const IOMSG: &str = "IO error ocurred";

    fn remove_path_if_exists(path: &str) -> PathBuf {
        let path = Path::new(path).to_path_buf();
        if path.exists() {
            remove_dir_all(&path).unwrap();
        }
        path
    }

    fn init_engine(path: &str) -> Engine {
        // env_logger::init();
        let path = remove_path_if_exists(path);
        Engine::new(path, DIM).unwrap()
    }

    fn gen_rand_points(n: INT) -> Vec<Point> {
        let vs = gen_rand_vecs(DIM, n);
        let mut points = vec![];
        for (idx, v) in vs.iter().enumerate() {
            points.push(Point::with_level_and_id(v, 0, idx));
        }
        points
    }

    fn gen_rand_point(dim: INT) -> Vec<Point> {
        let vs = gen_rand_vecs(dim, 1);
        let mut points = vec![];
        for (idx, v) in vs.iter().enumerate() {
            points.push(Point::with_level_and_id(v, 0, idx));
        }
        points
    }

    #[test]
    fn different_dimension_not_allowed() {
        let path = "./diff-dim-test";
        let mut e = init_engine(path);
        assert!(e.params.dim.is_none());

        let r = e.insert(&vec![gen_rand_point(4)[0].clone()]);
        assert!(r.is_ok());
        assert_eq!(e.params.dim, Some(4));

        let r = e.insert(&vec![gen_rand_point(8)[0].clone()]);
        assert!(r.is_err());
        remove_dir_all(path).unwrap()
    }

    #[test]
    fn add_payloads() {
        let path = "./add-payloads-test";
        let mut e = init_engine(path);
        let n = 8;
        for _ in 0..4 {
            let p = gen_rand_points(n);
            let r = e.insert(&p);
            assert!(r.is_ok());
            assert_eq!(r.unwrap().get_nb_points(), n as INT);
        }
        remove_dir_all(path).unwrap();
    }

    #[test]
    fn add_one_payload() {
        let path = "./add-one-payload-test";
        let mut e = init_engine(path);
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

        assert_eq!(e.get_current_manifest_list().unwrap().get_nb_points(), 8);

        for i in 0..8 {
            let a = e
                .get_point(i)
                .expect(IOMSG)
                .expect("The point was not found");
            let b = points[i as INT].clone();
            assert_eq!(a.get_vals(), b.get_vals());
            assert_eq!(a.id, b.id);
            assert_eq!(a.level, b.level);
        }

        assert!(e.get_point(8).expect(IOMSG).is_none());
        assert!(e.get_point(16).expect(IOMSG).is_none());
        assert!(e.get_point(1024).expect(IOMSG).is_none());
        assert!(e.get_point(INT::MAX).expect(IOMSG).is_none());

        remove_dir_all(path).unwrap();
    }

    #[test]
    fn get_points_uses_cache() {
        let path = "./get-points-cache-test";
        let mut e = init_engine(path);

        let points = gen_rand_points(8);
        let _ = e.insert(&points[..4]).unwrap();
        let _ = e.insert(&points[4..]).unwrap();

        assert_eq!(e.get_current_manifest_list().unwrap().get_nb_points(), 8);

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
    /// Inserts a number of points in the store
    /// via a random number of insertions until all
    /// points are inserted. This creates many manifests
    /// and data files. After optimize() is called,
    /// it should reduce the number of manifests
    /// and the number of data files to 1
    fn optimize_rand_additions() {
        let path = "./optimize-rand-test";
        let mut e = init_engine(path);

        let mut rng = rand::thread_rng();
        let n = 2_usize.pow(12);
        timeit!(let points = gen_rand_points(n));
        timeit!(let sample = Uniform::new(0, points.len())
            .sample_iter(rng.clone())
            .take(1000)
            .collect::<Vec<INT>>());
        let mut cursor = 0;
        let mut len = points.len();
        let mut insertions = 0;
        let mut insert_loop = |e: &mut Engine| {
            loop {
                insertions += 1;
                let jump = (len / 2) - rng.gen_range(0..(len / 2).max(1)) + 1;
                e.insert(&points[cursor..cursor + jump]).unwrap();
                len -= jump;
                cursor += jump;
                if cursor == points.len() {
                    break;
                }
            }
        };
        let mut check_sample = |e: &mut Engine| {
            for id in sample.iter().copied() {
                let p = points.get(id).unwrap();
                assert_eq!(id, p.id as INT);

                let a = e
                    .get_point(id as INT)
                    .expect(IOMSG)
                    .expect("The point was not found");

                assert_eq!(a.id, p.id);
                assert_eq!(a.get_vals(), p.get_vals());
            }
        };

        timeit!(insert_loop(&mut e));

        assert_eq!(e.get_number_of_manifests().expect(IOMSG), insertions);
        timeit!(e.optimize().unwrap());
        assert_eq!(e.get_number_of_manifests().expect(IOMSG), 1);

        timeit!(check_sample(&mut e));

        // remove_dir_all(path).unwrap();
    }
}
