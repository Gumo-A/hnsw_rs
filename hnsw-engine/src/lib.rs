use std::{
    fs::{self, File, create_dir, create_dir_all},
    io::{self, Write},
    path::PathBuf,
    str::FromStr,
};

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

    pub fn insert(&self, payloads: Vec<Point>) -> io::Result<Vec<ID>> {
        let p_ids: Vec<ID> = self.add_payloads(payloads);
        // self.add_one_id_to_index(p_id);
        Ok(p_ids)
    }

    fn add_payloads(&self, mut payloads: Vec<Point>) -> Vec<ID> {
        let last_pl_id = self.get_last_payload_id();
        println!("last id is {last_pl_id}");
        payloads
            .iter_mut()
            .enumerate()
            .for_each(|(idx, p)| p.id = 1 + idx as u32 + last_pl_id as u32);
        let ids = (last_pl_id + 1..last_pl_id + payloads.len() as ID + 1).collect();
        self.save_payloads(ids, payloads)
    }

    fn save_payloads(&self, ids: Vec<ID>, payloads: Vec<Point>) -> Vec<ID> {
        assert_eq!(ids.len(), payloads.len());
        let first_id = ids.get(0).unwrap();
        let last_id = ids.last().unwrap();
        let f_name = format!("{first_id}-{last_id}-payload.json");
        let path = self.loc.join("data-layer").join(f_name);
        println!("Creating {path:?}");
        File::create_new(path)
            .unwrap()
            .write(&serde_json::to_vec(&payloads).unwrap())
            .unwrap();
        self.overwrite_last_id_file(*last_id);
        ids
    }

    fn overwrite_last_id_file(&self, id: ID) {
        let path = self.loc.join("metadata-layer").join("last_id.txt");
        println!("Overwritting {path:?} with ID {id}");
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
                _ => panic!("Something else"),
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

fn check_or_create_dir_structure(loc: PathBuf) -> io::Result<()> {
    let metadata_layer = loc.join("metadata-layer");
    if !metadata_layer.exists() {
        create_dir(metadata_layer)?;
    }

    let data_layer = loc.join("data-layer");
    if !data_layer.exists() {
        create_dir(data_layer)?;
    }

    let index_layer = loc.join("index-layer");
    if !index_layer.exists() {
        create_dir(index_layer)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use points::point::Point;
    use vectors::gen_rand_vecs;

    use crate::Engine;

    fn gen_rand_points(n: usize) -> Vec<Point> {
        let vs = gen_rand_vecs(12, n);
        let mut points = vec![];
        for v in vs {
            points.push(Point::with_level_and_id(&v, 0, 0));
        }
        points
    }

    #[test]
    fn add_payloads() {
        let e = Engine::new(Path::new("./engine-test-index").to_path_buf()).unwrap();

        for _ in 0..8 {
            let p = gen_rand_points(4);
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
