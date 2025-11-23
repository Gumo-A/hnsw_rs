use points::point::Point;
use vectors::{VecTrait, gen_rand_vecs};

#[test]
fn distance_computation_quant() {
    for _ in 0..100 {
        let rand_vecs = gen_rand_vecs(128, 2);
        let a = Point::new_quant(0, 0, &rand_vecs[0].clone());
        let b = Point::new_quant(0, 0, &rand_vecs[1].clone());
        assert!(a.distance(&b) >= 0.0);
    }
}

#[test]
fn distance_precision_quant() {
    let a = Point::new_quant(0, 0, &vec![0.75, 0.75]);
    let b = Point::new_quant(0, 0, &vec![0.25, 0.25]);
    let dist = a.distance(&b);
    print!("{dist}");
    assert!(dist == (1.0 / (2.0f32).sqrt()));
}

#[test]
fn distance_computation_full() {
    for _ in 0..100 {
        let rand_vecs = gen_rand_vecs(128, 2);
        let a = Point::new_full(0, 0, rand_vecs[0].clone());
        let b = Point::new_full(0, 0, rand_vecs[1].clone());
        assert!(a.distance(&b) >= 0.0);
    }
}

#[test]
fn distance_precision_full() {
    let a = Point::new_full(0, 0, vec![0.75, 0.75]);
    let b = Point::new_full(0, 0, vec![0.25, 0.25]);
    let dist = a.distance(&b);
    print!("{dist}");
    assert!(dist == (1.0 / (2.0f32).sqrt()));
}
