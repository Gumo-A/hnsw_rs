(trap 'kill 0' SIGINT; \
 /home/gamal/rust/hnsw_rs/target/release/brute_force 100 400000 6 0 & \
 /home/gamal/rust/hnsw_rs/target/release/brute_force 100 400000 6 1 & \
 /home/gamal/rust/hnsw_rs/target/release/brute_force 100 400000 6 2 & \
 /home/gamal/rust/hnsw_rs/target/release/brute_force 100 400000 6 3 & \
 /home/gamal/rust/hnsw_rs/target/release/brute_force 100 400000 6 4 & \
 /home/gamal/rust/hnsw_rs/target/release/brute_force 100 400000 6 5 & \
 wait)
