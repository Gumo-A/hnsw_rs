(trap 'kill 0' SIGINT; \
 ./target/release/brute_force $1 $2 6 0 & \
 ./target/release/brute_force $1 $2 6 1 & \
 ./target/release/brute_force $1 $2 6 2 & \
 ./target/release/brute_force $1 $2 6 3 & \
 ./target/release/brute_force $1 $2 6 4 & \
 ./target/release/brute_force $1 $2 6 5 & \
 wait)
