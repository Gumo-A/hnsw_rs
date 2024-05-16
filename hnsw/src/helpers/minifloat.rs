// TODO:
// Write this type
// Get inspiration from https://github.com/jdh8/minifloat-rs

const EXPONENT_SIZE: u8 = 4;
const SIGNIFICAND_SIZE: u8 = 3;

pub struct F8(u8);

fn get_bit_at(input: u32, n: u8) -> bool {
    if n < 32 {
        input & (1 << n) != 0
    } else {
        false
    }
}
