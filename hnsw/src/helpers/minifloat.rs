// TODO:
// Write this type
// Get inspiration from https://github.com/jdh8/minifloat-rs

// bits for the exponent
const E: u8 = 4;
// bits for the significand
const S: u8 = 3;

pub struct F8(u8);

impl F8 {
    pub fn from_f32(x: f32) -> Self {
        let mut minifloat: u8 = if x.is_sign_positive() { 1u8 << 7 } else { 0u8 };

        // TODO: Exponent translation
        // TODO: Significand translation

        F8(minifloat)
    }
}
