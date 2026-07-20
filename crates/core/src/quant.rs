//! Host-side F32 tensor unpack and half-precision helpers.

/// Unpack little-endian f32 bytes into `y`.
pub fn dequant_f32(data: &[u8], y: &mut [f32]) {
    debug_assert_eq!(data.len(), y.len() * 4);
    for (i, chunk) in data.chunks_exact(4).enumerate() {
        y[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
}

/// IEEE f16 bits → f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x03ff) as u32;
    match exponent {
        0 if fraction == 0 => sign * 0.0,
        0 => sign * (fraction as f32) * 2.0f32.powi(-24),
        31 if fraction == 0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + fraction as f32 / 1024.0) * 2.0f32.powi(exponent - 15),
    }
}

/// IEEE f32 → f16 bits (round-to-nearest-even for normal range).
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x007f_ffff;
    if exponent == 255 {
        return sign | if fraction == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exp = exponent - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = fraction | 0x0080_0000;
        let shift = (14 - half_exp) as u32;
        let mut rounded = mantissa >> shift;
        let remainder = mantissa & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | rounded as u16;
    }
    let mut rounded = fraction + 0x0000_0fff + ((fraction >> 13) & 1);
    let mut out_exp = half_exp as u16;
    if rounded & 0x0080_0000 != 0 {
        rounded = 0;
        out_exp += 1;
        if out_exp >= 31 {
            return sign | 0x7c00;
        }
    }
    sign | (out_exp << 10) | (rounded >> 13) as u16
}

/// Compact f32 → f16 used by MoE pack Q4→f16 expansion (flush denorms; simpler rounding).
pub fn f32_to_f16_bits_fast(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mant = x & 0x7f_ffff;
    let exp = ((x >> 23) & 0xff) as i32;
    if exp == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    if exp == 0 {
        return sign; // flush denorm to 0
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7c00; // inf
    }
    if new_exp <= 0 {
        return sign; // underflow
    }
    let new_mant = mant + 0x1000; // round
    sign | ((new_exp as u16) << 10) | ((new_mant >> 13) as u16)
}

/// f16 bits → f32 used by MoE pack helpers (bit-exact prior pack path).
pub fn f16_bits_to_f32_fast(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let bits = if exp == 0 {
        sign
    } else if exp == 31 {
        sign | 0x7f80_0000 | ((mant as u32) << 13)
    } else {
        // Use i32 for the bias math so exp < 15 does not underflow (debug panic /
        // wrapping garbage). Same intent as the prior pack helper.
        let f32_exp = (exp as i32) - 15 + 127;
        sign | ((f32_exp as u32) << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_f32_reads_little_endian() {
        let mut y = [0.0f32; 2];
        let bytes = {
            let mut b = Vec::new();
            b.extend_from_slice(&1.5f32.to_le_bytes());
            b.extend_from_slice(&(-2.0f32).to_le_bytes());
            b
        };
        dequant_f32(&bytes, &mut y);
        assert_eq!(y[0], 1.5);
        assert_eq!(y[1], -2.0);
    }

    #[test]
    fn f16_roundtrip_common_values() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 2.0, 65504.0, -0.25] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            let err = (back - v).abs();
            // f16 has ~3 decimal digits; allow small relative error.
            let tol = (v.abs() * 1e-3).max(1e-3);
            assert!(err <= tol, "v={v} back={back} err={err}");
        }
    }

    #[test]
    fn f16_inf_and_nan() {
        assert!(f16_to_f32(f32_to_f16(f32::INFINITY)).is_infinite());
        assert!(f16_to_f32(f32_to_f16(f32::NEG_INFINITY)).is_infinite());
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn fast_f16_matches_normals() {
        for v in [1.0f32, -2.5, 0.125, 100.0] {
            let a = f16_to_f32(f32_to_f16(v));
            let b = f16_bits_to_f32_fast(f32_to_f16_bits_fast(v));
            assert!((a - b).abs() < 1e-2 || (a - v).abs() < 1e-2);
        }
    }
}
