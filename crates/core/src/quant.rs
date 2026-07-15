//! Host-side F32 tensor unpack (norms / biases).

/// Unpack little-endian f32 bytes into `y`.
pub fn dequant_f32(data: &[u8], y: &mut [f32]) {
    debug_assert_eq!(data.len(), y.len() * 4);
    for (i, chunk) in data.chunks_exact(4).enumerate() {
        y[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
}
