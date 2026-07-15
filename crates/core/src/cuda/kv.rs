//! Per-layer key/value cache.

use cudarc::driver::CudaSlice;

pub struct CudaKv {
    pub(crate) k: Vec<CudaSlice<f32>>,
    pub(crate) v: Vec<CudaSlice<f32>>,
    pub len: usize,
    pub max_seq: usize,
}

impl CudaKv {
    pub fn clear(&mut self) {
        self.len = 0;
    }
}
