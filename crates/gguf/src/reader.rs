//! Memory-mapped GGUF file reader.

use crate::error::{GgufError, Result};
use crate::types::{GgmlType, GgufTensorInfo};
use crate::value::{read_string, read_value, Value, ValueType};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// GGUF magic: "GGUF" little-endian = 0x46554747
const GGUF_MAGIC: u32 = 0x4655_4747;

/// Opened GGUF model file (header parsed, weights mmap'd).
pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<GgufTensorInfo>,
    pub tensor_index: HashMap<String, usize>,
    /// Absolute file offset where the tensor data section begins (aligned).
    pub data_offset: u64,
    mmap: Mmap,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|source| GgufError::Io {
            path: path.clone(),
            source,
        })?;
        // SAFETY: we treat the file as read-only and do not modify it while mapped.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| GgufError::Io {
            path: path.clone(),
            source,
        })?;

        let mut cursor = Cursor::new(&mmap[..]);

        let magic = cursor
            .read_u32::<LittleEndian>()
            .map_err(|_| GgufError::UnexpectedEof { context: "magic" })?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }

        let version = cursor
            .read_u32::<LittleEndian>()
            .map_err(|_| GgufError::UnexpectedEof { context: "version" })?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let n_tensors =
            cursor
                .read_u64::<LittleEndian>()
                .map_err(|_| GgufError::UnexpectedEof {
                    context: "tensor count",
                })?;
        let n_kv = cursor
            .read_u64::<LittleEndian>()
            .map_err(|_| GgufError::UnexpectedEof {
                context: "metadata kv count",
            })?;

        if n_tensors > 1_000_000 || n_kv > 1_000_000 {
            return Err(GgufError::Malformed(format!(
                "unreasonable counts: tensors={n_tensors} kv={n_kv}"
            )));
        }

        let mut metadata = HashMap::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = read_string(&mut cursor)?;
            let ty_id =
                cursor
                    .read_u32::<LittleEndian>()
                    .map_err(|_| GgufError::UnexpectedEof {
                        context: "metadata value type",
                    })?;
            let ty = ValueType::from_u32(ty_id)?;
            let value = read_value(&mut cursor, ty)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        let mut tensor_index = HashMap::with_capacity(n_tensors as usize);

        for i in 0..n_tensors {
            let name = read_string(&mut cursor)?;
            let n_dims =
                cursor
                    .read_u32::<LittleEndian>()
                    .map_err(|_| GgufError::UnexpectedEof {
                        context: "tensor n_dims",
                    })?;
            if n_dims > 4 {
                return Err(GgufError::Malformed(format!(
                    "tensor {name}: n_dims={n_dims} > 4"
                )));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let d =
                    cursor
                        .read_u64::<LittleEndian>()
                        .map_err(|_| GgufError::UnexpectedEof {
                            context: "tensor dim",
                        })?;
                dims.push(d);
            }
            let type_id =
                cursor
                    .read_u32::<LittleEndian>()
                    .map_err(|_| GgufError::UnexpectedEof {
                        context: "tensor type",
                    })?;
            let ggml_type =
                GgmlType::from_u32(type_id).ok_or(GgufError::UnknownTensorType(type_id))?;
            let offset =
                cursor
                    .read_u64::<LittleEndian>()
                    .map_err(|_| GgufError::UnexpectedEof {
                        context: "tensor offset",
                    })?;

            tensor_index.insert(name.clone(), i as usize);
            tensors.push(GgufTensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        // Data section starts at alignment boundary after header.
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32);
        let header_end = cursor.position();
        let data_offset = align_up(header_end, alignment);

        // Basic bounds check for each tensor.
        let file_len = mmap.len() as u64;
        for t in &tensors {
            let start = data_offset.saturating_add(t.offset);
            let end = start.saturating_add(t.nbytes());
            if end > file_len {
                return Err(GgufError::Malformed(format!(
                    "tensor {} data [{start}, {end}) exceeds file size {file_len}",
                    t.name
                )));
            }
        }

        Ok(Self {
            path,
            version,
            metadata,
            tensors,
            tensor_index,
            data_offset,
            mmap,
        })
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    pub fn name(&self) -> Option<&str> {
        self.metadata.get("general.name").and_then(|v| v.as_str())
    }

    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(|v| v.as_u64())
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32())
    }

    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Raw bytes for a tensor (mmap slice).
    pub fn tensor_data(&self, info: &GgufTensorInfo) -> &[u8] {
        let start = (self.data_offset + info.offset) as usize;
        let end = start + info.nbytes() as usize;
        &self.mmap[start..end]
    }

    pub fn file_size(&self) -> u64 {
        self.mmap.len() as u64
    }

    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.nbytes()).sum()
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) / alignment * alignment
}
