//! GGUF metadata value types.

use crate::error::{GgufError, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Read;

/// Raw GGUF value type IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl ValueType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            other => return Err(GgufError::UnknownValueType(other)),
        })
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array {
        item_type: ValueType,
        items: Vec<Value>,
    },
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Value::U32(v) => Some(*v),
            Value::U64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            Value::I32(v) if *v >= 0 => Some(*v as u32),
            Value::U8(v) => Some(*v as u32),
            Value::U16(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U64(v) => Some(*v),
            Value::U32(v) => Some(*v as u64),
            Value::I64(v) if *v >= 0 => Some(*v as u64),
            Value::I32(v) if *v >= 0 => Some(*v as u64),
            Value::U8(v) => Some(*v as u64),
            Value::U16(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            Value::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Short display for CLI dumps.
    pub fn display_short(&self) -> String {
        match self {
            Value::U8(v) => v.to_string(),
            Value::I8(v) => v.to_string(),
            Value::U16(v) => v.to_string(),
            Value::I16(v) => v.to_string(),
            Value::U32(v) => v.to_string(),
            Value::I32(v) => v.to_string(),
            Value::F32(v) => format!("{v}"),
            Value::Bool(v) => v.to_string(),
            Value::String(s) => {
                if s.len() > 80 {
                    format!("\"{}…\" ({} bytes)", &s[..80], s.len())
                } else {
                    format!("\"{s}\"")
                }
            }
            Value::Array { item_type, items } => {
                format!(
                    "array[{}; {} items]",
                    format_value_type(*item_type),
                    items.len()
                )
            }
            Value::U64(v) => v.to_string(),
            Value::I64(v) => v.to_string(),
            Value::F64(v) => format!("{v}"),
        }
    }
}

fn format_value_type(t: ValueType) -> &'static str {
    match t {
        ValueType::U8 => "u8",
        ValueType::I8 => "i8",
        ValueType::U16 => "u16",
        ValueType::I16 => "i16",
        ValueType::U32 => "u32",
        ValueType::I32 => "i32",
        ValueType::F32 => "f32",
        ValueType::Bool => "bool",
        ValueType::String => "string",
        ValueType::Array => "array",
        ValueType::U64 => "u64",
        ValueType::I64 => "i64",
        ValueType::F64 => "f64",
    }
}

pub fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = r
        .read_u64::<LittleEndian>()
        .map_err(|_| GgufError::UnexpectedEof {
            context: "string length",
        })?;
    if len > 64 * 1024 * 1024 {
        return Err(GgufError::Malformed(format!(
            "string length {len} is unreasonably large"
        )));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)
        .map_err(|_| GgufError::UnexpectedEof {
            context: "string bytes",
        })?;
    String::from_utf8(buf).map_err(|_| GgufError::InvalidUtf8 {
        context: "metadata/tensor name",
    })
}

pub fn read_value<R: Read>(r: &mut R, ty: ValueType) -> Result<Value> {
    Ok(match ty {
        ValueType::U8 => Value::U8(r.read_u8().map_err(|_| GgufError::UnexpectedEof {
            context: "u8 value",
        })?),
        ValueType::I8 => Value::I8(r.read_i8().map_err(|_| GgufError::UnexpectedEof {
            context: "i8 value",
        })?),
        ValueType::U16 => {
            Value::U16(
                r.read_u16::<LittleEndian>()
                    .map_err(|_| GgufError::UnexpectedEof {
                        context: "u16 value",
                    })?,
            )
        }
        ValueType::I16 => {
            Value::I16(
                r.read_i16::<LittleEndian>()
                    .map_err(|_| GgufError::UnexpectedEof {
                        context: "i16 value",
                    })?,
            )
        }
        ValueType::U32 => {
            Value::U32(
                r.read_u32::<LittleEndian>()
                    .map_err(|_| GgufError::UnexpectedEof {
                        context: "u32 value",
                    })?,
            )
        }
        ValueType::I32 => {
            Value::I32(
                r.read_i32::<LittleEndian>()
                    .map_err(|_| GgufError::UnexpectedEof {
                        context: "i32 value",
                    })?,
            )
        }
        ValueType::F32 => {
            Value::F32(
                r.read_f32::<LittleEndian>()
                    .map_err(|_| GgufError::UnexpectedEof {
                        context: "f32 value",
                    })?,
            )
        }
        ValueType::Bool => {
            let b = r.read_u8().map_err(|_| GgufError::UnexpectedEof {
                context: "bool value",
            })?;
            Value::Bool(b != 0)
        }
        ValueType::String => Value::String(read_string(r)?),
        ValueType::Array => {
            let item_ty_id =
                r.read_u32::<LittleEndian>()
                    .map_err(|_| GgufError::UnexpectedEof {
                        context: "array item type",
                    })?;
            let item_type = ValueType::from_u32(item_ty_id)?;
            let n = r
                .read_u64::<LittleEndian>()
                .map_err(|_| GgufError::UnexpectedEof {
                    context: "array length",
                })?;
            if n > 10_000_000 {
                return Err(GgufError::Malformed(format!(
                    "array length {n} is unreasonably large"
                )));
            }
            let mut items = Vec::with_capacity(n as usize);
            for _ in 0..n {
                items.push(read_value(r, item_type)?);
            }
            Value::Array { item_type, items }
        }
        ValueType::U64 => {
            Value::U64(
                r.read_u64::<LittleEndian>()
                    .map_err(|_| GgufError::UnexpectedEof {
                        context: "u64 value",
                    })?,
            )
        }
        ValueType::I64 => {
            Value::I64(
                r.read_i64::<LittleEndian>()
                    .map_err(|_| GgufError::UnexpectedEof {
                        context: "i64 value",
                    })?,
            )
        }
        ValueType::F64 => {
            Value::F64(
                r.read_f64::<LittleEndian>()
                    .map_err(|_| GgufError::UnexpectedEof {
                        context: "f64 value",
                    })?,
            )
        }
    })
}
