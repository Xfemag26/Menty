// gguf.rs - Zero-copy memory-mapped GGUF v2/v3 parser
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little endian

#[allow(non_camel_case_types, dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    Unknown(u32),
}

impl From<u32> for GGMLType {
    fn from(val: u32) -> Self {
        match val {
            0 => GGMLType::F32,
            1 => GGMLType::F16,
            2 => GGMLType::Q4_0,
            3 => GGMLType::Q4_1,
            6 => GGMLType::Q5_0,
            7 => GGMLType::Q5_1,
            8 => GGMLType::Q8_0,
            9 => GGMLType::Q8_1,
            10 => GGMLType::Q2_K,
            11 => GGMLType::Q3_K,
            12 => GGMLType::Q4_K,
            13 => GGMLType::Q5_K,
            14 => GGMLType::Q6_K,
            15 => GGMLType::Q8_K,
            16 => GGMLType::I8,
            17 => GGMLType::I16,
            18 => GGMLType::I32,
            19 => GGMLType::I64,
            20 => GGMLType::F64,
            x => GGMLType::Unknown(x),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub qtype: GGMLType,
    pub offset: u64,
    pub size_bytes: usize,
}

pub struct GGUFFile {
    pub mmap: Mmap,
    pub version: u32,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: HashMap<String, TensorInfo>,
    pub data_offset: usize,
    pub alignment: usize,
}

impl GGUFFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {e}"))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap failed: {e}"))? };

        if mmap.len() < 16 {
            return Err("File too small for GGUF header".into());
        }

        let mut offset = 0;
        let magic = read_u32(&mmap, &mut offset)?;
        if magic != GGUF_MAGIC {
            return Err(format!("Invalid magic number: 0x{:08x}", magic));
        }

        let version = read_u32(&mmap, &mut offset)?;
        if version != 2 && version != 3 {
            return Err(format!("Unsupported GGUF version: {version}"));
        }

        let tensor_count = read_u64(&mmap, &mut offset)? as usize;
        let metadata_kv_count = read_u64(&mmap, &mut offset)? as usize;

        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = read_string(&mmap, &mut offset)?;
            let val = read_metadata_val(&mmap, &mut offset)?;
            metadata.insert(key, val);
        }

        let alignment = match metadata.get("general.alignment") {
            Some(MetadataValue::Uint32(v)) => *v as usize,
            Some(MetadataValue::Uint64(v)) => *v as usize,
            _ => 32,
        };

        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(&mmap, &mut offset)?;
            let n_dims = read_u32(&mmap, &mut offset)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(&mmap, &mut offset)?);
            }
            let qtype_raw = read_u32(&mmap, &mut offset)?;
            let qtype = GGMLType::from(qtype_raw);
            let tensor_offset = read_u64(&mmap, &mut offset)?;

            let n_elements: usize = dims.iter().map(|&d| d as usize).product();
            let size_bytes = calc_tensor_bytes(qtype, n_elements);

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dims,
                    qtype,
                    offset: tensor_offset,
                    size_bytes,
                },
            );
        }

        let data_offset = (offset + alignment - 1) & !(alignment - 1);

        Ok(GGUFFile {
            mmap,
            version,
            metadata,
            tensors,
            data_offset,
            alignment,
        })
    }

    #[inline]
    pub fn get_tensor_data(&self, tensor: &TensorInfo) -> &[u8] {
        let start = self.data_offset + tensor.offset as usize;
        let end = start + tensor.size_bytes;
        &self.mmap[start..end]
    }

    pub fn get_str_metadata(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(MetadataValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn get_u32_metadata(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(MetadataValue::Uint32(v)) => Some(*v),
            Some(MetadataValue::Uint64(v)) => Some(*v as u32),
            Some(MetadataValue::Int32(v)) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn get_f32_metadata(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(MetadataValue::Float32(v)) => Some(*v),
            Some(MetadataValue::Float64(v)) => Some(*v as f32),
            _ => None,
        }
    }
}

fn calc_tensor_bytes(qtype: GGMLType, elements: usize) -> usize {
    match qtype {
        GGMLType::F32 => elements * 4,
        GGMLType::F16 => elements * 2,
        GGMLType::Q4_0 => (elements / 32) * 18,
        GGMLType::Q4_1 => (elements / 32) * 20,
        GGMLType::Q5_0 => (elements / 32) * 22,
        GGMLType::Q8_0 => (elements / 32) * 34,
        GGMLType::Q4_K => (elements / 256) * 144,
        GGMLType::Q6_K => (elements / 256) * 210,
        GGMLType::Q8_K => (elements / 256) * 292,
        _ => elements * 4,
    }
}

fn read_u32(buf: &[u8], offset: &mut usize) -> Result<u32, String> {
    if *offset + 4 > buf.len() {
        return Err("Unexpected EOF reading u32".into());
    }
    let val = u32::from_le_bytes(buf[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    Ok(val)
}

fn read_u64(buf: &[u8], offset: &mut usize) -> Result<u64, String> {
    if *offset + 8 > buf.len() {
        return Err("Unexpected EOF reading u64".into());
    }
    let val = u64::from_le_bytes(buf[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    Ok(val)
}

fn read_string(buf: &[u8], offset: &mut usize) -> Result<String, String> {
    let len = read_u64(buf, offset)? as usize;
    if *offset + len > buf.len() {
        return Err("Unexpected EOF reading string".into());
    }
    let s = std::str::from_utf8(&buf[*offset..*offset + len])
        .map_err(|e| format!("Invalid utf-8: {e}"))?
        .to_string();
    *offset += len;
    Ok(s)
}

fn read_metadata_val(buf: &[u8], offset: &mut usize) -> Result<MetadataValue, String> {
    let val_type = read_u32(buf, offset)?;
    match val_type {
        0 => {
            let v = buf[*offset];
            *offset += 1;
            Ok(MetadataValue::Uint8(v))
        }
        1 => {
            let v = buf[*offset] as i8;
            *offset += 1;
            Ok(MetadataValue::Int8(v))
        }
        2 => {
            let v = u16::from_le_bytes(buf[*offset..*offset + 2].try_into().unwrap());
            *offset += 2;
            Ok(MetadataValue::Uint16(v))
        }
        3 => {
            let v = i16::from_le_bytes(buf[*offset..*offset + 2].try_into().unwrap());
            *offset += 2;
            Ok(MetadataValue::Int16(v))
        }
        4 => Ok(MetadataValue::Uint32(read_u32(buf, offset)?)),
        5 => {
            let v = i32::from_le_bytes(buf[*offset..*offset + 4].try_into().unwrap());
            *offset += 4;
            Ok(MetadataValue::Int32(v))
        }
        6 => {
            let v = f32::from_le_bytes(buf[*offset..*offset + 4].try_into().unwrap());
            *offset += 4;
            Ok(MetadataValue::Float32(v))
        }
        7 => {
            let v = buf[*offset] != 0;
            *offset += 1;
            Ok(MetadataValue::Bool(v))
        }
        8 => Ok(MetadataValue::String(read_string(buf, offset)?)),
        9 => {
            let elem_type = read_u32(buf, offset)?;
            let count = read_u64(buf, offset)? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                let elem = match elem_type {
                    0 => {
                        let v = buf[*offset];
                        *offset += 1;
                        MetadataValue::Uint8(v)
                    }
                    1 => {
                        let v = buf[*offset] as i8;
                        *offset += 1;
                        MetadataValue::Int8(v)
                    }
                    2 => {
                        let v = u16::from_le_bytes(buf[*offset..*offset + 2].try_into().unwrap());
                        *offset += 2;
                        MetadataValue::Uint16(v)
                    }
                    3 => {
                        let v = i16::from_le_bytes(buf[*offset..*offset + 2].try_into().unwrap());
                        *offset += 2;
                        MetadataValue::Int16(v)
                    }
                    4 => MetadataValue::Uint32(read_u32(buf, offset)?),
                    5 => {
                        let v = i32::from_le_bytes(buf[*offset..*offset + 4].try_into().unwrap());
                        *offset += 4;
                        MetadataValue::Int32(v)
                    }
                    6 => {
                        let v = f32::from_le_bytes(buf[*offset..*offset + 4].try_into().unwrap());
                        *offset += 4;
                        MetadataValue::Float32(v)
                    }
                    7 => {
                        let v = buf[*offset] != 0;
                        *offset += 1;
                        MetadataValue::Bool(v)
                    }
                    8 => MetadataValue::String(read_string(buf, offset)?),
                    10 => MetadataValue::Uint64(read_u64(buf, offset)?),
                    11 => {
                        let v = i64::from_le_bytes(buf[*offset..*offset + 8].try_into().unwrap());
                        *offset += 8;
                        MetadataValue::Int64(v)
                    }
                    12 => {
                        let v = f64::from_le_bytes(buf[*offset..*offset + 8].try_into().unwrap());
                        *offset += 8;
                        MetadataValue::Float64(v)
                    }
                    x => return Err(format!("Unsupported array elem type: {x}")),
                };
                arr.push(elem);
            }
            Ok(MetadataValue::Array(arr))
        }
        10 => Ok(MetadataValue::Uint64(read_u64(buf, offset)?)),
        11 => {
            let v = i64::from_le_bytes(buf[*offset..*offset + 8].try_into().unwrap());
            *offset += 8;
            Ok(MetadataValue::Int64(v))
        }
        12 => {
            let v = f64::from_le_bytes(buf[*offset..*offset + 8].try_into().unwrap());
            *offset += 8;
            Ok(MetadataValue::Float64(v))
        }
        _ => Err(format!("Unknown metadata type: {val_type}")),
    }
}
