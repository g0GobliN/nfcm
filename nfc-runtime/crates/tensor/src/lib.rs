//! Lightweight tensor primitives for NFCM.
//!
//! Phase 1 uses simple owned buffers. Candle / ONNX backends can plug in later
//! behind the same shape metadata without changing the runtime API surface.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("empty tensor")]
    Empty,
    #[error("invalid dtype: {0}")]
    InvalidDtype(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    F32,
    F16,
    I8,
    I32,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 => 2,
            DType::I8 => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape {
    pub dims: Vec<usize>,
}

impl TensorShape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            0
        } else {
            self.dims.iter().product()
        }
    }

    pub fn nbytes(&self, dtype: DType) -> usize {
        self.numel().saturating_mul(dtype.size_bytes())
    }
}

/// Owned dense tensor used by the mock generator and runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub shape: TensorShape,
    pub dtype: DType,
    /// Phase 1: packed little-endian bytes. Real backends may replace this.
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn zeros(shape: TensorShape, dtype: DType) -> Self {
        let nbytes = shape.nbytes(dtype);
        Self {
            shape,
            dtype,
            data: vec![0u8; nbytes],
        }
    }

    pub fn from_f32(shape: TensorShape, values: &[f32]) -> Result<Self, TensorError> {
        if values.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: shape.dims.clone(),
                got: vec![values.len()],
            });
        }
        let mut data = Vec::with_capacity(values.len() * 4);
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Ok(Self {
            shape,
            dtype: DType::F32,
            data,
        })
    }

    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn validate(&self) -> Result<(), TensorError> {
        if self.shape.numel() == 0 {
            return Err(TensorError::Empty);
        }
        let expected = self.shape.nbytes(self.dtype);
        if self.data.len() != expected {
            return Err(TensorError::ShapeMismatch {
                expected: vec![expected],
                got: vec![self.data.len()],
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_allocates_expected_bytes() {
        let t = Tensor::zeros(TensorShape::new([2, 3]), DType::F32);
        assert_eq!(t.memory_bytes(), 24);
        t.validate().unwrap();
    }

    #[test]
    fn from_f32_rejects_mismatch() {
        let err = Tensor::from_f32(TensorShape::new([2, 2]), &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, TensorError::ShapeMismatch { .. }));
    }
}
