//! Reader endpoints — `BytesReader` and `Float64Reader`.
//!
//! Both are thin shells over [`RawReader`]; the only difference is the element
//! type they declare in the segment header and the units `entry_length` is
//! measured in.

use crate::python::errors::{check_out, IntoPyResult};
use crate::python::zerocopy::{HandleReader, Layout, PyReadHandle};
use crate::python::F64_BYTES;
use crate::{Dtype, RawReader};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::cell::Cell;

// ── Bytes channel ───────────────────────────────────────────────────────────
/// Reader end of a byte-oriented ZeroChannel.
#[pyclass(name = "BytesReader", module = "zerochannel", unsendable)]
pub(crate) struct PyBytesReader {
    pub(crate) inner: RawReader,
}

#[pymethods]
impl PyBytesReader {
    /// Create or attach to a shared-memory channel for reading bytes.
    ///
    /// Readers default to attaching rather than creating: geometry is omitted
    /// and `delayed_connect` is on, so constructing a reader before its writer
    /// exists succeeds and resolves on the first read.
    ///
    /// `enable_zero_copy` opts into `try_acquire()` and claims the channel's
    /// exclusive zero-copy reader role; it raises `ChannelRoleConflict` if a
    /// live process already holds it.
    #[new]
    #[pyo3(signature = (name, *, entry_length=None, entry_count=None, delayed_connect=true, enable_zero_copy=false))]
    fn new(
        name: &str,
        entry_length: Option<usize>,
        entry_count: Option<usize>,
        delayed_connect: bool,
        enable_zero_copy: bool,
    ) -> PyResult<Self> {
        let build = if enable_zero_copy {
            RawReader::new_zero_copy
        } else {
            RawReader::new
        };
        let inner = build(name, entry_length, entry_count, Dtype::U8, delayed_connect).or_py()?;
        Ok(Self { inner })
    }

    /// Bytes per entry (0 while a deferred channel is unconnected).
    #[getter]
    fn entry_length(&self) -> usize {
        self.inner.payload_bytes()
    }

    /// Sequence number of the last entry this reader returned (0 if none).
    #[getter]
    fn last_entry_seq(&self) -> u64 {
        self.inner.last_entry_seq()
    }

    /// The writer's high watermark, straight from the segment header.
    #[getter]
    fn writer_entry_seq(&self) -> u64 {
        self.inner.writer_entry_seq()
    }

    /// Read the next entry newer than the watermark.
    ///
    /// Returns `(entry_seq, payload)` for exactly one entry, or `None` when
    /// nothing newer is available. `payload` is a `bytes` object unless a
    /// pre-allocated `out` array is supplied, in which case it is filled in
    /// place and returned.
    ///
    /// `out` must be a contiguous 1-D `uint8` array of `entry_length` elements.
    /// `from_entry_seq` overrides the reader's own watermark.
    #[pyo3(signature = (*, out=None, from_entry_seq=None))]
    fn read<'py>(
        &mut self,
        py: Python<'py>,
        out: Option<Bound<'py, PyArray1<u8>>>,
        from_entry_seq: Option<u64>,
    ) -> PyResult<Option<(u64, Bound<'py, PyAny>)>> {
        if !self.inner.connect().or_py()? {
            return Ok(None);
        }
        let entry_length = self.inner.payload_bytes();

        // Safety: `unsendable` plus exclusive `&mut self` for this call.
        let inner_ptr = &mut self.inner as *mut RawReader as usize;

        match out {
            Some(arr) => {
                check_out(&arr, entry_length)?;
                // Safety: `arr` is contiguous and exclusively borrowed here.
                let dst_ptr = unsafe { arr.as_slice_mut()? }.as_mut_ptr() as usize;
                let seq = py.detach(|| {
                    let reader = unsafe { &mut *(inner_ptr as *mut RawReader) };
                    let dst =
                        unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, entry_length) };
                    reader.read_into(dst, from_entry_seq).or_py()
                })?;
                Ok(seq.map(|s| (s, arr.into_any())))
            }
            None => {
                let mut buf = vec![0u8; entry_length];
                let seq = py.detach(|| {
                    let reader = unsafe { &mut *(inner_ptr as *mut RawReader) };
                    reader.read_into(&mut buf, from_entry_seq).or_py()
                })?;
                Ok(seq.map(|s| (s, PyBytes::new(py, &buf).into_any())))
            }
        }
    }

    /// Acquire the next entry newer than the watermark (zero-copy path).
    ///
    /// Returns a `ReadHandle` borrowing the entry in place, or `None` when
    /// nothing newer is available. `from_entry_seq` overrides the reader's own
    /// watermark, exactly as for `read()`.
    ///
    /// Single-reader only: the slot is pinned by driving its `entry_seq`
    /// negative, a state only one reader can own. The handle must be released.
    #[pyo3(signature = (*, from_entry_seq=None))]
    fn try_acquire(
        slf: &Bound<'_, Self>,
        from_entry_seq: Option<u64>,
    ) -> PyResult<Option<PyReadHandle>> {
        let Some(handle) = slf.borrow_mut().inner.try_acquire(from_entry_seq).or_py()? else {
            return Ok(None);
        };
        Ok(Some(PyReadHandle {
            entry_seq: handle.entry_seq(),
            handle: Some(handle),
            reader: HandleReader::Bytes(slf.clone().unbind()),
            layout: Layout::BYTES,
            exports: Cell::new(0),
        }))
    }
}

// ── float64 channel ──────────────────────────────────────────────────────────

/// Reader end of a `float64` ZeroChannel.
#[pyclass(name = "Float64Reader", module = "zerochannel", unsendable)]
pub(crate) struct PyFloat64Reader {
    pub(crate) inner: RawReader,
}

#[pymethods]
impl PyFloat64Reader {
    /// Create or attach to a shared-memory channel for reading `float64`.
    ///
    /// Readers default to attaching rather than creating: geometry is omitted
    /// and `delayed_connect` is on, so constructing a reader before its writer
    /// exists succeeds and resolves on the first read.
    ///
    /// `enable_zero_copy` opts into `try_acquire()` and claims the channel's
    /// exclusive zero-copy reader role; it raises `ChannelRoleConflict` if a
    /// live process already holds it.
    #[new]
    #[pyo3(signature = (name, *, entry_length=None, entry_count=None, delayed_connect=true, enable_zero_copy=false))]
    fn new(
        name: &str,
        entry_length: Option<usize>,
        entry_count: Option<usize>,
        delayed_connect: bool,
        enable_zero_copy: bool,
    ) -> PyResult<Self> {
        let payload_bytes = entry_length.map(|n| n.saturating_mul(F64_BYTES));
        let build = if enable_zero_copy {
            RawReader::new_zero_copy
        } else {
            RawReader::new
        };
        let inner = build(
            name,
            payload_bytes,
            entry_count,
            Dtype::F64,
            delayed_connect,
        )
        .or_py()?;
        Ok(Self { inner })
    }

    /// Number of `float64` elements per entry (0 while deferred).
    #[getter]
    fn entry_length(&self) -> usize {
        self.inner.payload_bytes() / F64_BYTES
    }

    /// Sequence number of the last entry this reader returned (0 if none).
    #[getter]
    fn last_entry_seq(&self) -> u64 {
        self.inner.last_entry_seq()
    }

    /// The writer's high watermark, straight from the segment header.
    #[getter]
    fn writer_entry_seq(&self) -> u64 {
        self.inner.writer_entry_seq()
    }

    /// Read the next entry newer than the watermark.
    ///
    /// Returns `(entry_seq, payload)` for exactly one entry, or `None` when
    /// nothing newer is available. `payload` is a 1-D `float64` array of
    /// `entry_length` elements — the pre-allocated `out` array when one is
    /// supplied, otherwise a freshly allocated one.
    ///
    /// `out` must be a contiguous 1-D `float64` array of `entry_length`
    /// elements. `from_entry_seq` overrides the reader's own watermark.
    #[pyo3(signature = (*, out=None, from_entry_seq=None))]
    fn read<'py>(
        &mut self,
        py: Python<'py>,
        out: Option<Bound<'py, PyArray1<f64>>>,
        from_entry_seq: Option<u64>,
    ) -> PyResult<Option<(u64, Bound<'py, PyArray1<f64>>)>> {
        if !self.inner.connect().or_py()? {
            return Ok(None);
        }
        let entry_length = self.inner.payload_bytes() / F64_BYTES;

        let dst = match out {
            Some(arr) => {
                check_out(&arr, entry_length)?;
                arr
            }
            None => PyArray1::<f64>::zeros(py, [entry_length], false),
        };

        // Safety: `dst` is contiguous and exclusively borrowed for this call;
        // `unsendable` prevents this pyclass from crossing threads.
        let dst_ptr = unsafe { dst.as_slice_mut()? }.as_mut_ptr() as usize;
        let inner_ptr = &mut self.inner as *mut RawReader as usize;

        // Release the GIL for the copy.
        let seq = py.detach(|| {
            let reader = unsafe { &mut *(inner_ptr as *mut RawReader) };
            let slice =
                unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut f64, entry_length) };
            reader.read_into_as::<f64>(slice, from_entry_seq).or_py()
        })?;

        Ok(seq.map(|s| (s, dst)))
    }

    /// Acquire the next entry newer than the watermark (zero-copy path).
    ///
    /// Returns a `ReadHandle` borrowing the entry in place as a `float64`
    /// `memoryview`, or `None` when nothing newer is available.
    /// `from_entry_seq` overrides the reader's own watermark, exactly as for
    /// `read()`.
    ///
    /// Single-reader only: the slot is pinned by driving its `entry_seq`
    /// negative, a state only one reader can own. The handle must be released.
    #[pyo3(signature = (*, from_entry_seq=None))]
    fn try_acquire(
        slf: &Bound<'_, Self>,
        from_entry_seq: Option<u64>,
    ) -> PyResult<Option<PyReadHandle>> {
        let Some(handle) = slf.borrow_mut().inner.try_acquire(from_entry_seq).or_py()? else {
            return Ok(None);
        };
        Ok(Some(PyReadHandle {
            entry_seq: handle.entry_seq(),
            handle: Some(handle),
            reader: HandleReader::Float64(slf.clone().unbind()),
            layout: Layout::FLOAT64,
            exports: Cell::new(0),
        }))
    }
}
