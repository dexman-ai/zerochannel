//! Writer endpoints — `BytesWriter` and `Float64Writer`.
//!
//! Both are thin shells over [`RawWriter`]; the only difference is the element
//! type they declare in the segment header and the units `entry_length` is
//! measured in.

use crate::python::errors::IntoPyResult;
use crate::python::zerocopy::{HandleWriter, Layout, PyWriteHandle};
use crate::python::F64_BYTES;
use crate::{Dtype, RawWriter};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::cell::Cell;

// ── Bytes channel ───────────────────────────────────────────────────────────

/// Single-writer end of a byte-oriented ZeroChannel.
///
/// `entry_length` is measured in bytes and must be a multiple of 8, so that
/// every slot's `entry_seq` word stays naturally aligned.
#[pyclass(name = "BytesWriter", module = "zerochannel", unsendable)]
pub(crate) struct PyBytesWriter {
    pub(crate) inner: RawWriter,
}

#[pymethods]
impl PyBytesWriter {
    /// Create or attach to a shared-memory channel for writing bytes.
    #[new]
    #[pyo3(signature = (name, *, entry_length=None, entry_count=None, delayed_connect=false))]
    fn new(
        name: &str,
        entry_length: Option<usize>,
        entry_count: Option<usize>,
        delayed_connect: bool,
    ) -> PyResult<Self> {
        // Dtype::U8 is opaque: a bytes endpoint attaches to a channel of any
        // declared element type.
        let inner =
            RawWriter::new(name, entry_length, entry_count, Dtype::U8, delayed_connect).or_py()?;
        Ok(Self { inner })
    }

    /// Bytes per entry (0 while a deferred channel is unconnected).
    #[getter]
    fn entry_length(&self) -> usize {
        self.inner.payload_bytes()
    }

    /// Sequence number of the most recently committed entry (0 if none).
    #[getter]
    fn last_entry_seq(&self) -> u64 {
        self.inner.last_entry_seq()
    }

    /// Write one entry. `len(data)` must equal `entry_length`.
    fn write(&mut self, py: Python<'_>, data: &[u8]) -> PyResult<()> {
        let nbytes = data.len();
        let src_ptr = data.as_ptr() as usize;

        // Release the GIL for the fence/copy/fence section.
        // Safety: `unsendable` prevents this pyclass from crossing threads, and
        // `data` borrows a Python buffer that stays alive for this call.
        py.detach(|| {
            let bytes = unsafe { std::slice::from_raw_parts(src_ptr as *const u8, nbytes) };
            self.inner.write(bytes).or_py()
        })
    }

    /// Acquire the next ring slot for in-place construction (zero-copy path).
    ///
    /// Returns a `WriteHandle` whose `payload` is a writable `memoryview` of
    /// `entry_length` bytes, or `None` while a deferred channel is still
    /// unconnected. The entry becomes visible only on `commit()`.
    ///
    /// Raises `BlockingIOError` if every ring slot is held by a reader.
    fn try_acquire(slf: &Bound<'_, Self>) -> PyResult<Option<PyWriteHandle>> {
        let Some(handle) = slf.borrow_mut().inner.try_acquire().or_py()? else {
            return Ok(None);
        };
        Ok(Some(PyWriteHandle {
            handle: Some(handle),
            writer: HandleWriter::Bytes(slf.clone().unbind()),
            layout: Layout::BYTES,
            exports: Cell::new(0),
        }))
    }
}
// ── float64 channel ─────────────────────────────────────────────────────────

/// Single-writer end of a `float64` ZeroChannel.
///
/// `entry_length` is measured in `float64` elements.
#[pyclass(name = "Float64Writer", module = "zerochannel", unsendable)]
pub(crate) struct PyFloat64Writer {
    pub(crate) inner: RawWriter,
}

#[pymethods]
impl PyFloat64Writer {
    /// Create or attach to a shared-memory channel for writing `float64`.
    #[new]
    #[pyo3(signature = (name, *, entry_length=None, entry_count=None, delayed_connect=false))]
    fn new(
        name: &str,
        entry_length: Option<usize>,
        entry_count: Option<usize>,
        delayed_connect: bool,
    ) -> PyResult<Self> {
        let payload_bytes = entry_length.map(|n| n.saturating_mul(F64_BYTES));
        let inner = RawWriter::new(
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

    /// Sequence number of the most recently committed entry (0 if none).
    #[getter]
    fn last_entry_seq(&self) -> u64 {
        self.inner.last_entry_seq()
    }

    /// Write one entry. `len(data)` must equal `entry_length`.
    fn write<'py>(&mut self, py: Python<'py>, data: PyReadonlyArray1<'py, f64>) -> PyResult<()> {
        let slice = data.as_slice()?;
        let nbytes = std::mem::size_of_val(slice);
        let src_ptr = slice.as_ptr() as usize;

        // Release the GIL for the fence/copy/fence section.
        // Safety: `unsendable` prevents this pyclass from crossing threads.
        // The numpy buffer remains valid for the duration of the Python call.
        py.detach(|| {
            let bytes = unsafe { std::slice::from_raw_parts(src_ptr as *const u8, nbytes) };
            self.inner.write(bytes).or_py()
        })
    }

    /// Acquire the next ring slot for in-place construction (zero-copy path).
    ///
    /// Returns a `WriteHandle` whose `payload` is a writable `float64`
    /// `memoryview` of `entry_length` elements, or `None` while a deferred
    /// channel is still unconnected. The entry becomes visible only on
    /// `commit()`.
    ///
    /// Raises `BlockingIOError` if every ring slot is held by a reader.
    fn try_acquire(slf: &Bound<'_, Self>) -> PyResult<Option<PyWriteHandle>> {
        let Some(handle) = slf.borrow_mut().inner.try_acquire().or_py()? else {
            return Ok(None);
        };
        Ok(Some(PyWriteHandle {
            handle: Some(handle),
            writer: HandleWriter::Float64(slf.clone().unbind()),
            layout: Layout::FLOAT64,
            exports: Cell::new(0),
        }))
    }
}
