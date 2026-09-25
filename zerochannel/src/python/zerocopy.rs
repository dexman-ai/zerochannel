//! Zero-copy handles and the CPython buffer protocol.
//!
//! A handle is a borrowed view of one ring slot: pinned for reading, or owned
//! and unpublished for writing. Exposing that view to Python without copying
//! means exporting a `Py_buffer` that points straight into shared memory, so
//! the slot must not be recycled while any `memoryview` or `numpy` array
//! still references it.
//!
//! Two mechanisms enforce that. Every export stores the handle in `view.obj`,
//! so a consumer holds a strong reference and the handle cannot be collected;
//! and every export increments a counter the handle checks before releasing or
//! committing, so a premature finish raises `BufferError` instead of handing
//! a live slot back to the ring.

use crate::python::reader::{PyBytesReader, PyFloat64Reader};
use crate::python::writer::{PyBytesWriter, PyFloat64Writer};
use crate::python::F64_BYTES;
use crate::{RawReader, RawWriter, ReadHandle, WriteHandle};
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use std::cell::Cell;
use std::ffi::{c_char, c_int, c_void};
use std::ptr;
// ── Zero-copy handles ───────────────────────────────────────────────────────

/// Buffer-protocol format codes (NUL-terminated, as CPython expects).
const FORMAT_U8: &[u8] = b"B\0";
const FORMAT_F64: &[u8] = b"d\0";

/// `[shape[0], strides[0]]` for a one-dimensional export.
///
/// CPython requires both arrays to stay valid for the lifetime of the
/// `Py_buffer`, so they are allocated per export and handed back through
/// `internal` for `__releasebuffer__` to free.
type Dims = [ffi::Py_ssize_t; 2];

/// Describes how a handle exposes its ring slot to the buffer protocol.
#[derive(Clone, Copy)]
pub(crate) struct Layout {
    pub(crate) itemsize: usize,
    pub(crate) format: &'static [u8],
}

impl Layout {
    pub(crate) const BYTES: Layout = Layout {
        itemsize: 1,
        format: FORMAT_U8,
    };
    pub(crate) const FLOAT64: Layout = Layout {
        itemsize: F64_BYTES,
        format: FORMAT_F64,
    };
}

/// Fill `view` with a 1-D export of the `len` bytes at `data`.
///
/// `owner` is stored in `view.obj`, which is what makes the guarantee work: the
/// consumer holds a strong reference to the handle for as long as the export
/// lives, so the handle cannot be released — or even garbage-collected — while
/// a `memoryview` or `numpy` array still points into the slot.
///
/// # Safety
///
/// `data` must address `len` mapped bytes that stay valid for as long as
/// `owner` holds its ring slot.
unsafe fn fill_view(
    view: *mut ffi::Py_buffer,
    flags: c_int,
    owner: Bound<'_, PyAny>,
    data: *mut u8,
    len: usize,
    layout: Layout,
    readonly: bool,
) -> PyResult<()> {
    if view.is_null() {
        return Err(PyBufferError::new_err("buffer view is null"));
    }
    if readonly && (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
        return Err(PyBufferError::new_err(
            "a read handle exports a read-only buffer",
        ));
    }

    let dims = Box::into_raw(Box::new([
        (len / layout.itemsize) as ffi::Py_ssize_t,
        layout.itemsize as ffi::Py_ssize_t,
    ] as Dims))
    .cast::<ffi::Py_ssize_t>();

    // Safety: `view` is non-null and CPython owns it for the export's lifetime.
    unsafe {
        (*view).obj = owner.into_ptr();
        (*view).buf = data.cast::<c_void>();
        (*view).len = len as ffi::Py_ssize_t;
        (*view).readonly = c_int::from(readonly);
        (*view).itemsize = layout.itemsize as ffi::Py_ssize_t;
        (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
            layout.format.as_ptr().cast::<c_char>().cast_mut()
        } else {
            ptr::null_mut()
        };
        (*view).ndim = 1;
        (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
            dims
        } else {
            ptr::null_mut()
        };
        (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
            dims.add(1)
        } else {
            ptr::null_mut()
        };
        (*view).suboffsets = ptr::null_mut();
        (*view).internal = dims.cast::<c_void>();
    }
    Ok(())
}

/// Release the per-export shape/stride allocation made by [`fill_view`].
///
/// # Safety
///
/// `view` must be a `Py_buffer` previously filled by [`fill_view`].
unsafe fn free_view(view: *mut ffi::Py_buffer) {
    if view.is_null() {
        return;
    }
    // Safety: `internal` is either null or the `Dims` allocation we made.
    unsafe {
        let dims = (*view).internal.cast::<Dims>();
        if !dims.is_null() {
            drop(Box::from_raw(dims));
            (*view).internal = ptr::null_mut();
        }
    }
}

/// Wrap `obj` — a handle exporting the buffer protocol — in a `memoryview`.
fn memoryview_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    // Safety: `PyMemoryView_FromObject` returns a new reference or null, and
    // takes an export on `obj` through its buffer slots.
    let raw = unsafe { ffi::PyMemoryView_FromObject(obj.as_ptr()) };
    unsafe { Bound::from_owned_ptr_or_err(obj.py(), raw) }
}

/// Error raised when a handle is finished while views into it are still alive.
fn outstanding_exports(action: &str, n: usize) -> PyErr {
    PyBufferError::new_err(format!(
        "cannot {action} this handle: {n} buffer export{} still \
         outstanding. Every memoryview or numpy array derived from `payload` \
         must be dropped first (`del arr`), since the ring slot is recycled as \
         soon as the handle is finished. Copy the data out (`bytes(...)`, \
         `numpy.array(...)`) if it needs to outlive the handle.",
        if n == 1 { " is" } else { "s are" }
    ))
}

/// The reader a [`PyReadHandle`] hands its slot back to.
///
/// Holding the endpoint alive also keeps the shared-memory mapping alive for
/// as long as the handle exists.
pub(crate) enum HandleReader {
    Bytes(Py<PyBytesReader>),
    Float64(Py<PyFloat64Reader>),
}

impl HandleReader {
    /// Run `f` against the owning reader's channel end.
    fn with_raw<R>(&self, py: Python<'_>, f: impl FnOnce(&mut RawReader) -> R) -> PyResult<R> {
        Ok(match self {
            HandleReader::Bytes(r) => f(&mut r.bind(py).try_borrow_mut()?.inner),
            HandleReader::Float64(r) => f(&mut r.bind(py).try_borrow_mut()?.inner),
        })
    }
}

/// The writer a [`PyWriteHandle`] publishes through.
pub(crate) enum HandleWriter {
    Bytes(Py<PyBytesWriter>),
    Float64(Py<PyFloat64Writer>),
}

impl HandleWriter {
    /// Run `f` against the owning writer's channel end.
    fn with_raw<R>(&self, py: Python<'_>, f: impl FnOnce(&mut RawWriter) -> R) -> PyResult<R> {
        Ok(match self {
            HandleWriter::Bytes(w) => f(&mut w.bind(py).try_borrow_mut()?.inner),
            HandleWriter::Float64(w) => f(&mut w.bind(py).try_borrow_mut()?.inner),
        })
    }
}

/// A borrowed, in-place view of one committed entry (zero-copy read path).
///
/// Handed out by `BytesReader.try_acquire()` and
/// `Float64Reader.try_acquire()`. The slot's `entry_seq` is negative for as
/// long as the handle is held, so the writer skips it and the payload cannot
/// change underfoot. Releasing is mandatory — an unreleased slot is retired
/// from the ring — so prefer the context manager:
///
/// ```python
/// with reader.try_acquire() as handle:
///     total = numpy.asarray(handle.payload).sum()
/// ```
///
/// The handle exports the buffer protocol, so every view derived from
/// `payload` keeps it alive and counted. `release()` raises `BufferError`
/// rather than recycling a slot somebody is still reading, which means a
/// `numpy` array must not survive the `with` block:
///
/// ```python
/// with reader.try_acquire() as handle:
///     arr = numpy.asarray(handle.payload)
///     ...
///     del arr          # required: `arr` aliases the ring slot
/// ```
#[pyclass(name = "ReadHandle", module = "zerochannel", unsendable)]
pub(crate) struct PyReadHandle {
    pub(crate) handle: Option<ReadHandle>,
    pub(crate) reader: HandleReader,
    pub(crate) entry_seq: u64,
    pub(crate) layout: Layout,
    pub(crate) exports: Cell<usize>,
}

#[pymethods]
impl PyReadHandle {
    /// Sequence number of the held entry.
    #[getter]
    fn entry_seq(&self) -> u64 {
        self.entry_seq
    }

    /// The entry payload as a read-only `memoryview` over shared memory.
    ///
    /// Typed as `uint8` on a byte channel and `float64` on a `float64`
    /// channel, so `numpy.asarray(handle.payload)` is a zero-copy typed view.
    /// Each access creates a fresh view; the handle cannot be released while
    /// any of them is alive.
    #[getter]
    fn payload<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        memoryview_of(slf.as_any())
    }

    /// True once the slot has been handed back to the ring.
    #[getter]
    fn released(&self) -> bool {
        self.handle.is_none()
    }

    /// Number of `memoryview`/`numpy` views currently borrowing the payload.
    #[getter]
    fn exports(&self) -> usize {
        self.exports.get()
    }

    /// Return the slot to the ring. Idempotent.
    ///
    /// Raises `BufferError` if any view into `payload` is still alive; the
    /// handle stays valid and can be released once they are gone.
    fn release(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.handle.is_none() {
            return Ok(());
        }
        let live = self.exports.get();
        if live != 0 {
            return Err(outstanding_exports("release", live));
        }

        let slot = &mut self.handle;
        self.reader.with_raw(py, |raw| {
            if let Some(handle) = slot.take() {
                raw.release(handle);
            }
        })
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (exc_type=None, exc_value=None, traceback=None))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        exc_value: Option<&Bound<'_, PyAny>>,
        traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let _ = (exc_type, exc_value, traceback);
        self.release(py)?;
        Ok(false)
    }

    /// # Safety
    ///
    /// Called by CPython with a valid `Py_buffer` to populate.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let me = slf.try_borrow()?;
        let Some(handle) = me.handle.as_ref() else {
            return Err(PyBufferError::new_err(
                "this handle has already been released",
            ));
        };
        let bytes = handle.as_bytes();
        // Safety: the slot is pinned against the writer for as long as the
        // handle holds it, and `slf` keeps the handle alive.
        unsafe {
            fill_view(
                view,
                flags,
                slf.clone().into_any(),
                bytes.as_ptr().cast_mut(),
                bytes.len(),
                me.layout,
                true,
            )?;
        }
        me.exports.set(me.exports.get() + 1);
        Ok(())
    }

    /// # Safety
    ///
    /// Called by CPython with a `Py_buffer` filled by `__getbuffer__`.
    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        unsafe { free_view(view) };
        self.exports.set(self.exports.get().saturating_sub(1));
    }
}

impl Drop for PyReadHandle {
    /// Best-effort release: a handle dropped without releasing would retire its
    /// ring slot permanently.
    ///
    /// Exports are necessarily zero here — each one holds a strong reference to
    /// this object, so the last export must have gone before it can be dropped.
    fn drop(&mut self) {
        if self.handle.is_none() {
            return;
        }
        Python::attach(|py| {
            let _ = self.release(py);
        });
    }
}

/// An exclusive, not-yet-published view of one ring slot (zero-copy write path).
///
/// Handed out by `BytesWriter.try_acquire()` and
/// `Float64Writer.try_acquire()`. The slot's `entry_seq` is `0` while the
/// handle is held, so readers skip it. Fill `payload` in place and call
/// `commit()`; an abandoned handle simply leaves the slot empty for the next
/// write.
///
/// Used as a context manager, the entry is committed on a clean exit and
/// abandoned if the block raises:
///
/// ```python
/// with writer.try_acquire() as handle:
///     numpy.asarray(handle.payload)[:] = joint_state
/// ```
///
/// A freshly acquired slot is **not** zeroed: it still holds the bytes of the
/// entry written `entry_count` writes ago. Fill every byte, or zero it first.
///
/// As on the read side, `commit()` raises `BufferError` while any view into
/// `payload` is still alive.
#[pyclass(name = "WriteHandle", module = "zerochannel", unsendable)]
pub(crate) struct PyWriteHandle {
    pub(crate) handle: Option<WriteHandle>,
    pub(crate) writer: HandleWriter,
    pub(crate) layout: Layout,
    pub(crate) exports: Cell<usize>,
}

#[pymethods]
impl PyWriteHandle {
    /// The entry payload as a writable `memoryview` over shared memory.
    ///
    /// Typed as `uint8` on a byte channel and `float64` on a `float64`
    /// channel, so `numpy.asarray(handle.payload)[:] = ...` fills the slot in
    /// place. Each access creates a fresh view; the handle cannot be committed
    /// while any of them is alive.
    #[getter]
    fn payload<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        memoryview_of(slf.as_any())
    }

    /// True once the entry has been published.
    #[getter]
    fn committed(&self) -> bool {
        self.handle.is_none()
    }

    /// Number of `memoryview`/`numpy` views currently borrowing the payload.
    #[getter]
    fn exports(&self) -> usize {
        self.exports.get()
    }

    /// Publish the entry and return its sequence number.
    ///
    /// Raises `BufferError` if any view into `payload` is still alive — writing
    /// through one after the entry is visible would corrupt a published entry.
    fn commit(&mut self, py: Python<'_>) -> PyResult<u64> {
        if self.handle.is_none() {
            return Err(PyValueError::new_err("handle has already been committed"));
        }
        let live = self.exports.get();
        if live != 0 {
            return Err(outstanding_exports("commit", live));
        }

        let slot = &mut self.handle;
        let mut entry_seq = 0;
        self.writer.with_raw(py, |raw| {
            if let Some(handle) = slot.take() {
                entry_seq = raw.commit(handle);
            }
        })?;
        Ok(entry_seq)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (exc_type=None, exc_value=None, traceback=None))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        exc_value: Option<&Bound<'_, PyAny>>,
        traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let _ = (exc_value, traceback);
        if exc_type.is_none() && self.handle.is_some() {
            self.commit(py)?;
        }
        Ok(false)
    }

    /// # Safety
    ///
    /// Called by CPython with a valid `Py_buffer` to populate.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let mut me = slf.try_borrow_mut()?;
        let layout = me.layout;
        let Some(handle) = me.handle.as_mut() else {
            return Err(PyBufferError::new_err(
                "this handle has already been committed",
            ));
        };
        let bytes = handle.as_bytes_mut();
        let (data, len) = (bytes.as_mut_ptr(), bytes.len());
        // Safety: the slot is exclusively owned until the handle is committed,
        // and `slf` keeps the handle alive.
        unsafe {
            fill_view(
                view,
                flags,
                slf.clone().into_any(),
                data,
                len,
                layout,
                false,
            )?;
        }
        me.exports.set(me.exports.get() + 1);
        Ok(())
    }

    /// # Safety
    ///
    /// Called by CPython with a `Py_buffer` filled by `__getbuffer__`.
    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        unsafe { free_view(view) };
        self.exports.set(self.exports.get().saturating_sub(1));
    }
}
