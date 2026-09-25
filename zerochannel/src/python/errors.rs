//! Mapping ZeroChannel failures onto Python exceptions.
//!
//! Each `ZeroChannelError` variant has one natural Python counterpart, and the
//! distinction that matters to callers is whether retrying can help:
//! `BlockingIOError` is transient, `ChannelRoleConflict` is not.

use crate::ZeroChannelError;
use numpy::{PyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::{PyBlockingIOError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

// ── Error conversion ──────────────────────────────────────────────────────

pyo3::create_exception!(
    zerochannel,
    ChannelRoleConflict,
    PyRuntimeError,
    "An exclusive channel role is already held by a live process.\n\n\
     Raised when a second writer, or a second `enable_zero_copy` reader, tries \
     to attach to a channel that already has one. Deliberately not a subclass \
     of `BlockingIOError`: a role conflict is not transient and retrying will \
     not clear it."
);

/// Translate a channel failure into the Python exception that matches it.
///
/// This cannot be a `From` impl: both `ZeroChannelError` and `PyErr` are
/// foreign to this crate, so the orphan rule rules it out. The conversion is
/// therefore spelled out at each boundary with [`IntoPyResult::or_py`].
fn to_py_err(e: ZeroChannelError) -> PyErr {
    match e {
        ZeroChannelError::InvalidArgument(msg) => PyValueError::new_err(msg),
        ZeroChannelError::OsError(msg) => PyOSError::new_err(msg),
        ZeroChannelError::Busy(msg) => PyBlockingIOError::new_err(msg),
        ZeroChannelError::RoleConflict(msg) => ChannelRoleConflict::new_err(msg),
    }
}

/// Carries a channel result into Python's error space.
pub(crate) trait IntoPyResult<T> {
    /// Convert a channel failure into the matching Python exception.
    fn or_py(self) -> PyResult<T>;
}

impl<T> IntoPyResult<T> for Result<T, ZeroChannelError> {
    #[inline]
    fn or_py(self) -> PyResult<T> {
        self.map_err(to_py_err)
    }
}

/// Validate a caller-supplied `out` array: 1-D, contiguous and exactly
/// `expected` elements long.
pub(crate) fn check_out<T: numpy::Element>(
    arr: &Bound<'_, PyArray1<T>>,
    expected: usize,
) -> PyResult<()> {
    if !arr.is_contiguous() {
        return Err(PyValueError::new_err("out must be a contiguous array"));
    }
    if arr.len() != expected {
        return Err(PyValueError::new_err(format!(
            "out must hold exactly {expected} elements, got {}",
            arr.len()
        )));
    }
    Ok(())
}
