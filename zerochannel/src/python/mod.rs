//! PyO3 bindings — the `zerochannel` native extension module.
//!
//! Exposes the ZeroChannel classes as `zerochannel.BytesWriter`,
//! `BytesReader`, `Float64Writer`, `Float64Reader`, and the zero-copy
//! `ReadHandle` / `WriteHandle`.
//!
//! The split mirrors the Rust API: [`writer`] and [`reader`] hold the channel
//! endpoints, [`zerocopy`] holds the borrowed handles and the buffer-protocol
//! machinery that keeps a ring slot alive while Python still points into it,
//! and [`errors`] maps failures onto Python exceptions. This module owns only
//! the registration that assembles them into one extension module.

mod errors;
mod reader;
mod writer;
mod zerocopy;

use errors::{ChannelRoleConflict, IntoPyResult};
use pyo3::prelude::*;
use reader::{PyBytesReader, PyFloat64Reader};
use writer::{PyBytesWriter, PyFloat64Writer};
use zerocopy::{PyReadHandle, PyWriteHandle};

/// Bytes per `float64` element.
const F64_BYTES: usize = 8;

// ── Module registration ─────────────────────────────────────────────────────

/// Remove a channel's shared-memory segment by name.
///
/// The escape hatch for reclaiming a leaked segment: a channel abandoned by a
/// crashed process keeps its name until something removes it. A later writer
/// adopts the stale role claim and carries on, so this is only needed when the
/// segment itself must go — supervisors reclaiming a name, and test fixtures
/// that must not leak state between runs.
///
/// Processes already attached keep their mapping; only the name goes away, so
/// the next open creates a fresh segment. Does nothing if the segment is
/// absent.
#[pyfunction]
fn unlink(name: &str) -> PyResult<()> {
    crate::unlink(name).or_py()
}

/// The `zerochannel` native extension module.
///
/// The name here is load-bearing: PyO3 emits `PyInit_zerochannel` from it, and
/// that must match the final component of `module-name` in pyproject.toml.
#[pymodule]
fn zerochannel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBytesWriter>()?;
    m.add_class::<PyBytesReader>()?;
    m.add_class::<PyFloat64Writer>()?;
    m.add_class::<PyFloat64Reader>()?;
    m.add_class::<PyReadHandle>()?;
    m.add_class::<PyWriteHandle>()?;
    m.add_function(wrap_pyfunction!(unlink, m)?)?;
    m.add(
        "ChannelRoleConflict",
        m.py().get_type::<ChannelRoleConflict>(),
    )?;
    Ok(())
}
