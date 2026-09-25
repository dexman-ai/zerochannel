//! Errors returned by ZeroChannel operations.

use alloc::string::String;
use core::fmt;

/// Errors returned by ZeroChannel operations.
#[derive(Debug)]
pub enum ZeroChannelError {
    /// Invalid parameters (zero entry_length, zero entry_count, size mismatch).
    InvalidArgument(String),
    /// OS shared-memory error (creation, opening, mapping).
    OsError(String),
    /// No slot is currently available — every ring slot is held by a reader.
    Busy(String),
    /// An exclusive role on the channel is already held by a live process.
    ///
    /// Distinct from [`ZeroChannelError::Busy`]: `Busy` is transient and worth
    /// retrying, whereas a role conflict persists until the holder exits.
    RoleConflict(String),
}

impl fmt::Display for ZeroChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZeroChannelError::InvalidArgument(msg) => write!(f, "{}", msg),
            ZeroChannelError::OsError(msg) => write!(f, "{}", msg),
            ZeroChannelError::Busy(msg) => write!(f, "{}", msg),
            ZeroChannelError::RoleConflict(msg) => write!(f, "{}", msg),
        }
    }
}

impl core::error::Error for ZeroChannelError {}
