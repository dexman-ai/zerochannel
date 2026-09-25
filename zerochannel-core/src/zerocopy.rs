//! Zero-copy slot handles.
//!
//! Both ends of the channel borrow a ring slot in place rather than copying
//! through it: [`WriteHandle`] is an exclusive, not-yet-published slot the
//! writer fills directly, and [`ReadHandle`] is a committed slot pinned against
//! overwrite while a reader looks at it. They share this module because they
//! are the same idea pointed in opposite directions — a borrowed view into
//! shared memory whose validity is governed by the slot's `entry_seq` word.

use alloc::format;
use core::mem::size_of;

use crate::error::ZeroChannelError;
use crate::metadata::Element;

// ── Write handle ────────────────────────────────────────────────────────────

/// An exclusive, not-yet-published view of one ring slot.
///
/// Handed out by [`RawWriter::try_acquire`](crate::RawWriter::try_acquire). The
/// slot's `entry_seq` has already been set to `0`, so readers skip it while it
/// is being filled. Write into [`WriteHandle::as_bytes_mut`] (or
/// [`WriteHandle::as_slice_mut`]) and pass the handle to
/// [`RawWriter::commit`](crate::RawWriter::commit) to publish it.
///
/// Dropping a handle without committing simply leaves the slot empty; the next
/// write reclaims it.
pub struct WriteHandle {
    pub(crate) index: usize,
    pub(crate) ptr: *mut u8,
    pub(crate) len: usize,
}

// Safety: the pointer targets process-shared memory owned by the writer that
// issued the handle.
unsafe impl Send for WriteHandle {}

impl WriteHandle {
    /// Ring index this handle refers to.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Mutable byte view of the held payload.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // Safety: the slot is exclusively owned for the lifetime of the handle.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Mutable typed view of the held payload.
    pub fn as_slice_mut<T: Element>(&mut self) -> Result<&mut [T], ZeroChannelError> {
        let elem = size_of::<T>();
        if elem == 0 || !self.len.is_multiple_of(elem) {
            return Err(ZeroChannelError::InvalidArgument(format!(
                "payload of {} bytes is not a whole number of {elem}-byte elements",
                self.len
            )));
        }
        // Safety: slot payloads start on an 8-byte boundary, so any `Element`
        // is naturally aligned, and the slot is exclusively owned.
        Ok(unsafe { core::slice::from_raw_parts_mut(self.ptr.cast::<T>(), self.len / elem) })
    }
}

// ── Read handle ─────────────────────────────────────────────────────────────

/// A borrowed, in-place view of one committed ring slot.
///
/// Handed out by [`RawReader::try_acquire`](crate::RawReader::try_acquire),
/// which flips the slot's `entry_seq` to `-entry_seq` so the writer skips it.
/// The view stays valid until the handle is returned to
/// [`RawReader::release`](crate::RawReader::release) — which the holder
/// **must** do, or the slot is retired from the ring for good.
pub struct ReadHandle {
    pub(crate) index: usize,
    pub(crate) entry_seq: u64,
    pub(crate) ptr: *const u8,
    pub(crate) len: usize,
}

// Safety: the pointer targets process-shared memory owned by the reader that
// issued the handle.
unsafe impl Send for ReadHandle {}

impl ReadHandle {
    /// Sequence number of the held entry.
    #[inline]
    pub fn entry_seq(&self) -> u64 {
        self.entry_seq
    }

    /// Ring index this handle refers to.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Byte view of the held payload.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        // Safety: the slot is pinned against overwrite for the handle's lifetime.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Typed view of the held payload.
    pub fn as_slice<T: Element>(&self) -> Result<&[T], ZeroChannelError> {
        let elem = size_of::<T>();
        if elem == 0 || !self.len.is_multiple_of(elem) {
            return Err(ZeroChannelError::InvalidArgument(format!(
                "payload of {} bytes is not a whole number of {elem}-byte elements",
                self.len
            )));
        }
        // Safety: slot payloads start on an 8-byte boundary, so any `Element`
        // is naturally aligned, and the slot is pinned against overwrite.
        Ok(unsafe { core::slice::from_raw_parts(self.ptr.cast::<T>(), self.len / elem) })
    }
}
