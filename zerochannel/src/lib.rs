//! ZeroChannel — lock-free Single-Writer Multi-Reader IPC over shared memory.
//!
//! The core abstractions are the generic [`Writer<T>`] and [`Reader<T>`], with
//! concrete aliases per payload type ([`BytesWriter`], [`Float64Writer`], …),
//! for high-performance inter-process communication using a circular buffer in
//! POSIX/Windows shared memory.
//!
//! This crate is the OS-facing half of ZeroChannel: it turns a channel *name*
//! into a mapped segment, arbitrates segment ownership and the exclusive
//! writer / zero-copy reader roles using PID liveness, and exposes the Python
//! bindings. The ring protocol itself — header layout, `entry_seq` sequencing,
//! the one-copy and zero-copy access paths — lives in the dependency-free
//! `no_std` [`zerochannel_core`] crate and is re-exported here.
//!
//! Each ring slot carries a signed 64-bit **`entry_seq`** control word:
//!
//! | Value | Meaning |
//! |---|---|
//! | `0` | empty, or a writer is mid-write |
//! | `> 0` | committed entry with sequence number `entry_seq` |
//! | `< 0` | committed entry `-entry_seq` currently **held** by a reader |
//!
//! Sequence numbers are assigned by the writer and increase strictly by one.
//! The writer's high watermark is mirrored in the header so a reader can tell
//! whether anything newer than its own watermark exists without scanning the
//! ring.
//!
//! Two access paths are provided:
//!
//! * **One-copy** ([`RawReader::read_as`]) — copies the payload out and
//!   validates it with a double read of `entry_seq`. Uses plain atomic loads
//!   and stores only, so it works on any target.
//! * **Zero-copy** ([`RawReader::try_acquire`] / [`RawWriter::try_acquire`]) —
//!   hands out a borrowed view into shared memory. Acquiring a read handle flips
//!   `entry_seq` to `-entry_seq` with a compare-and-swap, so this path requires
//!   a target with lock-free 64-bit atomics (x86_64 and aarch64 both qualify)
//!   and is **single-reader** only.
//!
//! The `python` feature adds the PyO3 bindings that back the `zerochannel`
//! extension module. It is off by default so a Rust-only consumer can use the
//! channel without pulling PyO3 and numpy into its build graph.

mod lifecycle;
mod reader;
mod writer;

#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod tests;

use std::marker::PhantomData;
use std::mem::size_of;

pub use lifecycle::unlink;
pub use reader::RawReader;
pub use writer::RawWriter;
pub use zerochannel_core::{Dtype, Element, ReadHandle, WriteHandle, ZeroChannelError};

// ── Typed wrappers ──────────────────────────────────────────────────────────

/// Typed single-writer end of a ZeroChannel ring buffer.
///
/// A zero-cost wrapper over [`RawWriter`]: sizes are expressed in elements of
/// `T` rather than bytes, and the channel header records `T`'s type tag so a
/// mismatched peer fails loudly instead of reinterpreting bits.
pub struct Writer<T: Element> {
    raw: RawWriter,
    _marker: PhantomData<T>,
}

impl<T: Element> Writer<T> {
    /// Create or open a shared-memory channel for writing `entry_length`
    /// elements of `T` per entry.
    pub fn new(
        name: &str,
        entry_length: Option<usize>,
        entry_count: Option<usize>,
        delayed_connect: bool,
    ) -> Result<Self, ZeroChannelError> {
        let payload_bytes = entry_length.map(|n| n.saturating_mul(size_of::<T>()));
        Ok(Self {
            raw: RawWriter::new(name, payload_bytes, entry_count, T::DTYPE, delayed_connect)?,
            _marker: PhantomData,
        })
    }

    /// Number of `T` elements per entry (0 if deferred and unknown).
    #[inline]
    pub fn entry_length(&self) -> usize {
        self.raw.payload_bytes() / size_of::<T>()
    }

    /// Sequence number of the most recently committed entry (0 if none).
    #[inline]
    pub fn last_entry_seq(&self) -> u64 {
        self.raw.last_entry_seq()
    }

    /// Write one entry to the ring buffer (one-copy path).
    pub fn write(&mut self, data: &[T]) -> Result<(), ZeroChannelError> {
        // Safety: `Element` guarantees `T` has no padding, so the value slice
        // has an equivalent byte representation of exactly `size_of_val` bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        self.raw.write(bytes)
    }

    /// Acquire the next ring slot for in-place construction (zero-copy path).
    pub fn try_acquire(&mut self) -> Result<Option<WriteHandle>, ZeroChannelError> {
        self.raw.try_acquire()
    }

    /// Publish a held slot and return its sequence number.
    pub fn commit(&mut self, handle: WriteHandle) -> u64 {
        self.raw.commit(handle)
    }
}

/// Typed reader end of a ZeroChannel ring buffer.
pub struct Reader<T: Element> {
    raw: RawReader,
    _marker: PhantomData<T>,
}

impl<T: Element> Reader<T> {
    /// Create or open a shared-memory channel for reading `entry_length`
    /// elements of `T` per entry.
    ///
    /// Uses the one-copy path only; for zero-copy reads see
    /// [`Reader::new_zero_copy`].
    pub fn new(
        name: &str,
        entry_length: Option<usize>,
        entry_count: Option<usize>,
        delayed_connect: bool,
    ) -> Result<Self, ZeroChannelError> {
        let payload_bytes = entry_length.map(|n| n.saturating_mul(size_of::<T>()));
        Ok(Self {
            raw: RawReader::new(name, payload_bytes, entry_count, T::DTYPE, delayed_connect)?,
            _marker: PhantomData,
        })
    }

    /// Create or open a shared-memory channel for zero-copy reading.
    ///
    /// Claims the channel's exclusive zero-copy reader role, which
    /// [`Reader::try_acquire`] requires.
    pub fn new_zero_copy(
        name: &str,
        entry_length: Option<usize>,
        entry_count: Option<usize>,
        delayed_connect: bool,
    ) -> Result<Self, ZeroChannelError> {
        let payload_bytes = entry_length.map(|n| n.saturating_mul(size_of::<T>()));
        Ok(Self {
            raw: RawReader::new_zero_copy(
                name,
                payload_bytes,
                entry_count,
                T::DTYPE,
                delayed_connect,
            )?,
            _marker: PhantomData,
        })
    }

    /// Number of `T` elements per entry (0 if deferred and unknown).
    #[inline]
    pub fn entry_length(&self) -> usize {
        self.raw.payload_bytes() / size_of::<T>()
    }

    /// Sequence number of the last entry this reader returned (0 if none).
    #[inline]
    pub fn last_entry_seq(&self) -> u64 {
        self.raw.last_entry_seq()
    }

    /// The writer's high watermark.
    #[inline]
    pub fn writer_entry_seq(&self) -> u64 {
        self.raw.writer_entry_seq()
    }

    /// Read the next entry newer than the watermark (one-copy path).
    pub fn read(
        &mut self,
        from_entry_seq: Option<u64>,
    ) -> Result<Option<(u64, Vec<T>)>, ZeroChannelError> {
        self.raw.read_as::<T>(from_entry_seq)
    }

    /// Read the next entry newer than the watermark into a caller-owned buffer.
    pub fn read_into(
        &mut self,
        dst: &mut [T],
        from_entry_seq: Option<u64>,
    ) -> Result<Option<u64>, ZeroChannelError> {
        self.raw.read_into_as::<T>(dst, from_entry_seq)
    }

    /// Acquire the next entry in place (zero-copy path).
    ///
    /// Requires the reader to have been built with [`Reader::new_zero_copy`].
    pub fn try_acquire(
        &mut self,
        from_entry_seq: Option<u64>,
    ) -> Result<Option<ReadHandle>, ZeroChannelError> {
        self.raw.try_acquire(from_entry_seq)
    }

    /// Return a held slot to the ring.
    pub fn release(&mut self, handle: ReadHandle) {
        self.raw.release(handle)
    }
}

/// Byte-oriented channel writer (`entry_length` is measured in bytes).
pub type BytesWriter = Writer<u8>;
/// Byte-oriented channel reader (`entry_length` is measured in bytes).
pub type BytesReader = Reader<u8>;
/// `f64` channel writer (`entry_length` is measured in elements).
pub type Float64Writer = Writer<f64>;
/// `f64` channel reader (`entry_length` is measured in elements).
pub type Float64Reader = Reader<f64>;
