//! Named, role-aware reader over a shared-memory segment.
//!
//! [`RawReader`] pairs a `zerochannel_core::RawReader` — which knows the ring
//! protocol but not the operating system — with the mapping it reads from, the
//! channel name that mapping came from, and (for zero-copy readers) the
//! exclusive role that keeps two pinning readers from silently stealing
//! entries from each other.

use shared_memory::Shmem;
use zerochannel_core::{zc_reader_owner, Dtype, Element, ReadHandle, ZeroChannelError};

use crate::lifecycle::{claim_role, open_or_create, release_role, try_attach};

/// Reader end of a ZeroChannel ring buffer, operating on raw bytes.
///
/// A read returns **at most one** entry: the oldest committed entry whose
/// `entry_seq` is greater than the watermark. The watermark defaults to the
/// sequence number of the last entry this reader returned, and may be
/// overridden per call by callers that track their own position.
///
/// The one-copy path ([`RawReader::read_as`]) supports many concurrent readers.
/// The zero-copy path ([`RawReader::try_acquire`]) is single-reader only, and
/// must be opted into at construction so that exclusivity is claimed only by
/// readers that actually intend to use it.
pub struct RawReader {
    name: String,
    dtype: Dtype,
    delayed_connect: bool,
    enable_zero_copy: bool,
    /// The ring protocol, bound to `shmem` once a segment exists.
    inner: Option<zerochannel_core::RawReader>,
    shmem: Option<Shmem>,
}

// Safety: the core reader holds a raw pointer into process-shared memory whose
// lifetime is governed by the `shmem` field alongside it, so moving the pair to
// another thread keeps the pointer valid. Nothing here is `Sync`: a reader owns
// a mutable watermark.
unsafe impl Send for RawReader {}

impl RawReader {
    /// Create or open a shared-memory channel for reading.
    ///
    /// Supplying `payload_bytes`/`entry_count` lets the reader create the
    /// segment if it does not exist yet; omitting them — the usual case —
    /// attaches to whatever the writer published.
    ///
    /// The returned reader uses the one-copy path only. For zero-copy reads
    /// see [`RawReader::new_zero_copy`].
    pub fn new(
        name: &str,
        payload_bytes: Option<usize>,
        entry_count: Option<usize>,
        dtype: Dtype,
        delayed_connect: bool,
    ) -> Result<Self, ZeroChannelError> {
        Self::open(
            name,
            payload_bytes,
            entry_count,
            dtype,
            delayed_connect,
            false,
        )
    }

    /// Create or open a shared-memory channel for zero-copy reading.
    ///
    /// Claims the channel's exclusive zero-copy reader role, without which
    /// [`RawReader::try_acquire`] refuses to run. Acquiring a handle pins a
    /// slot and hides it from every other reader, so two readers doing it at
    /// once silently lose entries with no gap in the sequence numbers to show
    /// for it. Fails with [`ZeroChannelError::RoleConflict`] when a live
    /// process already holds the role; a role left by a crashed reader is
    /// adopted.
    ///
    /// One-copy readers are unaffected by this claim and remain unlimited.
    pub fn new_zero_copy(
        name: &str,
        payload_bytes: Option<usize>,
        entry_count: Option<usize>,
        dtype: Dtype,
        delayed_connect: bool,
    ) -> Result<Self, ZeroChannelError> {
        Self::open(
            name,
            payload_bytes,
            entry_count,
            dtype,
            delayed_connect,
            true,
        )
    }

    fn open(
        name: &str,
        payload_bytes: Option<usize>,
        entry_count: Option<usize>,
        dtype: Dtype,
        delayed_connect: bool,
        enable_zero_copy: bool,
    ) -> Result<Self, ZeroChannelError> {
        let mut this = Self {
            name: name.to_string(),
            dtype,
            delayed_connect,
            enable_zero_copy,
            inner: None,
            shmem: None,
        };
        if let Some((shmem, header)) =
            open_or_create(name, payload_bytes, entry_count, dtype, delayed_connect)?
        {
            this.bind(shmem, &header)?;
        }
        Ok(this)
    }

    /// Claim the zero-copy role if requested, then bind the ring protocol to a
    /// freshly mapped segment.
    fn bind(
        &mut self,
        shmem: Shmem,
        header: &zerochannel_core::Header,
    ) -> Result<(), ZeroChannelError> {
        let ptr = shmem.as_ptr();
        if self.enable_zero_copy {
            // SAFETY: the segment is mapped.
            claim_role(
                unsafe { zc_reader_owner(ptr) },
                "zero-copy reader",
                &self.name,
            )?;
        }
        // SAFETY: `shmem` stays alive in `self.shmem` for as long as `inner`.
        self.inner = Some(unsafe { zerochannel_core::RawReader::attach(ptr, header) });
        self.shmem = Some(shmem);
        Ok(())
    }

    /// Base address of the mapped segment, or null while unconnected.
    ///
    /// Only the tests need this: they reach past the public API to forge
    /// header state that a real crash would leave behind.
    #[cfg(test)]
    #[inline]
    pub(crate) fn ptr(&self) -> *mut u8 {
        match &self.shmem {
            Some(shmem) => shmem.as_ptr(),
            None => std::ptr::null_mut(),
        }
    }

    /// Payload bytes per entry (0 while a deferred channel is unconnected).
    #[inline]
    pub fn payload_bytes(&self) -> usize {
        self.inner.as_ref().map_or(0, |r| r.payload_bytes())
    }

    /// Sequence number of the last entry this reader returned (0 if none).
    #[inline]
    pub fn last_entry_seq(&self) -> u64 {
        self.inner.as_ref().map_or(0, |r| r.last_entry_seq())
    }

    /// The writer's high watermark, read straight from the header.
    #[inline]
    pub fn writer_entry_seq(&self) -> u64 {
        self.inner.as_ref().map_or(0, |r| r.writer_entry_seq())
    }

    /// Attach a deferred channel if the segment now exists.
    ///
    /// Returns `false` when the segment is still missing and the channel was
    /// opened in delayed mode. Callers that need `payload_bytes` before issuing
    /// a read use this to resolve the geometry first.
    pub fn connect(&mut self) -> Result<bool, ZeroChannelError> {
        self.ensure_connected()
    }

    /// Attach a deferred channel if the segment now exists.
    ///
    /// Returns `Ok(false)` when the segment is still missing and the channel
    /// was opened in delayed mode.
    fn ensure_connected(&mut self) -> Result<bool, ZeroChannelError> {
        if self.inner.is_some() {
            return Ok(true);
        }
        match try_attach(&self.name, self.dtype)? {
            Some((shmem, header)) => {
                self.bind(shmem, &header)?;
                Ok(true)
            }
            None if self.delayed_connect => Ok(false),
            None => Err(ZeroChannelError::OsError(
                "shared memory segment not available".into(),
            )),
        }
    }

    /// Copy the next entry into `dst` (one-copy path), as raw bytes.
    ///
    /// Returns the entry's sequence number, or `None` when nothing newer than
    /// the watermark is available.
    pub fn read_into(
        &mut self,
        dst: &mut [u8],
        from_entry_seq: Option<u64>,
    ) -> Result<Option<u64>, ZeroChannelError> {
        if !self.ensure_connected()? {
            return Ok(None);
        }
        self.inner
            .as_mut()
            .expect("connected")
            .read_into(dst, from_entry_seq)
    }

    /// Copy the next entry into `dst` (one-copy path), as elements of `T`.
    pub fn read_into_as<T: Element>(
        &mut self,
        dst: &mut [T],
        from_entry_seq: Option<u64>,
    ) -> Result<Option<u64>, ZeroChannelError> {
        if !self.ensure_connected()? {
            return Ok(None);
        }
        self.inner
            .as_mut()
            .expect("connected")
            .read_into_as(dst, from_entry_seq)
    }

    /// Read the next entry (one-copy path), allocating the destination.
    pub fn read(
        &mut self,
        from_entry_seq: Option<u64>,
    ) -> Result<Option<(u64, Vec<u8>)>, ZeroChannelError> {
        self.read_as::<u8>(from_entry_seq)
    }

    /// Read the next entry (one-copy path) as elements of `T`, allocating the
    /// destination.
    pub fn read_as<T: Element>(
        &mut self,
        from_entry_seq: Option<u64>,
    ) -> Result<Option<(u64, Vec<T>)>, ZeroChannelError> {
        if !self.ensure_connected()? {
            return Ok(None);
        }
        self.inner
            .as_mut()
            .expect("connected")
            .read_as::<T>(from_entry_seq)
    }

    /// Acquire the next entry in place (zero-copy path).
    ///
    /// Flips the slot's `entry_seq` from `s` to `-s` with a compare-and-swap so
    /// the writer skips the slot until [`RawReader::release`] hands it back.
    ///
    /// Requires the reader to have been constructed with `enable_zero_copy`,
    /// and a target with lock-free 64-bit atomics (x86_64 and aarch64 both
    /// qualify).
    pub fn try_acquire(
        &mut self,
        from_entry_seq: Option<u64>,
    ) -> Result<Option<ReadHandle>, ZeroChannelError> {
        if !self.enable_zero_copy {
            return Err(ZeroChannelError::InvalidArgument(
                "zero-copy reads require a reader built with new_zero_copy".into(),
            ));
        }
        if !self.ensure_connected()? {
            return Ok(None);
        }
        self.inner
            .as_mut()
            .expect("connected")
            .try_acquire(from_entry_seq)
    }

    /// Return a held slot to the ring, restoring `entry_seq` to `+entry_seq`.
    pub fn release(&mut self, handle: ReadHandle) {
        if let Some(inner) = self.inner.as_mut() {
            inner.release(handle);
        }
    }
}

impl Drop for RawReader {
    /// Hand the zero-copy reader role back, if this reader held it.
    fn drop(&mut self) {
        if !self.enable_zero_copy {
            return;
        }
        if let Some(shmem) = &self.shmem {
            // SAFETY: the mapping is still alive — `shmem` is dropped after
            // this body runs.
            release_role(unsafe { zc_reader_owner(shmem.as_ptr()) });
        }
    }
}
