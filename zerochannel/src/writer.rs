//! Named, role-aware writer over a shared-memory segment.
//!
//! [`RawWriter`] pairs a `zerochannel_core::RawWriter` — which knows the ring
//! protocol but not the operating system — with the mapping it writes into,
//! the channel name that mapping came from, and the exclusive writer role that
//! keeps a second writer from silently shredding the sequence numbering.

use shared_memory::Shmem;
use zerochannel_core::{writer_owner, Dtype, WriteHandle, ZeroChannelError};

use crate::lifecycle::{claim_role, open_or_create, release_role, try_attach};

/// Single-writer end of a ZeroChannel ring buffer, operating on raw bytes.
///
/// This is the byte-level entry point: it owns the segment mapping, the writer
/// role and deferred connection. Typed access is layered on top by the
/// zero-cost [`Writer<T>`](crate::Writer) wrapper.
pub struct RawWriter {
    name: String,
    dtype: Dtype,
    delayed_connect: bool,
    /// The ring protocol, bound to `shmem` once a segment exists.
    inner: Option<zerochannel_core::RawWriter>,
    shmem: Option<Shmem>,
}

// Safety: the core writer holds a raw pointer into process-shared memory whose
// lifetime is governed by the `shmem` field alongside it, so moving the pair to
// another thread keeps the pointer valid. Nothing here is `Sync`: the ring
// protocol assumes one writer at a time.
unsafe impl Send for RawWriter {}

impl RawWriter {
    /// Create or open a shared-memory channel for writing.
    ///
    /// `payload_bytes` must be a non-zero multiple of 8.
    ///
    /// Claims the channel's writer role, which is exclusive: if a live process
    /// already holds it the call fails with
    /// [`ZeroChannelError::RoleConflict`]. A role left behind by a crashed
    /// writer is adopted.
    pub fn new(
        name: &str,
        payload_bytes: Option<usize>,
        entry_count: Option<usize>,
        dtype: Dtype,
        delayed_connect: bool,
    ) -> Result<Self, ZeroChannelError> {
        let mut this = Self {
            name: name.to_string(),
            dtype,
            delayed_connect,
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

    /// Claim the writer role on a freshly mapped segment and bind the ring
    /// protocol to it.
    fn bind(
        &mut self,
        shmem: Shmem,
        header: &zerochannel_core::Header,
    ) -> Result<(), ZeroChannelError> {
        let ptr = shmem.as_ptr();
        // SAFETY: the segment is mapped.
        claim_role(unsafe { writer_owner(ptr) }, "writer", &self.name)?;
        // SAFETY: `shmem` stays alive in `self.shmem` for as long as `inner`.
        self.inner = Some(unsafe { zerochannel_core::RawWriter::attach(ptr, header) });
        self.shmem = Some(shmem);
        Ok(())
    }

    /// Payload bytes per entry (0 while a deferred channel is unconnected).
    #[inline]
    pub fn payload_bytes(&self) -> usize {
        self.inner.as_ref().map_or(0, |w| w.payload_bytes())
    }

    /// Sequence number of the most recently committed entry (0 if none).
    ///
    /// This is the writer's high watermark, mirrored in the segment header.
    #[inline]
    pub fn last_entry_seq(&self) -> u64 {
        self.inner.as_ref().map_or(0, |w| w.last_entry_seq())
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

    /// Write one entry to the ring buffer (one-copy path).
    ///
    /// If the channel is deferred and still not connected, returns `Ok(())`
    /// (no-op).
    pub fn write(&mut self, data: &[u8]) -> Result<(), ZeroChannelError> {
        if !self.ensure_connected()? {
            return Ok(());
        }
        self.inner.as_mut().expect("connected").write(data)
    }

    /// Acquire the next ring slot for in-place construction (zero-copy path).
    ///
    /// Returns `Ok(None)` only when a deferred channel is still unconnected.
    /// The handle must be passed to [`RawWriter::commit`] to become visible.
    pub fn try_acquire(&mut self) -> Result<Option<WriteHandle>, ZeroChannelError> {
        if !self.ensure_connected()? {
            return Ok(None);
        }
        self.inner
            .as_mut()
            .expect("connected")
            .try_acquire()
            .map(Some)
    }

    /// Publish a held slot and return its sequence number.
    pub fn commit(&mut self, handle: WriteHandle) -> u64 {
        self.inner.as_mut().expect("connected").commit(handle)
    }
}

impl Drop for RawWriter {
    /// Hand the writer role back so a successor can take it immediately.
    ///
    /// Only a clean exit reaches this; a crash leaves the PID behind, which is
    /// exactly what the liveness check in `claim_role` is there to resolve.
    fn drop(&mut self) {
        if let Some(shmem) = &self.shmem {
            // SAFETY: the mapping is still alive — `shmem` is dropped after
            // this body runs.
            release_role(unsafe { writer_owner(shmem.as_ptr()) });
        }
    }
}
