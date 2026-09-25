//! The single-writer end of the ring protocol, over a mapped segment.
//!
//! [`RawWriter`] owns the `entry_seq` protocol, ring index bookkeeping and
//! writer-restart recovery. It borrows a segment someone else mapped: naming,
//! creation and the exclusive writer role are all above this layer.

use alloc::format;
use core::sync::atomic::{fence, Ordering};

use crate::error::ZeroChannelError;
use crate::metadata::{high_watermark, payload_offset, seq_cell, Header, OFF_WRITE_INDEX};
use crate::zerocopy::WriteHandle;

/// Single-writer end of a ZeroChannel ring buffer, operating on raw bytes.
pub struct RawWriter {
    ptr: *mut u8,
    payload_bytes: usize,
    entry_count: usize,
    current_index: usize,
    last_entry_seq: u64,
}

// Safety: the raw pointer targets process-shared memory whose lifetime is
// governed by whoever mapped it.
unsafe impl Send for RawWriter {}

impl RawWriter {
    /// Bind a writer to an already-mapped segment described by `header`.
    ///
    /// Resume state is recovered from the segment, so a writer that replaces a
    /// crashed predecessor picks up its sequence numbering rather than
    /// restarting it.
    ///
    /// # Safety
    ///
    /// `ptr` must point at a mapped ZeroChannel segment whose geometry matches
    /// `header`, and must stay mapped for as long as the returned writer lives.
    pub unsafe fn attach(ptr: *mut u8, header: &Header) -> Self {
        let (last_entry_seq, current_index) = recover_state(
            ptr,
            header.payload_bytes,
            header.entry_count,
            header.write_index,
            header.last_entry_seq,
        );
        Self {
            ptr,
            payload_bytes: header.payload_bytes,
            entry_count: header.entry_count,
            current_index,
            last_entry_seq,
        }
    }

    /// Payload bytes per entry.
    #[inline]
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }

    /// Number of ring slots.
    #[inline]
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Sequence number of the most recently committed entry (0 if none).
    ///
    /// This is the writer's high watermark, mirrored in the segment header.
    #[inline]
    pub fn last_entry_seq(&self) -> u64 {
        self.last_entry_seq
    }

    /// Claim the next slot that is not held by a reader and mark it
    /// in-progress (`entry_seq = 0`).
    fn claim_slot(&mut self) -> Result<usize, ZeroChannelError> {
        for n in 0..self.entry_count {
            let index = (self.current_index + n) % self.entry_count;
            // Safety: `index < entry_count` and the segment is mapped.
            let cell = unsafe { seq_cell(self.ptr, self.payload_bytes, index) };

            // A negative `entry_seq` means a reader holds a zero-copy handle on
            // the slot; skip it rather than corrupting the view it is reading.
            if cell.load(Ordering::Acquire) < 0 {
                continue;
            }

            cell.store(0, Ordering::Relaxed);
            // Zeroed entry_seq is visible before the payload is mutated.
            fence(Ordering::Release);
            return Ok(index);
        }
        Err(ZeroChannelError::Busy(format!(
            "all {} ring slots are held by a reader",
            self.entry_count
        )))
    }

    /// Publish slot `index` under the next sequence number and advance the ring.
    fn publish(&mut self, index: usize) -> u64 {
        // Payload writes are visible before the new entry_seq.
        fence(Ordering::Release);

        let seq = self.last_entry_seq + 1;
        // Safety: `index < entry_count` and the segment is mapped.
        unsafe { seq_cell(self.ptr, self.payload_bytes, index) }
            .store(seq as i64, Ordering::Relaxed);
        self.last_entry_seq = seq;

        self.current_index = (index + 1) % self.entry_count;
        unsafe {
            core::ptr::write(
                self.ptr.add(OFF_WRITE_INDEX) as *mut u64,
                self.current_index as u64,
            );
            // Advertise the watermark only after the slot is committed, so a
            // reader never sees a promise it cannot fulfil.
            high_watermark(self.ptr).store(seq, Ordering::Release);
        }
        seq
    }

    /// Write one entry to the ring buffer (one-copy path).
    pub fn write(&mut self, data: &[u8]) -> Result<(), ZeroChannelError> {
        if data.len() != self.payload_bytes {
            return Err(ZeroChannelError::InvalidArgument(format!(
                "expected {} payload bytes, got {}",
                self.payload_bytes,
                data.len()
            )));
        }

        let index = self.claim_slot()?;
        // Safety: `index` is in range and the slot is exclusively ours.
        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.ptr.add(payload_offset(self.payload_bytes, index)),
                self.payload_bytes,
            );
        }
        self.publish(index);
        Ok(())
    }

    /// Acquire the next ring slot for in-place construction (zero-copy path).
    ///
    /// The handle must be passed to [`RawWriter::commit`] to become visible.
    pub fn try_acquire(&mut self) -> Result<WriteHandle, ZeroChannelError> {
        let index = self.claim_slot()?;
        Ok(WriteHandle {
            index,
            // Safety: `index` is in range and the segment is mapped.
            ptr: unsafe { self.ptr.add(payload_offset(self.payload_bytes, index)) },
            len: self.payload_bytes,
        })
    }

    /// Publish a held slot and return its sequence number.
    pub fn commit(&mut self, handle: WriteHandle) -> u64 {
        self.publish(handle.index)
    }
}

/// Recover the writer's resume state from an existing segment.
///
/// `last_entry_seq` in the header is the authoritative high watermark, but a
/// writer can die between committing a slot and persisting the header, so the
/// ring itself is scanned as well and the larger of the two wins.
///
/// A crash between committing `entry_seq` and persisting `current_write_index`
/// also leaves the header index pointing *at* the last committed slot instead
/// of past it. That is detected by comparing the slot's sequence number with
/// the recovered watermark.
///
/// Returns `(last_entry_seq, corrected_write_index)`.
///
/// # Safety
///
/// `ptr` must point at a mapped ZeroChannel segment with the given geometry.
unsafe fn recover_state(
    ptr: *mut u8,
    payload_bytes: usize,
    entry_count: usize,
    header_write_index: usize,
    header_last_entry_seq: u64,
) -> (u64, usize) {
    let mut max_seq = header_last_entry_seq;
    for i in 0..entry_count {
        let raw = seq_cell(ptr, payload_bytes, i).load(Ordering::Acquire);
        let seq = raw.unsigned_abs();
        if seq > max_seq {
            max_seq = seq;
        }
    }

    let write_index = header_write_index % entry_count.max(1);
    let at_index = seq_cell(ptr, payload_bytes, write_index)
        .load(Ordering::Acquire)
        .unsigned_abs();

    // The slot the header points at already holds the newest entry: the
    // previous writer committed but never advanced the index.
    let corrected_index = if at_index != 0 && at_index == max_seq {
        (write_index + 1) % entry_count
    } else {
        write_index
    };

    (max_seq, corrected_index)
}
