//! The reader end of the ring protocol, over a mapped segment.
//!
//! [`RawReader`] owns slot location, the torn-read double check and the
//! zero-copy compare-and-swap. It borrows a segment someone else mapped:
//! naming, creation and the exclusive zero-copy reader role are all above this
//! layer.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::size_of;
use core::sync::atomic::{fence, AtomicI64, Ordering};

use crate::error::ZeroChannelError;
use crate::metadata::{high_watermark, payload_offset, seq_cell, Element, Header, READ_RETRIES};
use crate::zerocopy::ReadHandle;

/// Reader end of a ZeroChannel ring buffer, operating on raw bytes.
///
/// A read returns **at most one** entry: the oldest committed entry whose
/// `entry_seq` is greater than the watermark. The watermark defaults to the
/// sequence number of the last entry this reader returned, and may be
/// overridden per call by callers that track their own position.
///
/// The one-copy path ([`RawReader::read_as`]) supports many concurrent
/// readers. The zero-copy path ([`RawReader::try_acquire`]) is single-reader
/// only; enforcing that is the caller's job.
pub struct RawReader {
    ptr: *mut u8,
    payload_bytes: usize,
    entry_count: usize,
    last_entry_seq: u64,
    last_read_idx: usize,
}

// Safety: the raw pointer targets process-shared memory whose lifetime is
// governed by whoever mapped it.
unsafe impl Send for RawReader {}

impl RawReader {
    /// Bind a reader to an already-mapped segment described by `header`.
    ///
    /// # Safety
    ///
    /// `ptr` must point at a mapped ZeroChannel segment whose geometry matches
    /// `header`, and must stay mapped for as long as the returned reader lives.
    pub unsafe fn attach(ptr: *mut u8, header: &Header) -> Self {
        Self {
            ptr,
            payload_bytes: header.payload_bytes,
            entry_count: header.entry_count,
            last_entry_seq: 0,
            last_read_idx: 0,
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

    /// Sequence number of the last entry this reader returned (0 if none).
    #[inline]
    pub fn last_entry_seq(&self) -> u64 {
        self.last_entry_seq
    }

    /// The writer's high watermark, read straight from the header.
    #[inline]
    pub fn writer_entry_seq(&self) -> u64 {
        // Safety: the segment is mapped for this reader's lifetime.
        unsafe { high_watermark(self.ptr) }.load(Ordering::Acquire)
    }

    /// Borrow the `entry_seq` control word of ring slot `idx`.
    #[inline]
    fn seq_at(&self, idx: usize) -> &AtomicI64 {
        // Safety: callers keep `idx < entry_count` and the segment is mapped.
        unsafe { seq_cell(self.ptr, self.payload_bytes, idx) }
    }

    /// Locate the oldest committed entry with `entry_seq > watermark`.
    ///
    /// The scan cannot stop at the first committed slot it meets: that slot may
    /// hold an arbitrarily old entry, because the writer skips slots held by a
    /// reader and because the ring wraps. Empty or mid-write slots
    /// (`entry_seq == 0`) and reader-held slots (`entry_seq < 0`) are skipped.
    fn locate(&self, watermark: u64) -> Option<(usize, u64)> {
        // Fast path: the writer normally lands in the slot right after the one
        // we read last, carrying the very next sequence number.
        let next_idx = (self.last_read_idx + 1) % self.entry_count;
        let next = self.seq_at(next_idx).load(Ordering::Acquire);
        if next > 0 && next as u64 == watermark.saturating_add(1) {
            return Some((next_idx, next as u64));
        }

        let mut best: Option<(usize, u64)> = None;
        for idx in 0..self.entry_count {
            let raw = self.seq_at(idx).load(Ordering::Acquire);
            if raw <= 0 {
                continue;
            }
            let seq = raw as u64;
            if seq <= watermark {
                continue;
            }
            if best.is_none_or(|(_, best_seq)| seq < best_seq) {
                best = Some((idx, seq));
            }
        }
        best
    }

    /// Resolve the effective watermark and answer the cheap "is there anything
    /// newer?" question from the header high watermark.
    ///
    /// Returns `None` when the caller is already up to date.
    fn watermark(&self, from_entry_seq: Option<u64>) -> Option<u64> {
        let watermark = from_entry_seq.unwrap_or(self.last_entry_seq);
        if self.writer_entry_seq() <= watermark {
            return None;
        }
        Some(watermark)
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
        if dst.len() != self.payload_bytes {
            return Err(ZeroChannelError::InvalidArgument(format!(
                "destination holds {} bytes, expected {}",
                dst.len(),
                self.payload_bytes
            )));
        }
        let Some(watermark) = self.watermark(from_entry_seq) else {
            return Ok(None);
        };

        for _ in 0..READ_RETRIES {
            let Some((index, seq)) = self.locate(watermark) else {
                return Ok(None);
            };

            // Safety: `Element` payloads are plain-old-data, and `index` is in
            // range.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    self.ptr.add(payload_offset(self.payload_bytes, index)),
                    dst.as_mut_ptr(),
                    self.payload_bytes,
                );
            }

            // The local copy completes before `entry_seq` is re-checked.
            fence(Ordering::Acquire);

            // Double read: an unchanged entry_seq proves the copy was not torn.
            if self.seq_at(index).load(Ordering::Relaxed) != seq as i64 {
                continue;
            }

            self.last_entry_seq = seq;
            self.last_read_idx = index;
            return Ok(Some(seq));
        }
        Ok(None)
    }

    /// Copy the next entry into `dst` (one-copy path), as elements of `T`.
    pub fn read_into_as<T: Element>(
        &mut self,
        dst: &mut [T],
        from_entry_seq: Option<u64>,
    ) -> Result<Option<u64>, ZeroChannelError> {
        // Safety: `Element` guarantees `T` has no padding and no invalid bit
        // patterns, so the destination is exactly `size_of_val` writable bytes.
        let bytes = unsafe {
            core::slice::from_raw_parts_mut(
                dst.as_mut_ptr().cast::<u8>(),
                core::mem::size_of_val(dst),
            )
        };
        self.read_into(bytes, from_entry_seq)
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
        let elem = size_of::<T>();
        if elem == 0 || !self.payload_bytes.is_multiple_of(elem) {
            return Err(ZeroChannelError::InvalidArgument(format!(
                "payload of {} bytes is not a whole number of {elem}-byte elements",
                self.payload_bytes
            )));
        }

        let mut buf = vec![T::ZERO; self.payload_bytes / elem];
        match self.read_into_as::<T>(&mut buf, from_entry_seq)? {
            Some(seq) => Ok(Some((seq, buf))),
            None => Ok(None),
        }
    }

    /// Acquire the next entry in place (zero-copy path).
    ///
    /// Flips the slot's `entry_seq` from `s` to `-s` with a compare-and-swap so
    /// the writer skips the slot until [`RawReader::release`] hands it back.
    ///
    /// Requires a target with lock-free 64-bit atomics (x86_64 and aarch64 both
    /// qualify), and a caller that guarantees it is the only zero-copy reader.
    pub fn try_acquire(
        &mut self,
        from_entry_seq: Option<u64>,
    ) -> Result<Option<ReadHandle>, ZeroChannelError> {
        let Some(watermark) = self.watermark(from_entry_seq) else {
            return Ok(None);
        };

        for _ in 0..READ_RETRIES {
            let Some((index, seq)) = self.locate(watermark) else {
                return Ok(None);
            };

            // The slot may have been recycled between the scan and here; the
            // CAS is what makes the handle safe.
            if self
                .seq_at(index)
                .compare_exchange(
                    seq as i64,
                    -(seq as i64),
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_err()
            {
                continue;
            }

            self.last_entry_seq = seq;
            self.last_read_idx = index;
            return Ok(Some(ReadHandle {
                index,
                entry_seq: seq,
                // Safety: `index` is in range and the segment is mapped.
                ptr: unsafe { self.ptr.add(payload_offset(self.payload_bytes, index)) },
                len: self.payload_bytes,
            }));
        }
        Ok(None)
    }

    /// Return a held slot to the ring, restoring `entry_seq` to `+entry_seq`.
    ///
    /// A plain release store suffices: only the single zero-copy reader ever
    /// drives a slot negative.
    pub fn release(&mut self, handle: ReadHandle) {
        self.seq_at(handle.index)
            .store(handle.entry_seq as i64, Ordering::Release);
    }
}
