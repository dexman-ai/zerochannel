//! On-the-wire layout of a ZeroChannel segment.
//!
//! Everything that describes *where* bytes live — header offsets, slot stride,
//! the magic latch, the element type tag — lives here, together with the
//! accessors that borrow individual header fields out of a mapped segment.
//! Nothing in this module allocates a mapping or talks to the OS; callers hand
//! in a base pointer that is already valid.

use alloc::format;
use core::sync::atomic::{fence, AtomicI64, AtomicU64, Ordering};

use crate::error::ZeroChannelError;

// ── Layout constants ────────────────────────────────────────────────────────

/// Size of the prefix header — nine `u64` fields.
///
/// The layout is split into an immutable half and a mutable half:
///
/// | Offset | Field | |
/// |---|---|---|
/// | `0..8` | `magic` + `version` | immutable |
/// | `8..16` | `payload_bytes` | immutable |
/// | `16..24` | `entry_count` | immutable |
/// | `24..32` | `dtype` | immutable |
/// | `32..40` | `segment_owner` — PID, `0` when unclaimed | mutable |
/// | `40..48` | `writer_owner` — PID, `0` when unclaimed | mutable |
/// | `48..56` | `zero_copy_reader_owner` — PID, `0` when unclaimed | mutable |
/// | `56..64` | `current_write_index` | mutable |
/// | `64..72` | `last_entry_seq` | mutable |
///
/// The first 32 bytes are written once at creation and validated once at
/// attach; everything that changes during the channel's life follows.
///
/// A multiple of [`SLOT_ALIGN`], so the first ring slot is aligned.
pub const HEADER_BYTES: usize = 72;

/// Byte offset of the combined `magic` + `version` word.
pub const OFF_MAGIC: usize = 0;

/// Byte offset of `payload_bytes`.
pub const OFF_PAYLOAD_BYTES: usize = 8;

/// Byte offset of `entry_count`.
pub const OFF_ENTRY_COUNT: usize = 16;

/// Byte offset of the `dtype` tag.
pub const OFF_DTYPE: usize = 24;

/// Byte offset of `segment_owner`, the PID of the process responsible for
/// unlinking the segment (`0` when unclaimed).
pub const OFF_SEGMENT_OWNER: usize = 32;

/// Byte offset of `writer_owner`, the PID of the process holding the writer
/// role (`0` when unclaimed).
pub const OFF_WRITER_OWNER: usize = 40;

/// Byte offset of `zero_copy_reader_owner`, the PID of the process holding the
/// zero-copy reader role (`0` when unclaimed).
pub const OFF_ZC_READER_OWNER: usize = 48;

/// Byte offset of `current_write_index` within the header.
pub const OFF_WRITE_INDEX: usize = 56;

/// Byte offset of `last_entry_seq` (the writer's high watermark) in the header.
pub const OFF_LAST_ENTRY_SEQ: usize = 64;

/// Identifies the segment as a ZeroChannel ring — ASCII `ZCHN`.
///
/// A segment name is just a string in a global namespace, so a channel can
/// collide with unrelated shared memory. The magic makes that a loud failure
/// instead of a reinterpretation of someone else's bytes.
pub const MAGIC: u32 = 0x5A43_484E;

/// On-the-wire layout revision, bumped whenever the byte layout changes.
///
/// Segments outlive the process that created them — a crashed writer leaves
/// one behind indefinitely — so a stale segment can easily be older than the
/// binary attaching to it. Nothing else catches that: the opener-visible
/// segment length is page-rounded by the OS, so a size check cannot
/// distinguish a 1064-byte v1 segment from a 1080-byte v2 one.
pub const VERSION: u32 = 1;

/// `magic` in the high half, `version` in the low half.
pub const MAGIC_VERSION: u64 = ((MAGIC as u64) << 32) | VERSION as u64;

/// Required alignment — and size granularity — of a ring slot, in bytes.
///
/// Every slot begins with an `AtomicI64` `entry_seq`. Keeping the slot stride a
/// multiple of 8 guarantees that word is always naturally aligned, which is
/// what makes the protocol sound. Payload sizes that are not a multiple of 8
/// are rejected rather than padded.
pub const SLOT_ALIGN: usize = 8;

/// How many times a one-copy read retries after observing a torn entry.
pub const READ_RETRIES: usize = 4;

// ── Element typing ──────────────────────────────────────────────────────────

/// Element type recorded in the channel header.
///
/// [`Dtype::U8`] means "opaque bytes": a participant requesting it will attach
/// to a channel of any declared type. Every other variant requires an exact
/// match, which is what distinguishes an `f64` channel from an `i64` one —
/// both have the same payload size, so geometry alone cannot tell them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// Opaque bytes.
    U8,
    /// Signed 64-bit integers.
    I64,
    /// IEEE-754 double-precision floats.
    F64,
}

impl Dtype {
    /// Stable on-the-wire code stored in the header.
    #[inline]
    pub fn code(self) -> u64 {
        match self {
            Dtype::U8 => 0,
            Dtype::I64 => 1,
            Dtype::F64 => 2,
        }
    }

    /// Human-readable name, used in error messages.
    pub fn name_of_code(code: u64) -> &'static str {
        match code {
            0 => "u8",
            1 => "i64",
            2 => "f64",
            _ => "unknown",
        }
    }

    /// Human-readable name of this type.
    #[inline]
    pub fn name(self) -> &'static str {
        Dtype::name_of_code(self.code())
    }
}

/// Marker for types that may be stored directly in a ZeroChannel payload.
///
/// # Safety
///
/// Implementors must be plain-old-data: no padding bytes, every bit pattern a
/// valid value, and `align_of::<Self>() <= SLOT_ALIGN`. These properties are
/// what make reinterpreting the shared-memory payload as `&[Self]` sound.
pub unsafe trait Element: Copy + 'static {
    /// Type tag written to — and validated against — the channel header.
    const DTYPE: Dtype;
    /// Zero value, used to pre-size read buffers.
    const ZERO: Self;
}

unsafe impl Element for u8 {
    const DTYPE: Dtype = Dtype::U8;
    const ZERO: u8 = 0;
}

unsafe impl Element for i64 {
    const DTYPE: Dtype = Dtype::I64;
    const ZERO: i64 = 0;
}

unsafe impl Element for f64 {
    const DTYPE: Dtype = Dtype::F64;
    const ZERO: f64 = 0.0;
}

// ── Geometry helpers ────────────────────────────────────────────────────────

/// Byte size of one ring slot: 8-byte `entry_seq` + payload.
#[inline]
pub fn entry_bytes(payload_bytes: usize) -> usize {
    SLOT_ALIGN + payload_bytes
}

/// Total shared-memory allocation: header + entry_count slots.
#[inline]
pub fn required_bytes(payload_bytes: usize, entry_count: usize) -> usize {
    HEADER_BYTES + entry_count * entry_bytes(payload_bytes)
}

/// Byte offset of the payload of ring slot `idx`.
#[inline]
pub fn payload_offset(payload_bytes: usize, idx: usize) -> usize {
    HEADER_BYTES + idx * entry_bytes(payload_bytes) + SLOT_ALIGN
}

/// Reject geometries that would misalign a slot timestamp.
pub fn check_geometry(payload_bytes: usize, entry_count: usize) -> Result<(), ZeroChannelError> {
    if payload_bytes == 0 || entry_count == 0 {
        return Err(ZeroChannelError::InvalidArgument(
            "entry_length and entry_count must be non-zero".into(),
        ));
    }
    if !payload_bytes.is_multiple_of(SLOT_ALIGN) {
        return Err(ZeroChannelError::InvalidArgument(format!(
            "payload must be a multiple of {SLOT_ALIGN} bytes to keep slot \
             entry_seq words aligned, got {payload_bytes}"
        )));
    }
    Ok(())
}

// ── Header ──────────────────────────────────────────────────────────────────

/// Decoded copy of the shared-memory prefix header.
#[derive(Debug, Clone, Copy)]
pub struct Header {
    /// Payload bytes per ring slot, excluding the `entry_seq` word.
    pub payload_bytes: usize,
    /// Number of ring slots.
    pub entry_count: usize,
    /// The writer's next slot index, as last persisted.
    pub write_index: usize,
    /// Element type code — see [`Dtype::code`].
    pub dtype: u64,
    /// The writer's high watermark, as last persisted.
    pub last_entry_seq: u64,
}

/// Validate that an existing header is compatible with the requested element
/// type. A request for [`Dtype::U8`] is treated as "opaque" and accepts any
/// declared type.
pub fn check_dtype(header: &Header, requested: Dtype) -> Result<(), ZeroChannelError> {
    if requested == Dtype::U8 || header.dtype == requested.code() {
        return Ok(());
    }
    Err(ZeroChannelError::InvalidArgument(format!(
        "dtype mismatch: channel holds {}, requested {}",
        Dtype::name_of_code(header.dtype),
        requested.name()
    )))
}

/// Validate that the header matches supplied sizes.
pub fn validate_header(
    header: &Header,
    payload_bytes: usize,
    entry_count: usize,
) -> Result<(), ZeroChannelError> {
    if header.payload_bytes != payload_bytes || header.entry_count != entry_count {
        return Err(ZeroChannelError::InvalidArgument(format!(
            "header mismatch: expected ({payload_bytes} payload bytes, {entry_count} entries), \
             found ({} payload bytes, {} entries)",
            header.payload_bytes, header.entry_count
        )));
    }
    Ok(())
}

/// Write the immutable half of the header and zero the rest.
///
/// The magic word goes down **last**: until it lands the segment reads as
/// uninitialized, so a peer that attaches mid-initialization backs off instead
/// of consuming a half-written geometry. `owner_pid` is stamped before the
/// magic for the same reason — whoever lays down the header owns the segment,
/// and publishing that atomically with the magic means no racing peer can
/// claim the deed out from under the creator.
///
/// # Safety
///
/// `base` must point at a mapped segment of at least
/// `required_bytes(payload_bytes, entry_count)` bytes.
pub unsafe fn init_header(
    base: *mut u8,
    payload_bytes: usize,
    entry_count: usize,
    dtype: Dtype,
    owner_pid: u64,
) {
    let size = required_bytes(payload_bytes, entry_count);
    core::ptr::write_bytes(base, 0, size);
    core::ptr::write(
        base.add(OFF_PAYLOAD_BYTES) as *mut u64,
        payload_bytes as u64,
    );
    core::ptr::write(base.add(OFF_ENTRY_COUNT) as *mut u64, entry_count as u64);
    core::ptr::write(base.add(OFF_DTYPE) as *mut u64, dtype.code());
    core::ptr::write(base.add(OFF_SEGMENT_OWNER) as *mut u64, owner_pid);
    // Geometry is visible before the magic that advertises it.
    fence(Ordering::Release);
    (*(base.add(OFF_MAGIC) as *const AtomicU64)).store(MAGIC_VERSION, Ordering::Release);
}

/// Decode the prefix header, or explain why it is unusable.
///
/// `Ok(None)` means the segment exists but carries no magic yet: either its
/// creator has not finished initializing it, or it died partway through.
///
/// # Safety
///
/// `base` must point at a mapped segment of at least [`HEADER_BYTES`] bytes.
pub unsafe fn read_header(base: *const u8) -> Result<Option<Header>, ZeroChannelError> {
    let magic = (*(base.add(OFF_MAGIC) as *const AtomicU64)).load(Ordering::Acquire);
    if magic == 0 {
        return Ok(None);
    }
    if magic != MAGIC_VERSION {
        if magic >> 32 != MAGIC as u64 {
            return Err(ZeroChannelError::InvalidArgument(
                "shared memory segment is not a ZeroChannel ring".into(),
            ));
        }
        return Err(ZeroChannelError::InvalidArgument(format!(
            "ZeroChannel layout version mismatch: segment is v{}, this build speaks v{}",
            magic & 0xFFFF_FFFF,
            VERSION
        )));
    }
    Ok(Some(Header {
        payload_bytes: core::ptr::read(base.add(OFF_PAYLOAD_BYTES) as *const u64) as usize,
        entry_count: core::ptr::read(base.add(OFF_ENTRY_COUNT) as *const u64) as usize,
        write_index: core::ptr::read(base.add(OFF_WRITE_INDEX) as *const u64) as usize,
        dtype: core::ptr::read(base.add(OFF_DTYPE) as *const u64),
        last_entry_seq: (*(base.add(OFF_LAST_ENTRY_SEQ) as *const AtomicU64))
            .load(Ordering::Acquire),
    }))
}

// ── Header field accessors ──────────────────────────────────────────────────

/// Borrow the header's `segment_owner` cell.
///
/// # Safety
///
/// `base` must point at a mapped ZeroChannel segment.
#[inline]
pub unsafe fn segment_owner(base: *mut u8) -> &'static AtomicU64 {
    &*(base.add(OFF_SEGMENT_OWNER) as *const AtomicU64)
}

/// Borrow the header's `writer_owner` cell.
///
/// # Safety
///
/// `base` must point at a mapped ZeroChannel segment.
#[inline]
pub unsafe fn writer_owner(base: *mut u8) -> &'static AtomicU64 {
    &*(base.add(OFF_WRITER_OWNER) as *const AtomicU64)
}

/// Borrow the header's `zero_copy_reader_owner` cell.
///
/// # Safety
///
/// `base` must point at a mapped ZeroChannel segment.
#[inline]
pub unsafe fn zc_reader_owner(base: *mut u8) -> &'static AtomicU64 {
    &*(base.add(OFF_ZC_READER_OWNER) as *const AtomicU64)
}

/// Borrow the header's `last_entry_seq` cell — the writer's high watermark.
///
/// # Safety
///
/// `base` must point at a mapped ZeroChannel segment.
#[inline]
pub unsafe fn high_watermark(base: *mut u8) -> &'static AtomicU64 {
    &*(base.add(OFF_LAST_ENTRY_SEQ) as *const AtomicU64)
}

/// Borrow the `entry_seq` control word of ring slot `idx`.
///
/// # Safety
///
/// `base` must point at a mapped ZeroChannel segment whose geometry matches
/// `payload_bytes`, and `idx` must be less than the segment's `entry_count`.
#[inline]
pub unsafe fn seq_cell(base: *mut u8, payload_bytes: usize, idx: usize) -> &'static AtomicI64 {
    let offset = HEADER_BYTES + idx * entry_bytes(payload_bytes);
    &*(base.add(offset) as *const AtomicI64)
}
