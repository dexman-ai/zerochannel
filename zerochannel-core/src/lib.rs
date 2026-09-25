//! ZeroChannel core — the lock-free ring protocol, with no OS in sight.
//!
//! This crate implements everything that happens *inside* an already-mapped
//! ZeroChannel segment: the header layout, the `entry_seq` protocol, the
//! one-copy and zero-copy access paths, and writer-restart recovery. It never
//! creates, names, opens or unlinks a mapping, and it never asks the operating
//! system a question — so it is `no_std` (with `alloc`) and portable to
//! anywhere a shared mapping can be handed to it.
//!
//! The companion `zerochannel` crate supplies the missing half: named POSIX or
//! Windows shared memory, exclusive role claims backed by PID liveness checks,
//! segment ownership, and the Python bindings.
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
//!   hands out a borrowed view into shared memory. Acquiring a read handle
//!   flips `entry_seq` to `-entry_seq` with a compare-and-swap, so this path
//!   requires a target with lock-free 64-bit atomics (x86_64 and aarch64 both
//!   qualify) and is **single-reader** only.

#![no_std]

extern crate alloc;

pub mod error;
pub mod metadata;
pub mod reader;
pub mod writer;
pub mod zerocopy;

pub use error::ZeroChannelError;
pub use metadata::{
    check_dtype, check_geometry, entry_bytes, high_watermark, init_header, payload_offset,
    read_header, required_bytes, segment_owner, seq_cell, validate_header, writer_owner,
    zc_reader_owner, Dtype, Element, Header, HEADER_BYTES, MAGIC, MAGIC_VERSION, OFF_DTYPE,
    OFF_ENTRY_COUNT, OFF_LAST_ENTRY_SEQ, OFF_MAGIC, OFF_PAYLOAD_BYTES, OFF_SEGMENT_OWNER,
    OFF_WRITER_OWNER, OFF_WRITE_INDEX, OFF_ZC_READER_OWNER, READ_RETRIES, SLOT_ALIGN, VERSION,
};
pub use reader::RawReader;
pub use writer::RawWriter;
pub use zerocopy::{ReadHandle, WriteHandle};
