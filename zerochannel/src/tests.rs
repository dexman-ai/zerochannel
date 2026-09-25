//! Integration tests for the OS-facing half of ZeroChannel.
//!
//! These exercise named channels end to end: segment creation and ownership,
//! the exclusive writer and zero-copy reader roles, deferred connection, and
//! the ring protocol as seen through the typed wrappers.

use super::*;
use shared_memory::ShmemConf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use zerochannel_core::{
    entry_bytes, init_header, required_bytes, segment_owner, writer_owner, HEADER_BYTES, MAGIC,
    OFF_MAGIC, SLOT_ALIGN, VERSION,
};

use crate::lifecycle::{current_pid, try_open};

static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

fn unique_name(prefix: &str) -> String {
    let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    format!("/zc_test_{prefix}_{pid}_{id}")
}

/// Drain every entry currently available, one scalar read at a time.
fn drain<T: Element + PartialEq + std::fmt::Debug>(reader: &mut Reader<T>) -> Vec<(u64, Vec<T>)> {
    let mut entries = Vec::new();
    while let Some(entry) = reader.read(None).unwrap() {
        entries.push(entry);
    }
    entries
}

#[test]
fn round_trip_single() {
    let name = unique_name("rt");
    let mut writer = Float64Writer::new(&name, Some(4), Some(16), false).unwrap();
    let mut reader = Float64Reader::new(&name, None, None, false).unwrap();

    writer.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();

    assert_eq!(
        reader.read(None).unwrap(),
        Some((1, vec![1.0, 2.0, 3.0, 4.0]))
    );
    assert_eq!(reader.read(None).unwrap(), None);
}

#[test]
fn empty_read() {
    let name = unique_name("empty");
    let mut writer = Float64Writer::new(&name, Some(3), Some(8), false).unwrap();
    let mut reader = Float64Reader::new(&name, None, None, false).unwrap();

    assert!(reader.read(None).unwrap().is_none());

    // Write then read
    writer.write(&[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(reader.read(None).unwrap(), Some((1, vec![1.0, 2.0, 3.0])));
}

#[test]
fn reads_return_one_entry_at_a_time() {
    let name = unique_name("multi");
    let mut writer = Float64Writer::new(&name, Some(2), Some(16), false).unwrap();
    let mut reader = Float64Reader::new(&name, None, None, false).unwrap();

    for i in 0..5 {
        writer.write(&[i as f64, (i * 10) as f64]).unwrap();
    }

    for i in 0..5u64 {
        assert_eq!(
            reader.read(None).unwrap(),
            Some((i + 1, vec![i as f64, (i * 10) as f64]))
        );
    }
    assert!(reader.read(None).unwrap().is_none());
}

#[test]
fn entry_seq_is_consecutive_and_mirrored_in_the_header() {
    let name = unique_name("seq");
    let mut writer = Float64Writer::new(&name, Some(1), Some(32), false).unwrap();
    let reader = Float64Reader::new(&name, None, None, false).unwrap();

    assert_eq!(writer.last_entry_seq(), 0);
    assert_eq!(reader.writer_entry_seq(), 0);

    for i in 0..20u64 {
        writer.write(&[i as f64]).unwrap();
        assert_eq!(writer.last_entry_seq(), i + 1);
        assert_eq!(reader.writer_entry_seq(), i + 1);
    }
}

#[test]
fn lap_recovery() {
    let name = unique_name("lap");
    let total = 8usize;
    let mut writer = Float64Writer::new(&name, Some(2), Some(total), false).unwrap();
    let mut reader = Float64Reader::new(&name, None, None, false).unwrap();

    // Write one entry, read it so the reader has a bookmark
    writer.write(&[0.0, 0.0]).unwrap();
    assert_eq!(reader.read(None).unwrap(), Some((1, vec![0.0, 0.0])));

    // Overwrite the entire buffer plus extra to force a lap
    for i in 1..=(total + 3) {
        writer.write(&[i as f64, (i * 100) as f64]).unwrap();
    }

    // Only the `total` most recent entries survive; the reader resumes at
    // the oldest one still in the ring rather than at its stale bookmark.
    let entries = drain(&mut reader);
    assert_eq!(entries.len(), total);
    for i in 1..entries.len() {
        assert_eq!(entries[i].0, entries[i - 1].0 + 1);
        assert!(entries[i].1[0] > entries[i - 1].1[0]);
    }
}

#[test]
fn rollover_ordering() {
    let name = unique_name("roll");
    let total = 4usize;
    let mut writer = Float64Writer::new(&name, Some(1), Some(total), false).unwrap();
    let mut reader = Float64Reader::new(&name, None, None, false).unwrap();

    // Write more entries than entry_count to force rollover
    for i in 0..(total + 2) {
        writer.write(&[(i + 1) as f64]).unwrap();
    }

    let entries = drain(&mut reader);
    assert_eq!(entries.len(), total);

    // Sequence numbers and values are both monotonically increasing.
    for i in 1..entries.len() {
        assert_eq!(entries[i].0, entries[i - 1].0 + 1);
        assert!(
            entries[i].1[0] > entries[i - 1].1[0],
            "entry {} ({}) not > entry {} ({})",
            i,
            entries[i].1[0],
            i - 1,
            entries[i - 1].1[0]
        );
    }
}

#[test]
fn delayed_connect_writer() {
    let name = unique_name("delay_w");
    let mut writer = Float64Writer::new(&name, None, None, true).unwrap();

    // Write should no-op when deferred
    writer.write(&[1.0]).unwrap();

    // Create via reader
    let mut reader = Float64Reader::new(&name, Some(1), Some(8), false).unwrap();

    // Now writer should connect and work
    writer.write(&[42.0]).unwrap();
    assert_eq!(reader.read(None).unwrap(), Some((1, vec![42.0])));
}

#[test]
fn delayed_connect_reader() {
    let name = unique_name("delay_r");
    let mut reader = Float64Reader::new(&name, None, None, true).unwrap();

    // Read should return nothing when deferred
    assert!(reader.read(None).unwrap().is_none());

    // Create via writer
    let mut writer = Float64Writer::new(&name, Some(2), Some(8), false).unwrap();
    writer.write(&[3.5, 2.75]).unwrap();

    // Reader should now connect and return data
    assert_eq!(reader.read(None).unwrap(), Some((1, vec![3.5, 2.75])));
}

#[test]
fn header_validation_mismatch() {
    let name = unique_name("hdr");
    let _writer = Float64Writer::new(&name, Some(4), Some(16), false).unwrap();

    // Opening with wrong sizes should fail
    let result = Float64Reader::new(&name, Some(3), Some(16), false);
    assert!(result.is_err());

    let result = Float64Reader::new(&name, Some(4), Some(8), false);
    assert!(result.is_err());
}

#[test]
fn zero_sizes_rejected() {
    let result = Float64Writer::new("bad_zero", Some(0), Some(10), false);
    assert!(result.is_err());

    let result = Float64Writer::new("bad_zero2", Some(5), Some(0), false);
    assert!(result.is_err());
}

#[test]
fn write_wrong_length() {
    let name = unique_name("wlen");
    let mut writer = Float64Writer::new(&name, Some(3), Some(8), false).unwrap();

    let result = writer.write(&[1.0, 2.0]);
    assert!(result.is_err());

    let result = writer.write(&[1.0, 2.0, 3.0, 4.0]);
    assert!(result.is_err());
}

#[test]
fn read_delayed_then_write_then_read() {
    let name = unique_name("rdwr");
    let mut writer = Float64Writer::new(&name, Some(2), Some(16), false).unwrap();
    let mut reader = Float64Reader::new(&name, None, None, false).unwrap();

    // Read from empty — should be empty
    assert!(reader.read(None).unwrap().is_none());

    // Write some data
    writer.write(&[10.0, 20.0]).unwrap();
    writer.write(&[30.0, 40.0]).unwrap();

    // Read should get the written data, oldest first
    assert_eq!(reader.read(None).unwrap(), Some((1, vec![10.0, 20.0])));
    assert_eq!(reader.read(None).unwrap(), Some((2, vec![30.0, 40.0])));

    // Subsequent read should be empty (nothing new)
    assert!(reader.read(None).unwrap().is_none());
}

#[test]
fn from_entry_seq_override() {
    let name = unique_name("fromseq");
    let mut writer = Float64Writer::new(&name, Some(1), Some(16), false).unwrap();
    let mut reader = Float64Reader::new(&name, None, None, false).unwrap();

    for i in 0..5 {
        writer.write(&[i as f64]).unwrap();
    }

    // Consume everything.
    assert_eq!(drain(&mut reader).len(), 5);
    assert_eq!(reader.last_entry_seq(), 5);

    // Rewind by supplying an explicit watermark.
    assert_eq!(reader.read(Some(0)).unwrap(), Some((1, vec![0.0])));
    assert_eq!(reader.read(Some(3)).unwrap(), Some((4, vec![3.0])));

    // The override also moves the reader's own bookmark forward.
    assert_eq!(reader.last_entry_seq(), 4);
    assert_eq!(reader.read(None).unwrap(), Some((5, vec![4.0])));
}

#[test]
fn writer_restart_resumes_index() {
    let name = unique_name("restart");

    // Reader creates the segment (keeps the name alive across writer
    // drops, since the shared_memory crate unlinks on creator drop).
    let mut reader = Float64Reader::new(&name, Some(2), Some(8), false).unwrap();

    // Writer 1 attaches and writes 3 entries
    let mut writer1 = Float64Writer::new(&name, Some(2), Some(8), false).unwrap();
    writer1.write(&[1.0, 10.0]).unwrap();
    writer1.write(&[2.0, 20.0]).unwrap();
    writer1.write(&[3.0, 30.0]).unwrap();

    // Read the 3 entries written by writer 1
    assert_eq!(drain(&mut reader).len(), 3);

    // Writer 1 is dropped (simulating process exit)
    drop(writer1);

    // Writer 2 opens the same segment (simulating process restart). It
    // recovers `last_entry_seq` and `current_write_index` from the header.
    let mut writer2 = Float64Writer::new(&name, Some(2), Some(8), false).unwrap();
    assert_eq!(writer2.last_entry_seq(), 3);

    writer2.write(&[4.0, 40.0]).unwrap();
    writer2.write(&[5.0, 50.0]).unwrap();

    // Reader should see the new entries immediately
    let entries = drain(&mut reader);
    assert_eq!(entries, vec![(4, vec![4.0, 40.0]), (5, vec![5.0, 50.0])]);
}

#[test]
fn read_into_fills_caller_buffer() {
    let name = unique_name("readinto");
    let mut writer = Float64Writer::new(&name, Some(2), Some(16), false).unwrap();
    let mut reader = Float64Reader::new(&name, None, None, false).unwrap();

    writer.write(&[1.0, 2.0]).unwrap();
    writer.write(&[3.0, 4.0]).unwrap();

    let mut buf = [0.0f64; 2];
    assert_eq!(reader.read_into(&mut buf, None).unwrap(), Some(1));
    assert_eq!(buf, [1.0, 2.0]);
    assert_eq!(reader.read_into(&mut buf, None).unwrap(), Some(2));
    assert_eq!(buf, [3.0, 4.0]);

    // Nothing new — the buffer is left untouched.
    assert_eq!(reader.read_into(&mut buf, None).unwrap(), None);
    assert_eq!(buf, [3.0, 4.0]);
}

#[test]
fn read_into_wrong_length_rejected() {
    let name = unique_name("readintolen");
    let mut writer = Float64Writer::new(&name, Some(2), Some(8), false).unwrap();
    let mut reader = Float64Reader::new(&name, None, None, false).unwrap();
    writer.write(&[1.0, 2.0]).unwrap();

    let mut too_small = [0.0f64; 1];
    assert!(reader.read_into(&mut too_small, None).is_err());

    let mut too_big = [0.0f64; 3];
    assert!(reader.read_into(&mut too_big, None).is_err());
}

// ── Zero-copy handles ───────────────────────────────────────────────────

#[test]
fn write_handle_commit_round_trip() {
    let name = unique_name("whandle");
    let mut writer = Float64Writer::new(&name, Some(3), Some(8), false).unwrap();
    let mut reader = Float64Reader::new(&name, None, None, false).unwrap();

    let mut handle = writer.try_acquire().unwrap().unwrap();
    handle
        .as_slice_mut::<f64>()
        .unwrap()
        .copy_from_slice(&[7.0, 8.0, 9.0]);

    // Uncommitted slots carry entry_seq == 0 and stay invisible.
    assert!(reader.read(None).unwrap().is_none());

    assert_eq!(writer.commit(handle), 1);
    assert_eq!(reader.read(None).unwrap(), Some((1, vec![7.0, 8.0, 9.0])));
}

#[test]
fn read_handle_pins_slot_against_the_writer() {
    let name = unique_name("rhandle");
    let entry_count = 4usize;
    let mut writer = Float64Writer::new(&name, Some(1), Some(entry_count), false).unwrap();
    let mut reader = Float64Reader::new_zero_copy(&name, None, None, false).unwrap();

    writer.write(&[1.0]).unwrap();

    let handle = reader.try_acquire(None).unwrap().unwrap();
    assert_eq!(handle.entry_seq(), 1);
    assert_eq!(handle.as_slice::<f64>().unwrap(), &[1.0]);

    // Lap the ring several times; the held slot must be skipped.
    for i in 2..=(entry_count as u64 * 3) {
        writer.write(&[i as f64]).unwrap();
    }
    assert_eq!(handle.as_slice::<f64>().unwrap(), &[1.0]);

    reader.release(handle);

    // Released slots re-enter the rotation.
    writer.write(&[99.0]).unwrap();
}

#[test]
fn read_handle_advances_the_bookmark() {
    let name = unique_name("rhandleseq");
    let mut writer = Float64Writer::new(&name, Some(1), Some(8), false).unwrap();
    let mut reader = Float64Reader::new_zero_copy(&name, None, None, false).unwrap();

    writer.write(&[10.0]).unwrap();
    writer.write(&[20.0]).unwrap();

    let first = reader.try_acquire(None).unwrap().unwrap();
    assert_eq!(first.entry_seq(), 1);
    reader.release(first);

    assert_eq!(reader.read(None).unwrap(), Some((2, vec![20.0])));
    assert!(reader.try_acquire(None).unwrap().is_none());
}

#[test]
fn writer_is_busy_when_every_slot_is_held() {
    let name = unique_name("busy");
    let mut writer = Float64Writer::new(&name, Some(1), Some(2), false).unwrap();
    let mut reader = Float64Reader::new_zero_copy(&name, None, None, false).unwrap();

    writer.write(&[1.0]).unwrap();
    writer.write(&[2.0]).unwrap();

    let a = reader.try_acquire(None).unwrap().unwrap();
    let b = reader.try_acquire(None).unwrap().unwrap();
    assert_eq!((a.entry_seq(), b.entry_seq()), (1, 2));

    assert!(matches!(
        writer.write(&[3.0]),
        Err(ZeroChannelError::Busy(_))
    ));

    reader.release(a);
    reader.release(b);
    assert!(writer.write(&[3.0]).is_ok());
}

// ── Byte channels ───────────────────────────────────────────────────────

#[test]
fn payload_not_multiple_of_eight_rejected() {
    for bad in [1usize, 4, 12, 100, 1001] {
        let name = unique_name("align");
        let result = BytesWriter::new(&name, Some(bad), Some(8), false);
        assert!(
            result.is_err(),
            "payload of {bad} bytes should have been rejected"
        );
    }
}

#[test]
fn payload_multiple_of_eight_accepted() {
    let name = unique_name("align_ok");
    let writer = BytesWriter::new(&name, Some(4096), Some(4), false);
    assert!(writer.is_ok());
}

#[test]
fn byte_round_trip() {
    let name = unique_name("bytes");
    let payload = 4096usize;
    let mut writer = BytesWriter::new(&name, Some(payload), Some(8), false).unwrap();
    let mut reader = BytesReader::new(&name, None, None, false).unwrap();

    let block: Vec<u8> = (0..payload).map(|i| (i % 251) as u8).collect();
    writer.write(&block).unwrap();

    let (seq, data) = reader.read(None).unwrap().unwrap();
    assert_eq!(seq, 1);
    assert_eq!(data, block);
}

#[test]
fn byte_wrong_length_rejected() {
    let name = unique_name("blen");
    let mut writer = BytesWriter::new(&name, Some(16), Some(8), false).unwrap();

    assert!(writer.write(&[0u8; 8]).is_err());
    assert!(writer.write(&[0u8; 24]).is_err());
    assert!(writer.write(&[0u8; 16]).is_ok());
}

#[test]
fn byte_lap_recovery() {
    let name = unique_name("blap");
    let total = 8usize;
    let mut writer = BytesWriter::new(&name, Some(8), Some(total), false).unwrap();
    let mut reader = BytesReader::new(&name, None, None, false).unwrap();

    writer.write(&0u64.to_le_bytes()).unwrap();
    assert_eq!(drain(&mut reader).len(), 1);

    // Overwrite the whole ring plus extra to force a lap
    for i in 1..=(total as u64 + 3) {
        writer.write(&i.to_le_bytes()).unwrap();
    }

    let entries = drain(&mut reader);
    assert_eq!(entries.len(), total);

    let value = |e: &Vec<u8>| u64::from_le_bytes(e[..8].try_into().unwrap());
    for i in 1..entries.len() {
        assert!(entries[i].0 > entries[i - 1].0);
        assert!(value(&entries[i].1) > value(&entries[i - 1].1));
    }
}

#[test]
fn f64_payload_readable_as_bytes() {
    let name = unique_name("xtype");
    let mut writer = Float64Writer::new(&name, Some(2), Some(8), false).unwrap();

    // Dtype::U8 is opaque and attaches to a channel of any declared type.
    let mut raw = RawReader::new(&name, None, None, Dtype::U8, false).unwrap();

    writer.write(&[1.5, -2.25]).unwrap();

    let (seq, data) = raw.read(None).unwrap().unwrap();
    assert_eq!(seq, 1);
    assert_eq!(data.len(), 16);

    let a = f64::from_le_bytes(data[0..8].try_into().unwrap());
    let b = f64::from_le_bytes(data[8..16].try_into().unwrap());
    assert_eq!(a, 1.5);
    assert_eq!(b, -2.25);
}

#[test]
fn dtype_mismatch_rejected() {
    let name = unique_name("dtype");
    let _writer = Float64Writer::new(&name, Some(2), Some(8), false).unwrap();

    // Same geometry (2 × 8 bytes), different element type.
    let result = Reader::<i64>::new(&name, Some(2), Some(8), false);
    assert!(result.is_err());

    // Pure attach with a concrete type is also checked.
    let result = Reader::<i64>::new(&name, None, None, false);
    assert!(result.is_err());

    // Opaque attach is allowed.
    assert!(BytesReader::new(&name, None, None, false).is_ok());
}

#[test]
fn slot_seq_words_are_aligned() {
    for payload in [8usize, 16, 24, 4096] {
        assert!(HEADER_BYTES.is_multiple_of(SLOT_ALIGN));
        for i in 0..64 {
            let offset = HEADER_BYTES + i * entry_bytes(payload);
            assert!(
                offset.is_multiple_of(SLOT_ALIGN),
                "slot {i} of a {payload}-byte channel is misaligned"
            );
        }
    }
}

// ── Exclusive roles ─────────────────────────────────────────────────────

#[test]
fn second_writer_is_rejected_while_the_first_is_alive() {
    let name = unique_name("wexcl");
    let _writer = Float64Writer::new(&name, Some(1), Some(8), false).unwrap();

    // Two live writers diverge silently: each keeps its own sequence
    // counter, so entries overwrite one another with no gap in the
    // sequence numbers to reveal the loss.
    assert!(matches!(
        Float64Writer::new(&name, Some(1), Some(8), false),
        Err(ZeroChannelError::RoleConflict(_))
    ));
}

#[test]
fn writer_role_is_released_on_drop() {
    let name = unique_name("wrelease");
    let writer = Float64Writer::new(&name, Some(1), Some(8), false).unwrap();
    drop(writer);

    // A cleanly departed writer leaves the role free for its successor.
    assert!(Float64Writer::new(&name, Some(1), Some(8), false).is_ok());
}

#[test]
fn a_dead_writers_role_is_adopted() {
    let name = unique_name("wadopt");
    let _keeper = Float64Reader::new(&name, Some(1), Some(8), false).unwrap();

    // Impersonate a writer that died without releasing its claim.
    // `u32::MAX` is not a valid PID on any supported platform, standing in
    // for the crash that `Drop` cannot cover.
    let owner = unsafe { writer_owner(_keeper.raw.ptr()) };
    owner.store(u32::MAX as u64, Ordering::Release);

    let writer = Float64Writer::new(&name, Some(1), Some(8), false).unwrap();
    assert_eq!(owner.load(Ordering::Acquire), current_pid());
    drop(writer);
}

#[test]
fn second_zero_copy_reader_is_rejected() {
    let name = unique_name("zcexcl");
    let _writer = Float64Writer::new(&name, Some(1), Some(8), false).unwrap();
    let _reader = Float64Reader::new_zero_copy(&name, None, None, false).unwrap();

    assert!(matches!(
        Float64Reader::new_zero_copy(&name, None, None, false),
        Err(ZeroChannelError::RoleConflict(_))
    ));
}

#[test]
fn one_copy_readers_are_unrestricted() {
    let name = unique_name("swmr");
    let mut writer = Float64Writer::new(&name, Some(1), Some(8), false).unwrap();
    let _zc = Float64Reader::new_zero_copy(&name, None, None, false).unwrap();

    // The zero-copy claim must not constrain the SWMR path.
    let mut readers: Vec<_> = (0..4)
        .map(|_| Float64Reader::new(&name, None, None, false).unwrap())
        .collect();

    writer.write(&[7.0]).unwrap();
    for reader in &mut readers {
        assert_eq!(reader.read(None).unwrap(), Some((1, vec![7.0])));
    }
}

#[test]
fn try_acquire_requires_the_zero_copy_constructor() {
    let name = unique_name("zcgate");
    let mut writer = Float64Writer::new(&name, Some(1), Some(8), false).unwrap();
    let mut reader = Float64Reader::new(&name, None, None, false).unwrap();
    writer.write(&[1.0]).unwrap();

    assert!(matches!(
        reader.try_acquire(None),
        Err(ZeroChannelError::InvalidArgument(_))
    ));
}

#[test]
fn zero_copy_role_is_released_on_drop() {
    let name = unique_name("zcrelease");
    let _writer = Float64Writer::new(&name, Some(1), Some(8), false).unwrap();

    let reader = Float64Reader::new_zero_copy(&name, None, None, false).unwrap();
    drop(reader);

    assert!(Float64Reader::new_zero_copy(&name, None, None, false).is_ok());
}

// ── Header identity ─────────────────────────────────────────────────────

#[test]
fn foreign_segment_is_rejected() {
    let name = unique_name("foreign");
    // Shared memory that is not a ZeroChannel ring at all.
    let shmem = ShmemConf::new().os_id(&name).size(4096).create().unwrap();
    unsafe { std::ptr::write(shmem.as_ptr() as *mut u64, 0xDEAD_BEEF_DEAD_BEEF) };

    assert!(matches!(
        Float64Reader::new(&name, None, None, false),
        Err(ZeroChannelError::InvalidArgument(_))
    ));
}

#[test]
fn stale_layout_version_is_rejected() {
    let name = unique_name("oldver");
    let _writer = Float64Writer::new(&name, Some(1), Some(8), false).unwrap();

    // A segment left behind by an older build: right magic, wrong layout.
    // The opener-visible segment length is page-rounded by the OS, so a
    // size check cannot catch this — only the version can.
    let stale = ((MAGIC as u64) << 32) | (VERSION as u64 - 1);
    let shmem = try_open(&name).unwrap();
    unsafe {
        (*(shmem.as_ptr().add(OFF_MAGIC) as *const AtomicU64)).store(stale, Ordering::Release)
    };

    assert!(matches!(
        Float64Reader::new(&name, None, None, false),
        Err(ZeroChannelError::InvalidArgument(_))
    ));
}

#[test]
fn unlink_removes_the_segment() {
    let name = unique_name("unlink");
    let writer = Float64Writer::new(&name, Some(1), Some(8), false).unwrap();
    drop(writer);

    unlink(&name).unwrap();

    // The name is gone, so a pure attach finds nothing.
    assert!(Float64Reader::new(&name, None, None, false).is_err());
    // Unlinking an absent segment is not an error.
    unlink(&name).unwrap();
}

// ── Segment ownership ───────────────────────────────────────────────────

/// Leave behind an initialized segment that nobody owns, as a process
/// killed outright would: the mapping exists, but its creator's destructor
/// never ran, so the name was never unlinked.
fn orphan_segment(name: &str, payload_bytes: usize, entry_count: usize) {
    let mut shmem = ShmemConf::new()
        .os_id(name)
        .size(required_bytes(payload_bytes, entry_count))
        .create()
        .unwrap();
    // SAFETY: the mapping was just created with exactly this geometry.
    unsafe {
        init_header(
            shmem.as_ptr(),
            payload_bytes,
            entry_count,
            Dtype::F64,
            current_pid(),
        );
        // `init_header` stamps the *calling* process as owner, but the test
        // process is very much alive. Overwrite the deed with a PID that
        // cannot be running so the segment reads as genuinely abandoned.
        segment_owner(shmem.as_ptr()).store(u32::MAX as u64, Ordering::Release);
    }
    shmem.set_owner(false);
}

#[test]
fn declaring_geometry_adopts_an_orphaned_segment() {
    let name = unique_name("adopt");
    orphan_segment(&name, 8, 8);
    assert!(try_open(&name).is_some(), "the orphan should have survived");

    // Declaring the shape claims the deed even though the mapping already
    // existed, so a clean exit reclaims the leaked name.
    let writer = Float64Writer::new(&name, Some(1), Some(8), false).unwrap();
    drop(writer);

    assert!(try_open(&name).is_none());
}

#[test]
fn attaching_without_geometry_never_claims_ownership() {
    let name = unique_name("tenant");
    orphan_segment(&name, 8, 8);

    // A pure attach is a tenancy: it leaves the segment as it found it.
    let reader = Float64Reader::new(&name, None, None, false).unwrap();
    drop(reader);

    assert!(try_open(&name).is_some());
    unlink(&name).unwrap();
}

#[test]
fn a_tenant_outlives_the_owner_without_inheriting_the_deed() {
    let name = unique_name("outlive");
    let writer = Float64Writer::new(&name, Some(1), Some(8), false).unwrap();
    let reader = Float64Reader::new(&name, None, None, false).unwrap();

    // The owner leaving unlinks the name, but the tenant keeps its mapping.
    drop(writer);
    assert!(try_open(&name).is_none());

    // Ownership does not transfer on the way out: the tenant drops without
    // trying to unlink a name it never held.
    drop(reader);
    unlink(&name).unwrap();
}

#[test]
fn ownership_is_exclusive_among_geometry_suppliers() {
    let name = unique_name("exclusive");
    // Both ends declare the shape, but only the first to arrive owns it.
    let reader = Float64Reader::new(&name, Some(2), Some(8), false).unwrap();
    let writer = Float64Writer::new(&name, Some(2), Some(8), false).unwrap();

    // The later arrival is a tenant, so its exit leaves the channel intact
    // for the peer that is still running.
    drop(writer);
    assert!(try_open(&name).is_some());

    drop(reader);
    assert!(try_open(&name).is_none());
}
