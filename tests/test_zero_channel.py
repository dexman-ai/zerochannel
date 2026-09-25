"""Integration tests for the ZeroChannel shared-memory IPC subsystem.

All tests exercise the Python-to-Python path via the native extension.
"""

import os
import uuid

import numpy as np
import pytest

from zerochannel import (
    BytesWriter,
    BytesReader,
    ChannelRoleConflict,
    Float64Writer,
    Float64Reader,
    unlink,
)


def _name() -> str:
    """Return a unique shared-memory name for each test invocation."""
    return f"/zc_py_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _drain(reader):
    """Read every available entry, one scalar read at a time."""
    entries = []
    while (entry := reader.read()) is not None:
        entries.append(entry)
    return entries


# ── Round-trip tests ─────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_write_then_read_single(self):
        name = _name()
        writer = Float64Writer(name, entry_length=4, entry_count=16)
        reader = Float64Reader(name)

        writer.write(np.array([1.0, 2.0, 3.0, 4.0]))
        entry_seq, data = reader.read()

        assert entry_seq == 1
        assert data.shape == (4,)
        np.testing.assert_array_equal(data, [1.0, 2.0, 3.0, 4.0])
        assert reader.read() is None

    def test_reads_return_one_entry_at_a_time(self):
        name = _name()
        writer = Float64Writer(name, entry_length=3, entry_count=32)
        reader = Float64Reader(name)

        for i in range(5):
            writer.write(np.array([float(i), float(i * 10), float(i * 100)]))

        entries = _drain(reader)
        assert len(entries) == 5
        for i, (entry_seq, data) in enumerate(entries):
            assert entry_seq == i + 1
            np.testing.assert_array_equal(
                data, [float(i), float(i * 10), float(i * 100)]
            )

    def test_incremental_reads(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=16)
        reader = Float64Reader(name)

        writer.write(np.array([1.0, 2.0]))
        entry_seq, data = reader.read()
        assert entry_seq == 1
        np.testing.assert_array_equal(data, [1.0, 2.0])

        writer.write(np.array([3.0, 4.0]))
        entry_seq, data = reader.read()
        assert entry_seq == 2
        np.testing.assert_array_equal(data, [3.0, 4.0])

    def test_watermarks(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=8)
        reader = Float64Reader(name)

        assert writer.last_entry_seq == 0
        assert reader.writer_entry_seq == 0
        assert reader.last_entry_seq == 0

        writer.write(np.array([1.0]))
        writer.write(np.array([2.0]))

        assert writer.last_entry_seq == 2
        assert reader.writer_entry_seq == 2
        assert reader.last_entry_seq == 0

        reader.read()
        assert reader.last_entry_seq == 1


# ── Empty read ───────────────────────────────────────────────────────────────


class TestEmptyRead:
    def test_read_before_write(self):
        name = _name()
        writer = Float64Writer(name, entry_length=3, entry_count=8)
        reader = Float64Reader(name)

        assert reader.read() is None
        del writer

    def test_read_after_all_consumed(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=8)
        reader = Float64Reader(name)

        writer.write(np.array([1.0, 2.0]))
        assert reader.read() is not None
        assert reader.read() is None


# ── Rollover / wrap-around ───────────────────────────────────────────────────


class TestRollover:
    def test_write_past_rollover_increasing_order(self):
        """Write more entries than entry_count; the reader returns the
        surviving window in increasing entry_seq order."""
        name = _name()
        total = 4
        writer = Float64Writer(name, entry_length=1, entry_count=total)
        reader = Float64Reader(name)

        for i in range(total + 3):
            writer.write(np.array([float(i + 1)]))

        entries = _drain(reader)
        assert len(entries) == total

        for i in range(1, len(entries)):
            assert entries[i][0] == entries[i - 1][0] + 1
            assert entries[i][1][0] > entries[i - 1][1][0]

    def test_rollover_preserves_latest_entries(self):
        """After rollover the reader should see the most recent entries."""
        name = _name()
        total = 4
        writer = Float64Writer(name, entry_length=1, entry_count=total)
        reader = Float64Reader(name)

        # Write exactly entry_count entries
        for i in range(total):
            writer.write(np.array([float(i + 1)]))

        assert len(_drain(reader)) == total

        # Write more, wrapping around
        for i in range(total, total + 2):
            writer.write(np.array([float(i + 1)]))

        entries = _drain(reader)
        assert len(entries) == 2
        # The newest entry should be the last one written
        assert entries[-1][1][0] == float(total + 2)
        assert entries[-1][0] == writer.last_entry_seq

    def test_lap_recovery(self):
        """Writer laps the reader; reader recovers and returns surviving window."""
        name = _name()
        total = 8
        writer = Float64Writer(name, entry_length=2, entry_count=total)
        reader = Float64Reader(name)

        # Read one entry to set a bookmark
        writer.write(np.array([0.0, 0.0]))
        assert reader.read()[0] == 1

        # Overwrite the entire buffer plus extra to force a lap
        for i in range(1, total + 4):
            writer.write(np.array([float(i), float(i * 100)]))

        # The stale bookmark points at an overwritten slot; the reader resumes
        # at the oldest entry still in the ring.
        entries = _drain(reader)
        assert len(entries) == total

        for i in range(1, len(entries)):
            assert entries[i][0] == entries[i - 1][0] + 1
            assert entries[i][1][0] > entries[i - 1][1][0]

    def test_gap_detection(self):
        """A jump in entry_seq tells the client how many entries it lost."""
        name = _name()
        total = 4
        writer = Float64Writer(name, entry_length=1, entry_count=total)
        reader = Float64Reader(name)

        writer.write(np.array([1.0]))
        first_seq = reader.read()[0]

        for i in range(2, 12):
            writer.write(np.array([float(i)]))

        next_seq = reader.read()[0]
        assert next_seq - first_seq > 1  # entries were dropped
        assert writer.last_entry_seq - next_seq == total - 1


# ── Read-delayed then write then read ────────────────────────────────────────


class TestReadDelayedWriteRead:
    def test_read_empty_then_write_then_read(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=16)
        reader = Float64Reader(name)

        # Read from empty channel
        assert reader.read() is None

        # Write some data
        writer.write(np.array([10.0, 20.0]))
        writer.write(np.array([30.0, 40.0]))

        # Read should return the new data, oldest first
        entries = _drain(reader)
        assert len(entries) == 2
        np.testing.assert_array_equal(entries[0][1], [10.0, 20.0])
        np.testing.assert_array_equal(entries[1][1], [30.0, 40.0])

    def test_alternating_read_write(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=16)
        reader = Float64Reader(name)

        for i in range(5):
            writer.write(np.array([float(i)]))
            entry_seq, data = reader.read()
            assert entry_seq == i + 1
            assert data[0] == float(i)


# ── Delayed connect ──────────────────────────────────────────────────────────


class TestDelayedConnect:
    def test_reader_delayed_connect(self):
        name = _name()
        reader = Float64Reader(name, delayed_connect=True)

        # Read should return None while unconnected
        assert reader.read() is None

        # Create writer which creates the segment
        writer = Float64Writer(name, entry_length=2, entry_count=8)
        writer.write(np.array([3.5, 2.75]))

        # Reader should now connect and return data
        entry_seq, data = reader.read()
        assert entry_seq == 1
        np.testing.assert_array_almost_equal(data, [3.5, 2.75])

    def test_delayed_connect_reader_with_out(self):
        name = _name()
        reader = Float64Reader(name, delayed_connect=True)
        out = np.zeros(2)

        assert reader.read(out=out) is None

        writer = Float64Writer(name, entry_length=2, entry_count=8)
        writer.write(np.array([7.0, 8.0]))

        entry_seq, data = reader.read(out=out)
        assert entry_seq == 1
        assert data is out
        np.testing.assert_array_equal(out, [7.0, 8.0])

    def test_writer_delayed_connect(self):
        name = _name()
        writer = Float64Writer(name, delayed_connect=True)

        # Write should no-op when deferred
        writer.write(np.array([1.0]))

        # Create reader which creates the segment
        reader = Float64Reader(name, entry_length=1, entry_count=8)

        # Now writer should connect and work
        writer.write(np.array([42.0]))
        entry_seq, data = reader.read()
        assert entry_seq == 1
        assert data[0] == 42.0


# ── Writer restart ───────────────────────────────────────────────────────────


class TestWriterRestart:
    def test_writer_restart_reader_sees_new_data(self):
        """When a writer drops and a new one attaches, the reader should
        immediately see data from the new writer."""
        name = _name()
        # Reader creates the segment so the name survives writer drops.
        reader = Float64Reader(name, entry_length=2, entry_count=8)

        # Writer 1 writes some entries
        writer1 = Float64Writer(name, entry_length=2, entry_count=8)
        writer1.write(np.array([1.0, 10.0]))
        writer1.write(np.array([2.0, 20.0]))
        writer1.write(np.array([3.0, 30.0]))

        assert len(_drain(reader)) == 3

        # Simulate writer process restart
        del writer1

        writer2 = Float64Writer(name, entry_length=2, entry_count=8)
        assert writer2.last_entry_seq == 3

        writer2.write(np.array([4.0, 40.0]))
        writer2.write(np.array([5.0, 50.0]))

        entries = _drain(reader)
        assert [e[0] for e in entries] == [4, 5]
        np.testing.assert_array_equal(entries[0][1], [4.0, 40.0])
        np.testing.assert_array_equal(entries[1][1], [5.0, 50.0])


# ── Error handling ───────────────────────────────────────────────────────────


class TestErrors:
    def test_zero_entry_length(self):
        with pytest.raises(ValueError):
            Float64Writer(_name(), entry_length=0, entry_count=10)

    def test_zero_entry_count(self):
        with pytest.raises(ValueError):
            Float64Writer(_name(), entry_length=5, entry_count=0)

    def test_header_mismatch(self):
        name = _name()
        writer = Float64Writer(name, entry_length=4, entry_count=16)

        with pytest.raises(ValueError):
            Float64Reader(name, entry_length=3, entry_count=16)

        with pytest.raises(ValueError):
            Float64Reader(name, entry_length=4, entry_count=8)

        del writer  # explicitly release after assertions

    def test_write_wrong_length(self):
        name = _name()
        writer = Float64Writer(name, entry_length=3, entry_count=8)

        with pytest.raises(ValueError):
            writer.write(np.array([1.0, 2.0]))

        with pytest.raises(ValueError):
            writer.write(np.array([1.0, 2.0, 3.0, 4.0]))

    def test_segment_not_found_no_sizes_no_delay(self):
        # Readers default to delayed_connect, so the absent segment only
        # surfaces as an error when the caller opts out of waiting for it.
        with pytest.raises(OSError):
            Float64Reader(_name(), delayed_connect=False)

    def test_reader_defers_to_an_absent_segment(self):
        reader = Float64Reader(_name())
        assert reader.read() is None

    def test_segment_not_found_writer_no_delay(self):
        with pytest.raises(OSError):
            Float64Writer(_name())


# ── from_entry_seq parameter ─────────────────────────────────────────────────


class TestFromEntrySeq:
    def test_from_entry_seq_zero_returns_all(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=16)
        reader = Float64Reader(name)

        for i in range(5):
            writer.write(np.array([float(i)]))

        first = reader.read(from_entry_seq=0)
        assert first[0] == 1
        assert len(_drain(reader)) == 4

    def test_from_entry_seq_rewinds_bookmark(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=16)
        reader = Float64Reader(name)

        # Write and read to establish bookmarks
        writer.write(np.array([0.0]))
        assert reader.read()[0] == 1

        # Write more
        writer.write(np.array([1.0]))
        writer.write(np.array([2.0]))

        # Rewind to the beginning; the already-read entry is returned again
        entry_seq, data = reader.read(from_entry_seq=0)
        assert entry_seq == 1
        assert data[0] == 0.0

        # The bookmark now follows the override
        assert [e[0] for e in _drain(reader)] == [2, 3]

    def test_from_entry_seq_skips_ahead(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=16)
        reader = Float64Reader(name)

        for i in range(5):
            writer.write(np.array([float(i)]))

        entry_seq, data = reader.read(from_entry_seq=3)
        assert entry_seq == 4
        assert data[0] == 3.0

    def test_from_entry_seq_at_watermark_returns_none(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=16)
        reader = Float64Reader(name)

        writer.write(np.array([1.0]))
        assert reader.read(from_entry_seq=writer.last_entry_seq) is None


# ── out parameter ────────────────────────────────────────────────────────────


class TestOutParameter:
    def test_float_out_is_filled_in_place(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=16)
        reader = Float64Reader(name)

        writer.write(np.array([1.0, 2.0]))

        out = np.zeros(2)
        entry_seq, data = reader.read(out=out)
        assert entry_seq == 1
        assert data is out
        np.testing.assert_array_equal(out, [1.0, 2.0])

    def test_float_out_reused_across_reads(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=16)
        reader = Float64Reader(name)

        for i in range(5):
            writer.write(np.array([float(i), float(i * 10)]))

        out = np.zeros(2)
        for i in range(5):
            entry_seq, data = reader.read(out=out)
            assert entry_seq == i + 1
            np.testing.assert_array_equal(data, [float(i), float(i * 10)])

        assert reader.read(out=out) is None
        # A miss leaves the buffer untouched
        np.testing.assert_array_equal(out, [4.0, 40.0])

    def test_float_out_wrong_length(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=16)
        reader = Float64Reader(name)
        writer.write(np.array([1.0, 2.0]))

        for bad in (np.zeros(1), np.zeros(3)):
            with pytest.raises(ValueError):
                reader.read(out=bad)

    def test_float_out_non_contiguous(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=16)
        reader = Float64Reader(name)
        writer.write(np.array([1.0, 2.0]))

        out = np.zeros(4)[::2]
        assert not out.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError):
            reader.read(out=out)

    def test_float_out_wrong_dtype(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=16)
        reader = Float64Reader(name)
        writer.write(np.array([1.0, 2.0]))

        with pytest.raises(TypeError):
            reader.read(out=np.zeros(2, dtype=np.float32))

    def test_bytes_out_is_filled_in_place(self):
        name = _name()
        writer = BytesWriter(name, entry_length=16, entry_count=8)
        reader = BytesReader(name)

        block = bytes(range(16))
        writer.write(block)

        out = np.zeros(16, dtype=np.uint8)
        entry_seq, data = reader.read(out=out)
        assert entry_seq == 1
        assert data is out
        assert out.tobytes() == block

    def test_bytes_out_wrong_length(self):
        name = _name()
        writer = BytesWriter(name, entry_length=16, entry_count=8)
        reader = BytesReader(name)
        writer.write(b"\x01" * 16)

        with pytest.raises(ValueError):
            reader.read(out=np.zeros(8, dtype=np.uint8))

    def test_bytes_out_wrong_dtype(self):
        name = _name()
        writer = BytesWriter(name, entry_length=16, entry_count=8)
        reader = BytesReader(name)
        writer.write(b"\x01" * 16)

        with pytest.raises(TypeError):
            reader.read(out=np.zeros(16, dtype=np.float64))


# ── Raw byte channels ────────────────────────────────────────────────────────


class TestBytes:
    def test_byte_round_trip(self):
        name = _name()
        writer = BytesWriter(name, entry_length=4096, entry_count=8)
        reader = BytesReader(name)

        block = bytes(i % 251 for i in range(4096))
        writer.write(block)

        entry_seq, data = reader.read()
        assert entry_seq == 1
        assert data == block
        assert reader.read() is None

    def test_multiple_byte_writes(self):
        name = _name()
        writer = BytesWriter(name, entry_length=16, entry_count=32)
        reader = BytesReader(name)

        blocks = [bytes([i]) * 16 for i in range(5)]
        for b in blocks:
            writer.write(b)

        entries = _drain(reader)
        assert [e[0] for e in entries] == [1, 2, 3, 4, 5]
        assert [e[1] for e in entries] == blocks

    def test_entry_length_must_be_multiple_of_eight(self):
        for bad in (1, 4, 12, 100, 1001):
            with pytest.raises(ValueError):
                BytesWriter(_name(), entry_length=bad, entry_count=8)

    def test_entry_length_accepts_multiple_of_eight(self):
        writer = BytesWriter(_name(), entry_length=4096, entry_count=4)
        assert writer.entry_length == 4096

    def test_wrong_byte_length_rejected(self):
        name = _name()
        writer = BytesWriter(name, entry_length=16, entry_count=8)

        with pytest.raises(ValueError):
            writer.write(b"\x00" * 8)

        with pytest.raises(ValueError):
            writer.write(b"\x00" * 24)

    def test_float_channel_readable_as_bytes(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=8)
        reader = BytesReader(name)

        assert writer.entry_length == 2
        assert reader.entry_length == 16

        writer.write(np.array([1.5, -2.25]))

        entry_seq, data = reader.read()
        assert entry_seq == 1
        np.testing.assert_array_equal(
            np.frombuffer(data, dtype=np.float64), [1.5, -2.25]
        )


# ── Zero-copy handles ────────────────────────────────────────────────────────


class TestZeroCopy:
    def test_float_handle_round_trip(self):
        name = _name()
        writer = Float64Writer(name, entry_length=4, entry_count=8)
        reader = Float64Reader(name, enable_zero_copy=True)

        handle = writer.try_acquire()
        np.asarray(handle.payload)[:] = [1.0, 2.0, 3.0, 4.0]
        assert handle.commit() == 1
        assert handle.committed

        handle = reader.try_acquire()
        assert handle.entry_seq == 1
        np.testing.assert_array_equal(np.asarray(handle.payload), [1.0, 2.0, 3.0, 4.0])
        handle.release()
        assert handle.released

    def test_bytes_handle_round_trip(self):
        name = _name()
        writer = BytesWriter(name, entry_length=16, entry_count=8)
        reader = BytesReader(name, enable_zero_copy=True)

        with writer.try_acquire() as handle:
            handle.payload[:] = bytes(range(16))

        with reader.try_acquire() as handle:
            assert bytes(handle.payload) == bytes(range(16))

    def test_payload_is_a_zero_copy_view(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=8)
        reader = Float64Reader(name, enable_zero_copy=True)

        writer.write(np.array([1.0, 2.0]))

        with reader.try_acquire() as handle:
            view = np.asarray(handle.payload)
            assert view.dtype == np.float64
            # A copy would own its data; a view borrows the ring slot.
            assert not view.flags["OWNDATA"]
            del view  # a view may not outlive the handle

    def test_try_acquire_returns_none_when_empty(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=8)
        reader = Float64Reader(name, enable_zero_copy=True)

        assert reader.try_acquire() is None

        writer.write(np.array([1.0, 2.0]))
        with reader.try_acquire() as handle:
            assert handle.entry_seq == 1
        assert reader.try_acquire() is None

    def test_context_manager_releases_on_exception(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=8)
        reader = Float64Reader(name, enable_zero_copy=True)
        writer.write(np.array([1.0]))

        handle = reader.try_acquire()
        with pytest.raises(RuntimeError):
            with handle:
                raise RuntimeError("boom")
        assert handle.released

    def test_write_handle_is_abandoned_on_exception(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=8)
        reader = Float64Reader(name, enable_zero_copy=True)

        handle = writer.try_acquire()
        with pytest.raises(RuntimeError):
            with handle:
                np.asarray(handle.payload)[:] = [1.0]
                raise RuntimeError("boom")

        assert not handle.committed
        assert reader.try_acquire() is None

    def test_release_and_commit_are_idempotent(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=8)
        reader = Float64Reader(name, enable_zero_copy=True)

        with writer.try_acquire() as handle:
            np.asarray(handle.payload)[:] = [1.0]
        with pytest.raises(ValueError):
            handle.commit()

        read_handle = reader.try_acquire()
        read_handle.release()
        read_handle.release()  # a second release is a no-op

    def test_payload_is_rejected_after_release(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=8)
        reader = Float64Reader(name, enable_zero_copy=True)
        writer.write(np.array([1.0]))

        handle = reader.try_acquire()
        handle.release()
        with pytest.raises(BufferError):
            handle.payload

    def test_writer_skips_a_held_slot(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=2)
        reader = Float64Reader(name, enable_zero_copy=True)

        writer.write(np.array([1.0]))
        handle = reader.try_acquire()
        assert handle.entry_seq == 1

        # The ring holds two slots; the pinned one must survive the lap.
        writer.write(np.array([2.0]))
        writer.write(np.array([3.0]))
        np.testing.assert_array_equal(np.asarray(handle.payload), [1.0])
        handle.release()

    def test_writer_is_busy_when_every_slot_is_held(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=2)
        reader = Float64Reader(name, enable_zero_copy=True)

        writer.write(np.array([1.0]))
        writer.write(np.array([2.0]))
        handles = [reader.try_acquire(), reader.try_acquire()]
        assert [handle.entry_seq for handle in handles] == [1, 2]

        with pytest.raises(BlockingIOError):
            writer.try_acquire()

        for handle in handles:
            handle.release()
        assert writer.try_acquire() is not None

    def test_from_entry_seq_rewinds_a_handle(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=8)
        reader = Float64Reader(name, enable_zero_copy=True)

        writer.write(np.array([1.0]))
        writer.write(np.array([2.0]))
        with reader.try_acquire() as handle:
            assert handle.entry_seq == 1
        with reader.try_acquire() as handle:
            assert handle.entry_seq == 2

        with reader.try_acquire(from_entry_seq=0) as handle:
            assert handle.entry_seq == 1

    def test_acquire_and_read_share_one_bookmark(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=8)
        reader = Float64Reader(name, enable_zero_copy=True)

        for i in range(3):
            writer.write(np.array([float(i)]))

        with reader.try_acquire() as handle:
            assert handle.entry_seq == 1
        assert reader.read()[0] == 2
        with reader.try_acquire() as handle:
            assert handle.entry_seq == 3


# ── Buffer-protocol enforcement ──────────────────────────────────────────────


class TestHandleExports:
    """The handle exports the buffer protocol, so views into a ring slot are
    counted and the slot cannot be recycled while any of them is alive."""

    def test_release_refuses_while_a_view_is_alive(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=8)
        reader = Float64Reader(name, enable_zero_copy=True)
        writer.write(np.array([1.0, 2.0]))

        handle = reader.try_acquire()
        arr = np.asarray(handle.payload)

        with pytest.raises(BufferError):
            handle.release()
        assert not handle.released
        np.testing.assert_array_equal(arr, [1.0, 2.0])

        del arr
        handle.release()
        assert handle.released

    def test_context_manager_refuses_to_leak_a_view(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=8)
        reader = Float64Reader(name, enable_zero_copy=True)
        writer.write(np.array([1.0, 2.0]))

        escaped = []
        with pytest.raises(BufferError):
            with reader.try_acquire() as handle:
                escaped.append(np.asarray(handle.payload))

    def test_commit_refuses_while_a_view_is_alive(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=8)
        reader = Float64Reader(name, enable_zero_copy=True)

        handle = writer.try_acquire()
        arr = np.asarray(handle.payload)
        arr[:] = [3.0, 4.0]

        with pytest.raises(BufferError):
            handle.commit()
        assert not handle.committed
        assert reader.try_acquire() is None  # nothing was published

        del arr
        assert handle.commit() == 1

    def test_exports_are_counted(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=8)
        reader = Float64Reader(name, enable_zero_copy=True)
        writer.write(np.array([1.0, 2.0]))

        with reader.try_acquire() as handle:
            assert handle.exports == 0
            view = handle.payload
            assert handle.exports == 1
            # numpy exports from the memoryview, which keeps that one alive.
            arr = np.asarray(view)
            assert handle.exports == 1
            del arr, view
            assert handle.exports == 0

    def test_slot_is_returned_when_the_last_view_dies(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=2)
        reader = Float64Reader(name, enable_zero_copy=True)
        writer.write(np.array([1.0]))
        writer.write(np.array([2.0]))

        first, second = reader.try_acquire(), reader.try_acquire()
        arr = np.asarray(first.payload)
        del first  # the array is now the only thing keeping the slot alive

        # Both slots are pinned, so the ring cannot rotate at all.
        with pytest.raises(BlockingIOError):
            writer.write(np.array([3.0]))
        np.testing.assert_array_equal(arr, [1.0])

        # Dropping the last export finalises the handle and frees its slot,
        # even though nothing ever called `release()` on it.
        del arr
        writer.write(np.array([3.0]))

        second.release()

    def test_read_payload_is_read_only(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=8)
        reader = Float64Reader(name, enable_zero_copy=True)
        writer.write(np.array([1.0, 2.0]))

        with reader.try_acquire() as handle:
            view = handle.payload
            assert view.readonly
            assert not np.asarray(view).flags["WRITEABLE"]
            with pytest.raises(TypeError):
                view[0] = 9.0
            del view

    def test_write_payload_is_writable(self):
        name = _name()
        writer = Float64Writer(name, entry_length=2, entry_count=8)
        reader = Float64Reader(name)

        with writer.try_acquire() as handle:
            view = handle.payload
            assert not view.readonly
            view[0] = 5.0
            view[1] = 6.0
            del view

        np.testing.assert_array_equal(reader.read()[1], [5.0, 6.0])


# ── Exclusive roles ──────────────────────────────────────────────────────────


class TestExclusiveRoles:
    """A channel has one writer and at most one zero-copy reader. Both are
    claimed in the segment header, so the conflict is caught across processes
    rather than only within one."""

    def test_second_writer_is_rejected(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=8)

        with pytest.raises(ChannelRoleConflict):
            Float64Writer(name, entry_length=1, entry_count=8)

        assert writer.last_entry_seq == 0

    def test_second_zero_copy_reader_is_rejected(self):
        name = _name()
        _writer = Float64Writer(name, entry_length=1, entry_count=8)
        _reader = Float64Reader(name, enable_zero_copy=True)

        with pytest.raises(ChannelRoleConflict):
            Float64Reader(name, enable_zero_copy=True)

    def test_role_conflict_is_not_a_blocking_io_error(self):
        # BlockingIOError means "retry later"; a role conflict never clears on
        # its own, so it must not be caught by a retry loop.
        name = _name()
        _writer = Float64Writer(name, entry_length=1, entry_count=8)

        assert not issubclass(ChannelRoleConflict, BlockingIOError)
        with pytest.raises(ChannelRoleConflict):
            Float64Writer(name, entry_length=1, entry_count=8)

    def test_one_copy_readers_are_unrestricted(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=8)
        _zero_copy = Float64Reader(name, enable_zero_copy=True)

        readers = [Float64Reader(name) for _ in range(4)]
        writer.write(np.array([7.0]))
        for reader in readers:
            assert reader.read() == (1, pytest.approx([7.0]))

    def test_try_acquire_requires_the_flag(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=8)
        reader = Float64Reader(name)
        writer.write(np.array([1.0]))

        with pytest.raises(ValueError):
            reader.try_acquire()

    def test_roles_are_released_when_the_holder_is_collected(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=8)
        del writer

        # The role is freed on drop, so a successor can take it immediately.
        Float64Writer(name, entry_length=1, entry_count=8)


# ── Segment lifecycle ────────────────────────────────────────────────────────


class TestUnlink:
    def test_unlink_removes_the_segment(self):
        name = _name()
        writer = Float64Writer(name, entry_length=1, entry_count=8)
        writer.write(np.array([1.0]))
        del writer

        unlink(name)

        # The name is gone, so an attach-only reader finds nothing.
        reader = Float64Reader(name)
        assert reader.read() is None

    def test_unlink_is_idempotent(self):
        name = _name()
        unlink(name)
        unlink(name)

    def test_unlink_frees_a_held_role(self):
        name = _name()
        _writer = Float64Writer(name, entry_length=1, entry_count=8)

        # The segment carries the claim, so removing it clears the role even
        # though the original holder never released it.
        unlink(name)
        Float64Writer(name, entry_length=1, entry_count=8)

