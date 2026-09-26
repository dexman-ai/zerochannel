# ZeroChannel

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

High-performance, lock-free single-writer multi-reader (SWMR) inter-process communication (IPC) over shared memory. Implemented in Rust with Python bindings via PyO3.

## Overview

**ZeroChannel** provides a fast, asynchronous, and safe IPC mechanism for passing fixed-size entries between processes using a circular buffer in shared memory. It offers two access paths:

- **One-copy**: Safe for any number of readers. Copies payload out of shared memory with a double read of sequence numbers.
- **Zero-copy**: Borrow a ring slot in place without copying. The writer may acquire a writable slot, fill it in place, and commit it; the reader may acquire a pinned slot, inspect it in place, and release it after use.

Both paths are fully lock-free and work on x86_64 and aarch64 targets.

## Features

- **Lock-free**: No mutexes or spinlocks; all synchronization via atomic operations
- **Zero-copy access**: Optional read and write paths that borrow slots in place without copying
- **Cross-platform**: Linux (x86_64, aarch64) and Windows (x86_64)
- **Type-safe**: Generic `Writer<T>`/`Reader<T>` over any fixed-size element, with concrete aliases for bytes and `f64`
- **`no_std` core**: The ring protocol lives in a dependency-free `zerochannel-core` crate; the OS-facing half is a thin wrapper
- **Python support**: Full PyO3 bindings expose the channel to Python, releasing the GIL for every copy
- **Lossy reads**: Readers can safely lag behind the writer; old entries are silently skipped

## Installation

```bash
pip install zerochannel
```

## Quick Start

### Python

Entries are fixed-size: every write must be exactly `entry_length` long, and a
read returns `(entry_seq, payload)` for a single entry — or `None` when nothing
newer than the reader's watermark is available.

```python
from zerochannel import BytesWriter, BytesReader

# The writer declares the geometry, so it creates the segment.
writer = BytesWriter("/my_channel", entry_length=64, entry_count=256)
# The reader attaches to whatever the writer published.
reader = BytesReader("/my_channel")

writer.write(b"Hello from ZeroChannel!".ljust(64, b"\0"))

entry = reader.read()
if entry is not None:
    entry_seq, payload = entry
    print(entry_seq, payload.rstrip(b"\0"))  # 1 b'Hello from ZeroChannel!'
```

`Float64Writer`/`Float64Reader` measure `entry_length` in elements rather than
bytes, and read straight into a `numpy` array:

```python
import numpy as np
from zerochannel import Float64Writer, Float64Reader

writer = Float64Writer("/telemetry", entry_length=7, entry_count=256)
reader = Float64Reader("/telemetry")

writer.write(np.arange(7, dtype=np.float64))

entry_seq, payload = reader.read()       # payload is a float64 ndarray

# Or read into a buffer you own, avoiding the per-read allocation:
out = np.empty(7, dtype=np.float64)
reader.read(out=out)
```

### Zero-Copy Writing

The writer can also fill a slot in place before publishing it. Acquire the next
slot, write directly into the returned buffer, then commit it to make the entry
visible to readers.

```python
import numpy as np
from zerochannel import Float64Writer

writer = Float64Writer("/telemetry", entry_length=7, entry_count=256)

handle = writer.try_acquire()
if handle is not None:
    with handle:
        # `handle.payload` is a writable NumPy view into the ring slot.
        np.asarray(handle.payload)[:] = np.linspace(0.0, 1.0, 7)
    writer.commit(handle)
```

The view is valid only while the handle is alive. Pass that buffer to a camera,
sensor, or DMA-style producer while you are filling it, then call `commit()` to
publish it. Do not retain a pointer or array view past the handle lifetime.

### Zero-Copy Reading

For reduced latency, opt the reader into zero-copy mode. This claims the
channel's exclusive zero-copy reader role, so only one such reader may exist at
a time; ordinary readers remain unlimited.

```python
import numpy as np
from zerochannel import Float64Reader

reader = Float64Reader("/telemetry", enable_zero_copy=True)

handle = reader.try_acquire()
if handle is not None:
    with handle:
        # `payload` is a memoryview straight into the ring slot. The writer
        # skips the slot for as long as the handle holds it.
        total = np.asarray(handle.payload).sum()
    # The slot is returned to the ring at the end of the with block.
```

Any `numpy` array or `memoryview` derived from `payload` must be gone before the
handle is released — otherwise it would alias a slot the writer is free to
recycle. Releasing while one is alive raises `BufferError` rather than allowing
it, so copy the data out (or `del` the view) inside the block.

## Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
zerochannel = "0.1"
```

```rust
use zerochannel::{BytesReader, BytesWriter};

fn main() -> Result<(), zerochannel::ZeroChannelError> {
    // entry_length, entry_count, delayed_connect
    let mut writer = BytesWriter::new("/my_channel", Some(64), Some(256), false)?;
    let mut reader = BytesReader::new("/my_channel", None, None, true)?;

    writer.write(&[0u8; 64])?;

    if let Some((entry_seq, payload)) = reader.read(None)? {
        println!("{entry_seq}: {} bytes", payload.len());
    }
    Ok(())
}
```

A consumer that maps its own shared memory — or has no operating system to map
it with — can depend on `zerochannel-core` instead, which is `#![no_std]`, has
no dependencies, and operates on a caller-supplied pointer.

## Building from source

### Prerequisites

| Tool | Version |
|------|---------|
| Rust | 1.87+ (stable) |
| Python | 3.11+ |

### Development build

```bash
# Build the extension
pip install maturin
maturin develop            # editable install into current venv

# Run tests
cargo test
pytest tests/
```

Note that the PyO3 bindings sit behind a non-default `python` feature, so a
lint or check sweep must pass `--all-features` to cover them:

```bash
cargo clippy --all-targets --all-features
```

### Release build

```bash
maturin build --release
```

## Supported platforms

| Platform | Architecture | Status |
|----------|-------------|--------|
| Linux | x86_64 | ✅ |
| Linux | aarch64 | ✅ |
| Windows | x86_64 | ✅ |

## License

MIT – see [LICENSE](LICENSE).
