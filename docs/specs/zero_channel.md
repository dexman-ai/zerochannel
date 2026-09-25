# ZeroChannel — Shared-Memory IPC Subsystem

## Overview

`ZeroChannel` is a high-performance, Single-Writer Multi-Reader (SWMR) IPC
subsystem implemented in Rust and exposed to Python via PyO3. It uses a
lock-free, monotonic **entry-sequence** Seqlock over a circular buffer in shared
memory. Writes are non-blocking. Reads are safe, lossy, and asynchronous.
Payloads are fixed-size opaque byte blocks; typed access (`f64`, `i64`, `u8`) is
layered on top. The Python GIL is never held during the critical copy/spin
sections.

Two access paths are offered:

| Path | Concurrency | Mechanism |
|---|---|---|
| **One-copy** (`read` / `read_into` / `write`) | SWMR — any number of readers | Seqlock double read; payload is copied out of shared memory |
| **Zero-copy** (`try_acquire` / `release` / `commit`) | SRSW — exactly one reader | Reader flips the slot's sequence number negative with a CAS so the writer skips it, then borrows the payload in place |

The one-copy path is portable. The zero-copy path requires lock-free 64-bit
atomics, which `x86_64` and `aarch64` — the supported targets — both provide,
and must be opted into at construction (`new_zero_copy` / `enable_zero_copy=True`)
because it claims the channel's single zero-copy reader slot (§2.1).
Both paths are available from Rust and from Python; on the Python side the
zero-copy handle is itself a buffer exporter, so the borrow is tracked and the
slot cannot be recycled while a view into it is alive (§4.7).

The writer and reader endpoints both accept a shared-memory name (string) as
their sole resource identifier. Either side can create the segment. When sizes
are omitted the constructor attaches to an existing segment and reads geometry
from a header prefix block embedded at the start of the shared memory. The
writer role and the zero-copy reader role are each exclusive and are claimed in
that header, so a second claimant is rejected across process boundaries (§2.1).

---

## 1. Memory Layout & Constraints

### 1.1 Buffer Geometry

| Symbol | Meaning |
|---|---|
| `payload_bytes` (stride / item size) | Payload bytes per entry; **must be a non-zero multiple of 8** |
| `entry_length` | Elements per entry for a typed channel: `payload_bytes / sizeof(T)` |
| `entry_count` | Number of entries in the circular buffer |
| `slot_align` | `8` — required alignment and size granularity of a ring slot |
| `entry_bytes` | `8 + payload_bytes` — size in bytes of one entry |
| `header_bytes` | `72` — size of the prefix header (see §1.2) |
| `required_bytes` | `header_bytes + entry_count × entry_bytes` — total allocation size |

### 1.2 Header Prefix Block

The first 72 bytes of the shared-memory region form a header. The **first 32
bytes are immutable** — written once at creation and only ever validated
afterwards. The **remainder is mutable** — role claims and writer state that
change over the channel's life. Keeping the two apart means the words a reader
validates once are never in a line that a writer is dirtying.

| Byte Range | Content |
|---|---|
| `0..8` | `AtomicU64` — `magic`/`version`: `0x5A43484E` (`"ZCHN"`) in the high 32 bits, layout version in the low 32 |
| `8..16` | `u64` — `payload_bytes` (stride): payload bytes per entry |
| `16..24` | `u64` — `entry_count`: number of entries in the ring |
| `24..32` | `u64` — `dtype`: element type tag (`0 = u8`, `1 = i64`, `2 = f64`) |
| `32..40` | `AtomicU64` — `writer_owner`: PID of the process holding the writer role (`0` if free) |
| `40..48` | `AtomicU64` — `zero_copy_reader_owner`: PID holding the zero-copy reader role (`0` if free) |
| `48..56` | `u64` — `current_write_index`: next slot to be written by the writer |
| `56..64` | `AtomicU64` — `last_entry_seq`: highest sequence number the writer has committed (`0` before the first write) |
| `64..72` | `AtomicU64` — `segment_owner`: PID of the process responsible for unlinking the segment (`0` if free) |

The header allows a process that did not create the segment to discover the
buffer geometry, the element type and the current write cursor by reading these
fields after attaching.

The `magic`/`version` word is the segment's identity, and it is the **only**
defence against misreading a segment that is not what the caller expects. A size
check cannot substitute for it: the length an opener observes is page-rounded by
the OS, so a stale 1064-byte segment reports 4096 bytes and passes any
`len() >= required_bytes` test. A segment whose magic does not match is rejected
as "not a ZeroChannel ring"; one whose version does not match is rejected as a
layout mismatch, naming both versions.

The magic word is also the creation latch. `init_header` zeroes the region and
writes the geometry and `segment_owner` first, then stores the magic **last**
behind a `Release` fence. A magic of `0` therefore means "reserved but not yet
initialized", which lets a peer that attaches mid-initialization — or one that
finds a segment whose creator died between reserving the name and writing the
header — distinguish an unfinished segment from a finished one. Publishing
`segment_owner` under the same latch means no racing peer can claim the deed out
from under the creator.

`last_entry_seq` is the **writer's high watermark**. It is stored with a
`Release` store *after* the slot it refers to has been committed, so a reader
that loads it with `Acquire` never sees a promise the ring cannot fulfil. Its
only purpose is to let a reader answer "is there anything newer than my
watermark?" with a single load, skipping the ring scan entirely in the common
idle case.

The two role fields carry the channel's **exclusive role claims**; see §2.1.
`segment_owner` carries the **segment deed**; see §2.3.

The `dtype` tag exists because geometry alone cannot distinguish an `f64`
channel from an `i64` one — both have the same payload size. A participant
requesting `u8` is treated as **opaque** and attaches to a channel of any
declared type; any other request must match the stored tag exactly or the
constructor raises `ValueError`.

### 1.3 Entry Layout (per entry, starting at offset `header_bytes`)

| Byte Range (relative to entry start) | Content |
|---|---|
| `0..8` | `AtomicI64` `entry_seq` (Seqlock) — see the state table below |
| `8..(8 + payload_bytes)` | Opaque payload of `payload_bytes` bytes |

| `entry_seq` | Meaning |
|---|---|
| `0` | Slot is empty, or the writer is mid-write on it |
| `> 0` | Slot holds a committed entry with that sequence number |
| `< 0` | Slot holds committed entry `-entry_seq`, currently held by a zero-copy reader |

Sequence numbers start at `1` and increase by exactly one per committed entry,
so a reader detects dropped entries by comparing consecutive returned values.
The signed representation lets the reader claim a slot against the writer with a
single compare-and-swap, without adding a second control word per slot.

### 1.4 Alignment

Every slot begins with an `AtomicI64` sequence number, and both the Seqlock and
the acquire CAS are only sound if that word is naturally aligned. Because slot `i`
starts at `header_bytes + i × entry_bytes`, alignment holds for all `i` exactly
when both `header_bytes` and `entry_bytes` are multiples of `slot_align`.
`header_bytes` is `64`, and `entry_bytes` is `8 + payload_bytes`, so the
requirement reduces to:

> `payload_bytes` **must** be a multiple of `8`.

This is enforced as a constructor precondition rather than satisfied by padding:
sizes that violate it are rejected with `ValueError`. A typed channel of `f64`
or `i64` satisfies it automatically since `entry_length × 8` is always a multiple
of 8; only raw byte channels can violate it.

All `unsafe` pointer arithmetic must use exact byte offsets derived from
`header_bytes` and `entry_bytes`.

### 1.5 Ordering Guarantees

All cross-process synchronization relies on `std::sync::atomic::fence` with
`Acquire` and `Release` semantics. This is required for correctness on
weakly-ordered architectures (ARMv8/v9). **Do not** use `Mutex`, `RwLock`, or
any OS-level locking primitive.

Exactly one operation uses a read-modify-write: `RawReader::try_acquire`, which flips
`entry_seq` from `s` to `-s` with `compare_exchange`. Every other access is a
plain atomic load or store paired with a fence. Because that CAS operates on a
64-bit word, the zero-copy path requires a target with lock-free 64-bit atomics;
`x86_64` and `aarch64` both qualify. The one-copy path uses no CAS and is
portable.

### 1.6 Buffer Initialization

After allocation the entire buffer region must be zeroed (all header, sequence,
and data bytes set to `0`). The creator then writes `payload_bytes`,
`entry_count` and `dtype` into the 64-byte header prefix block (see §1.2),
leaving `current_write_index`, `last_entry_seq` and both owner fields at `0`,
and finally stores the `magic`/`version` word behind a `Release` fence. Writing
the magic last is what makes the initialization atomic from a peer's point of
view: until it lands, the segment reads as reserved-but-unfinished and no
participant will trust its geometry. `shared_memory::ShmemConf` zero-fills on
creation, but `init_header` re-zeroes anyway, since a segment may also be
initialized in place after a creator died partway through.

---

## 2. Memory Ownership

The shared-memory handle is stored as an `Option<shared_memory::Shmem>`:

* `Some(shmem)` — the segment is open and the cached `ptr` is valid.
* `None` — not yet connected (deferred). The first `read()`/`write()` call will
  attempt to open the segment.

```rust
_shmem: Option<shared_memory::Shmem>,
```

> **Critical:** If the `Shmem` is dropped, the underlying memory is
> freed/unmapped and any subsequent access through the cached raw pointer is
> **Use-After-Free**. Always keep the `Option` alive alongside the pointer.

### 2.1 Exclusive Roles

Two roles on a channel admit exactly one holder:

| Role | Header field | Claimed by |
|---|---|---|
| Writer | `writer_owner` | every `RawWriter` |
| Zero-copy reader | `zero_copy_reader_owner` | a `RawReader` built with `new_zero_copy` |

Both are unenforceable from inside a single process, because the conflicting
party is usually in a *different* one. The claim therefore lives in the segment
itself: on attach, a candidate compare-and-swaps its own PID into the field,
from `0`. Success takes the role; `Drop` compare-and-swaps it back to `0`.

Neither restriction is stylistic. **Two writers corrupt the stream
undetectably:** each keeps a private sequence counter, so they overwrite one
another's entries while still producing a dense, gapless sequence — the reader's
only drop-detection signal never fires. **Two zero-copy readers lose entries the
same way:** a slot pinned by one reader is invisible to the other, which steps
over it permanently, and again the surviving sequence numbers are dense.

A failed claim raises `RoleConflict` (`ChannelRoleConflict` in Python), which is
deliberately *not* a `Busy`/`BlockingIOError`. `Busy` means "retry later"; a role
conflict does not clear until the holder exits, so it must not be swallowed by a
retry loop.

The one-copy read path is unaffected: any number of `RawReader`s may attach to a
channel concurrently, including alongside a zero-copy reader. Only `try_acquire`
is gated, and calling it on a reader that did not opt in raises
`InvalidArgument`.

### 2.2 Liveness and Stale Claims

A PID in an owner field means nothing on its own: a process killed with
`SIGKILL` — or one that faults — never runs `Drop`, and leaves its claim behind.
Were the claim taken at face value the channel would be permanently unusable.

So a claim that loses the CAS is probed. If the recorded PID belongs to a live
process the conflict is real and `RoleConflict` is raised. If not, the claim is
stale and the candidate compare-and-swaps its own PID in place of the dead one;
if another adopter wins that race, the loser re-probes rather than assuming.

Liveness is `kill(pid, 0)` on Unix (`EPERM` counts as alive — the process exists,
it simply is not ours) and `OpenProcess` + `GetExitCodeProcess` on Windows. The
exit-code check is not redundant: a handle can outlive the process it refers to,
so an `OpenProcess` that succeeds does not by itself prove the target is
running.

PID reuse is the known limitation. A recycled PID makes a stale claim look live,
which fails closed — a spurious `RoleConflict` rather than a silently corrupted
stream — and clears as soon as the unrelated process exits. `unlink` (§2.3) is
the escape hatch.

### 2.3 Segment Lifetime

A segment name is a global resource, and something has to be responsible for
returning it. That responsibility is the **segment deed**, recorded as a PID in
`segment_owner`.

**Declaring the geometry is the bid for the deed.** A participant that passes
`entry_length` and `entry_count` is stating what the segment should look like,
which is a creator's intent whether or not it wins the race to create the
mapping. A participant that attaches without geometry is a **tenant**: it takes
the channel as it finds it and never bids.

The deed is **exclusive** — at most one live process holds it:

| Situation at attach | Outcome |
|---|---|
| Segment did not exist; this process created it | Owner (stamped by `init_header`) |
| Segment existed, `segment_owner` is `0` | Owner |
| Segment existed, `segment_owner` is a **live** PID | Tenant |
| Segment existed, `segment_owner` is a **dead** PID | Owner — the deed is adopted |
| Attached without geometry | Tenant, always |

The bid is placed only after the header has been validated. Bidding earlier
would mean a rejected open unlinks a healthy segment on its way out, destroying
a working channel because the caller passed the wrong geometry.

**On a clean exit the owner unlinks; a tenant does not.** Unlinking removes the
name, not the memory: processes already attached keep their mapping and keep
working, and the region is freed once the last of them detaches. Only the
rendezvous name disappears, so the next process to open the name creates a fresh
segment.

**A process that dies does neither**, which is the point of the liveness probe in
§2.2. The segment survives the death of its owner, so a restarted creator
re-attaches, finds a dead PID in `segment_owner`, adopts the deed, and resumes
from the previous watermark. Without adoption a single hard kill would orphan
the name permanently: every later participant would open rather than create, so
nobody would be left who unlinks it.

**The deed never moves away from a live holder.** There is no implicit transfer
and no refcount — a tenant that outlives the owner does not inherit the deed. It
keeps a working mapping to a segment whose name is already gone, which is the
intended behaviour: the owner decided the channel was over.

`unlink(name)` is the manual override, for supervisors reclaiming a name and for
test fixtures that must not leak state between runs. It succeeds whether or not
the segment exists. Use it deliberately: unlinking a live channel leaves the
existing participants talking to a segment that new arrivals can no longer find,
and those arrivals will create a second segment under the same name.

---

## 3. Writer — `RawWriter`

### 3.1 State

| Field | Type | Notes |
|---|---|---|
| `name` | `String` | OS shared-memory identifier |
| `ptr` | `*mut u8` | Start of the shared-memory region (null when deferred) |
| `payload_bytes` | `usize` | Payload bytes per entry (`0` when deferred and unknown) |
| `entry_count` | `usize` | Entries in the ring (`0` when deferred and unknown) |
| `current_index` | `usize` | Next slot to write (starts at `0`) |
| `last_entry_seq` | `u64` | Highest sequence number this writer has committed (init `0`, reseeded on attach) |
| `dtype` | `Dtype` | Element type requested by this participant |
| `delayed_connect` | `bool` | Whether deferred connection mode is active |
| `_shmem` | `Option<Shmem>` | Keeps the memory alive (`None` when deferred) |

Because the struct stores a raw pointer it does not auto-implement `Send`.
Add `unsafe impl Send for RawWriter {}` with a safety comment
explaining that the pointer targets process-shared memory whose lifetime is
governed by `_shmem`.

### 3.2 Constructor

Both Python and Rust constructors accept a shared-memory **name** (string).
Sizes (`entry_length`, `entry_count`) are **optional**. The constructor behaviour
depends on whether sizes are supplied:

| Sizes supplied? | Segment exists? | Behaviour |
|---|---|---|
| Yes | No | **Create** the segment with `header_bytes + entry_count × entry_bytes` bytes. Write `payload_bytes`, `entry_count` and `dtype` into the 64-byte header, then the magic word last (§1.6). |
| Yes | Yes | **Open** the existing segment. Validate magic, version, the supplied sizes and the element type. Raise `ValueError` on mismatch. |
| No | Yes | **Open** the existing segment. Read `payload_bytes`, `entry_count`, `current_write_index`, `dtype` and `last_entry_seq` from the header. |
| No | No, `delayed_connect=false` (default) | Raise `OSError` — the segment does not exist and sizes are unknown. |
| No | No, `delayed_connect=true` | Succeed with `_shmem = None`. Connection is retried on every subsequent `write()` call. |

Creation is attempted **first** whenever sizes are supplied, so the OS settles
the create/open race atomically: the underlying `O_EXCL` (POSIX) /
`CREATE_NEW` (Windows) creation returns "already exists" to the loser, which
then falls back to opening. No header handshake is needed to decide who creates.

Once the segment is mapped the writer claims the exclusive writer role (§2.1),
raising `RoleConflict` if a live process already holds it. A deferred writer
claims the role when it connects, not when it is constructed. The role is
released on `Drop`.

When sizes are supplied, raise `ValueError` if `payload_bytes == 0`, if
`entry_count == 0`, or if `payload_bytes` is not a multiple of `8` (see §1.4).

### 3.3 Slot Claim — `fn claim_slot(&mut self) -> Result<usize>`

Both write paths start by claiming a slot. Starting at `current_index`, probe up
to `entry_count` slots forward:

| Step | Action | Rationale |
|---|---|---|
| 1 | **Acquire load** `entry_seq` at the candidate slot. If it is **negative**, a reader holds the slot for zero-copy access — skip to the next slot. | Never overwrite a payload a reader is looking at. |
| 2 | **Relaxed store** `0` to `entry_seq`. | Signals readers that a write is in progress. |
| 3 | `fence(Release)` | Guarantees the zeroed `entry_seq` is visible to readers **before** the payload region is mutated. |

If every slot is held, return `Busy` (`BlockingIOError` in Python). This can
only happen on the zero-copy path, where a reader holds slots without releasing
them.

### 3.4 Publish — `fn publish(&mut self, index: usize) -> u64`

| Step | Action | Rationale |
|---|---|---|
| 1 | `fence(Release)` | Guarantees the completed payload write is visible **before** the new `entry_seq` is published. Pairs with the reader's `Acquire` load. |
| 2 | `seq = last_entry_seq + 1`; **Relaxed store** `seq` to the slot's `entry_seq`; record `last_entry_seq = seq`. | Publishes the entry. Sequence numbers are dense and strictly increasing, so readers can measure gaps exactly. |
| 3 | `current_index = (index + 1) % entry_count`, then persist it to the header at `16..24`. | Advance the ring and let a restarted writer resume in the right place. |
| 4 | **Release store** `seq` to the header `last_entry_seq` at `32..40`. | Publish the high watermark only after the slot is committed, so a reader's fast-path check never promises more than the ring holds. |

`publish` returns `seq`.

### 3.5 Write Protocol — `fn write(&mut self, data: &[u8])`

| Step | Action |
|---|---|
| 0 | If `_shmem` is `None`, attempt to open the segment and read the header. If still unavailable **and** `delayed_connect` is `true`, **return immediately** (no-op). If `delayed_connect` is `false`, raise `OSError`. |
| 1 | Verify `data.len() == payload_bytes`. Raise `ValueError` on mismatch. |
| 2 | `index = claim_slot()?` (§3.3). |
| 3 | `copy_nonoverlapping(data.as_ptr(), ptr + payload_offset(index), payload_bytes)`. |
| 4 | `publish(index)` (§3.4). |

> The GIL is only needed at the boundary: extracting the input buffer pointer.
> The fence/copy section runs entirely in compiled Rust with no GIL interaction.

### 3.6 Zero-Copy Write — `try_acquire` / `commit`

To build an entry directly in shared memory and avoid the staging copy:

* `try_acquire() -> Result<Option<WriteHandle>>` performs step 0 and `claim_slot()`,
  then hands back a `WriteHandle` exposing the slot's payload as `&mut [u8]` (or
  `&mut [T]`). Returns `Ok(None)` only when a deferred channel is still
  unconnected.
* `commit(handle) -> u64` runs `publish` and returns the entry's sequence number.

While a `WriteHandle` is outstanding the slot's `entry_seq` is `0`, so readers
treat it as in-progress and skip it. Dropping a handle without committing leaves
the slot empty; the next claim reuses it.

A freshly acquired slot is **not** zeroed — it still holds the bytes of the
entry written `entry_count` writes ago. Fill every byte, or zero it first.

### 3.7 Writer Restart Recovery

A writer that attaches to an existing segment must resume rather than restart
from sequence `1`, otherwise it would republish sequence numbers already in the
ring and readers would ignore or misinterpret its output.

On attach, scan all `entry_count` slots and take the maximum of
`|entry_seq|` over the ring and the header's `last_entry_seq`:

* `last_entry_seq` is seeded with that maximum, so the next published entry is
  strictly greater than anything already visible. The absolute value is used so
  so a slot held by a surviving reader still counts.
* If the slot **at** `current_write_index` already holds that maximum, the
  previous writer committed an entry but died before persisting the advanced
  index. Correct for this by resuming at
  `(current_write_index + 1) % entry_count`.

---

## 4. Reader — `RawReader`

### 4.1 State

| Field | Type | Notes |
|---|---|---|
| `name` | `String` | OS shared-memory identifier |
| `ptr` | `*mut u8` | Start of the shared-memory region (null when deferred) |
| `payload_bytes` | `usize` | Payload bytes per entry (`0` when deferred and unknown) |
| `entry_count` | `usize` | Entries in the ring (`0` when deferred and unknown) |
| `last_entry_seq` | `u64` | Sequence number of the last entry this reader consumed (init `0`) |
| `last_read_idx` | `usize` | Index of the last successfully read entry (init `0`) |
| `dtype` | `Dtype` | Element type requested by this participant |
| `delayed_connect` | `bool` | Whether deferred connection mode is active |
| `enable_zero_copy` | `bool` | Whether this reader claimed the zero-copy role and may call `try_acquire` |
| `_shmem` | `Option<Shmem>` | Keeps the memory alive (`None` when deferred) |

Add `unsafe impl Send for RawReader {}` with the same safety rationale
as the writer.

### 4.2 Constructor

Same segment semantics as the writer (§3.2). Both Reader and Writer can create
or open the shared-memory segment. When sizes are supplied and the segment does
not exist, the Reader creates it (and writes the header). When sizes are
omitted, the Reader attaches to the existing segment and reads geometry from the
header.

The `delayed_connect` parameter works identically, but **defaults to `true` for
readers**: a reader that outlives or precedes its writer is the normal case, so
construction succeeds with `_shmem = None` and every `read()` call attempts to
connect until it succeeds. Pass `delayed_connect=false` to demand that the
segment already exist.

Two constructors are offered, because the zero-copy path is exclusive (§2.1):

| Rust | Python | Effect |
|---|---|---|
| `RawReader::new` | `enable_zero_copy=False` (default) | One-copy reads only. Unrestricted: any number may attach. |
| `RawReader::new_zero_copy` | `enable_zero_copy=True` | Claims the channel's single zero-copy reader role; raises `RoleConflict` if a live process holds it. Enables `try_acquire`. |

The split is a constructor pair rather than a defaulted argument because Rust
has no default arguments, and making every caller pass `false` would be noise.
A deferred zero-copy reader claims the role when it connects. The role is
released on `Drop`.

### 4.3 Watermark Check and Slot Location

Every read starts with two cheap steps before any payload is touched.

**Watermark check.** The effective watermark is `from_entry_seq` when supplied,
otherwise the reader's own `last_entry_seq` bookmark. **Acquire load** the
header's `last_entry_seq` (§1.2); if it is less than or equal to the effective
watermark the reader is already up to date and the call returns "nothing new"
without scanning the ring.

**Slot location — `fn locate(&self, watermark: u64) -> Option<(usize, u64)>`.**
Find the **oldest** committed entry strictly newer than the watermark, i.e. the
slot whose `entry_seq` is the minimum value satisfying `entry_seq > watermark`.

> The scan **must not** stop at the first committed slot it meets. The ring is
> not scanned in publication order: the writer skips reader-held slots, and
> the ring wraps. An arbitrary committed slot may hold an entry far older than
> the watermark, or far newer than the next one the reader owes.

Rules for each slot:

| `entry_seq` | Action |
|---|---|
| `0` | Skip — empty or writer mid-write |
| `< 0` | Skip — held by the zero-copy reader |
| `<= watermark` | Skip — already seen |
| `> watermark` | Candidate; keep the smallest such value |

As a fast path, the slot at `(last_read_idx + 1) % entry_count` is probed first:
if it holds exactly `watermark + 1` it is the answer and the full scan is
skipped. Otherwise all `entry_count` slots are scanned. If no candidate is
found, the reader is up to date (or every surviving entry is older than its
watermark) and the call returns "nothing new".

Because sequence numbers are dense, a jump of more than one between consecutive
returned values tells the caller exactly how many entries the writer lapped past
it. Lap recovery needs no special case: a stale bookmark simply resolves to the
oldest surviving entry.

### 4.4 One-Copy Read — `fn read_into(&mut self, dst: &mut [u8], from_entry_seq: Option<u64>) -> Result<Option<u64>>`

Returns the sequence number of the entry copied into `dst`, or `None` when
nothing newer than the watermark is available.

| Step | Action |
|---|---|
| 0 | If `_shmem` is `None`, attempt to open the segment and read the header. If still unavailable **and** `delayed_connect` is `true`, **return `None`**. If `delayed_connect` is `false`, raise `OSError`. |
| 1 | Verify `dst.len() == payload_bytes`. Raise `ValueError` on mismatch. |
| 2 | Resolve the watermark (§4.3). If the reader is up to date, return `None`. |
| 3 | `(index, seq) = locate(watermark)` (§4.3). If there is no candidate, return `None`. |
| 4 | Copy `payload_bytes` bytes from the slot payload into `dst`. |
| 5 | `fence(Acquire)` — ensures the copy is complete before `entry_seq` is re-checked. |
| 6 | **Relaxed load** `entry_seq` at `index`. If it differs from `seq`, the entry was **torn** (the writer recycled the slot during the copy) — retry from step 3. |
| 7 | Success: set `last_entry_seq = seq`, `last_read_idx = index`, return `Some(seq)`. |

Steps 3–6 are retried at most `READ_RETRIES` (4) times; after that the call
returns `None` rather than spinning against a writer that is lapping the reader.
No CAS is involved — the double read of `entry_seq` is sufficient, because the
payload has already been copied out of shared memory.

A typed read (`read_into_as::<T>`) is identical except that the destination is a
`&mut [T]` whose byte length must equal `payload_bytes`. This is sound only for
element types that are plain-old-data with no padding and no invalid bit
patterns — the condition encoded by the `Element` marker trait.

`read` / `read_as::<T>` are allocating conveniences: they allocate a buffer of
`payload_bytes` (or `payload_bytes / size_of::<T>()` elements), call
`read_into`, and return `Option<(u64, Vec<_>)>`.

All reads are **scalar**: one call yields at most one entry. A caller that wants
to drain the channel loops until the call returns `None`.

### 4.5 Zero-Copy Read — `try_acquire` / `release`

`try_acquire(from_entry_seq) -> Result<Option<ReadHandle>>` hands back a borrowed view
of a committed slot instead of copying it. It is available only on a reader
built with `new_zero_copy` (§4.2); calling it on a one-copy reader raises
`InvalidArgument`.

| Step | Action |
|---|---|
| 0–3 | As in §4.4 (connect, watermark check, `locate`). |
| 4 | `compare_exchange(seq, -seq, AcqRel, Relaxed)` on the slot's `entry_seq`. On failure the slot was recycled between the scan and the CAS — retry from step 3. |
| 5 | Success: set `last_entry_seq = seq`, `last_read_idx = index`, and return a `ReadHandle` exposing the payload as `&[u8]` (or `&[T]`). |

The negative `entry_seq` makes the writer skip the slot (§3.3), so the borrowed
payload cannot be mutated underneath the reader. `release(handle)` returns the
slot with a plain `Release` store of `+entry_seq`.

> **Single-reader only, and enforced.** A negative `entry_seq` says "some reader
> holds this", not *which* one, so a second zero-copy reader cannot tell a slot
> it holds from one a peer holds — it steps over the peer's slot and never comes
> back to it. The loss is silent: the sequence numbers it does see stay dense,
> so its drop detection never fires. This is why the role is claimed in the
> header (§2.1) rather than merely documented. The one-copy path remains
> multi-reader and is unaffected.
>
> **Handles must be released.** A dropped-but-unreleased handle retires that
> slot from the ring permanently; holding every slot makes the writer return
> `Busy`. This inverts the channel's usual lossy-but-never-blocks property, so
> the zero-copy path is for consume-and-release loops — a consumer that needs
> to hold data for a while should copy it out.

### 4.6 Python Return Values

The Python binding returns `(entry_seq, payload)` or `None`:

* `Float64Reader.read()` returns a **newly allocated** 1-D `float64` array of
  `entry_length` elements, or fills a caller-supplied `out` array in place and
  returns it.
* `BytesReader.read()` returns a `bytes` object of `entry_length` bytes, or
  fills a caller-supplied 1-D `uint8` `out` array in place and returns it.
* Both return `None` when there is nothing new, including while deferred and
  unconnected. On a `None` result an `out` buffer is left untouched.

### 4.7 Python Handles

`try_acquire()` exposes the zero-copy path to Python with the same SRSW
restriction as §3.6 and §4.5, and requires a reader constructed with
`enable_zero_copy=True` (§4.2). It returns a `WriteHandle` / `ReadHandle`
object, or `None` when there is nothing to acquire (nothing newer than the
watermark for a reader, a still-unconnected deferred channel for a writer).

The handle itself is the buffer exporter. `handle.payload` returns a fresh
`memoryview(handle)`, formatted as `uint8` on a byte channel and `float64` on a
`float64` channel, so `numpy.asarray(handle.payload)` is a typed view that
neither allocates nor copies. A `ReadHandle` exports a **read-only** buffer, so
the view cannot be used to write into the ring; a `WriteHandle` exports a
writable one.

Because the export names the handle as its owner, CPython holds a strong
reference to the handle for as long as any view derived from it is alive, and
the handle counts its outstanding exports. That turns the caller obligation of
§4.5 into an **enforced** one:

* `release()` and `commit()` raise `BufferError` while any export is
  outstanding, including via `__exit__`.
* Nothing can dangle. A view that escapes its `with` block keeps the handle
  alive and the slot pinned, so it continues to read valid data; the slot is
  returned automatically once the last view dies and the handle is finalised.

The cost is that the escape is reported rather than tolerated: binding
`arr = numpy.asarray(handle.payload)` inside a `with` block and leaving it bound
at block exit raises `BufferError`. Either `del arr` first, or copy out
(`bytes(...)`, `numpy.array(...)`) when the data needs to outlive the handle.
Temporaries and copies release their export immediately and are unaffected.

> **Stable ABI.** `__getbuffer__` / `__releasebuffer__` are only exposed by the
> limited API from CPython 3.11 on, so the extension is built against
> `abi3-py311` and the package requires Python ≥ 3.11.

Both handle types are context managers. `ReadHandle.__exit__` always releases;
`WriteHandle.__exit__` commits on a clean exit and abandons the slot if the body
raised. Releasing is mandatory — an unreleased read handle retires its slot from
the ring permanently — so `ReadHandle` also releases on `__del__` as a backstop.

---

## 5. Sequence Numbers

`entry_seq` is a dense, strictly increasing counter over committed entries,
starting at `1`. It is stored as a signed 64-bit integer so the sign bit can
carry the "held by a reader" state (§1.3); the magnitude is the sequence
number.

Sequence numbers are per-channel. They do not correlate across channels or
processes and carry no wall-clock or monotonic-clock meaning. A caller that
needs a timestamp puts one in the payload.

On a writer restart the counter resumes from the recovered maximum (§3.7)
rather than resetting, so a reader's bookmark stays meaningful across the
restart. At one million entries per second a `u64` counter takes over 500,000
years to wrap; wrap-around is not handled.

---

## 6. Error Handling

| Condition | Behaviour |
|---|---|
| `payload_bytes == 0` or `entry_count == 0` (when supplied) | `ValueError` from constructor |
| `payload_bytes` not a multiple of `8` (when supplied) | `ValueError` from constructor |
| OS SHM creation/open fails (and `delayed_connect=false`) | `OSError` from constructor |
| Segment not found, sizes omitted, `delayed_connect=false` | `OSError` from constructor |
| Segment's magic word is not `"ZCHN"` | `ValueError` — not a ZeroChannel ring |
| Segment's layout version differs from this build's | `ValueError` naming both versions |
| Header `payload_bytes`/`entry_count` mismatch with supplied sizes | `ValueError` from constructor |
| Header `dtype` differs from a non-`u8` requested element type | `ValueError` from constructor |
| A live process already holds the writer role | `ChannelRoleConflict` from constructor |
| A live process already holds the zero-copy reader role | `ChannelRoleConflict` from constructor |
| `try_acquire()` on a reader built without `enable_zero_copy` | `ValueError` |
| `write()` called while still deferred and segment unavailable | **No-op** — return immediately without error |
| `read()` called while still deferred and segment unavailable | Return `None` — no error |
| `write()` receives a wrong-length payload | `ValueError` |
| `read(out=...)` receives a non-contiguous array, or one whose length is not `entry_length` | `ValueError` |
| `read(out=...)` receives an array of the wrong dtype | `TypeError` (raised by the argument conversion) |
| Every ring slot is held by a reader when the writer claims one | `BlockingIOError` |
| `read()` finds nothing newer than the watermark | Return `None` — not an error |
| `handle.payload` accessed after `commit()` / `release()` | `BufferError` |
| `release()` / `commit()` called while a view into `payload` is alive | `BufferError` |
| `WriteHandle.commit()` called twice | `ValueError` |
| `ReadHandle` buffer requested with `PyBUF_WRITABLE` | `BufferError` — read handles export a read-only buffer |

All Python-facing methods return `PyResult<T>`. The Rust `ZeroChannelError`
variants map as follows: `InvalidArgument → ValueError`, `OsError → OSError`,
`Busy → BlockingIOError`, `RoleConflict → ChannelRoleConflict`.

`ChannelRoleConflict` subclasses `RuntimeError`, **not** `BlockingIOError`. The
distinction is load-bearing: `Busy` is transient and a retry loop is the correct
response, whereas a role conflict persists until the holder exits, so a retry
loop would spin forever. Keeping them in separate branches of the exception
hierarchy means an existing `except BlockingIOError` retry cannot swallow it.

---

## 7. GIL Discipline

| Operation | GIL held? |
|---|---|
| Extracting the input `ndarray` slice pointer in `write()` | Yes |
| Claim + copy + fences + sequence store in `write()` | **No** |
| Watermark check, ring scan, and payload copy in `read()` | **No** |
| Allocating the output array / `bytes` object in `read()` | Yes |
| Validating a caller-supplied `out` array in `read()` | Yes |
| `try_acquire()` / `commit()` / `release()` | Yes — each is a handful of atomic operations with no copy to hide |

Use `py.detach(|| { … })` to release the GIL around the critical sections.

---

## 8. Testing Strategy

| Test | Description |
|---|---|
| **Round-trip (single-thread)** | Write N entries, drain them one at a time, verify contents, count and sequence numbers. |
| **Dense sequence numbers** | Write entries in rapid succession; verify the returned `entry_seq` increases by exactly one per entry. |
| **Watermarks** | Verify `writer.last_entry_seq`, `reader.writer_entry_seq` and `reader.last_entry_seq` track the writer and the reader independently. |
| **Lap recovery** | Write `entry_count + K` entries without reading; verify the reader resumes at the oldest surviving entry and the sequence gap matches the number of dropped entries. |
| **Torn-read detection** | Spawn a writer thread that continuously writes; verify the reader never returns corrupted data. |
| **Empty read** | Call `read()` before any writes, and again after draining; expect `None`. |
| **Pre-allocated `out`** | Read into a caller-supplied array for both byte and float channels; verify the array is filled in place and returned, that it survives reuse across reads, and that a `None` result leaves it untouched. |
| **`out` validation** | Pass a wrong-length, non-contiguous, or wrong-dtype `out`; expect `ValueError`/`TypeError`. |
| **`from_entry_seq`** | Rewind to `0` to re-read consumed entries, skip ahead past unread ones, and pass the writer's watermark to get `None`. |
| **Writer acquire/commit** | Build an entry in place through a `WriteHandle`; verify the committed sequence number and payload, and that an uncommitted handle leaves the slot empty. |
| **Reader acquire/release** | Acquire an entry; verify the slot's `entry_seq` is negative, that the writer skips it, that the borrowed payload is correct, and that `release` restores the slot. |
| **Slot exhaustion** | Hold every slot; expect the writer to return `Busy`. |
| **Python handles** | Round-trip both channel types through `try_acquire()`; verify `payload` is a non-owning typed view, that the writer skips a held slot, that `try_acquire()` returns `None` on an empty channel and raises `BlockingIOError` once every slot is held, and that handles share the reader's bookmark with `read()`. |
| **Python handle lifecycle** | Verify `payload` raises after `release`/`commit`, that a second `release` is a no-op while a second `commit` raises, and that the context manager releases a read handle on an exception but abandons a write handle. |
| **Python export tracking** | Verify `release`/`commit` raise `BufferError` while a derived view is alive (including from `__exit__`), that exports are counted, that a read payload is read-only and a write payload is writable, and that the slot is returned to the ring once the last view dies without an explicit `release`. |
| **OS SHM backing** | Construct writer/reader over a named SHM segment; run a round-trip. |
| **Delayed connect** | Construct reader with `delayed_connect=True` before the writer creates the segment; verify that the reader connects on the first `read()` after the writer is up, including when an `out` array is supplied. |
| **Writer restart** | Drop a writer and attach a new one; verify it resumes from the recovered high watermark rather than from `1`. |
| **Header validation** | Create a segment with known sizes, then open with mismatched sizes; expect `ValueError`. |
| **Segment identity** | Attach to a segment whose first word is not the magic; expect `ValueError`. Overwrite the version field and expect a second, distinct `ValueError`. |
| **Exclusive roles** | Attach a second writer to a live channel, and a second `enable_zero_copy` reader; expect `ChannelRoleConflict` for each. Verify it is not a `BlockingIOError` subclass, that any number of one-copy readers attach alongside a zero-copy one, and that `try_acquire` on a one-copy reader raises. |
| **Role release and adoption** | Drop a role holder and verify a successor claims the role. Write a dead PID into an owner field and verify the next candidate adopts the stale claim rather than failing. |
| **Unlink** | Unlink a segment and verify an attach-only reader then finds nothing, that unlinking an absent segment succeeds, and that unlinking frees a role its holder never released. |
| **Segment ownership** | Verify a geometry-supplying participant unlinks on a clean drop while a tenant does not; that a second geometry supplier attaching to a live owner's channel is a tenant, so its exit leaves the channel intact; that a segment left with a dead PID in `segment_owner` is adopted by the next geometry supplier and reclaimed on its clean exit; and that a tenant outliving its owner does not inherit the deed. |
| **Alignment rule** | Construct a byte channel with a `payload_bytes` that is not a multiple of 8; expect `ValueError`. Verify every slot offset is 8-byte aligned for a range of legal payload sizes. |
| **Byte round-trip** | Write and read back a 4096-byte block on a raw byte channel; verify contents. |
| **Cross-type read** | Write `f64` entries, read them back through an opaque byte reader, and verify the little-endian decoding matches. |
| **Dtype tagging** | Attach an `i64` reader to an `f64` channel of identical geometry; expect `ValueError`. Verify an opaque `u8` reader attaches successfully. |

---

## 9. API Signatures

### 9.1 Python

The Python layer cannot expose a generic type, so it ships one concrete
writer/reader pair per element type. All four classes share the same
constructor signature and the same `entry_length` semantics — *elements per
entry* — where the element is `bytes` for `BytesWriter`/`BytesReader` and
`float64` for `Float64Writer`/`Float64Reader`.

`BytesWriter`/`BytesReader` declare the opaque `u8` element type, so they can
also attach to a channel created by a typed endpoint and observe its raw bytes.
A `Float64*` endpoint, by contrast, only attaches to a `float64` channel.

```python
class BytesWriter:
    """Single-writer end of a byte-oriented ZeroChannel."""

    def __init__(
        self,
        name: str,
        *,
        entry_length: int | None = None,
        entry_count: int | None = None,
        delayed_connect: bool = False,
    ) -> None:
        """Create or attach to a shared-memory channel for writing bytes.

        Parameters
        ----------
        name : str
            OS shared-memory identifier.
        entry_length : int, optional
            Payload **bytes** per entry. Must be a non-zero multiple of 8.
            Required when creating a new segment.
        entry_count : int, optional
            Number of entries in the circular buffer. Required when
            creating a new segment.
        delayed_connect : bool
            If ``True`` and the segment does not yet exist (and sizes are
            not supplied), construction succeeds and the first ``write()``
            call will attempt to connect.

        Raises
        ------
        ValueError
            If a size is zero, if ``entry_length`` is not a multiple of 8, or
            if the header of an existing segment does not match the
            supplied sizes.
        OSError
            If the segment cannot be created/opened and
            ``delayed_connect`` is ``False``.
        ChannelRoleConflict
            If a live process already holds the channel's writer role.
        """
        ...

    @property
    def entry_length(self) -> int:
        """Payload bytes per entry (0 while deferred and unconnected)."""
        ...

    @property
    def last_entry_seq(self) -> int:
        """Sequence number of the last entry this writer committed (0 if none)."""
        ...

    def write(self, data: bytes) -> None:
        """Write a single entry to the ring buffer.

        If the channel is in deferred mode and still not connected, this
        method silently returns without writing (no-op).

        Raises
        ------
        ValueError
            If ``len(data) != entry_length``.
        BlockingIOError
            If every ring slot is currently held by a reader.
        """
        ...

    def try_acquire(self) -> WriteHandle | None:
        """Acquire the next ring slot for in-place construction.

        Returns a ``WriteHandle`` whose ``payload`` is a writable ``uint8``
        ``memoryview`` of ``entry_length`` bytes, or ``None`` if the channel
        is in deferred mode and still not connected. The entry becomes
        visible to readers only on ``commit()``.

        The slot is not zeroed: it still holds the bytes of the entry
        written ``entry_count`` writes ago.

        Raises
        ------
        BlockingIOError
            If every ring slot is currently held by a reader.
        """
        ...


class BytesReader:
    """Multi-reader end of a byte-oriented ZeroChannel."""

    def __init__(
        self,
        name: str,
        *,
        entry_length: int | None = None,
        entry_count: int | None = None,
        delayed_connect: bool = True,
        enable_zero_copy: bool = False,
    ) -> None:
        """Create or attach to a shared-memory channel for reading bytes.

        Same parameters and errors as ``BytesWriter.__init__``, except that
        in deferred mode every ``read()`` call retries the connection, and
        that ``delayed_connect`` defaults to ``True`` — a reader outliving or
        preceding its writer is the normal case.

        Parameters
        ----------
        enable_zero_copy : bool
            Opt into ``try_acquire()``. Claims the channel's single zero-copy
            reader role; one-copy readers remain unrestricted.

        Raises
        ------
        ChannelRoleConflict
            If ``enable_zero_copy`` is ``True`` and a live process already
            holds the zero-copy reader role.
        """
        ...

    @property
    def entry_length(self) -> int:
        """Payload bytes per entry (0 while deferred and unconnected)."""
        ...

    @property
    def last_entry_seq(self) -> int:
        """Sequence number of the last entry this reader consumed (0 if none)."""
        ...

    @property
    def writer_entry_seq(self) -> int:
        """The writer's high watermark, read from the channel header."""
        ...

    def read(
        self,
        *,
        out: numpy.ndarray | None = None,
        from_entry_seq: int | None = None,
    ) -> tuple[int, bytes] | tuple[int, numpy.ndarray] | None:
        """Read the next entry newer than the watermark.

        Returns ``(entry_seq, payload)``, or ``None`` when nothing newer is
        available — including while still deferred and unconnected. At most
        one entry is returned per call; loop until ``None`` to drain the
        channel.

        Because sequence numbers are dense, a jump of more than one between
        consecutive ``entry_seq`` values tells the caller exactly how many
        entries were lost to the writer lapping the ring.

        Parameters
        ----------
        out : numpy.ndarray, optional
            A pre-allocated, contiguous 1-D ``uint8`` array of exactly
            ``entry_length`` elements. When given, the payload is copied
            into it in place and the same array is returned as the second
            tuple element. When omitted, a new ``bytes`` object is
            returned instead. On a ``None`` result ``out`` is untouched.
        from_entry_seq : int, optional
            Low-watermark sequence number. Only an entry with
            ``entry_seq > from_entry_seq`` is returned, and the oldest such
            entry is chosen. Overrides the reader's internal bookmark for
            this call; a successful read then advances the bookmark to the
            returned value.

        Raises
        ------
        ValueError
            If ``out`` is not contiguous or does not hold exactly
            ``entry_length`` elements.
        """
        ...

    def try_acquire(self, *, from_entry_seq: int | None = None) -> ReadHandle | None:
        """Acquire the next entry newer than the watermark, in place.

        Returns a ``ReadHandle`` borrowing the entry as a ``uint8``
        ``memoryview``, or ``None`` when nothing newer is available.
        ``from_entry_seq`` overrides the reader's bookmark exactly as for
        ``read()``.

        Single-reader only, and the handle must be released — see
        ``ReadHandle``.

        Raises
        ------
        ValueError
            If this reader was not constructed with ``enable_zero_copy``.
        """
        ...


class Float64Writer:
    """Single-writer end of a ``float64`` ZeroChannel."""

    def __init__(
        self,
        name: str,
        *,
        entry_length: int | None = None,
        entry_count: int | None = None,
        delayed_connect: bool = False,
    ) -> None:
        """Create or attach to a shared-memory channel for writing float64.

        Parameters
        ----------
        name : str
            OS shared-memory identifier.
        entry_length : int, optional
            Number of ``float64`` **elements** per entry; the channel
            reserves ``entry_length * 8`` payload bytes. Required when
            creating a new segment.
        entry_count : int, optional
            Number of entries in the circular buffer. Required when
            creating a new segment.
        delayed_connect : bool
            If ``True`` and the segment does not yet exist (and sizes are
            not supplied), construction succeeds and the first ``write()``
            call will attempt to connect.

        Raises
        ------
        ValueError
            If a size is zero, if the existing segment's header does not
            match the supplied sizes, or if it declares a different
            element type.
        OSError
            If the segment cannot be created/opened and
            ``delayed_connect`` is ``False``.
        ChannelRoleConflict
            If a live process already holds the channel's writer role.
        """
        ...

    @property
    def entry_length(self) -> int:
        """Number of float64 elements per entry (0 while deferred)."""
        ...

    @property
    def last_entry_seq(self) -> int:
        """Sequence number of the last entry this writer committed (0 if none)."""
        ...

    def write(self, data: numpy.ndarray) -> None:
        """Write a single entry to the ring buffer.

        If the channel is in deferred mode and still not connected, this
        method silently returns without writing (no-op). No error is
        raised in this case.

        Parameters
        ----------
        data : numpy.ndarray
            A 1-D ``float64`` array of length ``entry_length``.

        Raises
        ------
        ValueError
            If ``data.shape[0] != entry_length``.
        BlockingIOError
            If every ring slot is currently held by a reader.
        """
        ...

    def try_acquire(self) -> WriteHandle | None:
        """Acquire the next ring slot for in-place construction.

        Returns a ``WriteHandle`` whose ``payload`` is a writable ``float64``
        ``memoryview`` of ``entry_length`` elements, or ``None`` if the
        channel is in deferred mode and still not connected. The entry
        becomes visible to readers only on ``commit()``.

        The slot is not zeroed: it still holds the values of the entry
        written ``entry_count`` writes ago.

        Raises
        ------
        BlockingIOError
            If every ring slot is currently held by a reader.
        """
        ...


class Float64Reader:
    """Multi-reader end of a ``float64`` ZeroChannel."""

    def __init__(
        self,
        name: str,
        *,
        entry_length: int | None = None,
        entry_count: int | None = None,
        delayed_connect: bool = True,
        enable_zero_copy: bool = False,
    ) -> None:
        """Create or attach to a shared-memory channel for reading float64.

        Same parameters and errors as ``Float64Writer.__init__``, except
        that in deferred mode every ``read()`` call retries the connection,
        and that ``delayed_connect`` defaults to ``True`` — a reader
        outliving or preceding its writer is the normal case.

        Parameters
        ----------
        enable_zero_copy : bool
            Opt into ``try_acquire()``. Claims the channel's single zero-copy
            reader role; one-copy readers remain unrestricted.

        Raises
        ------
        ChannelRoleConflict
            If ``enable_zero_copy`` is ``True`` and a live process already
            holds the zero-copy reader role.
        """
        ...

    @property
    def entry_length(self) -> int:
        """Number of float64 elements per entry (0 while deferred)."""
        ...

    @property
    def last_entry_seq(self) -> int:
        """Sequence number of the last entry this reader consumed (0 if none)."""
        ...

    @property
    def writer_entry_seq(self) -> int:
        """The writer's high watermark, read from the channel header."""
        ...

    def read(
        self,
        *,
        out: numpy.ndarray | None = None,
        from_entry_seq: int | None = None,
    ) -> tuple[int, numpy.ndarray] | None:
        """Read the next entry newer than the watermark.

        Returns ``(entry_seq, payload)`` where ``payload`` is a 1-D
        ``float64`` array of ``entry_length`` elements, or ``None`` when
        nothing newer is available — including while still deferred and
        unconnected. At most one entry is returned per call; loop until
        ``None`` to drain the channel.

        Parameters
        ----------
        out : numpy.ndarray, optional
            A pre-allocated, contiguous 1-D ``float64`` array of exactly
            ``entry_length`` elements. When given, the payload is copied
            into it in place and the same array is returned as the second
            tuple element. When omitted, a new array is allocated. On a
            ``None`` result ``out`` is untouched.
        from_entry_seq : int, optional
            Low-watermark sequence number. Only an entry with
            ``entry_seq > from_entry_seq`` is returned, and the oldest such
            entry is chosen. Overrides the reader's internal bookmark for
            this call; a successful read then advances the bookmark to the
            returned value.

        Raises
        ------
        ValueError
            If ``out`` is not contiguous or does not hold exactly
            ``entry_length`` elements.
        """
        ...

    def try_acquire(self, *, from_entry_seq: int | None = None) -> ReadHandle | None:
        """Acquire the next entry newer than the watermark, in place.

        Returns a ``ReadHandle`` borrowing the entry as a ``float64``
        ``memoryview``, or ``None`` when nothing newer is available.
        ``from_entry_seq`` overrides the reader's bookmark exactly as for
        ``read()``.

        Single-reader only, and the handle must be released — see
        ``ReadHandle``.

        Raises
        ------
        ValueError
            If this reader was not constructed with ``enable_zero_copy``.
        """
        ...


class ReadHandle:
    """A borrowed, in-place view of one committed entry.

    The slot's ``entry_seq`` is negative for the lifetime of the handle, so
    the writer skips it and the payload cannot change underfoot. The handle
    is a buffer exporter: any view derived from ``payload`` keeps it alive
    and blocks ``release()`` until the view dies (§4.7).
    """

    @property
    def entry_seq(self) -> int:
        """Sequence number of the held entry."""
        ...

    @property
    def payload(self) -> memoryview:
        """A fresh read-only ``memoryview`` of the entry.

        Raises
        ------
        BufferError
            If the handle has already been released.
        """
        ...

    @property
    def released(self) -> bool:
        """True once the slot has been handed back to the ring."""
        ...

    @property
    def exports(self) -> int:
        """Number of outstanding buffer exports."""
        ...

    def release(self) -> None:
        """Return the slot to the ring. Idempotent.

        Raises
        ------
        BufferError
            If any view derived from ``payload`` is still alive.
        """
        ...

    def __enter__(self) -> ReadHandle: ...

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Always release, whether or not the block raised."""
        ...


class WriteHandle:
    """An exclusive, not-yet-published view of one ring slot.

    The slot's ``entry_seq`` is ``0`` while the handle is held, so readers
    skip it. An abandoned handle leaves the slot empty for the next write.
    The slot is not zeroed on acquisition: it still holds the bytes of the
    entry written ``entry_count`` writes ago.
    """

    @property
    def payload(self) -> memoryview:
        """A fresh writable ``memoryview`` of the entry.

        Raises
        ------
        BufferError
            If the handle has already been committed.
        """
        ...

    @property
    def committed(self) -> bool:
        """True once the entry has been published."""
        ...

    @property
    def exports(self) -> int:
        """Number of outstanding buffer exports."""
        ...

    def commit(self) -> int:
        """Publish the entry and return its sequence number.

        Raises
        ------
        ValueError
            If the handle has already been committed.
        BufferError
            If any view derived from ``payload`` is still alive.
        """
        ...

    def __enter__(self) -> WriteHandle: ...

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Commit on a clean exit; abandon the slot if the block raised."""
        ...


class ChannelRoleConflict(RuntimeError):
    """An exclusive channel role is already held by a live process.

    Raised when a second writer, or a second ``enable_zero_copy`` reader,
    attaches to a channel that already has one. Deliberately *not* a
    subclass of ``BlockingIOError``: a role conflict is not transient and
    retrying will not clear it.
    """


def unlink(name: str) -> None:
    """Remove a channel's shared-memory segment by name.

    The escape hatch for reclaiming a leaked segment: a channel abandoned by
    a crashed process keeps its name until something removes it. Processes
    already attached keep their mapping — only the name goes away, so the
    next open creates a fresh segment. Does nothing if the segment is absent.
    """
    ...
```

### 9.2 Rust

The Rust API is layered: a byte-oriented core (`RawWriter` / `RawReader`) that
owns the Seqlock protocol, and zero-cost typed wrappers (`Writer<T>` /
`Reader<T>`) on top.

#### Byte core

```rust
use shared_memory::Shmem;

/// Single-writer end of a ZeroChannel ring buffer, operating on raw bytes.
pub struct RawWriter {
    name: String,
    ptr: *mut u8,
    payload_bytes: usize,
    entry_count: usize,
    current_index: usize,
    last_entry_seq: u64,
    dtype: Dtype,
    delayed_connect: bool,
    _shmem: Option<Shmem>,
}

unsafe impl Send for RawWriter {}

/// A borrowed, in-place view of one uncommitted ring slot.
pub struct WriteHandle { /* .. */ }

unsafe impl Send for WriteHandle {}

impl WriteHandle {
    pub fn index(&self) -> usize { .. }
    pub fn as_bytes_mut(&mut self) -> &mut [u8] { .. }
    pub fn as_slice_mut<T: Element>(&mut self) -> Result<&mut [T], ZeroChannelError> { .. }
}

impl RawWriter {
    /// Create or open a shared-memory channel for writing.
    ///
    /// * If `payload_bytes` and `entry_count` are `Some`, creates the segment
    ///   (or validates an existing one). `payload_bytes` must be a non-zero
    ///   multiple of 8.
    /// * If both are `None`, attaches to an existing segment and reads
    ///   geometry from the header.
    /// * If the segment does not exist and `delayed_connect` is `true`,
    ///   returns a deferred writer that will attempt to connect on every
    ///   subsequent `write()` call (never errors on write when deferred).
    ///
    /// Claims the channel's exclusive writer role once mapped, returning
    /// `RoleConflict` if a live process already holds it. A deferred writer
    /// claims the role when it connects. The role is released on `Drop`.
    pub fn new(
        name: &str,
        payload_bytes: Option<usize>,
        entry_count: Option<usize>,
        dtype: Dtype,
        delayed_connect: bool,
    ) -> Result<Self, ZeroChannelError> { .. }

    /// Payload bytes per entry (`0` while deferred and unconnected).
    pub fn payload_bytes(&self) -> usize { .. }

    /// Sequence number of the last entry this writer committed (`0` if none).
    pub fn last_entry_seq(&self) -> u64 { .. }

    /// Write one entry to the ring buffer (one-copy path).
    ///
    /// Returns an error if `data.len() != payload_bytes`, or `Busy` if every
    /// slot is held by a reader. If the channel is deferred and still
    /// not connected, returns `Ok(())` (no-op).
    pub fn write(&mut self, data: &[u8]) -> Result<(), ZeroChannelError> { .. }

    /// Claim the next ring slot for in-place construction (zero-copy path).
    ///
    /// Returns `Ok(None)` only when a deferred channel is still unconnected.
    /// The handle must be passed to `commit` to become visible to readers.
    /// The slot is not zeroed: it still holds the bytes of the entry written
    /// `entry_count` writes ago.
    pub fn try_acquire(&mut self) -> Result<Option<WriteHandle>, ZeroChannelError> { .. }

    /// Publish a held slot and return its sequence number.
    pub fn commit(&mut self, handle: WriteHandle) -> u64 { .. }
}

/// Multi-reader end of a ZeroChannel ring buffer, operating on raw bytes.
pub struct RawReader {
    name: String,
    ptr: *mut u8,
    payload_bytes: usize,
    entry_count: usize,
    last_entry_seq: u64,
    last_read_idx: usize,
    dtype: Dtype,
    delayed_connect: bool,
    enable_zero_copy: bool,
    _shmem: Option<Shmem>,
}

unsafe impl Send for RawReader {}

/// A borrowed, in-place view of one committed ring slot.
///
/// The slot's `entry_seq` stays negative — and the writer keeps skipping it —
/// until the handle is handed back to `RawReader::release`.
pub struct ReadHandle { /* .. */ }

unsafe impl Send for ReadHandle {}

impl ReadHandle {
    pub fn entry_seq(&self) -> u64 { .. }
    pub fn index(&self) -> usize { .. }
    pub fn as_bytes(&self) -> &[u8] { .. }
    pub fn as_slice<T: Element>(&self) -> Result<&[T], ZeroChannelError> { .. }
}

impl RawReader {
    /// Create or open a shared-memory channel for one-copy reading.
    ///
    /// Same segment semantics as `RawWriter::new`. Unrestricted: any number
    /// of one-copy readers may attach to a channel concurrently. Calling
    /// `try_acquire` on a reader built this way returns `InvalidArgument`.
    pub fn new(
        name: &str,
        payload_bytes: Option<usize>,
        entry_count: Option<usize>,
        dtype: Dtype,
        delayed_connect: bool,
    ) -> Result<Self, ZeroChannelError> { .. }

    /// Create or open a shared-memory channel for zero-copy reading.
    ///
    /// As `new`, but claims the channel's exclusive zero-copy reader role and
    /// enables `try_acquire`. Returns `RoleConflict` if a live process already
    /// holds the role. A deferred reader claims it when it connects; the role
    /// is released on `Drop`.
    ///
    /// This is a second constructor rather than an argument on `new` because
    /// Rust has no default arguments, and the zero-copy path is the rarer
    /// case.
    pub fn new_zero_copy(
        name: &str,
        payload_bytes: Option<usize>,
        entry_count: Option<usize>,
        dtype: Dtype,
        delayed_connect: bool,
    ) -> Result<Self, ZeroChannelError> { .. }

    /// Payload bytes per entry (`0` while deferred and unconnected).
    pub fn payload_bytes(&self) -> usize { .. }

    /// Sequence number of the last entry this reader consumed (`0` if none).
    pub fn last_entry_seq(&self) -> u64 { .. }

    /// The writer's high watermark, loaded from the channel header.
    pub fn writer_entry_seq(&self) -> u64 { .. }

    /// Attach a deferred channel if the segment now exists.
    ///
    /// Returns `Ok(false)` when the segment is still missing. Callers use this
    /// to resolve `payload_bytes` before allocating a destination buffer.
    pub fn connect(&mut self) -> Result<bool, ZeroChannelError> { .. }

    /// Copy the next entry into `dst` (one-copy path), as raw bytes.
    ///
    /// * `dst` — must be exactly `payload_bytes` long.
    /// * `from_entry_seq` — optional low watermark. Only an entry with
    ///   `entry_seq > from_entry_seq` is returned, and the oldest such entry
    ///   is chosen. Overrides the internal bookmark for this call.
    ///
    /// Returns the entry's sequence number, or `None` when nothing newer is
    /// available or the channel is deferred and still unconnected.
    pub fn read_into(
        &mut self,
        dst: &mut [u8],
        from_entry_seq: Option<u64>,
    ) -> Result<Option<u64>, ZeroChannelError> { .. }

    /// Copy the next entry into `dst` (one-copy path), as elements of `T`.
    pub fn read_into_as<T: Element>(
        &mut self,
        dst: &mut [T],
        from_entry_seq: Option<u64>,
    ) -> Result<Option<u64>, ZeroChannelError> { .. }

    /// Read the next entry (one-copy path), allocating the destination.
    pub fn read(
        &mut self,
        from_entry_seq: Option<u64>,
    ) -> Result<Option<(u64, Vec<u8>)>, ZeroChannelError> { .. }

    /// Read the next entry as elements of `T`, allocating the destination.
    ///
    /// Errors if `payload_bytes` is not a whole number of `T` elements.
    pub fn read_as<T: Element>(
        &mut self,
        from_entry_seq: Option<u64>,
    ) -> Result<Option<(u64, Vec<T>)>, ZeroChannelError> { .. }

    /// Acquire the next entry in place (zero-copy path).
    ///
    /// Requires a reader built with `new_zero_copy`; otherwise returns
    /// `InvalidArgument`. Single-reader only, and requires a target with
    /// lock-free 64-bit atomics (`x86_64` and `aarch64` both qualify).
    pub fn try_acquire(
        &mut self,
        from_entry_seq: Option<u64>,
    ) -> Result<Option<ReadHandle>, ZeroChannelError> { .. }

    /// Return a held slot to the ring.
    pub fn release(&mut self, handle: ReadHandle) { .. }
}
```

#### Typed layer

`Writer<T>` and `Reader<T>` are zero-cost wrappers over `RawWriter`/`RawReader`
that express sizes in elements of `T` rather than bytes and record `T`'s type
tag in the channel header.

```rust
/// Element type recorded in the channel header.
pub enum Dtype { U8, I64, F64 }

/// Marker for types that may be stored directly in a ZeroChannel payload.
///
/// # Safety
///
/// Implementors must be plain-old-data: no padding bytes, every bit pattern a
/// valid value, and `align_of::<Self>() <= 8`.
pub unsafe trait Element: Copy + 'static {
    const DTYPE: Dtype;
    const ZERO: Self;
}

pub struct Writer<T: Element> { /* .. */ }
pub struct Reader<T: Element> { /* .. */ }

impl<T: Element> Writer<T> {
    /// `entry_length` counts elements of `T`; the byte payload is
    /// `entry_length × size_of::<T>()`.
    pub fn new(
        name: &str,
        entry_length: Option<usize>,
        entry_count: Option<usize>,
        delayed_connect: bool,
    ) -> Result<Self, ZeroChannelError> { .. }

    pub fn entry_length(&self) -> usize { .. }
    pub fn last_entry_seq(&self) -> u64 { .. }
    pub fn write(&mut self, data: &[T]) -> Result<(), ZeroChannelError> { .. }
    pub fn try_acquire(&mut self) -> Result<Option<WriteHandle>, ZeroChannelError> { .. }
    pub fn commit(&mut self, handle: WriteHandle) -> u64 { .. }
}

impl<T: Element> Reader<T> {
    /// One-copy reader. Unrestricted; `try_acquire` is unavailable.
    pub fn new(
        name: &str,
        entry_length: Option<usize>,
        entry_count: Option<usize>,
        delayed_connect: bool,
    ) -> Result<Self, ZeroChannelError> { .. }

    /// Zero-copy reader. Claims the channel's exclusive zero-copy reader role
    /// and enables `try_acquire`; returns `RoleConflict` if it is taken.
    pub fn new_zero_copy(
        name: &str,
        entry_length: Option<usize>,
        entry_count: Option<usize>,
        delayed_connect: bool,
    ) -> Result<Self, ZeroChannelError> { .. }

    pub fn entry_length(&self) -> usize { .. }
    pub fn last_entry_seq(&self) -> u64 { .. }
    pub fn writer_entry_seq(&self) -> u64 { .. }

    pub fn read(
        &mut self,
        from_entry_seq: Option<u64>,
    ) -> Result<Option<(u64, Vec<T>)>, ZeroChannelError> { .. }

    pub fn read_into(
        &mut self,
        dst: &mut [T],
        from_entry_seq: Option<u64>,
    ) -> Result<Option<u64>, ZeroChannelError> { .. }

    pub fn try_acquire(
        &mut self,
        from_entry_seq: Option<u64>,
    ) -> Result<Option<ReadHandle>, ZeroChannelError> { .. }

    pub fn release(&mut self, handle: ReadHandle) { .. }
}

pub type BytesWriter = Writer<u8>;
pub type BytesReader = Reader<u8>;
pub type Float64Writer = Writer<f64>;
pub type Float64Reader = Reader<f64>;

/// Errors returned across the whole API.
pub enum ZeroChannelError {
    /// A precondition on an argument, or on an existing segment's header,
    /// was violated. Maps to `ValueError`.
    InvalidArgument(String),
    /// The OS refused to create, open, or unlink a segment. Maps to `OSError`.
    OsError(String),
    /// Transient back-pressure: every ring slot is held by a reader. Worth
    /// retrying. Maps to `BlockingIOError`.
    Busy(String),
    /// An exclusive role on the channel is already held by a live process.
    /// Persists until the holder exits, so retrying is futile. Maps to
    /// `ChannelRoleConflict`.
    RoleConflict(String),
}

/// Remove a channel's shared-memory segment by name.
///
/// Processes already attached keep their mapping; only the name goes away, so
/// the next open creates a fresh segment. Succeeds whether or not the segment
/// exists.
pub fn unlink(name: &str) -> Result<(), ZeroChannelError> { .. }
```

> The PyO3 bindings sit on the **raw** layer rather than the typed one:
> `#[pyclass]` cannot be generic. Python therefore gets one concrete class per
> element type — `BytesWriter`/`BytesReader` and `Float64Writer`/`Float64Reader`
> — mirroring the Rust aliases above. The generic typed layer is for Rust
> consumers.

---

## 10. Usage Examples

### 10.1 Python — Basic Round-Trip

```python
import numpy as np
from zerochannel import Float64Writer, Float64Reader

# Writer creates the segment with 6 doubles per entry and 100 ring slots.
writer = Float64Writer("my_channel", entry_length=6, entry_count=100)

# Reader attaches to the same segment (sizes read from header).
reader = Float64Reader("my_channel")

# Write a single entry.
writer.write(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

# Read the next entry. Returns (entry_seq, payload) or None.
entry_seq, data = reader.read()
print(entry_seq, data)        # 1 [1. 2. 3. 4. 5. 6.]
assert reader.read() is None  # nothing newer
```

### 10.2 Python — Draining into a Reused Buffer

```python
import numpy as np
from zerochannel import Float64Reader

reader = Float64Reader("my_channel")
buf = np.empty(reader.entry_length)

expected = reader.last_entry_seq + 1
while (entry := reader.read(out=buf)) is not None:
    entry_seq, _ = entry          # `_` is `buf`, filled in place
    if entry_seq != expected:
        print(f"dropped {entry_seq - expected} entries")
    expected = entry_seq + 1
    consume(buf)
```

### 10.3 Python — Delayed Connect

```python
import numpy as np
from zerochannel import Float64Writer, Float64Reader

# Reader starts before the writer — segment doesn't exist yet. Readers
# default to delayed_connect, so this is simply the default behaviour.
reader = Float64Reader("late_channel")

# ... later, in another process or thread ...
writer = Float64Writer("late_channel", entry_length=3, entry_count=50)
writer.write(np.array([0.1, 0.2, 0.3]))

# Reader connects on first read() after the segment is available.
entry_seq, data = reader.read()   # (1, array([0.1, 0.2, 0.3]))
```

### 10.4 Rust — Basic Round-Trip

```rust
use zerochannel::Float64Writer;
use zerochannel::Float64Reader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Writer creates the segment.
    let mut writer = Float64Writer::new(
        "rust_channel",
        Some(4),   // entry_length
        Some(64),  // entry_count
        false,     // delayed_connect
    )?;

    // Reader attaches (sizes from header).
    let mut reader = Float64Reader::new(
        "rust_channel",
        None,   // entry_length — read from header
        None,   // entry_count — read from header
        false,
    )?;

    writer.write(&[1.0, 2.0, 3.0, 4.0])?;

    let (entry_seq, payload) = reader.read(None)?.expect("one entry");
    assert_eq!(entry_seq, 1);
    assert_eq!(payload, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(reader.read(None)?.is_none());

    Ok(())
}
```

### 10.5 Rust — Delayed Connect

```rust
use zerochannel::{Float64Reader, Float64Writer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Reader starts before the writer.
    let mut reader = Float64Reader::new(
        "late_rust",
        None,
        None,
        true,   // delayed_connect
    )?;

    // Writer creates the segment later.
    let mut writer = Float64Writer::new(
        "late_rust",
        Some(2),
        Some(32),
        false,
    )?;

    writer.write(&[9.0, 8.0])?;

    // Reader connects on first read().
    let (_seq, payload) = reader.read(None)?.expect("one entry");
    assert_eq!(payload, vec![9.0, 8.0]);

    Ok(())
}
```

### 10.6 Rust — Raw Byte Channel

```rust
use zerochannel::{BytesReader, BytesWriter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 4096-byte blocks — a multiple of 8, as required by §1.4.
    let mut writer = BytesWriter::new("audio", Some(4096), Some(64), false)?;
    let mut reader = BytesReader::new("audio", None, None, false)?;

    let block = vec![0u8; 4096];
    writer.write(&block)?;

    let (_seq, payload) = reader.read(None)?.expect("one entry");
    assert_eq!(payload.len(), 4096);

    Ok(())
}
```

### 10.7 Rust — Zero-Copy Handles (SRSW only)

```rust
use zerochannel::{Float64Reader, Float64Writer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = Float64Writer::new("zc", Some(3), Some(16), false)?;
    // `new_zero_copy` claims the channel's single zero-copy reader role; a
    // reader from plain `new` would reject `try_acquire`.
    let mut reader = Float64Reader::new_zero_copy("zc", None, None, false)?;

    // Build the entry directly in shared memory — no staging buffer.
    let mut handle = writer.try_acquire()?.expect("connected");
    handle.as_slice_mut::<f64>()?.copy_from_slice(&[1.0, 2.0, 3.0]);
    let entry_seq = writer.commit(handle);

    // Borrow the entry in place. The slot's entry_seq is now negative, so the
    // writer skips it until the handle is released.
    let handle = reader.try_acquire(None)?.expect("one entry");
    assert_eq!(handle.entry_seq(), entry_seq);
    assert_eq!(handle.as_slice::<f64>()?, &[1.0, 2.0, 3.0]);

    // Releasing is mandatory: an unreleased handle retires the slot for good.
    reader.release(handle);

    Ok(())
}
```

### 10.8 Python — Raw Byte Channel

```python
from zerochannel import BytesWriter, BytesReader

writer = BytesWriter("audio", entry_length=4096, entry_count=64)
reader = BytesReader("audio")

writer.write(b"\x00" * 4096)

while (entry := reader.read()) is not None:
    entry_seq, block = entry
    assert len(block) == 4096
```

### 10.9 Python — Zero-Copy Handles (SRSW only)

```python
import numpy as np
from zerochannel import Float64Reader, Float64Writer

writer = Float64Writer("zc", entry_length=3, entry_count=16)
# `enable_zero_copy` claims the channel's single zero-copy reader role; the
# default reader rejects `try_acquire` with ValueError.
reader = Float64Reader("zc", enable_zero_copy=True)

# Build the entry directly in shared memory — no staging array. The slot is
# not zeroed, so every element is assigned. Leaving the block commits;
# raising inside it abandons the slot.
with writer.try_acquire() as handle:
    np.asarray(handle.payload)[:] = [1.0, 2.0, 3.0]

# Borrow the entry in place. The slot's entry_seq is negative for the
# lifetime of the handle, so the writer skips it.
with reader.try_acquire() as handle:
    print(handle.entry_seq)
    total = np.asarray(handle.payload).sum()  # consumed before release

# `total` is a scalar copy; the view it came from died with the expression.
# Had it stayed bound, `__exit__` would raise BufferError rather than let a
# view outlive the slot:
with reader.try_acquire() as handle:
    arr = np.asarray(handle.payload)
    biggest = arr.max()
    del arr  # required — the export must be gone before the handle releases
```

---

## 11. File Placement

The implementation is split across two crates, divided by whether the code
needs an operating system. `zerochannel-core` is `#![no_std]` and
dependency-free: it operates entirely on a caller-supplied mapping and has
nothing to say about how that mapping came to be. `zerochannel` supplies the
mapping, arbitrates who may write to it, and exposes the result to Python.

| Path | Contents |
| --- | --- |
| `zerochannel-core/src/error.rs` | `ZeroChannelError` and its `Display`/`core::error::Error` impls. |
| `zerochannel-core/src/metadata.rs` | Header layout — field offsets, `Dtype`, the `Element` trait, geometry arithmetic, header init/validation, and the atomic accessors. |
| `zerochannel-core/src/writer.rs` | The pointer-level `RawWriter`: slot claim, publish, and crash-recovery of the write index. |
| `zerochannel-core/src/reader.rs` | The pointer-level `RawReader`: watermark tracking, slot location, and both read paths. |
| `zerochannel-core/src/zerocopy.rs` | `ReadHandle` and `WriteHandle` — borrowed, in-place views of one ring slot. |
| `zerochannel-core/src/lib.rs` | Module declarations, re-exports, and the protocol documentation. |
| `zerochannel/src/lifecycle.rs` | Segment naming, mapping, ownership, PID liveness, and the exclusive role claims. |
| `zerochannel/src/writer.rs` | The named `RawWriter`: a core writer plus its mapping, writer role, and deferred connection. |
| `zerochannel/src/reader.rs` | The named `RawReader`, plus the gate keeping the zero-copy path behind `new_zero_copy`. |
| `zerochannel/src/lib.rs` | The typed `Writer<T>`/`Reader<T>` wrappers, the concrete aliases, and the crate's public re-exports. |
| `zerochannel/src/python/` | The PyO3 `#[pyclass]` wrappers and the `#[pymodule]` entry point, compiled only when the `python` feature is on. |

The Python bindings mirror that same split: `python/writer.rs` and
`python/reader.rs` hold the channel endpoints, `python/zerocopy.rs` holds the
handles and the buffer-protocol machinery that keeps a ring slot alive while
Python still points into it, `python/errors.rs` maps failures onto Python
exceptions, and `python/mod.rs` owns only the registration that assembles them
into one extension module.

Because `ZeroChannelError` and `PyErr` are both foreign to the `zerochannel`
crate, the orphan rule forbids a `From` impl between them. The conversion is
instead a crate-local `IntoPyResult::or_py()` extension trait, applied
explicitly at each boundary.

The PyO3 bindings sit behind a non-default `python` feature (and
`extension-module`, which adds `pyo3/extension-module` on top of it). A
Rust-only consumer can therefore depend on `zerochannel` and keep the Python
toolchain out of its build graph entirely, or depend on `zerochannel-core`
alone and supply its own mapping. Because the feature is off by default, any
workspace-wide lint or test sweep must pass `--features python` (or
`--all-features`) to cover the `python` module.

Maturin builds the extension from `zerochannel/Cargo.toml` (see
`manifest-path` and `features` under `[tool.maturin]` in `pyproject.toml`). The
compiled module is the `zerochannel` native extension; the last component of
`module-name` must match the `#[pymodule]` function name, because PyO3 derives
the `PyInit_zerochannel` symbol from it.

Callers import classes directly from the top-level `zerochannel` module:

```python
from zerochannel import BytesWriter, BytesReader
from zerochannel import Float64Writer, Float64Reader
from zerochannel import ReadHandle, WriteHandle  # for annotations
```

Each `#[pyclass]` also sets `module = "zerochannel"` so that
`__module__` matches the real import path, which `repr()`, tracebacks, `pickle`
and documentation cross-references all rely on.

