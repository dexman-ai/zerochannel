//! Segment lifecycle — naming, mapping, ownership and exclusive roles.
//!
//! This is the half of ZeroChannel that talks to the operating system. It
//! turns a channel *name* into a mapped segment, decides who is responsible
//! for unlinking that segment, and arbitrates the exclusive writer and
//! zero-copy reader roles using PID liveness so a crashed peer cannot poison a
//! channel name permanently.
//!
//! Everything that happens *inside* the mapping — the `entry_seq` protocol,
//! the ring, the header layout — belongs to `zerochannel_core`.

use core::sync::atomic::{AtomicU64, Ordering};

use shared_memory::{Shmem, ShmemConf};
use zerochannel_core::{
    check_dtype, check_geometry, init_header, read_header, required_bytes, segment_owner,
    validate_header, Dtype, Header, ZeroChannelError,
};

// ── Process liveness ────────────────────────────────────────────────────────

/// Is `pid` a live process?
///
/// Used to tell a role claim left behind by a crashed process from one held by
/// a live peer. Without this, a single `SIGKILL` would poison a channel name
/// permanently: the claim sits in shared memory where no amount of restarting
/// can clear it.
///
/// PIDs are recycled by the OS, so a false positive is possible if a segment
/// outlives its owner long enough for the PID to be reused *and* the same
/// channel name is reused. That window is accepted here; closing it requires
/// the process start time, which has no portable accessor.
#[cfg(unix)]
pub(crate) fn process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    // Signal 0 performs the permission and existence checks without delivering
    // anything. `EPERM` means the process exists but belongs to another user.
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if rc == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(windows)]
pub(crate) fn process_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    if pid == 0 {
        return false;
    }
    // SAFETY: `OpenProcess` is a plain FFI call; a null return means failure.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if handle.is_null() {
        return false;
    }
    // A handle can outlive the process it refers to, so existence of the handle
    // is not enough — an exited process still answers `OpenProcess` while any
    // handle to it remains open.
    let mut code: u32 = 0;
    // SAFETY: `handle` is valid and `code` is a live local.
    let ok = unsafe { GetExitCodeProcess(handle, &mut code) };
    // SAFETY: `handle` came from `OpenProcess` and is not used afterwards.
    unsafe { CloseHandle(handle) };
    ok != 0 && code == STILL_ACTIVE as u32
}

/// This process's identifier, as stored in the header's owner fields.
#[inline]
pub(crate) fn current_pid() -> u64 {
    std::process::id() as u64
}

// ── Exclusive roles ─────────────────────────────────────────────────────────

/// Claim an exclusive role by compare-and-swapping this process's PID into
/// `cell`.
///
/// A claim whose PID belongs to a dead process is stale and gets adopted, which
/// is what lets a channel recover from a crashed peer. `role` and `name` only
/// shape the error message.
pub(crate) fn claim_role(cell: &AtomicU64, role: &str, name: &str) -> Result<(), ZeroChannelError> {
    let me = current_pid();
    loop {
        match cell.compare_exchange(0, me, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Ok(()),
            Err(holder) => {
                // A holder equal to `me` is still a conflict: it means another
                // instance in *this* process already took the role.
                if process_alive(holder as u32) {
                    return Err(ZeroChannelError::RoleConflict(format!(
                        "the {role} role on channel '{name}' is held by process {holder}"
                    )));
                }
                // Stale claim from a crashed process — adopt it. If another
                // process adopts first, the loop re-examines the new holder.
                if cell
                    .compare_exchange(holder, me, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return Ok(());
                }
            }
        }
    }
}

/// Release a role previously claimed by this process.
///
/// The compare-and-swap makes this a no-op when the claim has already been
/// adopted by someone else, so a late drop cannot evict a live owner.
pub(crate) fn release_role(cell: &AtomicU64) {
    let _ = cell.compare_exchange(current_pid(), 0, Ordering::AcqRel, Ordering::Acquire);
}

/// Try to claim responsibility for unlinking the segment, reporting whether
/// this process now holds it.
///
/// Unlike [`claim_role`], losing is not an error: a participant that declares
/// the geometry of a segment someone else already owns is simply a tenant. It
/// gets the channel it asked for and leaves the name alone on the way out.
///
/// A claim held by a dead process is adopted, which is what lets a restarted
/// creator reclaim a segment orphaned by a hard kill. The deed never moves
/// away from a *live* holder.
pub(crate) fn claim_segment(cell: &AtomicU64) -> bool {
    let me = current_pid();
    loop {
        match cell.compare_exchange(0, me, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(holder) => {
                if holder == me || process_alive(holder as u32) {
                    return false;
                }
                if cell
                    .compare_exchange(holder, me, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return true;
                }
            }
        }
    }
}

// ── Shared-memory open/create helpers ───────────────────────────────────────

/// Try to open an existing shared-memory segment.
pub(crate) fn try_open(name: &str) -> Option<Shmem> {
    ShmemConf::new().os_id(name).open().ok()
}

/// Decode the prefix header of a mapped segment.
pub(crate) fn header_of(shmem: &Shmem) -> Result<Option<Header>, ZeroChannelError> {
    // SAFETY: `shmem` is mapped, and the header is the first `HEADER_BYTES`.
    unsafe { read_header(shmem.as_ptr()) }
}

/// Stamp a fresh header into a mapped segment, claiming the deed for this
/// process.
///
/// # Safety
///
/// `base` must point at a mapped segment of at least
/// `required_bytes(payload_bytes, entry_count)` bytes.
pub(crate) unsafe fn init_segment(
    base: *mut u8,
    payload_bytes: usize,
    entry_count: usize,
    dtype: Dtype,
) {
    init_header(base, payload_bytes, entry_count, dtype, current_pid());
}

/// Create a new shared-memory segment and write the header.
///
/// `Ok(None)` means the name is already taken. The OS settles that race
/// atomically (`O_EXCL` on unix, `CREATE_NEW` on Windows), so no header-level
/// handshake is needed to decide who creates.
fn create_segment(
    name: &str,
    payload_bytes: usize,
    entry_count: usize,
    dtype: Dtype,
) -> Result<Option<Shmem>, ZeroChannelError> {
    let size = required_bytes(payload_bytes, entry_count);
    let shmem = match ShmemConf::new().os_id(name).size(size).create() {
        Ok(shmem) => shmem,
        Err(shared_memory::ShmemError::MappingIdExists) => return Ok(None),
        Err(e) => {
            return Err(ZeroChannelError::OsError(format!(
                "failed to create shared memory: {e}"
            )))
        }
    };
    // SAFETY: the mapping was just created with exactly `size` bytes.
    unsafe { init_segment(shmem.as_ptr(), payload_bytes, entry_count, dtype) };
    Ok(Some(shmem))
}

/// Open or create a shared-memory segment based on constructor arguments.
///
/// Supplying geometry is what marks a participant as willing to create the
/// segment; omitting it means "attach to what is already there". Creation is
/// attempted first so the OS arbitrates the race, and an existing segment is
/// adopted on the second pass.
///
/// Geometry is also a bid for segment *ownership*, which is exclusive: exactly
/// one live process is responsible for unlinking the segment, and it unlinks on
/// a clean exit. A participant that attaches without geometry never bids. A
/// participant that bids while a live owner holds the deed is a tenant — it
/// gets the channel it asked for and leaves the name alone on the way out.
/// Ownership never moves away from a live holder; a deed left by a process that
/// died is adopted by the next bidder, which is what reclaims a leaked name.
///
/// Returns `(shmem, header)` on success or `None` when `delayed_connect` is
/// true and the segment does not exist yet.
pub(crate) fn open_or_create(
    name: &str,
    payload_bytes: Option<usize>,
    entry_count: Option<usize>,
    dtype: Dtype,
    delayed_connect: bool,
) -> Result<Option<(Shmem, Header)>, ZeroChannelError> {
    match (payload_bytes, entry_count) {
        (Some(pb), Some(tc)) => {
            check_geometry(pb, tc)?;
            if let Some(shmem) = create_segment(name, pb, tc, dtype)? {
                let header = header_of(&shmem)?.ok_or_else(|| {
                    ZeroChannelError::OsError("freshly created segment has no header".into())
                })?;
                return Ok(Some((shmem, header)));
            }
            // The name was taken: either another creator won the race, or a
            // previous one died without unlinking.
            let Some(mut shmem) = try_open(name) else {
                return Err(ZeroChannelError::OsError(format!(
                    "shared memory segment '{name}' exists but could not be opened"
                )));
            };
            // Supplying geometry is a bid for ownership, not just a statement
            // of shape. Without it a segment orphaned by a hard kill would leak
            // permanently: every later participant would open rather than
            // create, and `Shmem`'s destructor only unlinks for an owner.
            //
            // The bid is placed only once the segment has been *accepted*.
            // Bidding earlier would mean a rejected open unlinks a healthy
            // segment on its way out, destroying a working channel because the
            // caller passed the wrong geometry.
            match header_of(&shmem)? {
                Some(header) => {
                    validate_header(&header, pb, tc)?;
                    check_dtype(&header, dtype)?;
                    // SAFETY: `shmem` is mapped and carries a valid header.
                    let won = claim_segment(unsafe { segment_owner(shmem.as_ptr()) });
                    shmem.set_owner(won);
                    Ok(Some((shmem, header)))
                }
                None => {
                    // A creator died between reserving the name and writing the
                    // header, leaving a valid mapping with no magic. We supplied
                    // geometry, so we are entitled to initialize it.
                    if shmem.len() < required_bytes(pb, tc) {
                        return Err(ZeroChannelError::InvalidArgument(format!(
                            "abandoned segment '{name}' is too small for the requested geometry"
                        )));
                    }
                    // SAFETY: the mapping is at least `required_bytes` long,
                    // as just checked.
                    unsafe { init_segment(shmem.as_ptr(), pb, tc, dtype) };
                    let header = header_of(&shmem)?.ok_or_else(|| {
                        ZeroChannelError::OsError("segment has no header after init".into())
                    })?;
                    // Laying down the header is what confers ownership, and
                    // `init_segment` just stamped this process into
                    // `segment_owner` under cover of the magic latch.
                    shmem.set_owner(true);
                    Ok(Some((shmem, header)))
                }
            }
        }
        (None, None) => match try_attach(name, dtype)? {
            Some(pair) => Ok(Some(pair)),
            None if delayed_connect => Ok(None),
            None => Err(ZeroChannelError::OsError(
                "shared memory segment does not exist and sizes were not supplied".into(),
            )),
        },
        _ => Err(ZeroChannelError::InvalidArgument(
            "entry_length and entry_count must both be supplied or both omitted".into(),
        )),
    }
}

/// Attempt to open an existing segment (used for deferred reconnect).
///
/// `Ok(None)` means "not there yet"; an error means the segment exists but is
/// incompatible with the requested element type or layout version.
pub(crate) fn try_attach(
    name: &str,
    dtype: Dtype,
) -> Result<Option<(Shmem, Header)>, ZeroChannelError> {
    let Some(shmem) = try_open(name) else {
        return Ok(None);
    };
    // A segment whose magic has not landed yet is treated as not-yet-existing:
    // its creator is still initializing, or died trying.
    let Some(header) = header_of(&shmem)? else {
        return Ok(None);
    };
    if header.payload_bytes == 0 || header.entry_count == 0 {
        return Ok(None);
    }
    check_dtype(&header, dtype)?;
    Ok(Some((shmem, header)))
}

/// Remove a channel's shared-memory segment from the system by name.
///
/// The escape hatch for reclaiming a leaked segment: a channel abandoned by a
/// crashed process keeps its name until something removes it. A later writer
/// adopts the stale role claim and carries on, so this is only needed when the
/// segment itself must go — supervisors reclaiming a name, and test fixtures
/// that must not leak state between runs.
///
/// Existing participants keep their mapping; only the name is removed, so a
/// subsequent open creates a fresh segment. Succeeds whether or not the segment
/// exists.
pub fn unlink(name: &str) -> Result<(), ZeroChannelError> {
    let Some(mut shmem) = try_open(name) else {
        return Ok(());
    };
    // `Shmem`'s destructor unlinks exactly when it believes it is the owner.
    shmem.set_owner(true);
    drop(shmem);
    Ok(())
}
