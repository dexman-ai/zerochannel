"""Interactive manual test for the ZeroChannel shared-memory IPC subsystem.

Run two terminals from the repository root:

    python tests/interactive_zero_channel.py s    # server - prints what it reads
    python tests/interactive_zero_channel.py c    # client - sends what you type

Both ends attach to the same shared-memory segment. Every entry carries its own
``entry_seq`` from the channel itself, so the server detects dropped entries by
watching for gaps in the returned sequence numbers - no framing header needed.
The payload stream is newline-terminated text split across entries. Each side
runs until you press Ctrl+C; either one can come and go independently of the
other.

The server reads through the zero-copy handle API: it borrows each entry in
place rather than copying it out of the ring. That pins the slot against the
writer for as long as the handle is held, so the loop releases it as soon as
the payload has been decoded.
"""

import sys
import time

from zerochannel import BytesWriter, BytesReader

NAME = "/zc_interactive_demo"
ENTRY_LENGTH = 8
ENTRY_COUNT = 64
POLL_SECONDS = 0.005


def run_server() -> None:
    """Print entries as they arrive and warn when sequence gaps indicate loss."""
    reader = BytesReader(NAME, enable_zero_copy=True)
    print(f"server listening on {NAME} (Ctrl+C to stop)")

    expected_seq: int | None = None

    try:
        while True:
            handle = reader.try_acquire()
            if handle is None:
                time.sleep(POLL_SECONDS)
                continue

            # Decode inside the `with` block: `handle.payload` borrows the ring
            # slot, which the writer may recycle the moment the handle is
            # released. `bytes()` is the copy that outlives it.
            with handle:
                entry_seq = handle.entry_seq
                text = bytes(handle.payload).rstrip(b"\x00").decode(
                    "utf-8", errors="replace"
                )

            if expected_seq is not None and entry_seq != expected_seq:
                dropped = entry_seq - expected_seq
                print(
                    f"\n[server warning] dropped {dropped} entr"
                    f"{'y' if dropped == 1 else 'ies'} "
                    f"(expected seq={expected_seq}, got={entry_seq})"
                )
            expected_seq = entry_seq + 1

            sys.stdout.write(text)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nserver stopped")


def run_client() -> None:
    """Take console input and write it to the shared channel."""
    writer = BytesWriter(NAME, entry_length=ENTRY_LENGTH, entry_count=ENTRY_COUNT)
    print(f"client connected to {NAME} (Ctrl+C to stop)")

    try:
        while True:
            print("> ", end="", flush=True)
            # readline() keeps the newline the user typed by hitting Enter.
            line = sys.stdin.readline()

            payload = line.encode("utf-8")
            for start in range(0, len(payload), ENTRY_LENGTH):
                chunk = payload[start : start + ENTRY_LENGTH]
                writer.write(chunk.ljust(ENTRY_LENGTH, b"\x00"))
    except KeyboardInterrupt:
        print("\nclient stopped")


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("s", "c"):
        print(f"usage: {sys.argv[0]} s|c   (s = server, c = client)")
        return 2

    if sys.argv[1] == "s":
        run_server()
    else:
        run_client()
    return 0


if __name__ == "__main__":
    sys.exit(main())
