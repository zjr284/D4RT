"""Resume the published OpenD4RT 32-frame checkpoint download.

The public checkpoint is nearly 14 GB. Small ranged requests work more
reliably than one long request through this WSL environment's proxy.
Completed chunks are retained across interruptions, then verified against
the publisher's SHA-256 before the final checkpoint is exposed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import subprocess
import time
import urllib.request
from pathlib import Path


FILENAME = "opend4rt.ckpt"
CONFIG_FILENAME = "model.yaml"
REMOTE = (
    "https://huggingface.co/Lijiaxin0111/OpenD4RT/resolve/7099b1fc760475de3b7409acd6b63f801b015d07/"
    "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"
)
CONFIG_REMOTE = REMOTE.rsplit("/", 1)[0] + "/" + CONFIG_FILENAME
EXPECTED_SIZE = 13_950_006_682
EXPECTED_SHA256 = "1f63305422fdc2000b057fbbc1d37459ac1a8063bbfcd0e3b7d473f5485943f5"
CONFIG_SHA256 = "c0b51efbb79ffacc16fdb78e2cf78a5759bc22b2a11a1eb7890d109e4af8f9eb"
CHUNK_SIZE = 16 * 1024 * 1024
DESTINATION = (
    Path(__file__).resolve().parents[1]
    / "external/open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG"
)


def fetch_chunk(index: int, parts_dir: Path) -> int:
    start = index * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, EXPECTED_SIZE) - 1
    expected_bytes = end - start + 1
    final = parts_dir / f"part-{index:05d}"
    if final.is_file() and final.stat().st_size == expected_bytes:
        return index

    temporary = parts_dir / f"part-{index:05d}.tmp"
    for attempt in range(12):
        received = temporary.stat().st_size if temporary.exists() else 0
        if received == expected_bytes:
            temporary.replace(final)
            return index
        if received > expected_bytes:
            raise RuntimeError(f"Chunk {index}: temporary file is too large")
        request_start = start + received
        remaining = expected_bytes - received
        if remaining <= 1024 * 1024:
            # Some proxied range streams stall after delivering all but a few
            # kilobytes. curl handles these short suffix requests reliably.
            suffix = parts_dir / f"part-{index:05d}.suffix"
            result = subprocess.run(
                [
                    "curl", "-fLsS", "--retry", "2", "--max-time", "30",
                    "--range", f"{request_start}-{end}", REMOTE,
                    "--output", str(suffix),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and suffix.is_file() and suffix.stat().st_size == remaining:
                with temporary.open("ab") as output:
                    output.write(suffix.read_bytes())
                suffix.unlink()
                temporary.replace(final)
                return index
            suffix.unlink(missing_ok=True)
            if attempt == 11:
                raise RuntimeError(f"Chunk {index}: curl suffix request failed: {result.stderr.strip()}")
            time.sleep(min(2**attempt, 10))
            continue
        request = urllib.request.Request(REMOTE, headers={"Range": f"bytes={request_start}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 206:
                    raise RuntimeError(f"Chunk {index}: expected HTTP 206, got {response.status}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {request_start}-{end}/"):
                    raise RuntimeError(f"Chunk {index}: wrong Content-Range: {content_range}")
                with temporary.open("ab") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            if temporary.stat().st_size == expected_bytes:
                temporary.replace(final)
                return index
        except Exception:
            if attempt == 11:
                raise
        time.sleep(min(2**attempt, 10))
    raise AssertionError("unreachable")


def assemble(parts_dir: Path, destination: Path, chunk_count: int) -> None:
    temporary = destination.with_suffix(".ckpt.assembling")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for index in range(chunk_count):
            with (parts_dir / f"part-{index:05d}").open("rb") as part:
                while block := part.read(4 * 1024 * 1024):
                    output.write(block)
                    digest.update(block)
        output.flush()
        os.fsync(output.fileno())
    if temporary.stat().st_size != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Checkpoint size or SHA-256 mismatch; retaining chunks for inspection")
    temporary.replace(destination)
    print(f"Verified SHA-256: {digest.hexdigest()}", flush=True)


def ensure_config(destination_dir: Path) -> None:
    config = destination_dir / CONFIG_FILENAME
    if config.is_file() and hashlib.sha256(config.read_bytes()).hexdigest() == CONFIG_SHA256:
        return
    with urllib.request.urlopen(CONFIG_REMOTE, timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != CONFIG_SHA256:
        raise RuntimeError("Downloaded model config failed SHA-256 verification")
    temporary = config.with_suffix(".yaml.tmp")
    temporary.write_bytes(content)
    temporary.replace(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 128:
        parser.error("--workers must be between 1 and 128")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    ensure_config(DESTINATION)
    destination = DESTINATION / FILENAME
    if destination.is_file():
        if destination.stat().st_size == EXPECTED_SIZE:
            digest = hashlib.sha256()
            with destination.open("rb") as existing:
                while block := existing.read(4 * 1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() == EXPECTED_SHA256:
                print(f"Checkpoint already present and verified: {destination}")
                return
            raise RuntimeError(f"Existing checkpoint failed SHA-256 verification: {destination}")
        raise RuntimeError(f"Existing checkpoint has wrong size: {destination}")

    parts_dir = DESTINATION / ".download_parts"
    parts_dir.mkdir(exist_ok=True)
    chunk_count = (EXPECTED_SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Downloading {chunk_count} chunks with {args.workers} workers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_chunk, index, parts_dir) for index in range(chunk_count)]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            future.result()
            if completed % 20 == 0 or completed == chunk_count:
                print(f"Chunks ready: {completed}/{chunk_count}", flush=True)
    assemble(parts_dir, destination, chunk_count)
    for index in range(chunk_count):
        (parts_dir / f"part-{index:05d}").unlink()
    parts_dir.rmdir()
    print(f"Ready: {destination}", flush=True)


if __name__ == "__main__":
    main()
