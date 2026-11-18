#!/usr/bin/env python3
"""Download an explicitly selected official BNCI corpus, with content checks.

Standard library only. Data are not bundled with this project. On Windows, an
SSL chain validation failure may use PowerShell's verified system trust path;
certificate checking is never disabled. No MAT contents or labels are parsed.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import urllib.error
import urllib.parse
import urllib.request

ALLOWED_HOSTS = {"lampx.tugraz.at", "bnci-horizon-2020.eu"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.password:
        raise ValueError("Source URL must use HTTPS on an official BNCI host")


def load_manifest(path: Path) -> list[dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest["files"]
    names = set()
    for entry in entries:
        if not re.fullmatch(r"A0[1-9][TE]\.mat", entry["filename"]):
            raise ValueError("Unexpected filename in manifest")
        if entry["filename"] in names:
            raise ValueError("Duplicate filename in manifest")
        names.add(entry["filename"])
        check_url(entry["url"])
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("Invalid SHA256 in manifest")
        if isinstance(entry["bytes"], bool) or not isinstance(entry["bytes"], int) or entry["bytes"] <= 0:
            raise ValueError("Invalid byte count in manifest")
    expected = {f"A{s:02d}{r}.mat" for s in range(1, 10) for r in ("T", "E")}
    if names != expected:
        raise ValueError("Manifest must list exactly the 18 official sessions")
    return entries


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def powershell_download(url: str, part: Path, timeout: int) -> dict:
    """Use system certificate verification without interpolating path into code."""
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        raise RuntimeError("No PowerShell available for the system-trust fallback")
    script = """$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$r = Invoke-WebRequest -UseBasicParsing -Uri $env:BCI_SOURCE_URL -OutFile $env:BCI_PART_PATH -PassThru -TimeoutSec ([int]$env:BCI_TIMEOUT)
$u = $env:BCI_SOURCE_URL
if ($r.BaseResponse.RequestMessage) { $u = $r.BaseResponse.RequestMessage.RequestUri.AbsoluteUri }
elseif ($r.BaseResponse.ResponseUri) { $u = $r.BaseResponse.ResponseUri.AbsoluteUri }
$h = @{}
foreach ($k in @('Content-Length','Content-Type','ETag','Last-Modified','Date')) {
    if ($r.Headers[$k]) { $h[$k] = [string]::Join(', ', @($r.Headers[$k])) }
}
@{status=[int]$r.StatusCode;final_url=$u;headers=$h;transport='PowerShell system trust'} | ConvertTo-Json -Compress
"""
    environment = os.environ.copy()
    environment.update(BCI_SOURCE_URL=url, BCI_PART_PATH=str(part.resolve()), BCI_TIMEOUT=str(timeout))
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        env=environment, text=True, capture_output=True, timeout=timeout + 15,
        check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    metadata = json.loads(result.stdout.strip().lstrip("\ufeff"))
    if metadata["status"] != 200:
        raise RuntimeError(f"Unexpected HTTP status {metadata['status']}")
    check_url(metadata["final_url"])
    return metadata


def fetch(url: str, part: Path, expected_bytes: int, timeout: int) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "eeg-fixedpoint-audit/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            check_url(response.geturl())
            if response.status != 200:
                raise RuntimeError(f"Unexpected HTTP status {response.status}")
            received = 0
            with part.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > expected_bytes:
                        raise ValueError("Response exceeds manifest byte count")
                    stream.write(block)
            return {"status": response.status, "final_url": response.geturl(),
                    "headers": {key: response.headers[key] for key in ("Content-Length", "Content-Type", "ETag", "Last-Modified", "Date") if key in response.headers},
                    "transport": "Python verified HTTPS"}
    except urllib.error.URLError as error:
        if os.name == "nt" and isinstance(error.reason, ssl.SSLCertVerificationError):
            metadata = powershell_download(url, part, timeout)
            metadata["fallback_from"] = "Python verified HTTPS"
            metadata["fallback_reason"] = str(error.reason)
            return metadata
        raise


def download_one(entry: dict, output: Path, retries: int, timeout: int, verify_only: bool) -> dict:
    destination = output / entry["filename"]
    result = {"filename": entry["filename"], "url": entry["url"], "started_at_utc": utc_now(), "attempts": []}
    if destination.exists():
        actual_sha = digest(destination)
        if destination.stat().st_size != entry["bytes"] or actual_sha != entry["sha256"]:
            raise ValueError(f"Existing {entry['filename']} differs from manifest; it was not overwritten")
        result.update(status="verified_existing", bytes=entry["bytes"], sha256=actual_sha, completed_at_utc=utc_now())
        return result
    if verify_only:
        raise FileNotFoundError(f"Missing {entry['filename']}")
    part = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            metadata = fetch(entry["url"], part, entry["bytes"], timeout)
            size = part.stat().st_size
            actual_sha = digest(part)
            if size != entry["bytes"] or actual_sha != entry["sha256"]:
                raise ValueError(f"Downloaded bytes/SHA256 do not match {entry['filename']}")
            part.replace(destination)
            result["attempts"].append({"attempt": attempt, "status": "complete"})
            result.update(status="downloaded", bytes=size, sha256=actual_sha, http=metadata, completed_at_utc=utc_now())
            return result
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            part.unlink(missing_ok=True)
            result["attempts"].append({"attempt": attempt, "error": str(error)})
    raise RuntimeError(f"Failed {entry['filename']} after {retries} attempts: {result['attempts'][-1]['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Explicit local raw-data directory")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "data_sources.json")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--retries", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--verify-only", action="store_true", help="Check local bytes/SHA256, without network access")
    parser.add_argument("--dry-run", action="store_true", help="Display the corpus size, without downloading")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    entries = load_manifest(args.manifest)
    expected_bytes = sum(entry["bytes"] for entry in entries)
    if args.dry_run:
        print(json.dumps({"files": len(entries), "bytes": expected_bytes, "output": str(args.output)}))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, entry, args.output, args.retries, args.timeout, args.verify_only): entry for entry in entries}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                record = future.result()
                records.append(record)
                print(f"{entry['filename']}: {record['status']}", flush=True)
            except Exception as error:
                failures.append({"filename": entry["filename"], "error": str(error)})
                print(f"{entry['filename']}: FAILED: {error}", flush=True)
    report = {"dataset": "BNCI2014_001", "created_at_utc": utc_now(),
              "expected_bytes": expected_bytes, "expected_files": 18,
              "content_parsed": False, "files": sorted(records, key=lambda record: record["filename"]), "failures": failures}
    temporary = args.output / "download_manifest.json.part"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output / "download_manifest.json")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
