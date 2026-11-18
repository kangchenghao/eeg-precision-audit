"""Run synthetic tests and bind the outcome to the actual project sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
os.environ.setdefault("MPLBACKEND", "Agg")


def hashes():
    result = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(p in {"__pycache__", ".git", ".venv", "build", "results"} for p in relative.parts):
            continue
        if path.suffix in {".py", ".v", ".sv", ".tcl", ".ps1", ".toml"} or path.name in {
                "CONTRACT.md", "requirements-win-py313.lock", "data_sources.json"}:
            result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output/"results.json"
    if report_path.exists():
        raise FileExistsError("Preserve previous evidence; use a new output directory")
    before = hashes()
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    log = output/"unittest.log"
    with log.open("x", encoding="utf8", newline="\n") as stream:
        outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    stable = before == hashes()
    passed = outcome.wasSuccessful() and outcome.testsRun >= 46 and stable
    report = {
        "status": "passed" if passed else "failed",
        "tests_run": outcome.testsRun, "failures": len(outcome.failures),
        "errors": len(outcome.errors), "skipped": len(outcome.skipped),
        "source_stable": stable, "source_hashes": before,
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "data_scope": "synthetic only; no official E data",
    }
    report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps({k:report[k] for k in ("status","tests_run","failures","errors","source_stable")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
