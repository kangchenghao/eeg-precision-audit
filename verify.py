"""Recompute published decision counts from aggregate evidence using Python 3.10+."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "portfolio/eeg-row-restoration"
RESULTS = ROOT / "output/bci-row-restoration-2026-11-18"
BASELINE = ROOT / "portfolio/eeg-fixedpoint-audit"


def run(*args):
    subprocess.run([sys.executable, "-I", "-B", *map(str, args)], check=True, cwd=ROOT)


def main():
    frozen = json.loads((BASELINE / "results/pilot-2026-11-15/results.json").read_bytes())
    for record in frozen["execution_source_hashes"]:
        source = BASELINE / record["path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Frozen experiment source differs: {record['path']}")
    run(STUDY / "check_public_results.py", RESULTS)
    run(STUDY / "budget_ceiling.py", RESULTS, "--check", RESULTS / "budget-ceiling.json")
    run("-m", "unittest", "discover", "-s", STUDY, "-p", "test_budget_ceiling.py")
    print("Verified aggregate results, fixed-mask bounds and frozen baseline sources.")


if __name__ == "__main__":
    main()
