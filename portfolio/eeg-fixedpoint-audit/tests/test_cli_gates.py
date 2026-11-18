"""Gate failures must precede the private evaluation reader.

All reports, source files and fixture bytes below are synthetic test artifacts.
These tests check report consistency, not authenticity against malicious authors.
"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ENTRY_PATH = Path(__file__).resolve().parents[1] / "run.py"
SPEC = importlib.util.spec_from_file_location("eeg_audit_cli_gate_test_subject", ENTRY_PATH)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)

MUTANTS = (
    "missing_product", "21_terms", "23_terms", "zero_extension", "narrow_accumulator",
    "reset_keeps_count", "gap_advances_count", "wrong_streamed_coefficient", "wrong_weight_index",
)


class EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.package = self.directory / "synthetic-package"
        self.private = self.directory / "synthetic-private"
        self.private.mkdir()
        for relative in ("rtl/dot22.v", "rtl/tb_dot22.sv", "scripts/verify_rtl.py", "src/core.py"):
            target = self.package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"constructed source for {relative}\n", encoding="utf-8")
        self.root_patch = patch.object(CLI, "PACKAGE_ROOT", self.package)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        fixtures = []
        for subject in range(1, 10):
            relative = f"rtl-fixtures/s{subject:02d}.npz"
            target = self.private / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"constructed fixture {subject}".encode("ascii"))
            fixtures.append({"name": target.name, "path": relative, "sha256": CLI.digest(target)})
        self.manifest = {
            "rtl_fixtures": fixtures, "execution_sources": CLI.source_manifest(),
            "freeze_sha256": "synthetic-freeze", "models": [],
        }
        names_and_hashes = [{"name": entry["name"], "sha256": entry["sha256"]} for entry in fixtures]
        self.rtl = {
            "status": "passed",
            "source_sha256": CLI.digest(self.package / "rtl/dot22.v"),
            "testbench_sha256": CLI.digest(self.package / "rtl/tb_dot22.sv"),
            "runner_sha256": CLI.digest(self.package / "scripts/verify_rtl.py"),
            "runs": [{
                "weight_bits": bits, "accumulator_bits": bits + 20,
                "real": {"source_session": "T", "dots": 2304,
                         "partial_checks": 2304 * 22, "fixtures": deepcopy(names_and_hashes)},
                "synthetic_dots": 10, "total_dots": 2314,
                "compile": {"exit_code": 0}, "run": {"exit_code": 0},
            } for bits in (4, 8, 12, 16)],
            "mutants": [{"name": name, "compiled": True, "detected": True} for name in MUTANTS],
        }
        self.tests = {
            "status": "passed", "tests_run": 46, "failures": 0, "errors": 0, "skipped": 0,
            "source_stable": True,
            "source_hashes": {entry["path"]: entry["sha256"]
                              for entry in self.manifest["execution_sources"]["package"]},
        }

    def report(self, payload, name):
        path = self.directory / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def verify_rtl(self, payload):
        return CLI.verify_rtl_evidence(self.report(payload, "rtl.json"), self.manifest, self.private)

    def verify_tests(self, payload):
        return CLI.verify_test_evidence(self.report(payload, "tests.json"), self.manifest)

    def test_consistent_synthetic_reports_are_accepted(self):
        self.assertEqual(self.verify_rtl(self.rtl)["mutants_detected"], 9)
        self.assertEqual(self.verify_tests(self.tests)["tests_run"], 46)

    def test_missing_reports_and_bare_passed_claims_are_rejected(self):
        with self.assertRaises(ValueError):
            CLI.verify_rtl_evidence(None, self.manifest, self.private)
        with self.assertRaises(ValueError):
            CLI.verify_test_evidence(None, self.manifest)
        with self.assertRaises(ValueError):
            self.verify_rtl({"status": "passed"})
        with self.assertRaises(ValueError):
            self.verify_tests({"status": "passed", "tests_run": 46})

    def test_missing_or_duplicate_width_is_rejected(self):
        missing = deepcopy(self.rtl)
        missing["runs"].pop()
        duplicate = deepcopy(self.rtl)
        duplicate["runs"][-1] = deepcopy(duplicate["runs"][0])
        for payload in (missing, duplicate):
            with self.subTest(widths=[item["weight_bits"] for item in payload["runs"]]), self.assertRaises(ValueError):
                self.verify_rtl(payload)

    def test_rtl_source_hash_mismatch_is_rejected(self):
        for key in ("source_sha256", "testbench_sha256", "runner_sha256"):
            payload = deepcopy(self.rtl)
            payload[key] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify_rtl(payload)

    def test_report_fixture_mismatch_and_changed_actual_fixture_are_rejected(self):
        payload = deepcopy(self.rtl)
        payload["runs"][0]["real"]["fixtures"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.verify_rtl(payload)
        (self.private / self.manifest["rtl_fixtures"][0]["path"]).write_bytes(b"changed fixture")
        with self.assertRaises(ValueError):
            self.verify_rtl(self.rtl)

    def test_wrong_session_partial_counts_or_accumulator_width_is_rejected(self):
        for location, key, value in (
            ("real", "source_session", "E"), ("real", "dots", 2303),
            ("real", "partial_checks", 2304 * 22 - 1),
            ("run", "accumulator_bits", 23), ("run", "synthetic_dots", 0),
            ("run", "total_dots", 2304),
        ):
            payload = deepcopy(self.rtl)
            target = payload["runs"][0]["real"] if location == "real" else payload["runs"][0]
            target[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify_rtl(payload)

    def test_passed_status_cannot_override_compile_failure_or_escaped_mutant(self):
        bad_compile = deepcopy(self.rtl)
        bad_compile["runs"][0]["compile"]["exit_code"] = 1
        escaped = deepcopy(self.rtl)
        escaped["mutants"][0]["detected"] = False
        missing_mutant = deepcopy(self.rtl)
        missing_mutant["mutants"].pop()
        for payload in (bad_compile, escaped, missing_mutant):
            with self.subTest(payload=payload.get("status")), self.assertRaises(ValueError):
                self.verify_rtl(payload)

    def test_failed_skipped_or_insufficient_tests_are_rejected(self):
        for key, value in (("status", "failed"), ("tests_run", 45),
                           ("failures", 1), ("errors", 1), ("skipped", 1),
                           ("source_stable", False)):
            payload = deepcopy(self.tests)
            payload[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify_tests(payload)

    def test_test_source_hash_mismatch_missing_or_added_source_is_rejected(self):
        for kind in ("changed", "missing", "extra"):
            payload = deepcopy(self.tests)
            key = next(iter(payload["source_hashes"]))
            if kind == "changed":
                payload["source_hashes"][key] = "0" * 64
            elif kind == "missing":
                del payload["source_hashes"][key]
            else:
                payload["source_hashes"]["unfrozen.py"] = "0" * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.verify_tests(payload)

    def test_bad_rtl_gate_precedes_evaluation_reader(self):
        bad_rtl = deepcopy(self.rtl)
        bad_rtl["runs"].pop()
        with patch.object(CLI, "verify_freeze", return_value=self.manifest), \
                patch.object(CLI, "load_session") as reader:
            with self.assertRaises(ValueError):
                CLI.evaluate(self.directory / "no-raw-data", self.private,
                             rtl_evidence=self.report(bad_rtl, "bad-rtl.json"),
                             test_evidence=self.report(self.tests, "good-tests.json"))
            reader.assert_not_called()
        self.assertFalse((self.private / "evaluation").exists())

    def test_bad_test_gate_precedes_evaluation_reader(self):
        bad_tests = deepcopy(self.tests)
        bad_tests["failures"] = 1
        with patch.object(CLI, "verify_freeze", return_value=self.manifest), \
                patch.object(CLI, "load_session") as reader:
            with self.assertRaises(ValueError):
                CLI.evaluate(self.directory / "no-raw-data", self.private,
                             rtl_evidence=self.report(self.rtl, "good-rtl.json"),
                             test_evidence=self.report(bad_tests, "bad-tests.json"))
            reader.assert_not_called()
        self.assertFalse((self.private / "evaluation").exists())


if __name__ == "__main__":
    unittest.main()
