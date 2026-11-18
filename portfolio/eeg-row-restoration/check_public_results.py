"""Check consistency of published counts without participant data or packages."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def require(condition, description):
    if not condition:
        raise ValueError(description)


def check(folder):
    source = Path(__file__).resolve().parent
    a = json.loads((folder / "aggregate.json").read_text(encoding="utf8"))
    c = json.loads((folder / "equal-storage.json").read_text(encoding="utf8"))
    for name, expected in [("analyze_row_restoration.py", a["code_sha256"]),
                           ("compare_static_precision.py", c["implementation_sha256"]),
                           ("CONTROL.md", c["protocol_sha256"])]:
        require(hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name)
    require(a["training_freeze"] == c["training_freeze"], "Different frozen models")
    subjects = a["subjects"]
    require(sorted(s["subject"] for s in subjects) == list(range(1, 10)), "Subjects")
    with (folder / "subset-counts.csv").open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = {(r["setting"], int(r["mask"])) for r in rows}
    require(len(keys) == len(rows) == 224, "Missing or duplicate subset")
    require(keys == {(scenario, m) for scenario in a["scenarios"] for m in range(16)}, "Subset grid")
    for row in rows:
        scenario, mask = row["setting"], int(row["mask"])
        mismatch = int(row["vs_fixed_input_Wfp"])
        trials, correct = int(row["trials"]), int(row["correct"])
        groups = [s["aggregates"][scenario] for s in subjects]
        require(mismatch == sum(s["reference_disagreements_by_mask"][mask] for s in groups), "Subject mismatches")
        require(correct == sum(s["correct_by_mask"][mask] for s in groups), "Subject correct counts")
        require(mismatch == sum(s["reference_disagreements_by_mask"][0] for s in groups)
                - int(row["repair"]) + int(row["induce"]), "Repair/new counts")
        require(trials == 1296 and math.isclose(float(row["balanced_accuracy"]), correct / trials), "Trial totals")
    for name, total in c["summary"].items():
        members = [r for r in c["per_subject"] if r["config"] == name]
        require(sorted(r["subject"] for r in members) == list(range(1, 10)), "Budget group")
        for field in ("trials", "correct", "float_disagreements", "input16_disagreements",
                      "correct_to_wrong", "wrong_to_correct"):
            require(total[field] == sum(r[field] for r in members), field)
        require(total["float_disagreements"] == total["correct_to_wrong"] + total["wrong_to_correct"], name)
        require(total["correct"] == 963 - total["correct_to_wrong"] + total["wrong_to_correct"], name)
        require(math.isclose(total["score_rmse_vs_input16"] ** 2 * total["trials"],
                             sum(r["score_sse"] for r in members)), "Score error")
    for k in (1, 2, 3):
        counts = {(r["subject"], r["config"]): r["input16_disagreements"] for r in c["per_subject"]}
        delta = [counts[s, f"mixed{k}"] - counts[s, f"uniform{8 + k}"] for s in range(1, 10)]
        pair = c["paired"][str(k)]
        require([pair["fewer"], pair["same"], pair["more"]] ==
                [sum(d < 0 for d in delta), delta.count(0), sum(d > 0 for d in delta)], "Paired comparisons")
        require(8 * pair["payload_bytes"] == 22 * (12 * k + 8 * (4 - k)) == 88 * (8 + k), "Payload")
    moments = json.loads((folder / "moment-check.json").read_text(encoding="utf8"))
    require(moments["training_freeze_sha256"] == a["training_freeze"], "Moment replay models")
    require(hashlib.sha256((folder / "subset-counts.csv").read_bytes()).hexdigest()
            == moments["checked_primary_csv_sha256"], "Independent replay checked a different table")
    primary = {(r["setting"], int(r["mask"])): r for r in rows}
    checked = 0
    for path, cases in moments["subsets"].items():
        lo, hi = path.split("_to_")
        setting = f"X16_W{lo}_to_W{'fp' if hi == 'float' else hi}"
        require(len(cases) == 16, "Moment replay grid")
        for mask, case in enumerate(cases):
            row = primary[setting, mask]
            for key, col in (("trials", "trials"), ("disagrees", "vs_fixed_input_Wfp"),
                             ("repaired", "repair"), ("introduced", "induce")):
                require(case[key] == int(row[col]), "Moment replay totals")
            checked += 1
    require(checked == 48 == moments["matching_primary_csv_rows"], "Moment replay coverage")
    print(f"Verified {len(rows)} subset totals, {len(c['per_subject'])} subject/configuration rows,"
          f" 3 paired budgets and {checked} independent replay configurations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    check(parser.parse_args().results)
