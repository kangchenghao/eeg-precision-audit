"""Verify a dot22 kernel against Python integers; optionally map it with Yosys.

Real training fixtures are private and passed explicitly. They are not included
in this repository. No CPU, memory, EEG front end, or classifier is synthesized.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (4, 8, 12, 16)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def run(command, log, timeout=180, cwd=None):
    start = time.perf_counter()
    p = subprocess.run([str(x) for x in command], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, encoding="utf8", errors="replace",
                       timeout=timeout, cwd=cwd,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    Path(log).write_text(p.stdout, encoding="utf8")
    return {"exit_code": p.returncode, "seconds": time.perf_counter()-start,
            "log": Path(log).name, "log_sha256": sha(log), "output": p.stdout}


class Stimulus:
    """Independent behavioral grouping oracle using unbounded Python integers."""
    def __init__(self, wb):
        self.wb, self.rows, self.pending, self.outputs = wb, [], [], 0

    def add(self, reset, valid, x=0, w=0):
        if not (-32768 <= x <= 32767 and -(1 << (self.wb-1)) <= w < (1 << (self.wb-1))):
            raise ValueError("stimulus outside physical signed port range")
        ov, result = 0, 0
        if not reset:
            self.pending = []
        elif valid:
            self.pending.append(int(x)*int(w))
            if len(self.pending) == 22:
                result, ov = sum(self.pending), 1
                self.pending = []
                self.outputs += 1
        self.rows.append([int(reset), int(valid), x, w, ov, result])

    def dot(self, xs, ws, gaps=False):
        if self.pending or len(xs) != 22 or len(ws) != 22:
            raise ValueError("expected complete dot22 boundaries")
        for i, (x, w) in enumerate(zip(xs, ws)):
            if gaps and i % 5 == 1:
                self.add(1, 0, 32767, (1 << (self.wb-1))-1)
            self.add(1, 1, int(x), int(w))

    def write(self, path):
        Path(path).write_text("".join(" ".join(map(str, r))+"\n" for r in self.rows), encoding="ascii")


def synthetic(wb):
    s, rng = Stimulus(wb), random.Random(20261115 + wb)
    lo, hi = -(1 << (wb-1)), (1 << (wb-1))-1
    s.add(0, 0)
    for x, w in [(0, 0), (-32768, lo), (32767, hi), (-32768, hi), (32767, lo), (-1, 1), (1, -1)]:
        s.dot([x]*22, [w]*22)
    s.dot([32767, -32768]*11, [lo, hi]*11, gaps=True)
    for _ in range(64):
        s.dot([rng.randint(-32768, 32767) for _ in range(22)],
              [rng.randint(lo, hi) for _ in range(22)], gaps=True)
    # Drop every possible incomplete group length; also reset just after output.
    for length in range(23):
        for _ in range(length):
            s.add(1, 1, -32768, lo)
        s.add(0, 0)
        s.add(1, 0)
        s.dot(list(range(-11, 11)), [hi]*22, gaps=True)
    s.add(1, 0)
    return s


def append_real(s, folder):
    import numpy as np
    files = sorted(Path(folder).glob("s*.npz"))
    if len(files) != 9:
        raise ValueError(f"expected exactly 9 frozen training fixtures, got {len(files)}")
    n, checks, sources = 0, 0, []
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            x, w, expected = z["qx"], z[f"qw_{s.wb}"], z[f"expected_{s.wb}"]
            if x.shape != (2, 22, 32) or w.shape != (4, 22) or expected.shape != (2, 4, 32):
                raise ValueError("incorrect real fixture shapes")
            for trial in range(2):
                for t in range(32):
                    for j in range(4):
                        xs, ws = x[trial, :, t].tolist(), w[j].tolist()
                        want = sum(int(a)*int(b) for a, b in zip(xs, ws))
                        if want != int(expected[trial, j, t]):
                            raise ValueError("fixture disagrees with independent bigint oracle")
                        running = 0
                        for a, b in zip(xs, ws):
                            running += int(a)*int(b)
                            if not -(1 << (s.wb+19)) <= running < (1 << (s.wb+19)):
                                raise OverflowError("real partial exceeds arithmetic contract")
                            checks += 1
                        s.dot(xs, ws, gaps=(n % 13 == 0))
                        n += 1
            sources.append({"name": path.name, "sha256": sha(path)})
    return {"dots": n, "partial_checks": checks, "fixtures": sources, "source_session": "T"}


def simulate(wb, source, stim, folder, args):
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / "simulation.vvp"
    c = run([args.iverilog, "-g2012", "-s", "tb_dot22", f"-Ptb_dot22.WB={wb}",
             "-o", binary, ROOT/"rtl/tb_dot22.sv", source], folder/"compile.log")
    if c["exit_code"]:
        raise RuntimeError("RTL compilation failed: " + c["output"][-2000:])
    r = run([args.vvp, binary, "+STIM="+str(stim.resolve()).replace("\\", "/")], folder/"run.log")
    return c, r


def mutants(build, args, source_text):
    wb, stim = 12, synthetic(12)
    base = build/"mutants"
    base.mkdir(parents=True, exist_ok=True)
    stim_path = base/"synthetic.txt"
    stim.write(stim_path)
    edits = {
        "missing_product": ("acc <= acc + product_ext;", "acc <= acc;"),
        "21_terms": ("count == 5'd21", "count == 5'd20"),
        "23_terms": ("count == 5'd21", "count == 5'd22"),
        "zero_extension": ("{(AB-PB){product[PB-1]}}", "{(AB-PB){1'b0}}"),
        "narrow_accumulator": ("AB = 16 + WB + 4", "AB = 16 + WB + 3"),
        "reset_keeps_count": ("count <= 5'd0;", "// MUTANT: count not reset"),
        "gap_advances_count": ("if (in_valid) begin", "if (in_valid || count != 0) begin"),
    }
    results = []
    for name, (old, new) in edits.items():
        if old not in source_text:
            raise ValueError("mutation target missing: " + name)
        folder = base/name
        folder.mkdir(exist_ok=True)
        source = folder/"dot22.v"
        source.write_text(source_text.replace(old, new, 1), encoding="utf8")
        c, r = simulate(wb, source, stim_path, folder, args)
        detected = r["exit_code"] != 0 and "mismatch" in r["output"]
        results.append({"name": name, "compiled": c["exit_code"] == 0,
                        "detected": detected, "run": {k:v for k,v in r.items() if k != "output"}})
    corrupted = base/"wrong_streamed_coefficient.txt"
    for row in stim.rows:
        if row[0] and row[1] and row[2] != 0:
            row[3] += 1 if row[3] < 2047 else -1
            break
    stim.write(corrupted)
    c, r = simulate(wb, ROOT/"rtl/dot22.v", corrupted, base/"wrong_streamed_coefficient", args)
    results.append({"name": "wrong_streamed_coefficient", "compiled": c["exit_code"] == 0,
                    "detected": r["exit_code"] != 0 and "mismatch" in r["output"],
                    "run": {k:v for k,v in r.items() if k != "output"}})
    # A distinct channel-index fault: swap two existing coefficients, without
    # changing their values or the expected output. Choose unequal x and w so
    # the dot product must change by (x_i-x_j)*(w_j-w_i), not by chance.
    indexed = synthetic(wb)
    group, swapped = [], False
    for row_index, row in enumerate(indexed.rows):
        if not row[0]:
            group = []
        elif row[1]:
            group.append(row_index)
            if row[4]:
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        a, b = indexed.rows[group[i]], indexed.rows[group[j]]
                        if a[2] != b[2] and a[3] != b[3]:
                            a[3], b[3] = b[3], a[3]
                            swapped = True
                            break
                    if swapped:
                        break
                group = []
        if swapped:
            break
    if not swapped:
        raise ValueError("no discriminating weight-index mutation found")
    index_path = base/"wrong_weight_index.txt"
    indexed.write(index_path)
    c, r = simulate(wb, ROOT/"rtl/dot22.v", index_path, base/"wrong_weight_index", args)
    results.append({"name":"wrong_weight_index", "compiled":c["exit_code"] == 0,
                    "detected":r["exit_code"] != 0 and "dot mismatch" in r["output"],
                    "run":{k:v for k,v in r.items() if k != "output"}})
    return results


def synthesize(wb, folder, args):
    folder.mkdir(parents=True, exist_ok=True)
    source, net = ROOT/"rtl/dot22.v", folder/"netlist.json"
    staged = folder/"dot22.v"
    staged.write_bytes(source.read_bytes())
    if sha(staged) != sha(source):
        raise ValueError("synthesis source snapshot differs")
    # Read dynamic ports; no learned coefficient is embedded in this netlist.
    script = (f'read_verilog dot22.v\nchparam -set WB {wb} dot22\n'
              'hierarchy -check -top dot22\n'
              'synth_xilinx -family xc7 -top dot22 -noiopad -noclkbuf\n'
              'write_json netlist.json\nstat\n')
    (folder/"synth.ys").write_text(script, encoding="utf8")
    # YoWASP's virtual filesystem accepts paths relative to the exposed cwd;
    # a native Windows drive-letter argument is not a portable Yosys path.
    r = run([args.yosys, "-s", "synth.ys"], folder/"synthesis.log", timeout=300, cwd=folder)
    if r["exit_code"]:
        raise RuntimeError("Yosys failed: " + r["output"][-2000:])
    top = json.loads(net.read_text(encoding="utf8"))["modules"]["dot22"]
    counts = Counter(c["type"] for c in top["cells"].values())
    if any(t.startswith("$") for t in counts):
        raise ValueError("unmapped generic cells remain")
    for port, width in [("x",16),("w",wb),("result",wb+20)]:
        if len(top["ports"][port]["bits"]) != width:
            raise ValueError("netlist width mismatch")
    return {"weight_bits": wb, "scope": "dynamic-input dot22 arithmetic kernel only",
            "cells": dict(sorted(counts.items())),
            "logic_lut": sum(n for t,n in counts.items() if re.fullmatch(r"LUT[1-6]",t)),
            "ff": sum(n for t,n in counts.items() if t.startswith("FD")),
            "dsp48": sum(n for t,n in counts.items() if t.startswith("DSP48")),
            "netlist_sha256": sha(net), "script_sha256": sha(folder/"synth.ys"),
            "staged_source_sha256": sha(staged),
            "log_sha256": sha(folder/"synthesis.log")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--fixtures", type=Path)
    ap.add_argument("--iverilog", required=True)
    ap.add_argument("--vvp", required=True)
    ap.add_argument("--yosys")
    args = ap.parse_args()
    build = args.build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = ROOT/"rtl/dot22.v"
    results = {"status":"running", "source_sha256":sha(source),
               "testbench_sha256":sha(ROOT/"rtl/tb_dot22.sv"),
               "runner_sha256":sha(__file__), "runs":[], "mutants":[], "synthesis":[]}
    save(build/"results.json", results)
    try:
        for wb in WIDTHS:
            folder = build/f"w{wb}"
            folder.mkdir(exist_ok=True)
            s = synthetic(wb)
            synthetic_outputs = s.outputs
            real = append_real(s, args.fixtures) if args.fixtures else None
            s.add(1,0)
            stim = folder/"stimulus.txt"
            s.write(stim)
            c,r = simulate(wb, source, stim, folder, args)
            if r["exit_code"] or f"outputs={s.outputs}" not in r["output"] or "PASS dot22" not in r["output"]:
                raise RuntimeError("positive RTL verification failed: " + r["output"][-2000:])
            results["runs"].append({"weight_bits":wb,"accumulator_bits":wb+20,
                "stimulus_rows":len(s.rows),"synthetic_dots":synthetic_outputs,
                "total_dots":s.outputs,"real":real,"stimulus_sha256":sha(stim),
                "compile":{k:v for k,v in c.items() if k!="output"},
                "run":{k:v for k,v in r.items() if k!="output"}})
            print(f"PASS WB={wb}: {s.outputs} dot products", flush=True)
        results["mutants"] = mutants(build,args,source.read_text(encoding="utf8"))
        if not all(x["detected"] for x in results["mutants"]):
            raise RuntimeError("a deliberate fault escaped detection")
        if args.yosys:
            v = run([args.yosys,"-V"],build/"yosys-version.log")
            results["yosys_version"] = v["output"].strip()
            for wb in WIDTHS:
                row = synthesize(wb, build/f"synth-w{wb}", args)
                results["synthesis"].append(row)
                print(f"MAPPED WB={wb}: LUT={row['logic_lut']} FF={row['ff']} DSP={row['dsp48']}",flush=True)
        if sha(source) != results["source_sha256"]:
            raise RuntimeError("source changed during verification")
        results["status"] = "passed"
    except Exception as e:
        results["status"], results["error"] = "failed", f"{type(e).__name__}: {e}"
        raise
    finally:
        save(build/"results.json",results)


if __name__ == "__main__":
    main()
