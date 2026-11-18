"""Render the two main comparisons from aggregate counts only."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.ticker import MaxNLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    folder = parser.parse_args().results
    attribution = json.loads((folder / "aggregate.json").read_text(encoding="utf-8"))
    control = json.loads((folder / "equal-storage.json").read_text(encoding="utf-8"))
    available = {f.name for f in fontManager.ttflist}
    chinese = next((f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei") if f in available), None)
    if chinese:
        plt.rcParams["font.family"] = chinese
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "path", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#829399"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.05), layout="constrained")
    required = attribution["scenarios"]["X16_W8_to_W12"]["minimum_rows_fix_original_disagreement"]
    labels = (["1 行", "2 行", "3 行", "4 行", "无法恢复"] if chinese else
              ["1 row", "2 rows", "3 rows", "4 rows", "Unrecovered"])
    counts = [required.get(str(k), 0) for k in (1, 2, 3, 4, "unreachable")]
    left.bar(labels, counts, width=.55, color=["#176f70"] * 4 + ["#9b653c"])
    for i, count in enumerate(counts):
        left.text(i, count+.25, str(count), ha="center", color="#152e35")
    left.set(title="23 次翻判各需几行升到 12 位？" if chinese else "Upgrades needed for each of the 23 flips",
             ylabel="试次数" if chinese else "Trials", ylim=(0, max(counts) * 1.18))
    right.set(title="同等权重存储下与浮点的分歧" if chinese else "Disagreements at equal weight payload",
              ylabel="分歧试次数 / 1,296" if chinese else "Disagreements / 1,296", ylim=(0, 15))
    summary = control["summary"]
    for offset, prefix, label, color in [
        (-.17, "mixed", "第一天选行" if chinese else "Rows selected on T", "#176f70"),
        (.17, "uniform", "统一位宽" if chinese else "Uniform width", "#405998"),
    ]:
        x = [k+offset for k in range(3)]
        y = [summary[prefix+str(k if prefix == "mixed" else 8+k)]["float_disagreements"] for k in (1, 2, 3)]
        right.bar(x, y, width=.31, color=color, label=label)
        for pos, height in zip(x, y):
            right.text(pos, height+.23, str(height), ha="center", color=color)
    right.set_xticks([0, 1, 2], ["99 B", "110 B", "121 B"])
    right.legend(loc="upper right", frameon=False)
    for axis in (left, right):
        axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=.18)
    for suffix in ("png", "svg"):
        fig.savefig(folder / ("row-restoration." + suffix), dpi=190)


if __name__ == "__main__":
    main()
