# 脑电空间滤波的行精度与分类决策

四行空间滤波权重决定如何把 22 路脑电合成四条波形。某一行的权重舍入，何时足以改变最后的左右手判断？能不能在第一天找出最需要保留精度的行，第二天一直使用这个选择？这项实验用九人共 1,296 次跨日判断检验这两个问题。[中文研究报告](../../output/bci-row-restoration-2026-11-18/README.md)

在 99、110、121 字节的权重预算下，每人固定把一、二、三行升到 12 位，其余用 8 位。即使允许看完整个第二天，再为每人选出让当天分歧最少的一种组合，三档仍分别有 **11、4、3 次**判断偏离浮点参考。用同样空间为四行统一分配 9、10、11 位，分别有 **11、3、0 次**分歧。这把“第一天选得不够好”和“这种固定分配本身还能改善多少”两个问题分开了。[按人、按预算的精确枚举](../../output/bci-row-restoration-2026-11-18/budget-ceiling.json)

## 先核对公开结果

从包含 `portfolio`、`tools` 和 `output` 的仓库根目录执行。以下命令只用 Python 标准库和仓库自带的汇总数据。它们分别核对 224 个组合和 72 条分人配置记录，重算 27 个「受试者 × 预算」的事后最少分歧数，以及运行针对枚举逻辑的合成测试。

```powershell
$Results = 'output/bci-row-restoration-2026-11-18'
python portfolio/eeg-row-restoration/check_public_results.py $Results
python portfolio/eeg-row-restoration/budget_ceiling.py $Results --check "$Results/budget-ceiling.json"
python portfolio/eeg-row-restoration/test_budget_ceiling.py
```

以下命令可在装有 Matplotlib 的环境中从汇总数字重新画图。

```powershell
python portfolio/eeg-row-restoration/plot_results.py $Results
```

## 从私有原始记录重做实验

原试验使用 CPython 3.13、锁定版本的 NumPy、SciPy、MNE 和 scikit-learn。依照[原实验环境与复现步骤](../eeg-fixedpoint-audit/README.md)取得有授权的官方数据，生成九个冻结模型以及 T 和 E 的评分所需记录。下面的 `$Run` 指含有 `training_manifest.json`、`models`、`evaluation` 的目录。示例从上述根目录运行，`$Raw` 和 `$Run` 指向自己的本地文件，`$Scratch` 和 `$Aggregates` 使用不同的新目录。命令示例假定当前 Python 已安装原试验依赖。

```powershell
$Pilot = 'portfolio/eeg-fixedpoint-audit'
$Rows = 'portfolio/eeg-row-restoration'
$Raw = '<官方MAT文件的本地目录>'
$Run = '<原实验冻结目录>'
$Scratch = '<本机私有的新输出目录>'
$Aggregates = '<仅用于本次重跑聚合结果的新目录>'
python "$Rows/analyze_row_restoration.py" --run $Run --private "$Scratch/trials" --public $Aggregates --independent-run
python "$Rows/compare_static_precision.py" --pilot $Pilot --raw $Raw --freeze $Run --private "$Scratch/choices" --output "$Aggregates/equal-storage.json"
python tools/check_bci_precision_attribution.py --raw-dir $Raw --frozen-dir $Run --output "$Scratch/moments.json" --public-output "$Aggregates/moment-check.json" --subset-counts "$Aggregates/subset-counts.csv"
python "$Rows/budget_ceiling.py" $Aggregates --write "$Aggregates/budget-ceiling.json"
python "$Rows/paired_budget_decisions.py" --raw $Raw --freeze $Run --selected "$Scratch/choices/training-selections.json" --results $Aggregates --write "$Aggregates/paired-budget-decisions.json"
```

最后一条命令从第二天的脑电窗口重算通道二阶矩，用它直接计算混合方案和统一方案的分类结果。输出给出每对方案共同和单独保留参考判断的次数，也按真实标签重复计数。可将 `--write` 换为 `--check` 并指定同一次运行保存的文件，验证 27 个个体预算组合。[配对复算程序](paired_budget_decisions.py) · [原实验各组合的配对结果](../../output/bci-row-restoration-2026-11-18/paired-budget-decisions.json)

对于原始归档，省略 `--independent-run` 即可严格核对历史输入文件哈希。自行重新训练产生的归档应明确记录其身份。各种计数、恢复行编号、抵消比例的定义在[逐行实验说明](ROW-ANALYSIS.md)，固定预算的选择规则在[实验约定](CONTROL.md)。

## 可以亲手验证的一个问题

在公开的组合表找出固定 16 位输入、权重从 8 位到 12 位、仅升级第 3 行的记录。先只看它消除了多少个原有分歧，自己算出预期剩余数，再用新增分歧修正。然后挑一位受试者，对照其四种单行升级方案，检验即使可以事后任选一行，是否还会留下分歧。[组合计数](../../output/bci-row-restoration-2026-11-18/subset-counts.csv) · [分人结果](../../output/bci-row-restoration-2026-11-18/budget-ceiling.json)

沿用原工程的[代码许可](../eeg-fixedpoint-audit/LICENSE)和[数据署名与使用说明](../eeg-fixedpoint-audit/DATA_NOTICE.md)。用户确定了研究问题、资源和收敛条件。上一轮 Claude 参与制定实验范围，本轮软件实现、分析和本地复算由 Codex 完成。公开材料只含代码、合成测试、示意和聚合结果；逐试次记录与模型保存在本机。
