# 统一研究工作流

本流程从不可变快照开始，所有下游运行都应绑定同一个 `snapshot_id`。命令在仓库根目录执行；完整参数见 [COMMANDS.md](COMMANDS.md)。

```text
step1 export/validate
        ↓
step2 states + step2 values
        ↓
step3 cluster
        ↓
step4 run  →  E1–E4 套件 / E5 / E6

step2 conditional → step2 select   （可选支线，不进入上图）
```

## 1. 准备环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\research_pipeline\requirements.txt
```

建议为本次研究记录变量：

```powershell
$SnapshotId = "20260831T074757Z_7a98fc5698"
$DataRoot = ".\research_data"
$RunsRoot = ".\research_runs"
```

## 2. 导出快照

若仓库外仍有原行情、股票池、因子运行和回测目录，可复制示例配置并把其中路径改为相对于仓库根目录的实际路径：

```powershell
Copy-Item .\research_pipeline\config.example.yaml .\research_pipeline\config.local.yaml
python -m research_pipeline step1 export `
  --config .\research_pipeline\config.local.yaml `
  --snapshot-id $SnapshotId
```

配置必须提供 `market_panel`、`pool_dir`；`factor_runs_root`、`backtests_root`、`pool_id` 可选，`output_root` 缺省为 `research_data`。导出成功后会写入 `snapshots\<snapshot_id>`，并更新 `manifest.json` 与 `current.txt` 指针。

已有快照时不必重新导出。除非确认要替换同名快照，否则不要使用 `--overwrite`。

## 3. 校验快照

```powershell
python -m research_pipeline step1 validate `
  --data-root $DataRoot `
  --snapshot-id $SnapshotId
```

继续运行前应确认输出中的 `"valid": true`。校验会检查六张必需表、主键/必需列以及清单中的 SHA-256。本地已有数据时，可从本步开始。

## 4. 生成市场状态

```powershell
python -m research_pipeline step2 states `
  --data-root $DataRoot `
  --runs-root $RunsRoot `
  --snapshot-id $SnapshotId
```

记下 JSON 输出中的 `run_dir`：

```powershell
$StateRun = ".\research_runs\state_<时间>_<哈希>"
```

## 5. 生成因子值

```powershell
python -m research_pipeline step2 values `
  --data-root $DataRoot `
  --runs-root $RunsRoot `
  --snapshot-id $SnapshotId
```

记下输出目录：

```powershell
$ValuesRun = ".\research_runs\values_<时间>_<哈希>"
```

若部分表达式失败，命令仍保存清单，但退出码为 `2`；先查看 `factor_values_manifest.json` 中的 `errors`。

## 6. 可选：条件评价与因子选择

该分支是可选分析，不进入步骤 3 和步骤 4。

```powershell
python -m research_pipeline step2 conditional `
  --data-root $DataRoot `
  --values-run $ValuesRun `
  --state-run $StateRun `
  --runs-root $RunsRoot `
  --snapshot-id $SnapshotId

$ConditionalRun = ".\research_runs\conditional_<时间>_<哈希>"
python -m research_pipeline step2 select `
  --data-root $DataRoot `
  --conditional-run $ConditionalRun `
  --runs-root $RunsRoot `
  --snapshot-id $SnapshotId `
  --state-id bull
```

## 7. 聚类与因子群

步骤 3 只计算一次 K-Means，并写出供 E1–E6 共用的聚类产物。测试起点、未来收益周期、簇数和候选因子在这里确定。

```powershell
python -m research_pipeline step3 cluster `
  --data-root $DataRoot `
  --values-run $ValuesRun `
  --state-run $StateRun `
  --output-root $RunsRoot `
  --snapshot-id $SnapshotId `
  --test-start 2025-01-01
```

记下聚类目录：

```powershell
$ClusterRun = ".\research_runs\cluster_<时间>_<哈希>"
```

未指定 `--test-start` 时，代码使用全部行情日期约 80% 处作为测试起点。正式对照实验建议显式指定该日期。

## 8. 运行 E1–E6

步骤 4 只读取 `--cluster-run`，不再重复指定因子值或状态路径，也不再各自重算 K-Means。

```powershell
python -m research_pipeline step4 run `
  --cluster-run $ClusterRun `
  --output-root $RunsRoot `
  --experiments E1,E2,E3,E4,E5,E6
```

- E1–E4 写入同一个 `experiments_*` 套件目录。
- E5、E6 各自写入 `e5_*`、`e6_*`，不改写 E1–E4 目录。
- 可传入子集，例如 `--experiments E1,E5`。

各级差异见 [E1](experiments/E1.md) 到 [E6](experiments/E6.md)。

## 9. 结果核对

- 每个运行目录都应记录相同的 `snapshot_id`。
- 步骤 3 查看 `cluster_manifest.json`。
- E1–E4 查看 `experiment_suite.json`；E5、E6 分别查看 `e5_manifest.json`、`e6_manifest.json`。
- 对照 `metrics.json` 前，先确认候选因子数、代表因子数、测试起点和成本参数一致。
- 文件含义见 [产物说明](reference/artifacts.md)，解释结果前先阅读 [已知限制](reference/limitations.md)。
