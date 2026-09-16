# 命令参考

以下命令均在仓库根目录的 Windows PowerShell 中执行，并使用相对路径。`<...>` 表示需要替换的值，不要原样输入。

唯一入口：

```powershell
python -m research_pipeline <step> <action> [参数]
```

- [step1 export](#step1-export)
- [step1 validate](#step1-validate)
- [step2 states](#step2-states)
- [step2 values](#step2-values)
- [step2 conditional](#step2-conditional)
- [step2 select](#step2-select)
- [step3 cluster](#step3-cluster)
- [step4 run](#step4-run)

## 步骤 1：快照

### step1 export

从原系统文件导出不可变研究快照。本地已有快照时可跳过本命令。

```powershell
Copy-Item .\research_pipeline\config.example.yaml .\research_pipeline\config.local.yaml
python -m research_pipeline step1 export `
  --config .\research_pipeline\config.local.yaml `
  --snapshot-id <snapshot_id>
```

- 必填：`--config`，支持 YAML 或 JSON。
- 可选：`--snapshot-id`；省略时自动生成。
- 可选：`--overwrite`，删除并重建同名快照目录；默认关闭。
- 配置键：`market_panel`、`pool_dir` 必填；`factor_runs_root`、`backtests_root`、`pool_id` 可选；`output_root` 默认 `research_data`。

### step1 validate

校验快照文件、必需列、唯一键和清单哈希。

```powershell
python -m research_pipeline step1 validate `
  --data-root .\research_data `
  --snapshot-id <snapshot_id>
```

- `--data-root` 默认 `research_data`。
- `--snapshot-id` 可选；省略时读取 `<data-root>\current.txt`。
- 有效返回码为 `0`，无效返回码为 `1`。

## 步骤 2：特征

### step2 states

由市场截面收益生成规则市场状态。

```powershell
python -m research_pipeline step2 states `
  --data-root .\research_data `
  --runs-root .\research_runs `
  --snapshot-id <snapshot_id>
```

- `--data-root` 默认 `research_data`；`--runs-root` 默认 `research_runs`。
- `--volatility-window` 默认 `20`。
- `--trend-threshold` 默认 `0.002`。
- `--volatility-threshold` 默认 `0.03`。
- `--snapshot-id` 可选。

### step2 values

按因子表达式生成分区因子值。

```powershell
python -m research_pipeline step2 values `
  --data-root .\research_data `
  --runs-root .\research_runs `
  --snapshot-id <snapshot_id>
```

- `--data-root` 默认 `research_data`；`--runs-root` 默认 `research_runs`。
- `--factor-id <id>` 可重复传入；默认处理目录中全部 accepted 因子。
- `--max-factors <n>` 可限制按 `factor_id` 排序后的前 `n` 个；默认不限制。
- `--snapshot-id` 可选。
- 全部成功返回码为 `0`，存在表达式失败时返回码为 `2`。

### step2 conditional

按市场状态和时间切分评价分区因子值。这是可选支线，不进入步骤 3 和步骤 4。

```powershell
python -m research_pipeline step2 conditional `
  --data-root .\research_data `
  --values-run .\research_runs\values_<run_id> `
  --state-run .\research_runs\state_<run_id> `
  --runs-root .\research_runs `
  --snapshot-id <snapshot_id>
```

- 必填：`--values-run`、`--state-run`。
- `--data-root` 默认 `research_data`；`--runs-root` 默认 `research_runs`。
- `--horizon` 默认 `20`；`--max-factors` 默认不限制。
- `--train-ratio` 默认 `0.6`；`--valid-ratio` 默认 `0.2`，余下日期为测试集。
- `--snapshot-id` 可选。
- 全部成功返回码为 `0`，存在分区失败时返回码为 `2`。

### step2 select

为一个市场状态和一个时间切分生成类别、因子与等权组合选择。这是可选支线，不进入步骤 3 和步骤 4。

```powershell
python -m research_pipeline step2 select `
  --data-root .\research_data `
  --conditional-run .\research_runs\conditional_<run_id> `
  --runs-root .\research_runs `
  --snapshot-id <snapshot_id> `
  --state-id bull
```

- 必填：`--conditional-run`、`--state-id`。
- `--data-root` 默认 `research_data`；`--runs-root` 默认 `research_runs`。
- `--category` 可选；省略时选择当前状态和切分中得分最高的类别。
- `--split` 可选值为 `train`、`valid`、`test`，默认 `train`。
- `--top-n` 默认 `10`；`--min-sample-count` 默认 `10`。
- `--correlation-threshold` 默认 `0.8`。当前 CLI 路径没有加载因子值，因此不会实际执行相关性剔除，详见 [已知限制](reference/limitations.md)。
- `--snapshot-id` 可选。

## 步骤 3：聚类

### step3 cluster

在训练期行为画像上执行 K-Means，选出代表因子，并保存完整因子群。E1–E6 共用这一份结果，不再各自重算。

```powershell
python -m research_pipeline step3 cluster `
  --data-root .\research_data `
  --values-run .\research_runs\values_<run_id> `
  --state-run .\research_runs\state_<run_id> `
  --output-root .\research_runs `
  --snapshot-id <snapshot_id> `
  --test-start 2025-01-01 `
  --n-clusters 5
```

- 必填：`--values-run`、`--state-run`。
- `--data-root` 默认 `research_data`；`--output-root` 默认 `research_runs`。
- `--snapshot-id`、可重复的 `--factor-id`、`--max-factors`、`--test-start` 均可选。
- `--test-start` 省略时采用行情日期约 80% 处。
- `--horizon` 默认 `20`；`--n-clusters` 默认 `20`；`--random-state` 默认 `42`。
- 实际簇数为 `min(n_clusters, 候选因子数)`。

## 步骤 4：实验

### step4 run

读取步骤 3 的聚类产物，运行指定的 E1–E6 实验。

```powershell
python -m research_pipeline step4 run `
  --cluster-run .\research_runs\cluster_<run_id> `
  --output-root .\research_runs `
  --experiments E1,E2,E3,E4,E5,E6
```

- 必填：`--cluster-run`，必须指向包含 `cluster_manifest.json` 的步骤 3 运行根目录。
- `--output-root` 默认 `research_runs`。
- `--experiments` 默认 `E1,E2,E3,E4,E5,E6`，也可传入子集，例如 `E1,E5`。
- `--top-n` 默认 `20`。
- `--factor-top-k` 省略时等于代表因子数，仅影响 E4。
- `--fee-rate 0.001`、`--slippage-rate 0.001`。
- `--rolling-window 40`、`--epochs 10`、`--random-state 42`。

E1–E4 写入同一个 `experiments_*` 套件目录；E5、E6 各自写入 `e5_*`、`e6_*`。步骤 4 不再接受 `--data-root`、`--values-run`、`--state-run`、`--test-start`、`--horizon`、`--n-clusters`，这些值已写在 `cluster_manifest.json` 中。

参数语义和实验差异见 [E1](experiments/E1.md) 到 [E6](experiments/E6.md)。
