# 数据契约

## 快照定位与一致性

数据根目录结构为：

```text
research_data\
  current.txt
  manifest.json
  snapshots\
    <snapshot_id>\
```

省略 `--snapshot-id` 时，代码读取 `current.txt`。状态、因子值、条件评价、聚类和实验运行都应记录同一 `snapshot_id`。步骤 3 会把 `data_root`、`values_run`、`state_run` 写入 `cluster_manifest.json`；步骤 4 按该清单重建上下文，并在对应运行清单存在时检查编号是否一致。

## 快照的六张必需表

`step1 validate` 要求以下文件全部存在，并与快照 `manifest.json` 中记录的 SHA-256 一致。

### `market.parquet`

- 主键：`date`、`code`，组合必须唯一。
- 必需列：`date`、`code`、`open`、`close`、`high`、`low`、`volume`、`amount`。
- 导出时会把 `timestamps` 重命名为 `date`，把 `vol` 重命名为 `volume`；证券代码规范为字符串。
- 可保留 `ChangePCT`、`Ifsuspend`、均线及技术指标等已知可选列。
- 市场状态至少使用 `date`、`code`、`close`，存在 `ChangePCT` 时优先将其除以 100 作为日收益。

### `universe.csv`

- 主键：`code`，必须唯一。
- 必需列：`code`、`pool_id`、`selected_at`、`source_signature`。
- 导出编码为 UTF-8 with BOM，便于 Windows 工具读取。

### `factor_catalog.parquet`

- 必需列：`factor_id`、`expression`、`pool_id`、`accepted`。
- 当前导出还写入 `canonical_expression`、`backtest_expression`、`factor_run_id`、`source_signature`、`expected_direction`、`category`、`category_source`、`rationale`、`source_path`。
- `step2 values` 只处理 `accepted` 为真的记录；`factor_id` 是下游分区键。

### `factor_metrics.parquet`

- 必需列：`factor_id`、`metric_scope`、`split`。
- 当前生成指标可包含 `horizon`、`score`、`ic`、`rank_ic`、`icir`、`rank_icir`、`coverage`、`sample_count`、`calculated_at`。

### `backtest_metrics.parquet`

- 必需列：`factor_id`、`backtest_run_id`、`split`、`start_date`、`end_date`。
- 可包含收益、夏普、回撤、换手、成本、交易次数、持仓数和来源路径。
- 没有可用历史回测时，导出允许生成只有表头的空表，并在清单中记录 warning。

### `backtest_runs.csv`

- 当前保存去重后的 `backtest_run_id` 与 `source_path`。
- 没有回测运行时允许为空，但文件仍必须存在。

## 快照元数据

- `manifest.json`：快照版本、编号、创建时间、股票池、证券数、因子数、warnings、六张表的行列和哈希。
- `source_metadata.json`：行情、股票池、因子运行、回测来源以及原股票池清单。
- 数据根目录的 `manifest.json` 和 `current.txt` 是当前快照指针，不属于单个快照的六张表。

## 因子值运行契约

```text
research_runs\values_<run_id>\
  factor_values_manifest.json
  factor_values\
    factor_id=<factor_id>\
      values.parquet
```

每个成功分区的 `values.parquet` 固定写出 `date`、`code`、`factor_id`、`value`。清单记录请求数、成功数、失败数以及每个表达式的行数、有效值数、覆盖率或错误。`--values-run` 必须指向该运行根目录，而不是 `factor_values\` 子目录。

## 市场状态运行契约

`market_states.parquet` 每个交易日一行，当前列包括：

- `date`
- `market_return`
- `cross_section_dispersion`
- `up_ratio`
- `market_volatility`
- `state_id`、`state_label`
- `state_method`
- `fit_window_end`

规则标签为 `sideways`、`high_volatility`、`bull` 或 `bear`。同目录的 `config.json` 和 `manifest.json` 记录快照、阈值和标签集合。`--state-run` 必须指向包含该文件的运行根目录。

## 条件评价契约

`conditional_metrics.parquet` 以因子、状态和时间切分为粒度，当前字段包括：

- `factor_id`、`category`、`category_source`
- `state_id`、`split`、`horizon`
- `sample_count`、`coverage`
- `ic_mean`、`rank_ic_mean`、`icir`、`hit_rate`、`stability`

`split` 按日期顺序产生 `train`、`valid`、`test`，不是随机切分。`conditional_metrics_manifest.json` 记录输入运行的解析路径、比例和错误。

## 聚类运行契约

```text
research_runs\cluster_<run_id>\
  cluster_manifest.json
  clustering.json
  daily_factor_performance.parquet
  behavior_features.parquet
  factor_clusters.parquet
  factor_groups.parquet
  factor_groups.json
  factor_groups\
    cluster_XXX.json
```

`cluster_manifest.json` 记录 `data_root`、`values_run`、`state_run`、`snapshot_id`、`test_start`、`horizon`、`selected_factor_ids`、`skipped_factor_ids` 和 `representatives`。`selected_factor_ids` 只包含训练期画像非空、实际参与聚类的因子；画像全空的因子写入 `skipped_factor_ids`，不进入步骤 4。`step4 run --cluster-run` 只接受该运行根目录，不再重复指定因子值或状态路径。

## E1–E6 共同输入契约

- 步骤 4 只读取 `cluster_run`；步骤 3 负责绑定快照、因子值和市场状态。
- 因子分区名中的 `factor_id` 决定可用候选池。
- 未来收益为同一证券内 `close(t + horizon) / close(t) - 1`，其中 shift 按交易记录执行。
- Alpha 表固定使用 `date`、`code`、`alpha`；因子权重表至少使用 `date`、`factor_id`、`weight`。
- 回测在下一交易日执行信号，并根据 `Ifsuspend`、`if_suspend` 或 `suspended` 中第一个存在的字段冻结停牌持仓。

各运行实际文件见 [产物说明](artifacts.md)，契约边界见 [已知限制](limitations.md)。
