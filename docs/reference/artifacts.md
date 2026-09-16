# 产物说明

运行目录名由 UTC 时间戳和配置哈希组成；同一秒内重复相同配置可能因目录已存在而失败。JSON 文件均以 UTF-8 写出。

## step1 export

在 `<output_root>\snapshots\<snapshot_id>\` 生成：

- 六张必需表：`market.parquet`、`universe.csv`、`factor_catalog.parquet`、`factor_metrics.parquet`、`backtest_metrics.parquet`、`backtest_runs.csv`。
- `source_metadata.json`：解析后的来源路径和股票池元数据。
- `manifest.json`：快照编号、统计、warnings、文件列、日期范围和 SHA-256。

成功后还会更新 `<output_root>\manifest.json` 和 `<output_root>\current.txt`。

## step1 validate

不写运行目录，只向标准输出打印校验 JSON，包括 `valid`、`errors`、快照路径、快照编号、清单有效标志、证券数和因子数。

## step2 states

目录：`state_<UTC时间>_<哈希>\`

- `market_states.parquet`：每日市场特征、规则状态、方法和拟合截止日期。
- `config.json`：窗口、阈值、快照和运行编号。
- `manifest.json`：状态行数、标签集合和文件清单。

## step2 values

目录：`values_<UTC时间>_<哈希>\`

- `factor_values\factor_id=<id>\values.parquet`：每个成功因子的 `date`、`code`、`factor_id`、`value`。
- `factor_values_manifest.json`：每个因子的表达式、状态、覆盖率和错误。

表达式失败的因子不会创建其值分区，但其他因子继续处理。

## step2 conditional

目录：`conditional_<UTC时间>_<哈希>\`

- `conditional_metrics.parquet`：按因子、市场状态和 train/valid/test 切分汇总的条件 IC 指标。
- `conditional_metrics_manifest.json`：输入运行、比例、成功/失败数和错误。

## step2 select

目录：`selection_<UTC时间>_<哈希>\`

- `selection.json`：类别得分、选中因子、拒绝原因和等权组合。
- `manifest.json`：快照、状态、类别、切分、选中数量和文件清单。

## step3 cluster

目录：`cluster_<UTC时间>_<哈希>\`

- `daily_factor_performance.parquet`：每个日期、因子的 IC、Rank IC、slope、样本数和标签成熟日期。
- `behavior_features.parquet`：训练期因子行为摘要，包含三类指标的均值、标准差和 IR。
- `factor_clusters.parquet`：全部候选因子的 `factor_id` 与簇标签。
- `clustering.json`：簇数、特征列、中心、惯性、代表因子、随机种子和训练截止日期。
- `factor_groups.parquet`：因子所属簇、代表因子、是否代表、到中心距离和群大小。
- `factor_groups.json`：所有群的汇总成员清单。
- `factor_groups\cluster_XXX.json`：单个群的编号、代表因子、大小和成员。
- `cluster_manifest.json`：`data_root`、`values_run`、`state_run`、`snapshot_id`、`test_start`、`horizon`、代表因子和文件清单。

E1–E6 都读取这一份聚类产物，不再各自重算 K-Means。

## step4 run：E1–E4

目录：`experiments_<UTC时间>_<哈希>\`

从 `--cluster-run` 复制公共文件：

- `daily_factor_performance.parquet`
- `behavior_features.parquet`
- `factor_clusters.parquet`
- `clustering.json`
- `experiment_suite.json`：套件运行编号、快照、候选/代表因子、测试起点、成本配置、公共文件及本次实际运行的实验指标和目录。

每个被选中的 `E1\` 到 `E4\` 子目录均有：

- `config.json`：组件、快照、代表因子、时间边界和参数。
- `factor_weights.parquet`：每日代表因子权重。
- `alpha.parquet`：测试期股票 Alpha。
- `daily.parquet`：下一交易日执行后的日度回测。
- `metrics.json`：汇总指标。

E2、E3、E4 额外生成 `model.pt`；E1 不生成模型文件。若 `--experiments` 只包含部分 E1–E4，套件目录也只创建对应子目录。

## step4 run：E5

目录：`e5_<UTC时间>_<哈希>\`

- `daily_factor_performance.parquet`、`behavior_features.parquet`、`clustering.json`：从步骤 3 复制的行为与聚类信息。
- `factor_groups.parquet`：因子所属簇、代表因子、是否代表、到中心距离和群大小。
- `factor_groups.json`：所有群的汇总成员清单。
- `factor_groups\cluster_XXX.json`：单个群的编号、代表因子、大小和成员。
- `representative_weights.parquet`：E2 式外层市场路由的每日代表因子权重。
- `selected_groups.parquet`：每日 Top-1 代表因子、簇、外层权重和群大小。
- `factor_weights.parquet`：被选群内全部因子的最终等权。
- `model.pt`：外层市场路由参数。
- `alpha.parquet`、`daily.parquet`、`metrics.json`：Alpha、日度回测和指标。
- `config.json`、`e5_manifest.json`：配置与审计清单。

## step4 run：E6

目录：`e6_<UTC时间>_<哈希>\`

- 行为、聚类、因子群及分群 JSON 与 E5 同类，同样来自步骤 3。
- `outer_representative_weights.parquet`：外层市场路由权重。
- `selected_groups.parquet`：每日硬 Top-1 选群。
- `inner_factor_weights.parquet`：所有簇、所有日期的内层动态权重。
- `factor_weights.parquet`：仅保留当日被选簇并重新归一后的最终权重。
- `models\outer_router.pt`：外层模型参数。
- `models\inner\cluster_XXX.pt`：各簇内层模型参数。
- `inner_router_configs.json`：各内层模型的成员、随机种子和学习到的市场融合系数。
- `alpha.parquet`、`daily.parquet`、`metrics.json`：Alpha、日度回测和指标。
- `config.json`、`e6_manifest.json`：配置与审计清单。

## 回测字段与指标

`daily.parquet` 当前包含 `date`、`signal_date`、`gross_return`、`turnover`、`cost`、`net_return`、`holding_count`、`weights`、`equity`。

`metrics.json` 当前包含：

- `total_return`、`annual_return`
- `sharpe`、`max_drawdown`、`calmar`、`win_rate`
- `average_turnover`、`total_cost`、`annual_cost`
- `number_of_trades`、`average_holding_count`

数据字段约束见 [数据契约](data-contracts.md)。
