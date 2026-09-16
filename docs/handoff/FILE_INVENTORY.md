# 代码文件清单（已完成内容）

包根：`research_pipeline/`。下列文件均为**已完成、在用实现**。没有未接线的草稿模块。`__init__.py` 只做包标记，不单独说明。

## 入口

| 文件 | 状态 | 作用与关键符号 |
| --- | --- | --- |
| `__main__.py` | 完成 | `python -m research_pipeline` → `cli.main` |
| `cli.py` | 完成 | `build_parser` / `main` / `configure_utf8_output`。Windows 下 stdout/stderr 设 UTF-8。按 step/action 分发到各 `run_*` |
| `config.example.yaml` | 完成 | 仅 `step1 export` 用。键：`market_panel`、`pool_dir`、可选 `factor_runs_root`、`backtests_root`、`pool_id`、`output_root` |
| `requirements.txt` | 完成 | pandas、numpy、pyarrow、PyYAML、scikit-learn、torch、pytest |

`cli.py` 子命令与函数对应：

| 命令 | 调用 |
| --- | --- |
| `step1 export` | `export_snapshot` |
| `step1 validate` | `validate_snapshot` |
| `step2 states` | `run_state_analysis` |
| `step2 values` | `materialize_factor_values` |
| `step2 conditional` | `run_conditional_evaluation` |
| `step2 select` | `run_selection` |
| `step3 cluster` | `run_cluster` |
| `step4 run` | `run_experiments` |

---

## `common/` 跨步骤复用（不含实验编排）

| 文件 | 状态 | 作用与关键符号 |
| --- | --- | --- |
| `schema.py` | 完成 | 快照表名、必需列、`SNAPSHOT_VERSION`。validate/export 的契约源 |
| `io.py` | 完成 | `normalize_code`、`normalize_market_frame`、`validate_unique_keys`、`file_sha256`、`write_json`、`resolve_snapshot`、`load_snapshot_table`。省略 snapshot-id 时读 `current.txt` |
| `expression_engine.py` | 完成 | `evaluate_expression`、`ExpressionError`。支持时序/截面算子、中缀运算、`$close`、Alphagen 天数后缀。`step2 values` 用它物化因子 |
| `features.py` | 完成 | `make_forward_return`（按证券 `shift(-horizon)` 个**交易记录**）、`market_state_features`、`assign_rule_states`（bull/bear/sideways/high_volatility） |
| `behavior.py` | 完成 | `build_daily_factor_performance`（日度 IC/Rank IC/slope + `label_available_date`）、`build_rolling_performance_features`。成熟日用 **日历日** `date + horizon days`（与交易日 shift 口径不一致，见 limitations，保持原样） |
| `models.py` | 完成 | `FactorRouter`（隐藏维 32）、`gumbel_topk_mask`、`turnover_penalty` |
| `portfolio.py` | 完成 | `backtest_long_only`：信号下一交易日执行、停牌冻结、成本=`turnover*(fee+slippage)`、换手为权重绝对变化之和（未除以 2） |
| `context.py` | 完成 | `ExperimentContext`、`check_snapshot`、`behavior_frame`、`prepare_experiment_context`。step3/step4 的统一加载入口 |

---

## `step1_snapshot/` 快照

| 文件 | 状态 | 作用与关键符号 |
| --- | --- | --- |
| `export.py` | 完成 | `ExportSources`、`export_snapshot`。写出六张表 + manifest + `current.txt`。已有快照时用户可跳过 |
| `validate.py` | 完成 | `validate_snapshot`。检查文件存在、SHA-256、必需列、行情/股票池主键 |

本地快照：`research_data/snapshots/20260831T074757Z_7a98fc5698`。

---

## `step2_features/` 特征

| 文件 | 状态 | 作用与关键符号 |
| --- | --- | --- |
| `states.py` | 完成 | `run_state_analysis` → `state_* / market_states.parquet` |
| `values.py` | 完成 | `materialize_factor_values` → `values_* / factor_values/factor_id=*/values.parquet`。只处理 catalog 中 `accepted` 为真的因子。表达式失败记清单，退出码 2 |
| `conditional.py` | 完成（旁路） | `assign_time_splits`、`run_conditional_evaluation`。按日期比例切 train/valid/test，不进 step3/4 |
| `select.py` | 完成（旁路） | `score_categories`、`select_factors`、`equal_weight_portfolio`、`run_selection`。打分 `0.7*rank_ic_mean+0.3*icir`。CLI **未传入因子值**，`--correlation-threshold` 不真正剔除 |
| `classification.py` | 完成 | `infer_category`：按表达式猜 trend/volatility/liquidity/price 等 |
| `evaluation.py` | 完成 | `daily_ic`、`conditional_ic`，供条件评价使用 |

主线只要 `states` + `values`。

---

## `step3_clustering/` 聚类（E1–E6 共用一份）

| 文件 | 状态 | 作用与关键符号 |
| --- | --- | --- |
| `cluster.py` | 完成 | `run_cluster`。调 `prepare_experiment_context` → K-Means → 写 `cluster_*`：performance/behavior/clusters/clustering.json/factor_groups/`cluster_manifest.json` |
| `clustering.py` | 完成 | `fit_behavior_kmeans`、`select_representatives`、`_feature_matrix`。行全 NaN 则 `ValueError: each factor needs at least one finite behavior feature` |
| `grouping.py` | 完成 | `build_factor_group_table`、`select_top_representative_group`（硬 Top-1 选群）、`expand_selected_groups_to_equal_weights`（E5）、`select_active_inner_weights`（E6）、`save_factor_groups` |

`cluster_manifest.json` 记录 `data_root`、`values_run`、`state_run`、`snapshot_id`、`test_start`、`horizon`、代表因子。step4 只读这个目录。

---

## `step4_experiments/` 实验

| 文件 | 状态 | 作用与关键符号 |
| --- | --- | --- |
| `run.py` | 完成 | `parse_experiments`、`run_experiments`、`_run_e1_e4`、`_run_e5`、`_run_e6`。E1–E4 复制聚类公共文件（不含 factor_groups）；E5/E6 另复制因子群 |
| `specs.py` | 完成 | `build_experiment_specs`：E1–E4 的 experiment_id / name / components |
| `routing.py` | 完成 | `rolling_table`、`market_feature_frame`、`performance_tensor`、`target_tensor`、`fit_router`、`route_weights`。route：`equal` / `market` / `fused` / `full` |
| `alpha.py` | 完成 | `build_alpha`：因子值 × 权重 → `date,code,alpha` |

`run.py` 内部约定：

- E1–E4 Alpha 过滤为**代表因子**的 values。
- E5/E6 Alpha 用 `context.factor_values` 全候选。
- E6 内层：每个 cluster 独立 `fit_router(..., use_performance=True)`，seed=`random_state+cluster_id+1`，rolling 特征用全部 `selected_ids` 再按成员切片。
- 产物：E1–E4 → `experiments_*` + `experiment_suite.json`；E5 → `e5_*` + `e5_manifest.json` + `model.pt`；E6 → `e6_*` + `models/outer_router.pt` + `models/inner/cluster_XXX.pt`。

---

## `tests/` 已改到新路径

| 文件 | 覆盖 |
| --- | --- |
| `helpers.py` | 合成快照/values/state，给 step3/step4 测 |
| `test_cli.py` | `--help` 含 step1–4；select/cluster/run 参数解析；validate 无效清单 |
| `test_exporter.py` | 导出六张表；禁止覆盖 |
| `test_io_schema.py` | 行情别名、主键、sha256 |
| `test_expression_engine.py` | 表达式与 `materialize_factor_values` |
| `test_state_cli.py` | `run_state_analysis` 写表 |
| `test_conditional_run.py` | 时间切分、类别、条件评价、snapshot mismatch |
| `test_selection.py` | 类别打分、选择、run_selection |
| `test_research_metrics.py` | 未来收益、状态特征、条件 IC |
| `test_behavior.py` | 日度绩效、滚动特征只用成熟标签 |
| `test_clustering.py` | K-Means 可复现、代表因子最近中心 |
| `test_models.py` | Gumbel Top-k、FactorRouter 两路 + alpha |
| `test_portfolio.py` | 次日执行、成本、停牌冻结 |
| `test_step3_cluster.py` | `run_cluster` 产物 |
| `test_step4_e1_e4.py` | specs 四级组件；套件写出 metrics，E1 无 model.pt、E2 有 |
| `test_step4_e5.py` | 选群等权；E5 产物与群内 0.5 |
| `test_step4_e6.py` | 内层只激活外层选中群；E6 双路由器文件 |
| `test_old_modules_removed.py` | 旧包 ModuleNotFoundError |

验收：`python -m pytest research_pipeline/tests -q`

---

## 已删除、禁止再引入

不要重建这些路径或兼容层：

```text
research_pipeline/research_system/
research_pipeline/MoE/
research_pipeline/factor_moe/
research_pipeline/tests/test_compatibility_contracts.py
research_pipeline/tests/test_experiments.py   （已由 test_step4_e1_e4.py 替代）
step4_experiments/context.py                  （已并入 common/context.py）
```

旧命令不要写回文档或 CLI：

```text
python -m research_pipeline.research_system ...
python -m research_pipeline.MoE.run_e5
python -m research_pipeline.MoE.run_e6
```

---

## 改文件时的对应关系（减少误改）

| 用户要改的行为 | 应打开的文件 |
| --- | --- |
| 命令行参数 / 帮助 | `cli.py` |
| 快照格式、校验 | `common/schema.py`、`step1_snapshot/` |
| 表达式算子 | `common/expression_engine.py` |
| 市场状态规则 | `common/features.py`、`step2_features/states.py` |
| 聚类、代表因子 | `step3_clustering/clustering.py`、`cluster.py` |
| 选群 / 群内权重 | `step3_clustering/grouping.py` |
| 路由器训练与推理 | `step4_experiments/routing.py`、`common/models.py` |
| E1–E6 产物与流程 | `step4_experiments/run.py` |
| 回测指标 | `common/portfolio.py` |
| 训练期画像为空导致 cluster 失败 | `common/context.py` 的 `behavior_frame`、`common/behavior.py` 的成熟日 |
