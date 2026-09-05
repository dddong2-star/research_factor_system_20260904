# E5：代表因子选组与组内等权

## 1. E5 流程

E5 不修改原来的 E1-E4，单独执行以下流程：

1. 用训练期行为特征执行 K-Means。
2. 保存每个簇的全部因子，离簇中心最近的因子仍作为代表因子。
3. 完全复用 E2 的市场路由逻辑，只为各代表因子计算每日权重。
4. 每日选择权重最高的代表因子，并找到该代表因子所属的完整因子群。
5. 对该因子群中的全部因子等权，其他因子不参与当日 Alpha 合成。
6. 使用原项目的 Alpha 构造和含交易成本回测逻辑生成 E5 结果。

E2 的代表因子权重只负责选组，不会继续作为组内因子的权重。若当日选中的
因子群包含 `m` 个因子，则每个因子的最终权重均为 `1 / m`。

## 2. 运行方式

在项目根目录 `D:\MoE\research_factor_system_20260904` 下运行：

```powershell
python -m research_pipeline.MoE.run_e5 `
  --data-root "D:\MoE\research_factor_system_20260904\research_data" `
  --snapshot-id "20260831T074757Z_7a98fc5698" `
  --values-run "D:\MoE\research_factor_system_20260904\research_runs\values_20260905T033210Z_96fc830c" `
  --state-run "D:\MoE\research_factor_system_20260904\research_runs\state_20260905T033257Z_043b23e6" `
  --output-root "D:\MoE\research_factor_system_20260904\research_runs" `
  --test-start "2025-01-01" `
  --n-clusters 5 `
  --epochs 10
```

请按实际研究区间修改 `--test-start`。也可以用重复的
`--factor-id 因子编号` 限定因子，或用 `--max-factors` 限制因子数量。

运行前必须存在：

- `research_data/snapshots/<snapshot_id>/market.parquet`
- `<values_run>/factor_values/factor_id=<因子编号>/values.parquet`
- `<state_run>/market_states.parquet`

运行时程序会校验快照编号，并在上述数据文件缺失时直接报错。

## 3. 输出文件

每次运行会生成独立的 `e5_<时间>_<哈希>` 目录：

- `factor_groups.parquet`：所有因子的簇、代表因子、中心距离及群大小。
- `clustering.json`：K-Means 中心、特征列和代表因子等聚类信息。
- `factor_groups.json`：全部因子群的汇总清单。
- `factor_groups/cluster_XXX.json`：每个因子群单独保存的完整成员清单。
- `representative_weights.parquet`：E2 为代表因子计算的每日权重。
- `selected_groups.parquet`：每日权重最高的代表因子及被选中的簇。
- `factor_weights.parquet`：被选因子群展开后的组内等权结果。
- `model.pt`：E2 市场路由模型参数。
- `alpha.parquet`：E5 合成的股票 Alpha。
- `daily.parquet`、`metrics.json`：原回测器产生的日度结果与指标。
- `config.json`、`e5_manifest.json`：配置及可审计运行清单。

## 4. 测试

```powershell
python -m pytest research_pipeline/MoE/tests/test_e5_experiment.py -q
```

## 5. E6 双重 MoE

E6 在 E5 外层选群后新增独立的群内动态因子路由，且不会改变 E5。
设计、运行命令和输出说明见 `E6_README.md`。
