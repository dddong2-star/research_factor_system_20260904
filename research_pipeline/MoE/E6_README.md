# E6：双重 MoE 动态因子路由

## 1. 功能定义

E6 保留 E5 的 K-Means 因子群和第一级选群逻辑，但取消所选群内的等权分配，
改为第二级动态因子路由。

完整流程如下：

1. 在训练集上构造因子行为特征并执行 K-Means。
2. 保存每个簇的全部成员，离中心最近的因子作为代表因子。
3. 第一级路由沿用 E5/E2，只使用市场状态为代表因子分配权重。
4. 每个交易日选择权重最高的代表因子，即硬 Top-1 选择一个因子群。
5. 每个因子群拥有一个独立的第二级路由器。
6. 第二级路由同时输入市场状态和群内各因子的滚动绩效特征。
7. 第二级路由对群内全部因子执行 Softmax，得到每日动态权重。
8. 只启用第一级选中群的第二级权重，合成 Alpha 并执行原有成本回测。

因此 E6 一共训练 `1 + K` 个模型，其中 `K` 为实际聚类数量：

- 1 个外层代表因子路由器；
- K 个独立的群内因子路由器。

## 2. 两级路由关系

第一级输出代表因子权重 `g_t`，并选择：

```text
selected_group_t = argmax(g_t)
```

第二级路由器为选中群内各因子输出权重 `q_t`：

```text
q_t = Softmax(FactorRouter(市场状态, 因子滚动绩效))
```

最终只有所选群中的因子权重非零，且该群全部因子的权重和为 1。

每个内层路由器都使用全部训练日期独立训练，而不是只使用外层曾经选中该群的
日期。这样可以避免某些群因外层早期选择次数过少而缺少训练样本。

## 3. 运行命令

在项目根目录 `D:\MoE\research_factor_system_20260904` 下运行：

```powershell
python -m research_pipeline.MoE.run_e6 `
  --data-root ".\research_data" `
  --snapshot-id "20260831T074757Z_7a98fc5698" `
  --values-run ".\research_runs\values_20260905T033210Z_96fc830c" `
  --state-run ".\research_runs\state_20260905T033257Z_043b23e6" `
  --output-root ".\research_runs" `
  --test-start "2025-01-01" `
  --n-clusters 5 `
  --epochs 10
```

## 4. 主要输出

每次运行生成独立的 `e6_<时间>_<哈希>` 目录：

- `factor_groups.parquet`：完整因子群、中心距离和代表因子。
- `outer_representative_weights.parquet`：第一级代表因子权重。
- `selected_groups.parquet`：每日硬 Top-1 选中的因子群。
- `inner_factor_weights.parquet`：所有群、所有日期的内层动态权重。
- `factor_weights.parquet`：按每日选群结果激活后的最终因子权重。
- `models/outer_router.pt`：第一级路由模型。
- `models/inner/cluster_XXX.pt`：每个因子群独立的第二级路由模型。
- `inner_router_configs.json`：每个内层模型的成员、随机种子及融合系数。
- `alpha.parquet`：E6 最终股票 Alpha。
- `daily.parquet`、`metrics.json`：回测日度结果和评价指标。
- `config.json`、`e6_manifest.json`：配置及可审计运行清单。

E5 不会被 E6 覆盖，仍可通过 `python -m research_pipeline.MoE.run_e5` 单独运行。
