# 任务状态与接手约束

仓库：`D:\MoE\research_factor_system_20260904`  
GitHub：`https://github.com/dddong2-star/research_factor_system_20260904.git`  
系统：Windows；注释中文 UTF-8；不主动 commit。

## 已完成（不要当成未做）

1. **按 step1–step4 彻底重构**：删除 `research_pipeline/research_system`、`MoE`、`factor_moe` 及旧兼容测试。
2. **唯一 CLI**：`research_pipeline/cli.py` + `__main__.py`，子命令为 `step1 export/validate`、`step2 states/values/conditional/select`、`step3 cluster`、`step4 run`。
3. **step3 只聚类一次**：写出 `cluster_*`，E1–E6 都读 `--cluster-run`，不再各自重算 K-Means。
4. **E1–E6 编排在** `step4_experiments/run.py`：E1–E4 共用 `experiments_*`；E5、E6 各自 `e5_*` / `e6_*`。
5. **测试改到新导入路径**：`research_pipeline/tests/`，含旧模块无法再导入的检查。
6. **用户文档已按新命令重写**：根 `README.md` 与 `docs/`。

未做、也不要擅自做的：新实验、修 E4 换手实现、给 `select` 真正做相关剔除、改标签成熟口径、加 KeepTopK / 可训练噪声（曾加过又按用户要求回滚）。

## 硬约束（用户反复确认）

- 不确定先问；**保留原有功能，不新增功能**。
- 改函数时先读原逻辑，在原基础上改，不要删掉原有行为。
- 已正确的功能尽量不要改。
- `step2 conditional/select` 是可选支线，**不进入** step3/step4。
- 不要提交 `research_data/`、`research_runs/`（`.gitignore` 已忽略）。

## 流水线数据流

```text
step1 export/validate
        ↓
step2 states + step2 values          ← 主线特征
        ↓
step3 cluster                        ← 一份聚类，E1–E6 共用
        ↓
step4 run --cluster-run ...          ← E1–E6

step2 conditional → step2 select     ← 旁路
```

步骤 4 **不再接受** `--data-root` / `--values-run` / `--state-run` / `--test-start` / `--horizon` / `--n-clusters`。这些写在 `cluster_manifest.json` 里。

## 默认值（沿用原实现）

| 参数 | 默认 |
| --- | --- |
| horizon | 20 |
| n_clusters | 20（实际 `min(n, 因子数)`） |
| top_n | 20（step4 回测）；select 里 top_n=10 |
| fee_rate / slippage_rate | 0.001 |
| rolling_window | 40 |
| epochs | 10 |
| random_state | 42 |
| test_start 省略时 | 行情日期约 80% 处 |

## E1–E6 行为契约（算法不要「改进」）

| 实验 | 已完成内容 |
| --- | --- |
| E1 | 代表因子每日等权；无 `FactorRouter`；无 `model.pt`；Alpha **只用代表因子值** |
| E2 | 仅 4 维市场特征；绩效输入置零；Softmax 稠密权重；有 `model.pt` |
| E3 | 市场 + 9 维滚动绩效；可学习融合 `sigmoid(alpha_logit)` |
| E4 | 在 E3 上 Gumbel Top-k（温度 0.5）+ 换手惩罚系数 0.01；`--factor-top-k` 省略=代表因子数（不稀疏） |
| E5 | 外层同 E2；每日硬 Top-1 选群；群内等权；Alpha 用**全因子**；`e5_manifest.json` |
| E6 | 外层硬 Top-1；每簇独立 fused 内层（seed=`random_state+cluster_id+1`）；内层用**全部训练日**；`1+K` 个模型 |

E4 换手是已知简化：各日权重和一个共享 `previous` 比，不是相邻日递推。文档在 `docs/reference/limitations.md`。不要修成「正确换手」除非用户明确要求。

## 本地已有数据（Git 不跟踪）

快照：`research_data/snapshots/20260831T074757Z_7a98fc5698`（`current.txt` 指向它）  
行情约 `2021-06-18`～`2026-06-16`，1202 个交易日。

| 用途 | 目录 |
| --- | --- |
| 因子值（20 个全成功） | `research_runs/values_20260905T033210Z_96fc830c` |
| 市场状态 | `research_runs/state_20260905T033257Z_043b23e6` |
| 聚类（test_start=2025-01-01，5 簇） | `research_runs/cluster_20260905T162602Z_f60cac72` |

正式对照常用：`--test-start 2025-01-01 --n-clusters 5`，step4 `--factor-top-k 3`。

## 已知会踩的坑

### 1. `ValueError: each factor needs at least one finite behavior feature`

出现在 `step3_clustering/clustering.py` 的 `_feature_matrix`。  
画像只统计 `date < test_start` 且 `label_available_date < test_start`。`label_available_date = date + horizon 个日历日`（不是交易日）。`test_start` 太早、horizon 太大、或某个因子训练期 IC 全 NaN（截面样本不足 / 因子值截面常数），**只要一个因子整行空就会失败**。

同一套本地数据用 `2025-01-01` 可以成功；不要把 test_start 设到样本开头附近。

### 2. `step2 select --correlation-threshold`

CLI 没有把因子值传入 `run_selection`，相关剔除不会真正执行，只记进清单。保持原行为。

### 3. 旧包不得再出现

测试 `test_old_modules_removed.py` 断言这些无法导入：

- `research_pipeline.research_system`
- `research_pipeline.MoE`
- `research_pipeline.factor_moe`

## 验收命令

```powershell
python -m research_pipeline --help
python -m pytest research_pipeline/tests -q
```

复跑实验（已有 values/state 时跳过 step2）：

```powershell
python -m research_pipeline step3 cluster `
  --data-root .\research_data `
  --values-run .\research_runs\values_20260905T033210Z_96fc830c `
  --state-run .\research_runs\state_20260905T033257Z_043b23e6 `
  --output-root .\research_runs `
  --snapshot-id 20260831T074757Z_7a98fc5698 `
  --test-start 2025-01-01 `
  --n-clusters 5

python -m research_pipeline step4 run `
  --cluster-run .\research_runs\cluster_<新目录> `
  --output-root .\research_runs `
  --experiments E1,E2,E3,E4,E5,E6 `
  --factor-top-k 3
```

若聚类已有且未改 values/state，可直接 step4，`--cluster-run .\research_runs\cluster_20260905T162602Z_f60cac72`。

## 用户文档 vs 本交接文档

| 给谁 | 路径 |
| --- | --- |
| 使用系统的人 | `docs/WORKFLOW.md`、`docs/COMMANDS.md` |
| 下一个写代码的 agent | `docs/handoff/`（本目录） |
