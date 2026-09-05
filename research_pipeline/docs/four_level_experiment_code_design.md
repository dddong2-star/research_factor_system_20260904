# 四级因子动态组合实验代码设计说明

## 1. 文档目的

本文档介绍 `research_pipeline` 独立研究系统中“四级实验”代码的设计、数据流、模块职责、核心算法、运行方式、产物结构和当前限制。

这套代码服务于研究阶段，目标是验证以下问题：

1. 通过因子行为相似性进行 K-Means 聚类，并用每个簇的代表因子压缩候选池，是否可以建立稳定的基准组合。
2. 加入市场状态信息后，动态选择因子是否优于静态等权组合。
3. 加入严格滞后的因子历史绩效后，绩效路由是否能进一步改善样本外表现。
4. 加入 Gumbel Top-k 和换手惩罚后，模型是否能在收益、换手和交易成本之间取得更好的平衡。

代码采用逐级增加组件的实验设计。E1、E2、E3、E4 使用同一快照、同一测试起点、同一代表因子池和同一成本模型，便于进行消融比较。

## 2. 总体架构

系统分为五层：

```text
独立研究快照
    |
    +-- 行情 market.parquet
    +-- 因子值 factor_id=<id>/values.parquet
    +-- 市场状态 market_states.parquet
    |
    v
数据与标签层
    +-- 未来收益 target
    +-- 日度 IC / Rank IC / slope
    +-- 严格成熟日期 label_available_date
    |
    v
候选池压缩层
    +-- 训练期行为特征
    +-- 标准化 K-Means
    +-- 每个簇选择代表因子
    |
    v
动态路由层
    +-- E1 等权
    +-- E2 市场路由 MACoE
    +-- E3 市场路由 + 绩效路由 PACoE
    +-- E4 融合 + Gumbel Top-k + 换手惩罚
    |
    v
组合与评估层
    +-- 因子截面标准化
    +-- 股票 Alpha 合成
    +-- 下一交易日执行
    +-- 停牌冻结与成本扣除
    +-- 测试期净值和指标
```

代码入口是：

```text
research_pipeline/research_system/cli.py
```

核心实验运行器是：

```text
research_pipeline/research_system/experiments.py
```

## 3. 目录结构

```text
research_pipeline/
  config.example.yaml
  config.local.yaml
  requirements.txt
  docs/
    four_level_experiment_code_design.md
  research_system/
    behavior.py
    classification.py
    cli.py
    clustering.py
    conditional_run.py
    evaluation.py
    experiments.py
    exporter.py
    expression_engine.py
    factor_values.py
    features.py
    io.py
    models.py
    portfolio.py
    research_run.py
    schema.py
    selection.py
  tests/
    test_behavior.py
    test_clustering.py
    test_models.py
    test_portfolio.py
    test_experiments.py
```

`research_pipeline` 与原有 `custom_backtester`、`single_factor` 运行时分离。原系统只负责导出数据快照，四级实验代码只读取独立研究目录中的输入文件。

## 4. 输入数据设计

### 4.1 行情数据

行情数据位于独立快照中：

```text
research_data/snapshots/<snapshot_id>/market.parquet
```

最少需要以下字段：

```text
date
code
open
close
high
low
volume
amount
```

回测和停牌判断还可以使用：

```text
Ifsuspend
if_suspend
suspended
```

行情表的主键是 `(date, code)`。快照通过 `snapshot_id` 绑定，实验运行时会检查输入的值运行和状态运行是否属于同一快照。

### 4.2 因子值

因子值采用分区文件保存：

```text
research_runs/<values_run>/factor_values/
  factor_id=<factor_id>/values.parquet
```

每个分区包含：

```text
date
code
factor_id
value
```

采用“一个因子一个分区”的原因是避免把 1,264 个因子的全部股票级数据一次性加载到内存。实验运行器可以使用 `--max-factors` 或多个 `--factor-id` 限制候选因子范围。

### 4.3 市场状态

市场状态运行位于：

```text
research_runs/<state_run>/market_states.parquet
```

当前状态特征包括：

```text
market_return
market_volatility
up_ratio
cross_section_dispersion
state_id
```

动态路由使用前四个数值特征。`state_id` 主要用于解释和后续条件分析。

## 5. 行为特征与标签

### 5.1 未来收益标签

未来收益由 `features.make_forward_return` 生成。对每只股票，给定信号日 `t` 和持有期 `H`：

```text
target(t) = close(t + H) / close(t) - 1
```

代码在每只股票内部进行 shift，因此不会把不同股票的时间序列混在一起。最后 `H` 个交易日没有成熟标签，会保留为缺失值。

### 5.2 日度因子表现

`behavior.py` 中的：

```python
build_daily_factor_performance(values, target, horizon=20)
```

对每个 `(date, factor_id)` 计算：

| 字段 | 含义 |
|---|---|
| `ic` | 因子值与未来收益的 Pearson 截面相关系数 |
| `rank_ic` | 因子值排名与未来收益排名的相关系数 |
| `slope` | 截面回归斜率，因子值作为自变量、未来收益作为因变量 |
| `sample_count` | 当日同时具有有效因子值和有效标签的股票数 |
| `label_available_date` | 该日未来收益标签成熟的日期 |

其中回归斜率使用最小二乘闭式解：

```text
slope = sum((x - mean(x)) * (y - mean(y))) / sum((x - mean(x))^2)
```

有效样本不足或截面没有变化时，相关系数和斜率会保留为缺失值，而不是填充成零。

### 5.3 PACoE 的九维历史绩效特征

```python
build_rolling_performance_features(performance, as_of_date, window=40)
```

对每个因子，从历史表现中取最近 `window` 条已经成熟的记录，分别计算 IC、Rank IC 和 slope 的：

```text
均值
标准差
信息比率 IR = 均值 / 标准差
```

因此得到 9 个数值特征：

```text
ic_mean, ic_std, ic_ir
rank_ic_mean, rank_ic_std, rank_ic_ir
slope_mean, slope_std, slope_ir
```

严格可用条件是：

```text
performance.date < as_of_date
label_available_date < as_of_date
```

这里使用严格小于号，避免在标签刚好成熟的日期同日使用该标签。这个边界是防止 PACoE 未来数据泄漏的关键。

## 6. K-Means 候选池压缩

### 6.1 为什么先压缩

原始 accepted 因子数量为 1,264 个。如果直接构造股票级、因子级、时间级的动态注意力张量，内存和训练时间都会快速增长。因此代码先在因子层面进行行为聚类，再把完整模型限制在每个簇的代表因子上。

### 6.2 实现模块

```text
research_system/clustering.py
```

核心函数：

```python
fit_behavior_kmeans(feature_frame, n_clusters, random_state)
select_representatives(labels, features, centers, per_cluster=1)
```

处理过程：

1. 从行为表中提取数值列。
2. 只在传入的训练期行为矩阵上计算中位数填补和标准化参数。
3. 使用 scikit-learn `KMeans`，固定 `random_state`，`n_init=20`。
4. 根据簇标签和中心计算因子到中心的距离。
5. 每个簇选择距离中心最近的因子作为代表。

当前 `experiments.py` 会把成熟标签限制在 `date < test_start` 后再构造行为摘要，因此聚类拟合不会读取测试期标签。

### 6.3 聚类产物

实验运行根目录会保存：

```text
behavior_features.parquet
factor_clusters.parquet
clustering.json
```

其中 `clustering.json` 记录簇数、特征列、中心、惯性、随机种子、训练结束日期和代表因子列表。

## 7. 动态路由模型

### 7.1 模型模块

```text
research_system/models.py
```

核心类：

```python
FactorRouter(
    market_dim,
    performance_dim,
    factor_count,
    hidden_dim,
)
```

模型输出：

```text
market_scores
performance_scores
fused_scores
alpha
```

其中：

```text
alpha = sigmoid(alpha_logit)
fused_scores = alpha * market_scores + (1 - alpha) * performance_scores
```

`alpha_logit` 是可学习参数，经过 sigmoid 后被限制在 0 到 1 之间。

### 7.2 市场路由 MACoE

市场路由使用：

```text
market_return
market_volatility
up_ratio
cross_section_dispersion
```

市场特征先经过一个小型 MLP，再由 `market_head` 输出每个代表因子的分数。`FactorRouter` 的市场编码器支持 `[batch, time, market_dim]` 输入，并且包含时间注意力层；这为后续接入过去 40 日市场状态序列保留了接口。

当前 `run_experiment_suite` 的第一版使用每日 4 维市场特征 `[batch, market_dim]`，尚未在运行器中构造并传入 40 日时间窗口。因此当前代码属于简化 MACoE，而不是完整论文形式的时间注意力 MACoE。

### 7.3 绩效路由 PACoE

PACoE 使用上一节生成的 9 维滚动绩效向量。每个代表因子的 9 维输入经过共享 MLP，再由 `performance_head` 输出一个因子分数。

当前实验运行器只把已经成熟的滚动表转换为张量。没有成熟历史特征的日期使用数值零作为模型输入，但这只表示“该日期没有可用历史绩效特征”，不修改原始因子值，也不会把缺失的日度 IC 伪造成有效绩效记录。

### 7.4 训练目标

当前代码使用训练期日度 Rank IC 作为监督目标，令路由分数逼近因子当日 Rank IC：

```text
loss = mean((route_score - rank_ic_target)^2)
```

只在训练期日期计算损失。缺失监督标签不会直接进入有效损失位置；输入张量中的缺失绩效特征才会在张量化时用零表示。

模型训练使用 Adam，学习率为 `0.01`，训练轮数由 CLI 的 `--epochs` 控制，随机种子由 `--random-state` 控制。

## 8. Gumbel Top-k 与换手惩罚

### 8.1 Gumbel Top-k

```python
gumbel_topk_mask(scores, k, temperature, training)
```

训练模式：

1. 给分数添加 Gumbel 噪声。
2. 用 temperature 控制选择的平滑程度。
3. 生成 soft mask。
4. 使用 straight-through 形式返回硬选择结果，但保留可微反向路径。

推理模式：

1. 不添加随机噪声。
2. 对分数执行确定性 `topk`。
3. 返回恰好包含 `k` 个激活位置的 mask。

### 8.2 换手惩罚

```python
turnover_penalty(current_weights, previous_weights)
```

使用当前因子权重与上一时点因子权重的 L1 距离：

```text
turnover_penalty = sum(abs(w_t - w_(t-1)))
```

E4 在训练损失中加入换手惩罚，并在训练阶段启用 Gumbel Top-k。推理阶段通过 `--factor-top-k` 控制最终保留的代表因子数量。

需要注意，当前版本的换手惩罚是一个研究骨架实现：训练过程中以模型输出的相邻权重近似换手，没有实现完整的可交易股票权重级换手约束和专家负载平衡损失。

## 9. 股票 Alpha 与成本回测

### 9.1 Alpha 合成

实验运行器先读取代表因子的股票级值，然后按日期对每个因子做截面标准化：

```text
z_factor(t, i) = (value(t, i) - cross_section_mean(t)) / cross_section_std(t)
```

再按动态因子权重合成股票 Alpha：

```text
alpha(t, i) = sum(w_factor(t, f) * z_factor(t, i, f))
```

缺失因子值不会被直接当作有效零值，合成时只在有效因子上聚合并按有效权重归一。

### 9.2 下一交易日执行

```text
signal_date = t
execution_date = next_trading_date(t)
```

信号日只负责生成股票排名，组合在下一个交易日执行。回测入口为：

```python
backtest_long_only(
    market,
    alpha,
    top_n,
    fee_rate,
    slippage_rate,
)
```

默认选择 Alpha 最高的 `top_n` 只股票，等权持有。

### 9.3 停牌冻结

当股票在当前交易日被标记为停牌时，如果它在上一时点仍有持仓，代码会冻结该持仓，不把它当作正常可交易股票换出。可识别字段包括：

```text
Ifsuspend
if_suspend
suspended
```

当前回测已覆盖停牌冻结，但尚未把涨停、跌停、开盘不可成交等全部交易约束纳入统一成交引擎。

### 9.4 成本模型

换手定义为当前权重和前一时点权重的绝对差之和：

```text
turnover(t) = sum(abs(w_t - w_(t-1)))
```

交易成本为：

```text
cost(t) = turnover(t) * (fee_rate + slippage_rate)
```

净收益为：

```text
net_return(t) = gross_return(t) - cost(t)
```

回测输出：

```text
gross_return
turnover
cost
net_return
equity
weights
holding_count
```

### 9.5 统计指标

当前回测指标包括：

```text
total_return
annual_return
sharpe
max_drawdown
calmar
win_rate
average_turnover
total_cost
annual_cost
number_of_trades
average_holding_count
```

## 10. 四级实验定义

| 实验 | 使用组件 | 目的 |
|---|---|---|
| E1 | K-Means、代表因子、等权、成本回测 | 建立最简单的可解释基准 |
| E2 | E1 + MACoE 市场路由 | 检查市场信息是否带来增益 |
| E3 | E2 + PACoE 绩效路由 | 检查严格滞后绩效信息是否带来增益 |
| E4 | E3 + Gumbel Top-k + 换手惩罚 | 检查稀疏选择和交易成本约束的作用 |

四级实验由：

```python
build_experiment_specs(factor_ids, test_start)
```

生成配置。公共配置包括代表因子列表和测试起点，避免不同实验偷偷使用不同候选池。

### E1

每个日期给代表因子相同权重：

```text
w_f(t) = 1 / number_of_representatives
```

这是判断复杂路由是否有效的基准。

### E2

训练 `FactorRouter`，只使用市场分数：

```text
route_score = market_scores
```

绩效输入被关闭，因此 E2 只测试市场状态路由。

### E3

同时使用市场输入和严格滞后的绩效输入：

```text
route_score = fused_scores
```

融合权重由模型中的可学习 `alpha` 决定。

### E4

在 E3 基础上：

1. 训练阶段添加 Gumbel Top-k mask。
2. 训练损失加入换手惩罚。
3. 推理阶段执行确定性 Top-k。
4. 通过 `factor_top_k` 控制最终稀疏因子权重。

## 11. 主运行流程

`run_experiment_suite` 的实际执行顺序如下：

1. 解析并校验研究快照。
2. 校验 `values_run` 和 `state_run` 的 `snapshot_id`。
3. 读取候选因子分区。
4. 根据行情生成未来收益标签。
5. 逐因子计算日度 IC、Rank IC 和 slope。
6. 只用测试起点之前已经成熟的表现构造训练期行为摘要。
7. 在训练期行为摘要上拟合 K-Means。
8. 按簇中心距离选择代表因子。
9. 读取代表因子股票级值，构造滚动 PACoE 特征。
10. 读取市场状态特征并构造市场输入张量。
11. 运行 E1-E4；E2-E4 训练各自的路由模型。
12. 只取测试起点之后的因子权重生成测试期 Alpha。
13. 使用下一交易日执行规则进行成本回测。
14. 保存每级模型、因子权重、Alpha、每日回测和指标。
15. 写入统一 `experiment_suite.json`。

## 12. 时间边界与防泄漏设计

系统的关键时间约束如下：

| 环节 | 可使用的数据 |
|---|---|
| K-Means 拟合 | 测试起点之前的成熟行为特征 |
| 行为摘要 | 训练期日度表现，不含未成熟标签 |
| PACoE 在日期 t 的输入 | `label_available_date < t` 的历史表现 |
| 路由模型训练 | 训练期市场输入和训练期监督目标 |
| 测试 Alpha | 测试起点及之后的因子值和模型输出 |
| 测试回测 | 测试期信号的下一交易日执行结果 |

代码明确避免以下错误：

- 随机打乱交易日后切分训练和测试。
- 用测试期表现拟合 K-Means。
- 在标签成熟前使用未来收益计算 PACoE。
- 用测试期指标调节模型后再报告同一测试期结果。
- 把原始缺失因子值直接填成零。
- 混用不同 `snapshot_id` 的行情、因子值和状态数据。

## 13. 命令行使用

### 13.1 查看入口

```powershell
python -m research_pipeline.research_system --help
```

### 13.2 运行四级实验

```powershell
python -m research_pipeline.research_system experiment-suite `
  --data-root research_data `
  --values-run research_runs/values_<run_id> `
  --state-run research_runs/state_<run_id> `
  --output-root research_runs `
  --snapshot-id <snapshot_id> `
  --max-factors 50 `
  --test-start 2025-01-01 `
  --horizon 20 `
  --n-clusters 10 `
  --factor-top-k 5 `
  --top-n 20 `
  --fee-rate 0.001 `
  --slippage-rate 0.001 `
  --rolling-window 40 `
  --epochs 10 `
  --random-state 42
```

主要参数：

| 参数 | 作用 |
|---|---|
| `--max-factors` | 限制进入聚类的候选因子数 |
| `--n-clusters` | K-Means 簇数 |
| `--factor-top-k` | E4 路由阶段保留的代表因子数 |
| `--top-n` | 股票组合持仓数量 |
| `--horizon` | 未来收益标签持有期 |
| `--rolling-window` | PACoE 历史绩效窗口 |
| `--epochs` | 路由模型训练轮数 |
| `--fee-rate` | 手续费率 |
| `--slippage-rate` | 滑点率 |
| `--random-state` | K-Means 和 PyTorch 的随机种子 |

研究阶段建议先使用：

```text
max_factors = 20 到 100
n_clusters = 5 到 20
factor_top_k = 3 到 10
epochs = 5 到 30
```

正式实验应在验证期选择这些参数，测试期只运行最终配置。

## 14. 输出产物

一次实验运行目录如下：

```text
research_runs/experiments_<run_id>/
  daily_factor_performance.parquet
  behavior_features.parquet
  factor_clusters.parquet
  clustering.json
  experiment_suite.json
  E1/
    config.json
    factor_weights.parquet
    alpha.parquet
    daily.parquet
    metrics.json
  E2/
    config.json
    model.pt
    factor_weights.parquet
    alpha.parquet
    daily.parquet
    metrics.json
  E3/
    config.json
    model.pt
    factor_weights.parquet
    alpha.parquet
    daily.parquet
    metrics.json
  E4/
    config.json
    model.pt
    factor_weights.parquet
    alpha.parquet
    daily.parquet
    metrics.json
```

### 14.1 公共产物

`daily_factor_performance.parquet` 是逐日逐因子行为表；`behavior_features.parquet` 是训练期聚类输入；`factor_clusters.parquet` 保存每个候选因子的簇标签；`clustering.json` 保存聚类参数和代表因子。

### 14.2 单级产物

- `config.json`：实验组件、快照、时间边界、成本和随机种子。
- `model.pt`：E2-E4 的 PyTorch 模型参数。
- `factor_weights.parquet`：每日代表因子权重。
- `alpha.parquet`：测试期股票 Alpha。
- `daily.parquet`：每日收益、成本、换手、持仓和净值。
- `metrics.json`：汇总评价指标。

## 15. 当前真实烟雾实验

已用真实快照完成小规模烟雾运行：

```text
research_runs/experiments_20260902T102339Z_ccebe07c/
```

配置为：

```text
候选因子：6
代表因子：2
K-Means 簇数：2
factor_top_k：2
训练 epoch：1
快照：20260831T074757Z_7a98fc5698
```

运行结果：

- E1、E2、E3、E4 均成功生成完整产物。
- 输入快照校验通过。
- E2、E3、E4 均生成 `model.pt`。
- E4 测试期每日最多保留 2 个因子。
- E2、E3 的回测结果出现差异。

由于本次实验只有 2 个代表因子，且 `factor_top_k=2`，所以 E4 的 Top-k 没有进一步压缩候选因子，不能据此判断 Gumbel Top-k 的有效性。该运行只证明代码链路可执行，不是模型性能结论。

## 16. 测试与验证

当前测试覆盖：

```text
行为指标和成熟日期
K-Means 可复现性和代表因子选择
下一日执行和交易成本
停牌冻结
Gumbel Top-k 梯度和推理选择数量
FactorRouter 输出形状和可学习 alpha
四级配置和合成端到端流程
CLI 参数解析
```

验证命令：

```powershell
python -m pytest research_pipeline/tests -q
python -m compileall -q research_pipeline
python -m research_pipeline.research_system validate `
  --data-root research_data `
  --snapshot-id <snapshot_id>
```

当前实现验证结果为：

```text
38 passed
compileall: ok
snapshot valid=true
```

测试运行在 Windows 临时目录清理阶段可能出现 pytest 的 `WinError 145` warning，但不影响测试退出码和断言结果。

## 17. 当前实现边界

这份代码是可复现的研究实验骨架，已经打通从因子值到样本外成本回测的主要链路，但仍有以下边界：

1. 当前实验运行器使用日度市场特征；完整 40 日市场时间注意力序列还没有接入主运行流程。
2. 当前没有自动执行验证期超参数搜索，`n_clusters`、`factor_top_k`、窗口和训练轮数需要通过外部实验配置选择。
3. 当前模型是在整个训练期张量上进行简单训练，不是滚动窗口重新拟合的 walk-forward 训练。
4. 当前模型目标主要是拟合 Rank IC，尚未将 IC、Rank IC、回归斜率和组合收益构造成完整多任务损失。
5. 当前 E4 的换手惩罚主要约束因子权重变化，尚未完整约束股票持仓换手。
6. 当前回测支持停牌冻结，但尚未实现完整涨跌停、开盘不可成交、成交量容量和冲击成本模型。
7. 当前 E4 的完整效果需要至少 3 个以上代表因子，并设置 `factor_top_k < representative_factor_count` 才能观察稀疏选择差异。
8. 真实烟雾实验只使用很少候选因子和很少训练轮数，不能用于判断最终收益能力。

## 18. 后续推荐顺序

建议按照以下顺序继续增强：

1. 将过去 40 日市场状态拼成时间序列，接入 `FactorRouter` 的时间注意力输入。
2. 将训练、验证、测试切分封装为统一的 walk-forward 运行器。
3. 增加验证期自动选择簇数、代表因子数量、Top-k、窗口和损失权重。
4. 将股票级换手直接加入训练目标和回测反馈。
5. 增加涨跌停、成交量容量、滑点随成交额变化的交易约束。
6. 对不同随机种子进行重复实验，报告均值、标准差和置信区间。
7. 扩大真实烟雾实验到 20-50 个代表因子，再进行正式测试期评估。

这套设计的核心不是让 E4 一次性变得复杂，而是保证每增加一个模型组件，都能通过 E1 到 E4 的对照实验回答一个明确的研究问题。
