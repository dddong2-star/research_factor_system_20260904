# 统一文档导航

所有命令默认在仓库根目录运行，路径采用 Windows PowerShell 相对路径。

## 从这里开始

1. [WORKFLOW.md](WORKFLOW.md)：从 `step1 export` 到 `step4 run` 的完整顺序。
2. [COMMANDS.md](COMMANDS.md)：唯一入口 `python -m research_pipeline <step> <action>` 的全部子命令。
3. 按需查看各实验的目的、输入、模型与产物：
   - [E1：K-Means 代表因子等权基准](experiments/E1.md)
   - [E2：市场感知路由](experiments/E2.md)
   - [E3：市场与绩效融合路由](experiments/E3.md)
   - [E4：稀疏路由与换手惩罚](experiments/E4.md)
   - [E5：代表因子选群、群内等权](experiments/E5.md)
   - [E6：外层选群、内层动态路由](experiments/E6.md)

## 参考

- [数据契约](reference/data-contracts.md)：快照表、运行输入及关键字段。
- [产物说明](reference/artifacts.md)：各命令实际写出的文件。
- [已知限制](reference/limitations.md)：当前实现边界和使用风险。
- [代码交接（给后续 agent）](handoff/README.md)：已完成范围、文件清单、硬约束。

## 代码与步骤对照

唯一命令：`python -m research_pipeline`

```text
research_pipeline\
  cli.py                   按 step / action 分发
  common\                  跨步骤复用，不含实验编排
    io.py / schema.py
    expression_engine.py
    features.py / behavior.py
    models.py / portfolio.py
    context.py             加载快照、因子值、状态和行为画像
  step1_snapshot\
    export.py / validate.py
  step2_features\
    states.py / values.py
    conditional.py / select.py
    classification.py / evaluation.py
  step3_clustering\
    cluster.py             写出一份共用聚类产物
    clustering.py / grouping.py
  step4_experiments\
    run.py                 读取 cluster_run，运行指定实验
    specs.py / routing.py / alpha.py
  tests\
```

数据流：`step1` → `step2 states/values` → `step3 cluster` → `step4 run`。`step2 conditional/select` 是旁路，不进入步骤 3 和步骤 4。
