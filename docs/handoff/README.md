# Agent 交接：从这里开始

本文档给**下一个负责本仓库代码的 agent**。用户文档在 `docs/`，本目录只讲「代码做到哪了、每个文件干什么、接手时不要破坏什么」。

| 文件 | 读它做什么 |
| --- | --- |
| [STATUS.md](STATUS.md) | 任务完成度、硬约束、本地数据、已知行为（含缺陷）、怎么验收 |
| [FILE_INVENTORY.md](FILE_INVENTORY.md) | 每个 `.py` 的职责、关键函数、是否已完成 |

用户向文档（命令、实验、契约）仍以这些为准，不要另起旧入口说明：

- [../WORKFLOW.md](../WORKFLOW.md)
- [../COMMANDS.md](../COMMANDS.md)
- [../experiments/E1.md](../experiments/E1.md)–[E6.md](../experiments/E6.md)
- [../reference/limitations.md](../reference/limitations.md)

## 一句话状态

**step1–step4 彻底重构已完成。** 只保留一套实现：`common` + `step1_snapshot` + `step2_features` + `step3_clustering` + `step4_experiments`。旧包 `research_system` / `MoE` / `factor_moe` 已删除。唯一入口：

```powershell
python -m research_pipeline <step> <action>
```

算法、默认值、产物字段与重构前 E1–E6 一致，**不新增能力**。pytest 全量曾通过（约 49 passed）。

## 接手后先做

1. 读完本目录两篇 STATUS / FILE_INVENTORY。
2. 不要恢复旧包，不要改已正确的实验算法「修成新功能」。
3. 改代码前先跑：`python -m pytest research_pipeline/tests -q`
4. 注释用中文 UTF-8；环境是 Windows；不主动 git commit，除非用户明确要求。
