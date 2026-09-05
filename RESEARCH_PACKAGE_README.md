# 独立因子研究系统交付说明

## 包含内容

- `research_pipeline/`：研究代码、测试、依赖和设计文档。
- `research_data/`：当前只读研究快照，包含 500 只股票行情、股票池、1,264 个因子目录及历史指标。
- `research_runs/values_20260831T074840Z_ff40c015/`：1,264 个逐因子值。
- `research_runs/state_20260831T081146Z_043b23e6/`：市场状态。
- `research_runs/conditional_20260831T083739Z_4c33fe73/`：条件 IC 评价。
- `research_runs/experiments_20260902T102339Z_ccebe07c/`：最近一次完整四级实验示例。

所有数据均绑定快照：

```text
20260831T074757Z_7a98fc5698
```

## 环境安装

推荐使用 Python 3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r research_pipeline\requirements.txt
```

## 数据校验

在解压后的根目录执行：

```powershell
python -m research_pipeline.research_system validate `
  --data-root research_data `
  --snapshot-id 20260831T074757Z_7a98fc5698
```

预期返回：

```text
"valid": true
```

## 运行四级实验

```powershell
python -m research_pipeline.research_system experiment-suite `
  --data-root research_data `
  --values-run research_runs\values_20260831T074840Z_ff40c015 `
  --state-run research_runs\state_20260831T081146Z_043b23e6 `
  --output-root research_runs `
  --snapshot-id 20260831T074757Z_7a98fc5698 `
  --max-factors 50 `
  --test-start 2025-01-01 `
  --horizon 20 `
  --n-clusters 10 `
  --factor-top-k 5 `
  --top-n 20 `
  --epochs 10 `
  --random-state 42
```

## 运行测试

```powershell
python -m pytest research_pipeline\tests -q
python -m compileall -q research_pipeline
```

## 注意事项

- 本包是独立研究系统，不包含原网页系统 `custom_backtester/` 和原因子系统 `single_factor/`。
- 现有快照和逐因子值可以直接用于研究，不需要连接原系统。
- `config.example.yaml` 仅用于从原系统重新导出快照；直接运行现有研究数据时无需修改它。
- 正式研究应使用训练期拟合、验证期调参、测试期一次性评估，不应根据测试结果反复修改参数。
