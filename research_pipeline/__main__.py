"""允许 `python -m research_pipeline` 调用唯一入口。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
