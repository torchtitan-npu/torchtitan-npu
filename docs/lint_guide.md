# Lint 指南

torchtitan-npu 基于 pre-commit 执行代码样式及质量检查。提交 PR 前，需要确保以下命令运行通过。

```bash
pre-commit run --all-files
```

为了正确运行 pre-commit 检查，需要安装开发依赖：

```bash
pip install -r requirements.txt -r requirements_dev.txt
# 下载依赖并运行检查
pre-commit run --all-files
```

推荐安装 git 钩子，在每次执行commit时自动对增量文件执行检查。

```bash
pre-commit install
```

各项检查的详细配置参见 [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) 和 [`pyproject.toml`](../pyproject.toml)。
