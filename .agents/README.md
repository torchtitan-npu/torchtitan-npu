# torchtitan-npu skill集合

该目录包含 `torchtitan-npu` 的skill定义。


## 目录

- [torchtitan-sync skill](#torchtitan-sync-skill)

## torchtitan-sync skill

将上游 `torchtitan` 分支变更同步到 `torchtitan-npu`，并确保 Ascend NPU 侧兼容性。

### 快速开始

1. 在仓库根目录执行：
   ```bash
   bash .agents/setup_agent.sh
   ```
2. 在 `torchtitan-npu` 开发环境中发起会话。
3. 调用skill（例如 `/skills` 选择torchtitan-sync，或者关键词自动触发）。
