# torchtitan-npu skill 集合

该目录包含 `torchtitan-npu` 的 skill 定义。基于昇腾NPU的训练调试经验总结，支持Agent快速定位和解决问题。

## skill列表

| Skill | 说明 |
| --- | --- |
| `accuracy-debug` | 有基线对照的训练精度异常定位（loss 偏离、NaN/Inf），基于代码审查 + detect_anomaly + msprobe dump/compare 流程 |
| `oom-analysis` | NPU 训练 OOM 问题诊断，按日志分类 → 静态内存估算 → Memory Snapshot 深度分析 → 优化建议的流程定位和解决问题 |
| `torchtitan-sync` | 上游 torchtitan 分支同步与适配，读取 versioning_policy.md 分支同步表，生成变更分析并完成代码适配 |

## 快速开始

1. 在仓库根目录执行：

   ```bash
   bash .agents/setup_agent.sh
   ```

2. 在 `torchtitan-npu` 开发环境中发起会话。
3. 调用 skill（例如 `/skills` 选择对应 skill，或通过关键词自动触发）。
