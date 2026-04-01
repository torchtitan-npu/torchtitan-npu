---
name: torchtitan-sync
description: "当需要将 torchtitan-npu 与上游 torchtitan 指定分支进行同步（sync/adapt/rebase）时使用：读取 docs/source/community/versioning_policy.md 的分支同步表，对比旧基线与目标提交，生成 torchtitan_changes.md，分级展示 torchtitan 主要改动点，完成 torchtitan_npu 适配，并在结束后提醒更新 versioning_policy.md。"
---

# torchtitan-sync 技能

该技能用于将上游 `torchtitan` 变更按分支映射同步到 `torchtitan-npu`。

## 适用场景

- 用户要求将 `torchtitan-npu` 与上游 `torchtitan` 做同步、适配、rebase 或对齐。
- 用户要求评估上游某个提交区间对 `torchtitan-npu` 的影响。

## 不适用场景

- 与上游同步无关的单点 bug 修复。
- 仅文档改动。
- 不需要上游提交对比的本地重构。

## 必须依赖的事实源

首先读取 [docs/source/community/versioning_policy.md](../../../docs/source/community/versioning_policy.md)。

使用其中的 `分支同步表` 表。每个 `torchtitan-npu` 分支至少需要以下字段：
- `torchtitan_npu_branch`
- `torchtitan_repo_url`
- `torchtitan_branch`
- `torchtitan_commit`（当前已同步基线）

如果当前分支在表中没有对应行，或关键字段为空，先停止并要求用户补全
`versioning_policy.md` 后再继续。

## 工作流

### 1. 解析分支映射

```bash
current_branch="$(git rev-parse --abbrev-ref HEAD)"
```

- 在 `docs/source/community/versioning_policy.md` 的 `分支同步表` 中匹配 `current_branch`。
- 读取 `torchtitan_repo_url`、`torchtitan_branch`、`torchtitan_commit`。
- 校验：
  - `<torchtitan_repo_url>` 可访问：
    - `git ls-remote --exit-code <torchtitan_repo_url> HEAD`
  - 远端存在 `<torchtitan_branch>`：
    - `git ls-remote --exit-code <torchtitan_repo_url> <torchtitan_branch>`
  - 若无法确认 old commit 是否存在，后续通过临时拉取验证。

### 2. 解析目标上游提交

默认目标（来自远端）：

```bash
git ls-remote <torchtitan_repo_url> <torchtitan_branch>
```

- 如果用户明确指定了目标 commit/tag，则优先使用用户指定目标。
- 若 `target_commit == torchtitan_commit`，直接报告“已是最新”并结束。

### 3. 准备临时本地镜像用于差异分析

由于默认不依赖本地上游仓，先创建临时镜像：

```bash
tmp_repo="$(mktemp -d /tmp/torchtitan-sync-XXXXXX)"
git clone --filter=blob:none --no-checkout <torchtitan_repo_url> "$tmp_repo"
```

然后确保 old/new commit 在本地可分析：

```bash
git -C "$tmp_repo" fetch origin <torchtitan_branch> --tags
```

### 4. 对比上游变更

在临时仓执行：

```bash
git -C "$tmp_repo" log --oneline <old_commit>..<new_commit>
git -C "$tmp_repo" diff --name-status <old_commit> <new_commit>
git -C "$tmp_repo" diff --name-only <old_commit> <new_commit> -- torchtitan/
```

分析完成后清理临时仓：

```bash
rm -rf "$tmp_repo"
```

优先关注 `torchtitan/` 下的源码变更。

### 5. 生成 `torchtitan_changes.md`

创建或更新 `torchtitan_changes.md`，至少包含：
- old/new commit 与提交数量
- 优先级分类（`P0`..`P4`）
- 受影响上游文件与对应 `torchtitan-npu` 文件
- 重命名/迁移文件说明

| 优先级 | 分类 | 说明 |
|----------|----------|-------------|
| **P0** | 破坏性变更 | 不适配会导致运行时错误的 API/行为变更 |
| **P1** | 重要变更 | 影响功能或性能的变更 |
| **P2** | 中等变更 | 需要兼容性评估的变更 |
| **P3** | 模型变更 | 新模型或模型更新 |
| **P4** | 次要变更 | 配置、文档或轻量重构 |

### 6. 适配 torchtitan-npu 代码

对每个相关上游变更，评估并修改：
- `torchtitan_npu/converters/`
- `torchtitan_npu/patches/`
- `torchtitan_npu/models/`
- `torchtitan_npu/distributed/`
- `torchtitan_npu/config/`
- `torchtitan_npu/train.py`
- `torchtitan_npu/entry.py`

重点检查：
- ModelConverter 注册是否仍匹配上游接口。
- patch 目标在上游重构后是否仍存在。
- NPU 特有优化是否保持有效。
- 依赖约束是否仍兼容。

### 7. 验证

按改动范围执行对应验证：
- 受影响路径的针对性测试
- 可行时执行训练入口 smoke 测试
- 若涉及并行模块，补充分布式/并行验证

### 8. 最终交付与强制提醒

同步任务结束时，必须提醒用户更新
`docs/source/community/versioning_policy.md`（`分支同步表`）：
- 将 `torchtitan_commit` 更新为本次同步目标 commit
- 若同步目标分支/tag 变化，更新 `torchtitan_branch`
- 更新 `updated_at`
- 更新 `notes`（如提交区间、关键风险、遗留事项）

最终总结必须包含：
- 上游 old -> new commit
- 本次修改的 `torchtitan-npu` 文件
- 已执行测试/检查及结果

## 关键路径

| 项目 | 路径 |
|---------|------|
| 分支映射事实源 | `docs/source/community/versioning_policy.md` |
| torchtitan-npu 源码 | `torchtitan_npu/` |
| ModelConverter 插件 | `torchtitan_npu/converters/` |
| Patch 实现 | `torchtitan_npu/patches/` |
| NPU 模型实现 | `torchtitan_npu/models/` |
| 分布式实现 | `torchtitan_npu/distributed/` |
| NPU 配置 | `torchtitan_npu/config/` |
| 训练入口 | `torchtitan_npu/entry.py` |
| 主训练脚本 | `torchtitan_npu/train.py` |

## 参考

- [torchtitan Documentation](https://github.com/pytorch/torchtitan)
- [torchtitan-npu README](../../../README.md)
