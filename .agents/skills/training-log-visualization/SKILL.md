---
name: training-log-visualization
description: 当用户提到训练日志作图、loss/grad_norm 曲线、两份日志对比、误差曲线，或需要从 torchtitan/torchtitan-npu 的日志按 step 提取并可视化指标（含 memory/tps/tflops/mfu/elapsed_time_per_step）时，优先使用本技能；即使用户只说“画日志曲线”“对比两份日志”也应触发。
---

# training-log-visualization 技能

用于从训练 stdout 日志中解析指标并绘制可视化曲线。

## 适用场景

- 用户希望从日志文件绘制 `loss`、`grad_norm` 曲线。
- 用户希望在同一张图中对比两份日志（正常 vs 异常）。
- 用户希望追加 `memory`、`tps`、`tflops`、`mfu`、`elapsed_time_per_step` 曲线。
- 用户希望在双日志对比中查看 `loss` 的绝对误差与相对误差曲线。

## 所需输入

- 主日志路径（必需）
- 对比日志路径（可选）
- 可选指标列表（可空）
- 输出图片路径（可选）

## 执行流程

### Step 1：与用户交互确认输入

按顺序询问：

1. 主日志路径（必填）
2. 是否需要第二份日志做对比（可选）
3. 是否追加可选指标（可选：`memory`(等价 `memory_gib`)、`memory_pct`、`tps`、`tflops`、`mfu`(等价 `mfu_pct`)、`elapsed_time_per_step`）
4. 输出路径（可选，不填则自动命名）

### Step 2：调用绘图脚本

脚本路径：

- `.agents/skills/training-log-visualization/scripts/plot_training_logs.py`

单日志示例：

```bash
python .agents/skills/training-log-visualization/scripts/plot_training_logs.py \
  --log-a /path/to/train.log \
  --metrics memory,tps \
  --output /tmp/train_single.png \
  --no-show
```

> 说明：`memory` 会映射到 `memory_gib`。

双日志示例：

```bash
python .agents/skills/training-log-visualization/scripts/plot_training_logs.py \
  --log-a /path/to/baseline.log \
  --log-b /path/to/problem.log \
  --metrics memory,tps \
  --baseline b \
  --output /tmp/train_compare.png \
  --no-show
```

> 说明：双日志没有共同 step 时，脚本会报错退出，避免生成误导性对比图。

### Step 3：绘制完成后询问是否生成 PR 贴图低分辨率图

- 首次绘图完成后，**必须先询问用户**是否需要额外生成一张 `1024x768` 的低分辨率 PNG 用于 PR 贴图。
- 不得默认生成低分辨率图。
- 用户确认需要后，再次调用脚本并附加参数：
  - `--generate-pr-image`
  - 可选：`--pr-image-output /path/to/pr_image.png`
- 若未提供 `--pr-image-output`，默认输出为：`<主输出文件名>_pr_1024x768.png`。

低分辨率图示例（在原命令基础上追加）：

```bash
python .agents/skills/training-log-visualization/scripts/plot_training_logs.py \
  --log-a /path/to/train.log \
  --metrics memory,tps \
  --output /tmp/train_single.png \
  --generate-pr-image \
  --pr-image-output /tmp/train_single_pr.png \
  --no-show
```

### Step 4：返回结果

输出必须包含：

- 主图路径
- （若用户要求）PR 低分辨率图路径
- 解析到的 step 范围和关键摘要
- 对齐告警（例如双日志 step 不一致）
- 指标缺失告警（若某些可选指标不存在）

## 输出约束

- 所有曲线横轴统一为 `step`。
- 无论单日志还是双日志，`loss` 与 `grad_norm` 都必须绘制。
- 双日志模式下必须额外绘制：
  - `loss abs error`
  - `loss rel error`
