# 贡献指南

感谢关注torchtitan-npu！我们欢迎任何形式的贡献，包括但不限于报告问题、提出建议、改进文档、提交代码。为了让您的贡献过程更加顺畅，请仔细阅读以下指南。

## Pull Requests

请注意以下原则，以确保 PR 顺利合入：

1. Fork 仓库，从 `master` 创建你的分支，从 Fork 仓库提交 PR。
2. 若新增/修改代码可能影响现有功能，请补充或更新 UT 测试。详见 [测试文档](./docs/test_guides/test_design_zh.md) 。
3. 确保格式检查通过（`pre-commit run --all-files`）。详见 [Lint 指南](./docs/lint_guide.md) 。
4. 确保同步更新文档（`README.md` / `docs/`）。
5. 确保所有单元测试和冒烟测试通过。
6. 确保 PR 描述中清楚描述改动的动机、方案与验证结果。
7. 确保签署贡献者许可协议（CLA）。如果未签署，请按照 PR 机器人的引导完成签署。

## Issues

如果使用过程中遇到任何问题，或有任何需求，欢迎提 [Issue](https://gitcode.com/cann/torchtitan-npu/issues)。请根据 Issue 模版给出清晰的复现流程，torchtitan-npu 社区会及时响应。

## 贡献原则

提交改动时，请注意包括但不限于以下内容：

- torchtitan-npu 对 torchtitan 原生的训练与分布式能力进行 NPU 适配与优化。对于与 NPU 平台无关的通用功能，请考虑向 torchtitan 贡献。
- 开发者应清晰描述改动对 torchtitan-npu 仓库的影响以及价值。包括以下内容：
    - 若改动不应影响数值精度（例如重构、新增优化、等价替换），在固定随机种子、确定性开关下对比改动前后的 loss 曲线或其他指标，需要确保一致。
    - 若改动可能影响数值精度，请提供端到端训练实验结果，并说明数据集、任务配置、对比基线与观察到的差异。
    - 如改动涉及性能，请在 PR 中提供：
        - 相关量化指标，如：吞吐量（tokens/sec/NPU）、显存占用峰值、MFU等。
        - 测试环境说明，如：NPU 型号、数量等。
- 除必要的文档配图外，不要提交二进制文件。
- 不要提交数据集。如果需要，在文档中提供下载链接与操作指引。


## License

向 torchtitan-npu 贡献的代码将以本仓库根目录的 [`LICENSE`](./LICENSE) 所描述的许可协议进行许可。
