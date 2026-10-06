# 模型适配学习指南

[项目首页](../README.zh-CN.md) · [安装与启动](getting-started.md) · [API](api.md)

本项目保留了 Breeze 的官方推理代码和适配后的 vLLM-Omni 源码。
下面五篇文档介绍模型的生成过程、适配中的主要改动，以及接入其他模型时通常要做的工作。
阅读需要 Python、PyTorch 和 Transformer 基础，不需要先安装环境或下载权重。

## 阅读顺序

| 顺序 | 文档 | 内容 |
| --- | --- | --- |
| 1 | [Breeze 如何生成语音](learning/01-model.md) | 文本条件、时间 AR、帧内 Depth 和音频解码 |
| 2 | [Breeze 在 vLLM-Omni 中的实现](learning/02-adaptation.md) | 阶段划分、模型接口、注册与权重加载 |
| 3 | [一条语音请求的执行过程](learning/03-request.md) | 数据传递、完整帧反馈、分块输出与请求清理 |
| 4 | [适配结果的验证](learning/04-correctness.md) | 与官方实现的对照方法、现有测试及其范围 |
| 5 | [适配其他模型的步骤](learning/05-porting.md) | 从分析参考代码到完成接入和验证 |

建议按顺序阅读。正文中的源码链接指向相关实现，可以在需要查看细节时打开。
如果已经了解模型，想定位适配代码，可以直接看第 2、3 篇。

## 其他文档

安装命令和 API 用法单独放在参考文档中，避免打断阅读。

| 内容 | 文档 |
| --- | --- |
| 环境准备、启动服务 | [安装与启动](getting-started.md) |
| 请求参数、客户端示例 | [API 指南](api.md) |
| 详细架构与运行限制 | [架构参考](architecture.md) |
| 验证命令与已有结果 | [验证指南](validation.md)、[CPU 验证记录](core-validation.md) |
| 后续计划与贡献方式 | [路线图](roadmap.md)、[贡献指南](../CONTRIBUTING.md) |

`breeze-tts/` 是官方参考代码，`vllm-omni/` 是修改后的框架源码。
上游版本和适配文件清单见 [project-manifest.json](../project-manifest.json)。

## 当前范围

文档对应 `v0.1.0-dev` 的核心实现：支持 S0 单说话人的文本和可选指令输入，
尚未实现音色克隆、多说话人与多分支 CFG。
已有测试使用小型模型在 CPU 上验证局部计算和回调行为；
真实 checkpoint、CUDA 执行与并发服务仍待验证。最新进度见项目首页和验证记录。

[开始阅读：Breeze 如何生成语音 →](learning/01-model.md)
