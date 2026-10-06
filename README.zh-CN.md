# Breeze TTS 2 for vLLM-Omni

本项目为 Breeze TTS 2 添加 vLLM-Omni 支持，也供大家学习模型适配。
仓库中保留了官方推理代码和适配后的框架源码，可以对照着看具体改了哪些地方。

[English](README.md) · [学习指南](docs/learning-guide.md) · [安装与启动](docs/getting-started.md) · [架构](docs/architecture.md) · [路线图](docs/roadmap.md)

官方参考实现在 `breeze-tts/`，适配代码在 `vllm-omni/`。
如果想了解一个模型从官方实现接入推理框架的过程，可以从下面的学习指南开始。

## 当前进度

当前 tag 是 `v0.1.0-dev`，核心适配代码已基本完成。
模型注册、文本/指令条件、原生自回归（AR）执行、Depth 补齐码本、整帧反馈、阶段传输、
Mimi 音频解码和 `/v1/audio/speech` 适配均已实现。

目前通过了 35 项核心 CPU 测试（分别使用 Transformers 4.57.3 和 5.10.1），
以及 18 项客户端和打包工具测试，见[验证记录](docs/core-validation.md)。
测试用的是小型模型，真实 checkpoint 加载、CUDA 数值一致性、语音质量和并发服务还没验证，
也还没有性能数据或稳定版本。

| 范围 | 当前状态 |
| --- | --- |
| 文本/指令到语音的核心链路 | 代码已实现，真实模型端到端验证待完成 |
| 增量音频 | Mimi 保留历史并重解码完整前缀；小型模型连续性测试通过，效率待优化 |
| 请求隔离与清理 | CPU 测试已覆盖，真实服务的取消、异常与并发验证待完成 |
| 音色克隆、多说话人、CFG | 暂不支持 |
| 执行配置 | 目前要求 TP/PP=1、eager 模式、关闭 prefix caching |

`git show v0.1.0-dev` 可以查看这个版本，后续改动也可以与它比较。

## 后续工作

| 计划版本 | 工作内容 |
| --- | --- |
| `0.1.0` | 验证真实权重加载、官方 AR/Depth 输出对照、中英文语音和服务运行，记录 Linux/CUDA 环境 |
| `0.2.0` | 实现 Mimi 解码状态缓存、Depth 批量解码，限制请求状态占用，处理异常和取消，测量性能 |
| `0.3.0` | 添加参考音频/文本、说话人条件和多分支 CFG，补充 API 与测试 |
| `0.4.0` | 分别验证 CUDA graph、prefix caching、TP/PP、量化，补充权重预检和启动诊断 |

这些工作还没有确定的完成时间。具体内容见[路线图](docs/roadmap.md)，
流式优化的实现要求见[任务文档](docs/next-task-prompt.md)。

## 如何学习

[学习指南](docs/learning-guide.md)按顺序介绍 Breeze 的生成过程、框架中的适配实现、
一条请求的执行过程和验证方法。最后一篇整理了适配其他模型的步骤，
包括分析参考代码、选择接口、加载权重和处理请求状态。

阅读需要 Python、PyTorch 和 Transformer 基础，无需先安装环境或准备 GPU。
正文附有相关源码链接；安装命令和验证操作见对应文档。

## 快速上手

在项目根目录执行基础检查，只需 Python 3.10+：

```bash
python scripts/check_project.py
python -m unittest discover -s tests -v
python examples/speech_client.py --help
```

核心模型 CPU 测试建议使用独立的 Python 3.12 环境，无需 vLLM 或下载权重：

```bash
python -m pip install -r tests/core/requirements.txt
python -m pytest tests/core -q
```

实际推理请按[安装与启动](docs/getting-started.md)准备环境与 checkpoint，安装本项目修改过的
`vllm-omni/` 源码，再使用[客户端示例](examples/speech_client.py)和 [API 指南](docs/api.md)。
当前启动配置还需要验证，检查步骤见[验证指南](docs/validation.md)。

## 项目结构与文档

| 路径 | 用途 |
| --- | --- |
| `breeze-tts/` | 官方参考实现，用于理解和对照模型行为 |
| `vllm-omni/` | 包含 Breeze 原生适配的框架源码 |
| `docs/learning-guide.md`、`docs/learning/` | 学习指南、适配说明与相关源码 |
| `docs/` | 安装、架构、API、验证、路线图、[变更记录](docs/changelog.md)与[发布指南](docs/releasing.md) |
| `tests/` | 工具测试与核心数值、传输、生命周期 CPU 测试 |
| `examples/`、`scripts/` | 语音客户端、项目检查与源码打包 |
| `project-manifest.json` | 上游基线与适配文件清单 |

欢迎补充学习文档、可复现的问题和适配改进，见[贡献指南](CONTRIBUTING.md)。

## 许可证与致谢

项目源码和文档采用 [Apache-2.0](LICENSE)，第三方组件说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。Breeze 模型材料适用独立的
[研究与非商业使用协议](breeze-tts/MODEL_LICENSE)。源码发布不包含模型权重或生成音频。

感谢 [vLLM-Omni](https://github.com/vllm-project/vllm-omni)、
[vLLM](https://github.com/vllm-project/vllm) 与
[BreezeBlue](https://github.com/breezeblue-ai/breeze-tts)。本项目为独立适配项目。
