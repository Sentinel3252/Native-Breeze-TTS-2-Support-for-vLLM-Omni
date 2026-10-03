# Breeze TTS 2 for vLLM-Omni

将 Breeze TTS 2 的自回归生成接入 vLLM 原生执行路径，并通过两阶段流水线输出音频。

[English](README.md) · [安装与启动](docs/getting-started.md) · [架构](docs/architecture.md) · [版本路线](docs/roadmap.md) · [参与贡献](CONTRIBUTING.md)

项目在 `vllm-omni/` 源码中实现 Breeze 原生集成：Stage 0 使用 vLLM 原生 attention
执行 AR，并用官方结构的 depth decoder 补齐整帧码本；Stage 1 通过 Mimi 合成音频。
`breeze-tts/` 保留官方推理代码，作为模型行为和正确性对照。

**当前开发版本：`0.1.0-dev`，基础实现阶段。** 本项目是为 Breeze TTS 2 提供
vLLM-Omni 支持的独立开源适配项目，尚未发布稳定版本。模型注册、阶段传输、depth decoder、
文本/指令 conditioning、整帧 AR 反馈和保留历史的 Mimi 已实现，并有 CPU 回归测试；
真实 checkpoint 加载、原生 AR 数值一致性、完整模型的音频质量和并发服务仍需
Linux/CUDA 验证。当前没有可发布的性能结果或已验证的部署配置。

初版目标是提供文本/指令到语音的基础适配。高效流式解码、音色克隆与高级执行路径
按后续版本逐步加入。此版本号用于标识当前开发基线，不代表已经创建 Git tag、
发布正式版本或验证了真实运行环境。

## 已有能力与支持边界

| 能力 | 状态 |
| --- | --- |
| vLLM 原生 AR 模型 | 已实现；权重加载与数值一致性待验证 |
| 请求级整帧码本传输 | Async 与 full-payload 回调已实现，有 CPU 回归测试 |
| `/v1/audio/speech` 文本/指令适配 | 已实现；完整 checkpoint 的语音生成待验证 |
| 增量音频输出 | 保留完整历史并重解码前缀，小型 Mimi 的分块连续性测试通过 |
| 官方 text encoder conditioning 与 prompt 模板 | 已接入，投影与官方方法的 CPU 对照测试通过 |
| 音色克隆、多说话人、CFG | 暂不支持，相关请求会明确报错 |
| 并发与请求取消 | CPU 请求隔离/清理测试已覆盖，实际服务验证待完成 |
| 并行与图捕获 | 初版要求 TP/PP=1、eager 执行、关闭 prefix caching |
| 延迟、吞吐、显存需求 | 尚无本集成的测量结果 |

使用 vLLM 的调度器和 KV cache 执行路径是代码层面的设计，不代表已经完成模型兼容性或性能验证。
官方模型具备的能力也不等于本集成已支持的能力。

## 版本规划与保留方式

| 版本 | 主要内容 | 状态与完成条件 |
| --- | --- | --- |
| `0.1.0-dev` | 基础实现：文本条件、原生 AR、Depth 补帧、整帧反馈、Mimi 前缀重解码与语音 API | 当前基线；CPU 检查通过，真实 checkpoint 的 Linux/CUDA 验证待完成 |
| `0.1.0` | 可复现的基础模型支持 | 完成真实权重加载、可理解的语音输出、流式与服务生命周期验证，记录可复现环境 |
| `0.2.0` | 高效流式与服务运行 | Mimi 真正流式解码、Depth 批量解码、请求隔离、状态资源上限、异常与取消清理 |
| `0.3.0` | 音色克隆与 CFG | 参考音频/文本、说话人条件、多分支 CFG，完成官方行为对照与 API 验证 |
| `0.4.0` | 高级执行路径与开发维护工具 | 逐项验证执行优化，完善 checkpoint 预检、启动诊断和集成验证工具 |

后续版本为规划，没有预定发布日期或已支持承诺。开发与维护工具、回归测试会贯穿各版本补充。
高级功能需分别验收，再开放相应配置；不会因加入某一种优化就宣称所有多卡、量化或图捕获组合可用。

保留版本采用 **Git tag + 带校验和的源码归档 + 对应验证记录**，日常维护同一份源码。
当前项目根目录尚未初始化 Git，因此先保存源码快照；正式仓库建立后再创建版本 tag。
版本细节见[路线图](docs/roadmap.md)、[变更记录](docs/changelog.md)与[发布指南](docs/releasing.md)。
后续 `0.2.0` 开发可直接使用[任务 prompt](docs/next-task-prompt.md)。

## 开始使用

文档检查与工具测试只需要 Python 3.10+，不需要 GPU 或第三方 Python 包：

```bash
python scripts/check_project.py
python -m unittest discover -s tests -v
python examples/speech_client.py --help
```

核心数值测试需要独立的 Python 3.12 环境，无需 vLLM、GPU 或下载模型权重：

```bash
python -m pip install -r tests/core/requirements.txt
python -m pytest tests/core -q
```

已记录的基线结果：**35 项核心测试分别在 Transformers 4.57.3 与 5.10.1 下通过，
18 项客户端/打包工具测试通过**。测试使用小型模型与 CPU 回调测试环境，
覆盖边界见 [CPU 测试说明](tests/core/README.md) 与 [当前验证记录](docs/core-validation.md)。

推理开发面向 Linux 与 NVIDIA CUDA GPU。应安装本项目的 `vllm-omni/` 源码，
完整步骤见[安装与启动](docs/getting-started.md)。官方参考实现与原生集成的依赖不同，
请使用独立环境。

准备好兼容 checkpoint 并安装运行环境后，可使用以下开发启动命令：

```bash
vllm serve /path/to/breeze-tts-2 --omni \
  --stage-configs-path vllm-omni/vllm_omni/deploy/breeze_tts.yaml \
  --host 127.0.0.1 --port 8000
```

这条命令用于开始 checkpoint 验证，目前尚不保证直接运行官方 checkpoint 即可成功。
服务能输出有效音频后，可运行客户端：

```bash
python examples/speech_client.py \
  --model /path/to/breeze-tts-2 \
  --text "你好，欢迎使用 Breeze TTS。" \
  --output outputs/hello.wav
```

客户端支持 WAV 输出、`--stream` PCM 流式输出和通过 `BREEZE_API_KEY` 设置鉴权。
使用约定见 [API 指南](docs/api.md)。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `vllm-omni/` | 带 Breeze 集成的上游源码快照 |
| `breeze-tts/` | 官方参考实现 |
| `docs/` | 安装、API、设计、验证、版本路线、变更记录和发布指南 |
| `examples/` | 独立语音请求客户端 |
| `scripts/` | 项目检查和源码打包工具 |
| `tests/` | 工具测试与核心数值、回调、生命周期 CPU 回归测试 |
| `project-manifest.json` | 上游基线与集成文件清单 |

本地目录中有两个独立 Git 仓库。发布时请按[发布指南](docs/releasing.md)导出完整源码，
避免直接添加嵌套仓库导致集成代码未被收录。

## 贡献与路线图

优先工作是真实 checkpoint 的原生 AR 一致性、语音质量，以及请求取消和并发服务验证。
Mimi 当前采用前缀重解码，后续再实现高效的完整流式状态缓存。
欢迎提交可复现的问题、文档修复和代码改进；见[贡献指南](CONTRIBUTING.md)、
[验证要求](docs/validation.md)与[路线图](docs/roadmap.md)。

## 许可证与致谢

项目源码和文档采用 [Apache-2.0](LICENSE)，第三方组件保留各自许可证和归属，
见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。Breeze 模型材料适用独立的
[研究与非商业使用协议](breeze-tts/MODEL_LICENSE)，不因本项目源码开源而改变。
源码发布不包含模型权重或生成音频。

感谢 [vLLM-Omni](https://github.com/vllm-project/vllm-omni)、
[vLLM](https://github.com/vllm-project/vllm) 与
[BreezeBlue](https://github.com/breezeblue-ai/breeze-tts)。本项目是独立集成，不代表上游官方支持。
