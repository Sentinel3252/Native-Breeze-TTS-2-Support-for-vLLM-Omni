# 模型适配学习指南

[项目首页](../README.zh-CN.md) · [架构细节](architecture.md) · [安装与启动](getting-started.md) · [验证要求](validation.md)

下面按阅读顺序介绍 Breeze TTS 2 的适配代码，对应 `v0.1.0-dev` 版本。
主要看模型怎么执行、框架哪些地方需要改，以及测试怎么做。
阅读需要一些 Python、PyTorch 和 Transformer 基础，可以先只看这里列出的文件。

## 1. 模型怎么生成音频

Breeze 是模型本身，vLLM 负责 AR 调度和 KV cache，vLLM-Omni 负责阶段间传输和音频输出。
适配时，需要把官方生成代码中的计算接到框架接口上，同时保持计算结果一致。

这里的 AR（自回归）沿时间生成音频帧；Depth decoder 在一帧内部补齐多个码本；
RVQ 码本是音频的离散表示，Mimi 再将这些离散编码还原为波形。
Prefill 处理提示输入，Decode 逐步生成后续帧；EOS 表示生成结束。

```text
文本 / 可选指令
  → API 模板与 tokenizer → text encoder 条件
  → Stage 0：原生 AR → 采样第 0 个码本 → Depth 补齐整帧
                ↑                          │
                └──── 整帧 embedding 反馈 ──┘
  → 按 request_id 传输完整帧
  → Stage 1：Mimi → 增量波形 / 最终尾部 → API 音频响应
```

Depth 在 Stage 0 中执行。Stage 1 保留了 `breeze_depth_codec` 这个旧名字，实际负责 Mimi 解码。
当前 Mimi 每次重解码完整前缀来保留上下文；分块输出已实现，解码状态缓存仍是后续工作。

## 2. 先看官方实现

先看文本/指令合成分支，克隆和 CFG 的代码可以暂时跳过：

| 源码入口 | 重点问题 |
| --- | --- |
| [输入模板](../breeze-tts/breeze_infer/templates.py) | `[S0]`、指令与文本如何拼接并编码？ |
| [模型结构](../breeze-tts/models/breeze.py) | text encoder、backbone、Depth、音频 embedding 如何连接？重点看 `convert_input_ids_to_embeds`、`_project_segments` 和 `forward` |
| [生成循环](../breeze-tts/models/generation_breeze.py) | `_sample` 如何从时间步 hidden state 生成整帧，再反馈给下一步？何时停止？ |

阅读时留意输入模板、配置字段、权重名称和形状，以及采样、停止条件。
这些地方稍有差异，就可能出现权重能加载但生成结果不对的情况。
如果要运行参考代码，需要单独准备环境，它与原生运行环境的依赖不同，见[安装指南](getting-started.md)。

## 3. 适配改了哪些文件

下表的路径相对 `vllm-omni/vllm_omni/`。
完整文件清单在 [project-manifest.json](../project-manifest.json)。

| 环节 | 文件 | 这里做了什么 |
| --- | --- | --- |
| 配置与架构识别 | [engine/arg_utils.py](../vllm-omni/vllm_omni/engine/arg_utils.py)、[模型 registry](../vllm-omni/vllm_omni/model_executor/models/registry.py) | 注册 HF config、识别 `model_type` 与 checkpoint architecture，映射到原生类 |
| 模型配置 | [configuration_breeze_tts.py](../vllm-omni/vllm_omni/model_executor/models/breeze_tts/configuration_breeze_tts.py)、同目录 `configuration_breeze_base.py` | 保留 backbone、Depth、codec、特殊 token 等配置 |
| 阶段编排 | [pipeline.py](../vllm-omni/vllm_omni/model_executor/models/breeze_tts/pipeline.py)、[pipeline registry](../vllm-omni/vllm_omni/config/pipeline_registry.py) | 声明两阶段、执行类型、输入来源、传输回调和最终音频输出 |
| 请求入口 | [TTS adapter](../vllm-omni/vllm_omni/entrypoints/openai/tts_adapters/breeze_tts.py)、同目录 `__init__.py` | 校验参数、构造提示与采样设置，将 adapter 纳入框架发现入口 |
| Stage 0 模型计算 | [breeze_tts_talker.py](../vllm-omni/vllm_omni/model_executor/models/breeze_tts/breeze_tts_talker.py) | 原生 AR、文本预处理、整帧反馈、请求状态与权重加载 |
| 文本与帧内计算 | [text_conditioning.py](../vllm-omni/vllm_omni/model_executor/models/breeze_tts/text_conditioning.py)、同目录 `t5gemma2_encoder.py`、`depth_decoder.py`、`frame_decoder.py` | 文本编码/投影与 DimFusion、Depth attention、整帧采样和 embedding |
| 采样时机 | [worker/gpu_ar_model_runner.py](../vllm-omni/vllm_omni/worker/gpu_ar_model_runner.py) | 通过 `_run_post_sample_talker_mtp` 将本步 token 与 hidden state 交给模型补帧 |
| 阶段间传输 | [stage_input_processors/breeze_tts.py](../vllm-omni/vllm_omni/model_executor/stage_input_processors/breeze_tts.py) | 按请求积累完整帧，支持增量/full-payload 传输和结束标志 |
| Stage 1 音频 | [breeze_depth_codec.py](../vllm-omni/vllm_omni/model_executor/models/breeze_tts/breeze_depth_codec.py)、[codec_stream.py](../vllm-omni/vllm_omni/model_executor/models/breeze_tts/codec_stream.py) | Mimi 权重、历史前缀、音频增量、尾部刷新与清理 |
| 权重与运行约束 | [weight_coverage.py](../vllm-omni/vllm_omni/model_executor/models/breeze_tts/weight_coverage.py)、[deploy/breeze_tts.yaml](../vllm-omni/vllm_omni/deploy/breeze_tts.yaml) | 检查权重覆盖，配置分块与各阶段资源、执行限制 |

适配其他模型时，需要修改的文件不一定相同。通常先看现有接口能否满足要求；
如果需要改公共 runner，还要检查会不会影响其他模型。

## 4. 一条请求的执行过程

假设输入文本是“你好”，使用默认 S0 说话人：

1. Adapter 校验请求，用 `speech_prompt_ids` 构造官方模板。
   实际文本 ID 在 `additional_information` 中传递；调度器收到的是等长占位 ID。
2. Talker 的 `preprocess` 编码完整提示，按 chunked prefill 位置取切片；
   `forward` 将逐层文本条件接入原生 backbone。占位 ID 本身不代表真实文本。
3. vLLM 采样第 0 个码本后，runner 调用 `post_sample_talker_mtp`，
   使用同一步 hidden state 完成 Depth 解码。帧内 KV cache 只用于当前帧。
4. 所有码本 embedding 求和作为下一步 AR 输入；
   传输回调则按请求积累整数 `[frames, codebooks]`，默认每 8 帧传一次。
5. Stage 1 保留该请求的完整历史，重解码前缀并只输出新增波形。
   流式模式暂留两帧对应的尾部，结束时刷新。
6. EOS 不进入 Depth/Mimi；不足一个块的末帧也要传输。
   空的最终块仍携带结束信号；完成、取消或异常都需要释放请求状态。

这里有几个容易出错的地方：

- 补帧必须用同一步 hidden state；首帧和达到 token 上限时的末帧都不能遗漏。
- 码本值 0 合法，backbone EOS 为 `vocab_size`。KV 抢占后要重放历史完整帧，不能重新抽样 Depth。
- 文本缓存、帧历史、随机数生成器和音频历史都要按请求保存。文本条件不同，占位 ID 也可能相同，因此不能只用占位 ID 作为缓存键。

具体实现见[架构文档](architecture.md)。

## 5. 怎么验证

在项目根目录先运行不依赖 GPU 的基础检查，Python 3.10+ 即可：

```bash
python scripts/check_project.py
python -m unittest discover -s tests -v
python examples/speech_client.py --help
```

核心 CPU 测试需要在独立的 Python 3.12 环境安装依赖：

```bash
python -m pip install -r tests/core/requirements.txt
python -m pytest tests/core -q
```

读代码时可以一起看这些测试：

| 要理解的行为 | 测试入口 |
| --- | --- |
| Depth logits、完整帧和采样 | [test_numerics.py](../tests/core/test_numerics.py) |
| 文本模板、投影与官方方法对照 | [test_conditioning_parity.py](../tests/core/test_conditioning_parity.py) |
| 采样回调、首末帧、KV 抢占重放与清理 | [test_lifecycle.py](../tests/core/test_lifecycle.py) |
| 请求隔离、块边界与结束传输 | [test_transport.py](../tests/core/test_transport.py) |
| 音频连续性与空末块刷新 | [test_numerics.py](../tests/core/test_numerics.py) 中的 `test_mimi_split_decode_matches_whole_and_flushes_empty_terminal` |
| Stage 1 波形归属与请求清理 | [test_codec_stage.py](../tests/core/test_codec_stage.py) |
| 权重覆盖、请求参数边界 | [test_weight_coverage.py](../tests/core/test_weight_coverage.py)、[test_api.py](../tests/core/test_api.py) |

这些测试使用小型随机权重模型；runner 回调通过 CPU 测试环境执行，未运行真实 CUDA 调度链路。
目前跑过哪些测试，见[验证记录](core-validation.md)。

有 Linux/CUDA 环境后，按[安装指南](getting-started.md)分别准备参考与原生环境，使用同一 revision 的权重。
真实模型的验证顺序如下，详细步骤见[验证指南](validation.md)：

```text
完整权重加载 → 文本条件/AR logits/Depth 对照 → 单请求完整音频
→ 分块与全量音频一致性 → 连续请求、并发、取消和异常 → 性能测量
```

对照结果时先用确定性设置，找到最早出现差异的中间结果，再检查随机采样。
同时记下权重 revision、依赖、硬件、输入和采样设置，方便复现。
服务能启动之后，还需要检查生成的音频和请求结束后的状态。

## 6. 可以做的练习

刚开始可以选一个 EOS、空末块或 KV 重放测试，对照源码看输入、状态和输出怎么变化。
然后在临时实验配置里改一下 `depth_chunk_frames`，检查不足一块和刚好整除时的行为，
确认帧数、顺序和结束标志都没变。

有 GPU 环境后，可以整理真实 checkpoint 的权重名称和形状，记录加载失败的位置。
不要跳过缺失权重，否则后面的输出对照就没有意义了。

想继续做代码改进，可以选 Mimi 状态缓存或 Depth 批量解码。
这两项都要先检查输出是否一致、请求之间是否互相影响，再测性能。
具体要求见[路线图](roadmap.md)与[流式优化任务](next-task-prompt.md)。
