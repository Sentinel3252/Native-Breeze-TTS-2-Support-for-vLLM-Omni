# 2. Breeze 在 vLLM-Omni 中的实现

[← 模型生成过程](01-model.md) · [学习指南](../learning-guide.md) · [请求执行过程 →](03-request.md)

官方推理代码在生成循环中依次调用 backbone、采样和 Depth。
接入 vLLM-Omni 后，请求由框架调度，这些操作分别放到模型接口、采样回调和阶段处理器中。

## 模型与框架的分工

vLLM 负责时间 AR 的调度、KV cache、paged attention 和首码本采样。
Paged attention 使用按块管理的 KV cache，配合框架调度执行 attention。
vLLM-Omni 负责阶段编排、阶段间传输和音频服务。

适配代码复用 vLLM 的 Llama/Qwen3 backbone 层，
另外实现 Breeze 的文本条件、Depth、完整帧 embedding 和 Mimi 解码。

| 工作 | 接入位置 |
| --- | --- |
| 校验请求、构造提示与采样参数 | TTS adapter |
| 编码文本、准备历史帧 embedding | Talker `preprocess` |
| 执行时间 backbone、计算 logits | Talker `forward`、`compute_logits` |
| 首码本采样后补齐完整帧 | runner 调用 `post_sample_talker_mtp` |
| 按请求传递完整帧与结束标志 | stage input processor |
| 生成波形 | Stage 1 的 Mimi decoder |

Depth 必须在首码本采样后执行，使用同一步 hidden state。
如果推迟到下一次 `preprocess`，达到 token 上限时可能没有下一次调用，最后一帧就会遗漏。
因此，runner 在当前采样步调用模型的补帧方法。

这个回调由模型属性启用。修改公共 runner 时，需要保留未启用该回调的模型行为。
接口名中的 `mtp` 沿用了框架命名，Breeze 在此完成帧内生成；当前实现不支持推测解码。

## 两个执行阶段

Stage 0 包括文本条件、时间 AR 和 Depth，输出完整 RVQ 帧。
Depth 的结果会反馈给下一步 AR，因此两者放在同一阶段。
Stage 1 接收完整帧，用 Mimi 生成波形，并在请求内保留跨块历史。
阶段定义见 [pipeline.py](../../vllm-omni/vllm_omni/model_executor/models/breeze_tts/pipeline.py)。

Stage 1 仍叫 `breeze_depth_codec`，对应类名为 `BreezeDepthCodecDecoder`；
传输函数也保留了 `talker2depth` 的名称。这些是旧命名，当前 Depth 实际在 Stage 0 执行。

## 配置与注册

框架启动时，需要能够解析配置、找到模型类，并选择对应的 pipeline。

| 注册内容 | 文件 |
| --- | --- |
| `model_type=breeze` 对应 `BreezeTTSConfig` | [arg_utils.py](../../vllm-omni/vllm_omni/engine/arg_utils.py) |
| 官方与原生 architecture 名称对应原生模型类 | [模型 registry](../../vllm-omni/vllm_omni/model_executor/models/registry.py) |
| Breeze 对应两阶段 pipeline | [pipeline registry](../../vllm-omni/vllm_omni/config/pipeline_registry.py) |
| speech 请求使用 Breeze adapter | [TTS adapter](../../vllm-omni/vllm_omni/entrypoints/openai/tts_adapters/breeze_tts.py)、[adapter 注册入口](../../vllm-omni/vllm_omni/entrypoints/openai/tts_adapters/__init__.py) |

`BreezeTTSConfig` 保留完整配置，通过 `get_text_config` 向 vLLM 提供时间 backbone 的尺寸、
RoPE 和音频词表等设置。配置处理中还会检查 backbone、Depth 与 codec 的尺寸是否匹配。
文本 tokenizer 的词表和音频采样词表在这里需要分别处理。

## 权重加载

Stage 0 加载 backbone、文本条件、Depth 和 LM head，Stage 1 加载 `codec_model.*`。
Talker 的 `WeightsMapper` 把官方参数前缀改为原生模块名称，再交给框架 loader。

权重覆盖检查还有几种特殊情况：

- 原生 QKV 和 gate/up 可能合并存放，需要确认所有来源分片都已加载。
- 共享 embedding 会有多个参数名称，需要识别它们指向同一份参数。
- Mimi 的量化器包含持久 buffer，它们也必须从 checkpoint 加载。

这些检查位于 [weight_coverage.py](../../vllm-omni/vllm_omni/model_executor/models/breeze_tts/weight_coverage.py)。
注册和加载完成后，还需要对照实际计算结果。

## 当前运行限制

当前要求 TP/PP=1、eager、关闭 prefix caching、同步 AR 调度，
并拒绝量化与推测解码。这些执行方式尚未完成适配，例如推测解码一次可能返回多个 token，
当前补帧接口只处理每个请求的一个采样结果。

[部署 YAML](../../vllm-omni/vllm_omni/deploy/breeze_tts.yaml) 开启了 `async_chunk`，
同时关闭 Stage 0 的 `async_scheduling`。前者控制阶段间分块传输，
后者控制 AR 调度，两者作用不同。
