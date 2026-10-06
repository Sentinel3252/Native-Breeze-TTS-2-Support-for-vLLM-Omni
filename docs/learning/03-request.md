# 3. 一条语音请求的执行过程

[← 框架中的实现](02-adaptation.md) · [学习指南](../learning-guide.md) · [适配验证 →](04-correctness.md)

以输入“你好”、默认 S0、无额外指令的请求为例。
token 值、码本数量和采样率取自 tokenizer 与 checkpoint。

## 1. 准备请求

[BreezeTTSAdapter](../../vllm-omni/vllm_omni/entrypoints/openai/tts_adapters/breeze_tts.py)
校验参数，调用 `speech_prompt_ids` 得到 `[S0]你好` 的完整提示 ID，长度记为 `L`。
这些 ID 放入 `additional_information.breeze_prompt_ids`，
调度输入则是长度相同的 `[0, ..., 0]`。

时间 AR 使用音频词表，文本 tokenizer 的 ID 可能超出这个范围。
占位 ID 供调度器记录提示长度和位置，实际文本在模型预处理时单独编码。
不同文本可能对应相同的占位序列，因此不能按占位 ID 缓存文本条件。
当前文本条件按 `request_id` 保存，prefix caching 关闭。

Adapter 将停止 ID 设为 backbone EOS，忽略 tokenizer 的默认 EOS。
Depth 的采样参数随请求传入，首码本使用框架的采样参数。

## 2. 处理提示

Prefill 处理提示并建立时间 backbone 的 KV cache。
[Talker](../../vllm-omni/vllm_omni/model_executor/models/breeze_tts/breeze_tts_talker.py)
的 `preprocess` 首次处理提示时，编码完整文本，再按本次调度的位置切片。
提示分多次执行时，各次切片来自同一份完整编码，保留 text encoder 的上下文。

预处理返回初始 embedding，以及可选的逐层文本特征和位置 mask。
`forward` 将这些条件接入原生 backbone。
后续层使用 DimFusion 时，先合并延迟保存的 residual，再替换文本激活的后半维度并归一化。
这个顺序与参考实现一致。prefill 完成后释放文本条件缓存。

## 3. 生成与反馈

提示最后位置的 hidden state 用于首帧生成；后续 Decode 每步生成一帧。
`compute_logits` 屏蔽无效编码，只保留有效 codec 值和 backbone EOS，
框架从中采样首码本 `c_t,0`。

runner 的 [_run_post_sample_talker_mtp](../../vllm-omni/vllm_omni/worker/gpu_ar_model_runner.py)
将当前首码本和同一步 hidden state 交给 Talker。
Depth 补齐其余码本，完整帧写入请求历史，同时作为多模态输出交给传输层。

下一次 `preprocess` 取出完整帧，对各码本 embedding 求和，作为时间 AR 的输入。
如果 KV 抢占后需要重算历史，使用已保存的完整帧，并检查其首码本与调度 ID 是否一致。
重新采样 Depth 会改变原有序列，不能用于历史重放。

| 数据 | 形状 | 含义 |
| --- | --- | --- |
| 文本 ID | `[L]` | 完整提示，调度输入使用等长占位 ID |
| 输入 embedding | `[N, H]` | 本次执行的 N 个位置，H 为 backbone 维度 |
| 回调 hidden state | `[B, H]` | B 个有效请求各自的采样条件 |
| 回调完整帧 | `[B, K]` | 每个请求一帧，每帧 K 个码本 |
| 单请求传输块 | `[F, K]` | 按时间排序的 F 帧 |
| Mimi 输入 | `[1, K, T]` | 单个请求累计的 T 帧 |

回调按请求顺序返回数据。部分 prefill 尚未产生有效采样时，
runner 返回空输出，防止重发上一帧；EOS 对应的负值帧标记在传输前移除。

## 4. 分块传输

[传输处理器](../../vllm-omni/vllm_omni/model_executor/stage_input_processors/breeze_tts.py)
按请求积累待传帧，默认达到 8 帧后发送。
payload 的 `codes.audio` 保存帧，`meta` 保存结束标志和流式配置。
Stage 1 的占位 token 触发执行，完整 codec 数据从 payload 读取。

结束时发送剩余帧，即使没有剩余数据，也要发送空最终块。
`meta.finished` 供接收层更新调度状态，
另一个标志 `stream_finished` 保留给 Mimi，确保它能刷新暂留的波形尾部。
全量传输在请求完成时发送完整帧序列，使用相同的数据格式。

## 5. 解码与清理

[Stage 1](../../vllm-omni/vllm_omni/model_executor/models/breeze_tts/breeze_depth_codec.py)
将新帧加入请求历史，
[MimiPrefixDecoder](../../vllm-omni/vllm_omni/model_executor/models/breeze_tts/codec_stream.py)
重解码完整前缀，只返回尚未输出的波形。

流式模式暂留两帧对应的尾部，结束时刷新，并检查已输出的前缀是否保持稳定。
非流式模式等到请求结束后解码一次。
逐块独立解码可能丢失 transformer、上采样和卷积上下文，
当前重解码方式保留了这些上下文，但有重复计算开销。

Stage 1 返回按请求排列的音频张量和采样率，speech 服务层将其编码为请求指定的格式。

| 状态 | 保存位置 | 释放时机 |
| --- | --- | --- |
| 提示条件 | Talker `_prompt_cache` | prefill 完成或请求清理；重放提示时重建 |
| 完整帧历史、Depth RNG | Talker 的请求字典 | `on_requests_finished` |
| 待传帧、传输元数据 | connector 的请求映射 | 最终块；取消时由框架清理 |
| codec 历史、已发波形 | Mimi decoder 的请求状态 | 最终刷新、解码异常或请求清理 |

完成、取消和异常都需要调用相应的清理入口。
CPU 测试覆盖了部分回调和状态处理，真实服务中终止事件能否传到所有层，还需要运行验证。
