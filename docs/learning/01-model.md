# 1. Breeze 如何生成语音

[学习指南](../learning-guide.md) · [下一篇：框架中的实现 →](02-adaptation.md)

Breeze 沿时间逐帧生成音频编码。每一帧包含多个码本：
时间 backbone 先产生第一个码本，Depth decoder 补齐其余码本，再将完整帧作为下一步输入。
Mimi 负责把这些编码还原成波形。

## 文本条件

当前适配使用 S0 说话人，输入模板为 `[S0]文本`，
带指令时为 `[S0]<ins_bos>指令<ins_eos>文本`。
官方代码先带特殊 token 编码，再解码为保留这些 token 的字符串，最后重新编码。
适配保留了相同处理，避免改变提示中的特殊 token。

文本 ID 经 text encoder 和投影后送入 backbone。
投影方式由 checkpoint 决定，当前实现包括 linear、MLP 和 DimFusion。
DimFusion 还会在 backbone 的相应层融合文本特征，融合只作用于文本位置，
生成音频的位置保留原有激活。因此，文本条件既可能影响初始 embedding，也可能参与逐层计算。

## 时间 AR 与帧内 Depth

AR 指自回归，即每一步根据已有序列生成后续内容。
Breeze 的时间 backbone 是 Transformer 主干，沿音频时间推进；
Depth decoder 则在同一帧内依次生成各个码本。

音频 codec（编解码器）用 RVQ（残差向量量化）表示音频，每个码本提供一个离散索引。
设时间为 `t`，码本数量为 `K`，一帧写作：

```text
C_t = [c_t,0, c_t,1, ..., c_t,K-1]
```

这组索引还需要 Mimi 解码才能成为波形。
Depth 生成时使用当前 backbone 的 hidden state，以及这一帧已经生成的码本。
它的 KV cache 在帧内使用，完成一帧后丢弃；时间 backbone 的 KV cache 则跨时间步保留。

## 完整帧反馈

下面以三个码本为例，实际数量取自 checkpoint：

```text
文本条件 / 上一帧 embedding
            │
            ▼
      时间 backbone → h_t → LM head → 采样 c_t,0
                       │                    │
                       └───── Depth ────────┘
                                │
                       C_t = [c_t,0, c_t,1, c_t,2]
                          │                 │
                          ▼                 ▼
                   完整帧 embedding       Mimi → 波形
                          │
                          └── 下一时间步 backbone 输入
```

第一帧从提示最后位置的 hidden state 开始生成。随后，各码本的 embedding 求和，
得到下一时间步的输入：`E(C_t) = Σ E_k(c_t,k)`。
原生实现用一张 embedding 表，通过码本偏移选取各自的区域。
只反馈第 0 个码本会丢失其余帧信息，与官方计算不同。

Depth 使用的 `h_t` 必须与当前首码本来自同一个采样步。
如果使用前一步或后一步的 hidden state，张量形状可能仍然正确，但生成条件已经改变。
后面的 runner 回调就是为了在这个时机完成整帧。

## 停止条件

本实现的 backbone EOS 是 `vocab_size`，位于有效 codec 编码范围之外。
采样到 EOS 后，跳过 Depth 和 Mimi。codec 值 `0` 是合法编码，
文本 tokenizer 的 EOS 也不能直接用于停止音频生成。

达到生成上限时，最后一个已经采样出的有效首码本仍需补成完整帧。
它与 EOS 的处理不同：前者包含需要输出的音频，后者只表示结束。

## 相关源码

| 文件 | 关注位置 |
| --- | --- |
| [官方模板](../../breeze-tts/breeze_infer/templates.py) | 模板拼接、`_prepare_one` 的编码处理 |
| [官方模型](../../breeze-tts/models/breeze.py) | `convert_input_ids_to_embeds`、`_project_segments`、`forward` |
| [官方生成代码](../../breeze-tts/models/generation_breeze.py) | `_sample` 中首码本、Depth 与停止处理 |
| [原生帧生成](../../vllm-omni/vllm_omni/model_executor/models/breeze_tts/frame_decoder.py) | `complete_frame`、`embed_audio_frame` |
