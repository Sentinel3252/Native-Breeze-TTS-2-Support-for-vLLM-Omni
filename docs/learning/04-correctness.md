# 4. 适配结果的验证

[← 请求执行过程](03-request.md) · [学习指南](../learning-guide.md) · [适配其他模型 →](05-porting.md)

适配后需要检查两方面：计算是否与官方实现一致，请求是否在框架中正确执行和结束。
服务能启动，只说明配置、注册和初始化没有阻止启动。

## 对照条件

官方与原生实现应使用相同的 checkpoint revision、tokenizer、输入模板和采样设置，
同时记录代码版本。本仓库两套实现的 Transformers 依赖不同，需要分别准备环境，
版本和命令见[安装指南](../getting-started.md)。

先使用相同权重、输入和精度，比较 greedy 结果。
首码本和 Depth 分别采样，两处都需要控制。
固定 seed 不能保证随机序列一致，因为两套实现可能以不同顺序消费随机数。
比较浮点结果时，还要记录精度、执行后端和数值容差。

## 检查顺序

| 检查项 | 对照内容 | Breeze 中容易出错的地方 |
| --- | --- | --- |
| 配置与权重 | 参数形状、加载覆盖和持久 buffer | backbone 配置、融合参数的来源分片、共享 embedding、Mimi buffer |
| 文本条件 | 模板 ID、encoder 特征和投影结果 | 特殊 token、分段编码、DimFusion 层选择与融合顺序 |
| 时间 AR | 对应位置的 hidden state 和 logits | attention、位置、文本条件、有效词表和 EOS |
| Depth | 相同 hidden state 与首码本下的完整帧 | attention、投影、码本位置和采样 |
| 反馈与传输 | 帧数、顺序和请求对应关系 | 首末帧、KV 重放、重复传输和空最终块 |
| 波形与服务 | 音频连续性、请求隔离和状态清理 | 分块/全量对照、尾部、并发、取消和异常 |

从上游向下游查找最早出现差异的位置。
如果文本条件已不同，先查模板和 encoder；
如果 AR 结果一致而 Depth 不同，再查补帧所用的 hidden state 和帧内计算。

codec 值 `0`、EOS、首帧、token 上限末帧和空最终块需要单独检查。
这些情况可能不改变张量形状，却会影响帧数或输出时机。
波形除了非空和有限值，还需要检查听感与可懂度。

## 现有测试的范围

核心 CPU 测试使用小型模型和随机权重，执行实际 Depth、文本模块和小型 Mimi。
部分 adapter/runner 方法从源码中通过 AST 提取，在 CPU 测试环境中运行，
因此没有执行完整的 CUDA 调度链路。

- 数值和文本条件：[test_numerics.py](../../tests/core/test_numerics.py)、[test_conditioning_parity.py](../../tests/core/test_conditioning_parity.py)。
- 回调、传输和清理：[test_lifecycle.py](../../tests/core/test_lifecycle.py)、[test_transport.py](../../tests/core/test_transport.py)、[test_codec_stage.py](../../tests/core/test_codec_stage.py)。
- 权重覆盖和请求参数：[test_weight_coverage.py](../../tests/core/test_weight_coverage.py)、[test_api.py](../../tests/core/test_api.py)。

Depth 在 Transformers 4 环境中与官方小模型对照，
在 Transformers 5 环境中与其保存的参考输出对照。
真实 CUDA paged attention、预训练权重和服务行为仍待验证。
已有结果见[CPU 验证记录](../core-validation.md)，运行命令见[验证指南](../validation.md)。

## 性能与后续优化

真实权重加载、数值对照和单请求音频通过后，再检查分块输出、并发和终止处理，
最后测量延迟、吞吐和资源占用。性能对比应使用相同输入和相当的输出质量。

Mimi 状态缓存可以减少前缀重解码的重复计算，但需要正确保存跨块上下文，
并保留尾部刷新和请求清理。实现缓存后，应重新对照波形。
CUDA graph、prefix caching、并行和量化也需要各自的验证结果，
不能直接沿用当前 eager、TP/PP=1 路径的结论。
