# 语言与生成式模型领域

## 范围

覆盖文本编码、Transformer、分类、序列标注、问答、生成和参数高效微调。前沿大模型以 Native CUDA/ROCm 服务为主，浏览器只运行轻量模型或交互演示。

## Tokenizer

支持 BPE、WordPiece、Unigram、词表和特殊 token。Tokenizer manifest 固定 Unicode normalization、pre-tokenization、added token、BOS/EOS/PAD、byte fallback 和版本；训练与推理必须使用相同摘要。

## 模型能力

- Embedding、RoPE/position、Attention、LayerNorm/RMSNorm、FFN 和 MoE。
- causal/padding mask、prefill、decode、batching 和 streaming generation。
- paged KV Cache、prefix cache、会话取消和确定性采样种子。
- 权重分片、量化、LoRA/adapter 和模型包转换。

## 超显存执行

语言 MoE 直接使用 runtime 的 L0-L4 资源规划、专家缓存、预取和回退。模型定义只描述专家和路由语义，不编码 GPU 层数、页面或设备放置。

## 验收

测试覆盖 tokenizer round-trip、logits/生成 golden、KV 与无缓存等价、batch/单请求等价、量化质量、分层权重正确性和 checkpoint 恢复。基准记录 prefill/decode 吞吐、首 token/p95、KV 字节、专家等待、缓存命中和不同上下文/并发曲线。
