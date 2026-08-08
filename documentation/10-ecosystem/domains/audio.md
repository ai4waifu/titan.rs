# 音频与语音领域

## 范围

覆盖音频解码、重采样、频域变换、语音活动检测、自动语音识别、说话人任务、文本转语音和流式 Native 推理。

## 数据管线

Audio schema 固定 sample rate、channel、sample format、时间基准和 loudness。WAV/FLAC 解码后执行 channel mix、resample、frame/window、STFT、Mel 和 MFCC；流式 chunk 必须保留 overlap 与 filter state。

## 模型能力

- 变长 waveform/spectrogram 与 mask。
- convolution、FFT/STFT、filterbank、sequence encoder 和 decoder。
- streaming encoder cache、endpointing、增量解码和时间戳对齐。
- TTS 声学模型、vocoder 和实时播放 buffer 契约。

## 验收

测试覆盖解码 golden、重采样频响、chunk/full 等价、边界 padding、流式 state 恢复和文本/时间戳输出。基准报告 real-time factor、首结果延迟、chunk p95、峰值内存、音频丢帧和不同并发曲线。
