# 预取、页化与异步传输

## 预取类型

- 确定性预取：当前层路由已完成，对明确需要的专家提交最高优先级传输。
- 预测性预取：依据请求特征、路由历史和相邻层统计预测下一层专家。
- 保守预取：只使用剩余带宽、空闲槽位和可取消 buffer。
- 取消：路由未命中、请求取消或资源压力上升时撤销未开始 I/O，并在安全边界释放在途资源。

每个 PrefetchRequest 包含 model/layer/expert/block、目标层级、deadline、优先级、预测置信度、字节数、budget token 和取消 token。

## 双缓冲与事件

当前专家计算和下一批传输通过独立 compute/transfer stream 重叠。buffer A 被 kernel lease 持有时只能写 buffer B；复用前必须等待传输完成和最后一个消费者 event。多缓冲数量由资源规划器决定。

## 页化粒度

页化层次为 `Layer -> Expert -> Projection -> WeightBlock -> DevicePage`。页大小由量化 group、文件对齐、DMA 粒度、kernel tile、互连和显存碎片共同决定，不固定为操作系统页大小。

Autotune 可以选择 block 大小、预取深度、并发 I/O、buffer 数量和 pinned 比例，结果写入 `.tune`，key 必须包含模型布局、存储设备和后端指纹。

## 错误处理

预测失误只影响性能，不得改变结果。读取迟到时请求按 deadline 决定等待、切换 L2/L3 副本、降低 batch 或拒绝；禁止用未完成或错误版本的权重继续执行。
