# 数据面

数据面只承载张量、梯度、激活和 collective 控制帧。每条数据流都绑定一个已提交的 `TopologySnapshot`，控制面变更不会隐式影响正在执行的 collective。

## 传输层

### 本地确定性传输

当前实现仅提供进程内 `LocalTransport`，不建立网络连接。每个 frame 都绑定 `run_id`、epoch、单调 sequence、payload checksum 和调用方给出的超时；接收方在读取 payload 前逐项验证。run、epoch、sequence、超时或 checksum 不一致均返回明确错误。

## Frame 格式

固定头部包含 magic、协议版本、`run_id`、epoch、源 rank、目标 rank、`collective_seq`、`tensor_id`、分片序号、总分片数、dtype、字节数和 checksum。头部后是 payload，尾部可带设备事件句柄。

## 缓冲区生命周期

1. Planner 预留发送和接收 buffer，标注设备、对齐和可复用范围。
2. Producer 写入后发布 release event；transport 只读取已完成区域。
3. Receiver 校验头部和 checksum，完成后发布 acquire event。
4. 消费者等待 acquire event，使用结束后归还池。

任何错误都必须释放对应 token，防止 buffer 永久占用。跨设备的 host staging buffer 属于显式资源，纳入峰值内存统计。

## 超时与背压

单个 frame 的发送超时、collective 超时和会话超时分层配置。发送队列达到上限时 producer 进入背压，禁止无限制分配新 buffer。超时包含 `run_id`、epoch、序号和最近 transport 状态，控制面据此广播统一失败。

## 安全边界

当前本地实现没有监听地址、远端身份或证书边界。它仅用于在单一进程中验证确定性协议契约。
