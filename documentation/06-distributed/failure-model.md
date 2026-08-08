# 故障模型

## 本地故障分类

- 输入故障：空 world、长度不一致或 checkpoint 不能解码。
- 本地传输故障：run、epoch、sequence、checksum 不匹配或调用超时。
- 本地 checkpoint 故障：manifest 未提交、字段无效或恢复前 checksum 校验失败。

## 状态机

```text
Ready -> Running -> Suspect -> Draining -> Recovering -> Rejoining -> Running
                 \-> Failed -> Aborted
```

该状态机是未来扩展的设计草图，并非当前本地 crate 的运行时状态机。当前错误通过同步返回值报告，不启动重试、重连、成员变更或远程恢复。

## 一致失败

当前本地实现没有 rank 间广播或 future 协调。调用方必须处理每个本地操作返回的确定性错误。

## 重试边界

本地操作不会建立连接或提交设备操作。恢复前必须调用 manifest 校验；校验失败时不得解码或继续恢复。

## 可诊断性

每次故障记录 rank、epoch、step、collective_seq、tensor id、transport、设备事件、最近 checkpoint 和所有 rank 的最后进度。诊断记录写入本地不可变事件文件，并可由 `tt debug` 汇总。
