# 强化学习领域

## 范围

覆盖环境接口、向量化环境、并行采样、经验回放、DQN、PPO、SAC、TD3、Actor-Learner、离线数据集和策略部署。

## 环境协议

`Environment` 定义 observation/action schema、reset、step、termination、truncation、reward 和 info。每个 environment instance 有独立 RNG 和 episode id；vectorized wrapper 保留每个实例的结束状态。

## 采样与回放

Trajectory 记录 policy version、environment seed、observation/action/reward、logprob、value 和 done。Replay Buffer 支持容量、优先级、采样种子、分片、快照和一致性恢复；过期 policy 数据按算法规则过滤。

## Actor-Learner

Actor 只消费已提交 policy version，Learner 通过 artifact 协议发布新权重。队列背压限制在途 trajectory，故障恢复同时恢复 learner、replay、actor cursor 和 RNG。

## 验收

测试覆盖环境确定性、episode 边界、GAE/return、replay 分布、policy version 和恢复。基准报告环境 steps/s、inference latency、queue wait、learner throughput、样本新鲜度和多 actor 扩展效率。
