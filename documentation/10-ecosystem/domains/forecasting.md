# 时间序列领域

## 范围

覆盖规则与不规则采样、单/多变量、多步预测、异常检测、在线更新和预测区间；同时支持统计模型与神经模型。

## 数据语义

TimeSeries schema 固定 timezone、时间单位、频率、缺失值、重复 timestamp、feature/target、静态协变量和未来已知变量。切分严格按时间，禁止随机泄漏未来数据。

## 模型能力

- sliding window、lag、calendar feature、resample 和 mask。
- ARIMA、ETS、Kalman Filter 等状态模型。
- LSTM、TCN、Transformer 和统计/神经混合模型。
- point/quantile/distribution 输出、在线 state 和滚动更新。

## 评估

Backtest 记录 cutoffs、horizon、stride、训练窗口和数据版本；指标支持 MAE、RMSE、MAPE/sMAPE、MASE、pinball loss 和 interval coverage。

## 验收

测试覆盖时区/DST、不规则间隔、缺失、无泄漏切分、在线 state 恢复和预测区间。基准报告序列数/秒、horizon 延迟、state 内存、更新成本和不同窗口长度曲线。
