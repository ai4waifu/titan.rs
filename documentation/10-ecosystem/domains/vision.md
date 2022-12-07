# 视觉领域

## 范围

覆盖图像、视频和空间视觉数据，支持分类、检测、分割、姿态估计和视觉 Transformer。主要部署路径为 Native batch/streaming 服务和边缘设备。

## 数据管线

数据 schema 记录图像尺寸、通道、颜色空间、帧率、标注坐标系和 augmentation seed。读取、解码、resize、crop、normalize、颜色变换和 batch 由 DataLoader graph 表达，CPU/GPU 预处理必须保持约定容差。

## 基础能力

- Conv、pool、normalization、activation、patch embedding 和 attention。
- NCHW/NHWC/blocked layout 与 layout-aware kernel。
- 变长图像的 padding/mask 和视频时序维度。
- detection box、mask、keypoint 的结构化输出与后处理。

## 完整流程

训练产物包含类别/标注 schema、预处理、模型权重、optimizer、RNG 和评估配置。部署 manifest 固定输入像素契约、动态 batch/shape 范围、后处理和输出坐标系。

## 验收

测试覆盖解码/预处理 golden、augmentation 确定性、算子数值、训练恢复、动态图 shape、导出等价和 Native 性能。基准报告吞吐、单样本 p95、峰值显存、预处理占比和 batch 扩展曲线。
