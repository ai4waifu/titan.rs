# tt 工具链

`titan-tools` 编译出唯一命令 `tt.exe`。它面向 debug 和集群运维，业务代码不依赖该 crate；Rust 用户直接使用 Titan 库的公共 API。命令行解析依赖 `clap` 只存在于 `titan-tools`。

## `tt debug`

`tt debug --output <path>` 只读取本地运行目录，输出固定的 artifact 状态。若 `checkpoint.titan` 存在，还会输出其确定性 checksum 和 `checkpoint.recovery=ready`；缺失时输出 `checkpoint.recovery=unavailable`。该命令不写入运行目录，也不访问远端服务。

输出是本地诊断描述，不解析或修改未实现的遥测、集群或存储后端。

## `tt cluster`

`tt cluster --nodes <n> --rank <r>` 仅输出本地确定性拓扑描述：rank、world 和 `cluster.transport=local`。它不连接控制面、不接受 endpoint，也不探测远端设备或存储。

## 退出码

`0` 表示通过，`2` 表示参数或配置错误，`3` 表示发现可修复问题，`4` 表示数据一致性失败，`5` 表示连接或权限失败，`6` 表示内部错误。退出码是公共契约，变更需同步文档和测试。
