# Kernel 与编译缓存

## CacheKey

缓存键包含 kernel schema、kernel source hash、ABI version、graph operator、shape substitution、dtype、layout、device fingerprint、driver、编译器版本、launch config、precision policy 和 determinism policy。

## 缓存层级

- 进程内：最近使用的已加载 kernel。
- 本地磁盘：编译结果、日志、source map 和 checksum。
- 团队缓存：经过签名的跨任务共享 binary。
- 对象存储：发布版本和可回滚 artifact。

未签名、checksum 不符、能力不足、版本不兼容或编译选项不匹配的缓存不得加载。

## 失效

以下变化强制失效：kernel source、ABI、driver、device capability、dtype/layout、编译器、launch 约束、数值策略和 runtime major version。失效必须记录原因，不能静默重新编译。
