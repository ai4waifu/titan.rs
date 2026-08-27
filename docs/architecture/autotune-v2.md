# Autotune v2

`.tune` v2 只接受 exact-shape TuneKey。key 必须包含 schema/semantic hash、shape、stride、layout、dtype、strategy version、workspace policy、device fingerprint、precision、determinism 和 compiler/artifact identity。

候选顺序固定为：compile、capability、workspace、ABI、correctness、benchmark、selection。任何失败候选必须产生 rejection record，不能隐式回退。

GPU benchmark 只能使用 stream event；首次编译、upload、download 与全设备同步不得计入 kernel 时间。winner 先按 median 选择，2% 内按 p95、workspace、artifact size 打破平局。预算耗尽时运行已验证 generated baseline，并标记 provisional。

持久化采用 header version 2、canonical JSONL、逐记录 checksum、exclusive lock、临时文件、fsync、atomic rename 和父目录 fsync。cache hit 不得重新 compile 或 benchmark。
