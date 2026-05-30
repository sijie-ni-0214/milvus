# QN 恢复加载并发排查记录

## 背景

- 环境：4am，实例 `yanliang-10kcp-5`，namespace `qa-milvus`。
- 数据集：1 个 collection，单 shard，约 10 万 partition，每个 partition 1 到 2 个 sealed segment，总 segment 数约 17 万。
- 目标：QN 恢复时尽量提高 segment load 并发，直到 QN 或 SN CPU 能被打满，定位当前并发上不去的瓶颈。
- 当前测试入口：重启 QueryNode 后观察恢复加载曲线、QN/SN CPU、QueryCoord dispatch、SN delegator、QN worker load 和本地 LoadPool/cgo/storage pool。

## 当前部署配置

- 镜像：`harbor.milvus.io/milvusdb/milvus:optimize-load-partition-20260529-d4bb170b3d-v2-amd64`
- QN：1 副本，limit 32C/64Gi，request 16C/32Gi。
- SN：1 副本，limit 16C/48Gi，request 8C/24Gi。
- QC：
  - `queryNodeTaskParallelismFactor: 200`
  - `taskExecutionCap: 360`
  - `dispatchScanBudgetFactor: 8`
  - `checkSegmentInterval: 3000`
  - `balanceSegmentBatchSize: 1000000`
- QN：
  - `workerPooling.size: 50`
  - `delegatorPostLoadConcurrencyFactor: 20`
  - `mmap.vectorField/vectorIndex/scalarField/scalarIndex: true`
  - `warmup.vectorField/vectorIndex/scalarField/scalarIndex: disable`
  - `tieredStorage.evictionEnabled: false`
- common:
  - `threadCoreCoefficient.highPriority: 50`
  - `threadCoreCoefficient.middlePriority: 50`
  - `threadCoreCoefficient.lowPriority: 1`
  - `threadCoreCoefficient.maxThreadsSize: 2000`

## 已知代码改动含义

- `d4bb170b3d enhance: reduce segment load resource lock contention`
  - 将 segment load resource 预估移出 `segmentLoader.mut` 临界区，减少多个 LoadSegments RPC 同时进入 `requestResource()` 时的串行等待。
- `555361e8c1 enhance: skip stale segment load tasks earlier`
  - 在 QC task add/promote 阶段检查 segment 是否已经出现在 dist。
  - 若 segment 已加载完成但旧 grow task 仍残留在队列中，则提前 cancel/remove，减少恢复后期对 wait/process queue 的重复扫描和 promote 成本。
- 当前 worktree 还有未提交改动：在 `requestResource()` 上补充 5 秒聚合的 estimate/lock wait/lock hold/total 耗时日志，尚未进入当前部署镜像。

## 排查假设

1. 若 QN `LoadPool` active 明显低于 capacity，且 cgo/storage pool active 也低，瓶颈更可能在 QC dispatch、SN delegator post-load、gRPC worker 并发或 QN requestResource 串行段。
2. 若 `LoadPool` active 高但 cgo/storage active 低，瓶颈可能在 Go 层 load skeleton、manifest/index 注册、PK/BM25/delta 处理或锁。
3. 若 cgo/storage pool active 高且 CPU 仍低，可能是对象存储 IO、mmap page fault、文件系统/MinIO、或大量小 segment 带来的调度/序列化开销。
4. 若 QC dispatch 已能持续提交大量 task，但 QN active 卡住，则优先看 SN -> QN worker RPC 并发、QN `requestResource()` 和 delegator post-load 限制。

## 2026-05-30 Baseline 前状态

- 集群 Healthy，QN 当前 pod `yanliang-10kcp-5-milvus-querynode-1-75c498dcdf-lfww2` 已运行约 25h，当前不是恢复高峰。
- QN metrics 显示已加载 sealed segment 数：`170434`。
- SN metrics 显示自身也有 load 侧指标：`internal_cgo_execute_duration_seconds_count{pool="load"}=44186`，说明 streaming node 侧也参与了部分 segment load/serve 路径，需要后续同时监控 QN 和 SN。
- 当前静态容量：
  - QN `LoadPool` capacity：`1600`
  - cgo load pool size：`1600`
  - storage high/middle pool capacity：`1600/1600`
  - `BM25LoadPool` capacity：`32`
- 当前空闲态 CPU：QN 约 `11m`，SN 约 `210m`，mixcoord 约 `472m`。

## 下一步

1. 使用当前镜像重启 QN，采样恢复阶段 metrics baseline。
2. 若恢复时 QN active 仍只有几十到一百左右，补充或部署 requestResource/LoadSegments 聚合日志镜像。
3. 基于证据调大或拆除对应限制点，例如 worker pooling、post-load semaphore、requestResource 串行段、BM25/PK/delta 阶段并发，或将 QC dispatch 与 QN load 能力解耦。

## 2026-05-30 Baseline：当前镜像重启 QN

操作：

- `kubectl apply -f /Users/zilliz/.kube/chaos_querynode_pod_kill.yaml`
- 立即 `kubectl delete -f /Users/zilliz/.kube/chaos_querynode_pod_kill.yaml`
- 每 5 秒采样 QN metrics：sealed segment 数、`LoadPool` active/queue、cgo load executing/inflight、storage high/middle active、pod CPU/memory。

结果：

- QN restart count 从 3 变为 4。
- QN sealed segment 从首次出现 `14520` 到恢复到 `170434`，大约 `153s`。
- QN CPU 峰值约 `20C`，未打满 32C。
- QN `LoadPool` capacity 为 `1600`，恢复期 active 主要在 `50~120`，queue 始终为 `0`。
- cgo load pool size 为 `1600`，恢复期多数采样点 `executing=0`，只有少数突发点在 `27~91`。
- storage high/middle active 也只在突发点出现几十到一百多，未持续占满。
- 结论：当前瓶颈不是 QN `LoadPool` 容量，也不是 cgo/storage pool 容量。上游没有持续喂满本地 load pool，或者每个 LoadSegments RPC 大量时间花在 Go/post-load/注册分布等非 cgo 阶段。

异常：

- QN metrics 已显示 `170434` sealed segment loaded 后，QC metrics 仍长期显示：
  - `milvus_querycoord_task_num{querycoord_task_type="segment_grow"} = 11169`
  - `milvus_querycoord_task_num{querycoord_task_type="leader_grow"} = 1110`
- 这些值至少 90 秒内没有自然下降。需要继续确认它们是真残留任务，还是 metrics 更新/节点维度导致的滞后。

## 2026-05-30 新增观测代码

目的：在 `log.level=warn` 下也能看到聚合耗时，不打开 Info 级 per-segment 日志。

新增：

- QC scheduler 侧 `scheduler dispatch timing stats`，每 5 秒聚合输出：
  - dispatch 调用次数
  - 平均 scanned/toProcess/committed/toRemove task 数
  - promote/preprocess/process/dispatch 总耗时
  - 最新 process/wait queue 长度和 segment/channel task 总数
- QN worker 侧 `segment load timing stats`，每 5 秒聚合输出：
  - `avg/max LoadSegment`
  - `avg/max LoadDeltaLogs`
  - `avg/max PK candidate`
  - `avg/max manager.Put`
  - `avg/max notifyLoadFinish`
  - `LoadPool cap/running/waiting`
- QN `request resource timing stats`，每 5 秒聚合输出：
  - resource estimate 耗时
  - `segmentLoader.mut` lock wait/hold
  - requestResource 总耗时
- SN delegator 侧 `delegator load segments timing stats`，每 5 秒聚合输出：
  - `worker.LoadSegments` RPC 耗时
  - post-load 总耗时
  - post-load 中 `LoadBloomFilterSet`、`loadBM25Stats`、`loadStreamDelete`
  - post-load 其他耗时约等于 semaphore wait + 轻量逻辑开销
  - SN delegator 当前活跃 LoadSegments request 数
  - post-load semaphore cap/current

本地校验：

- `git diff --check` 通过。
- `go test ./internal/querynodev2/segments`：本地链接阶段失败，原因是 `libmilvus_core.dylib` 缺少 `LC_RPATH`，Go 编译已通过到 test binary 启动阶段。
- `go test ./internal/querynodev2/delegator`：当前 worktree 下失败于既有测试依赖 `undefined: streaming.SetupNoopWALForTest`，不是本次日志代码引入的编译错误。

提交：

- `099dd902d0 enhance: add qn recovery load timing logs`
- 已推送到 `origin/optimize-load-partition`。
- `fe4a62d6fd enhance: add qc scheduler dispatch timing logs`
- 已推送到 `origin/optimize-load-partition`。

镜像构建：

- 已触发社区镜像构建，buildRecordId：`4404`。
- 目标 tag：`harbor.milvus.io/milvusdb/milvus:optimize-load-partition-20260530-099dd902d0-amd64`。
- 构建分支：`sijie-ni-0214/milvus:optimize-load-partition`。
- 已触发包含 QC scheduler 日志的第二个构建，buildRecordId：`4405`。
- 新目标 tag：`harbor.milvus.io/milvusdb/milvus:optimize-load-partition-20260530-fe4a62d6fd-amd64`。
- 后续测试以 `fe4a62d6fd` 镜像为准。
