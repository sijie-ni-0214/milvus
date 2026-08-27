# 当前 `milvus.yaml` 与 Milvus 官方 `master` 配置差异

生成日期：2026-08-25（UTC）

## 结论

- 排除敏感配置路径后，当前配置纳入比较的叶子路径有 832 个，官方 `master` 有 819 个。
- 共发现 70 项 YAML 语义差异：仅当前配置存在 26 项、仅 `master` 存在 13 项、同路径值不同 31 项。
- 差异最多的一级配置域是 `queryNode`（17 项）、`common`（12 项）、`rootCoord`（10 项）和 `proxy`（7 项）。
- 相比 `master`，当前配置呈现明显的大规模、高并发调优特征：collection/partition 容量上限更高，QueryNode/Proxy 线程与并发系数更激进，mmap 和 tiered-storage eviction 已开启。
- `proxy.maxBloomFilter*` 与 `master` 的 `proxy.maxMembershipFilter*` 是配置键升级：两组值相同，且 `master` 源码保留旧键 fallback；下方仍按叶子路径计为 4 项差异。

## 比较基线

- 当前配置：工作区磁盘上的 `configs/milvus.yaml`，分支 `codex/load-1m-segments-pr-stack-rebased-qv-work`，HEAD `7734adeba64588944cae0ece531436c3e62e0cb5`。
- 当前文件 SHA-256：`bfe0752a955224b139e031d1219f44a725d463e87eddf4c9d0e116b270a72f06`。该文件被标记为 `skip-worktree`，本报告直接读取磁盘内容，而不是依赖 `git status`。
- 官方基线：[`milvus-io/milvus@5f7f1f19`](https://github.com/milvus-io/milvus/blob/5f7f1f199f62d6f6977ae2324ff3cb847aca319b/configs/milvus.yaml)，即 2026-08-25 刷新后的 `upstream/master`。
- 未采用 `origin/master`：该引用仍停留在 2022 年，不能代表当前官方配置。
- 比较方式：先整体排除敏感配置路径，再解析 YAML 并按叶子路径比较；注释、键顺序、空行和排版差异不计入统计。被排除的路径不出现在统计、结论或明细中。

> 本文比较的是两个 YAML 文件，不是运行中进程的最终生效配置。路径缺失只表示文件中未显式配置；程序仍可能使用源码默认值、兼容旧键，或受到环境变量、etcd 动态配置和启动参数影响。

## 汇总

| 项目 | 数量 |
| --- | ---: |
| 当前配置非敏感叶子路径 | 832 |
| `master` 配置非敏感叶子路径 | 819 |
| 仅当前配置存在 | 26 |
| 仅 `master` 配置存在 | 13 |
| 两边都有但值不同 | 31 |
| 语义差异合计 | 70 |

### 按一级配置域统计

| 配置域 | 仅当前 | 仅 `master` | 值不同 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| `common` | 5 | 2 | 5 | 12 |
| `dataCoord` | 4 | 0 | 0 | 4 |
| `dataNode` | 1 | 0 | 0 | 1 |
| `etcd` | 0 | 0 | 2 | 2 |
| `indexNode` | 0 | 0 | 1 | 1 |
| `metastore` | 1 | 0 | 0 | 1 |
| `minio` | 1 | 0 | 3 | 4 |
| `proxy` | 2 | 3 | 2 | 7 |
| `queryCoord` | 3 | 0 | 3 | 6 |
| `queryNode` | 6 | 0 | 11 | 17 |
| `quotaAndLimits` | 0 | 0 | 2 | 2 |
| `rootCoord` | 0 | 8 | 2 | 10 |
| `streaming` | 3 | 0 | 0 | 3 |
| **合计** | **26** | **13** | **31** | **70** |

## 重点差异

### 1. 容量上限

| 配置 | 当前值 | `master` 值 |
| --- | ---: | ---: |
| `quotaAndLimits.limits.maxCollectionNum` | 10,000,000 | 65,536 |
| `quotaAndLimits.limits.maxCollectionNumPerDB` | 10,000,000 | 65,536 |
| `rootCoord.maxGeneralCapacity` | 2,000,002 | 65,536 |
| `rootCoord.maxPartitionNum` | 100,002 | 1,024 |

当前文件放宽了 collection、单 DB collection、综合容量和 partition 上限。这里只确认配置值，不代表已完成对应规模的运行验证。

### 2. 并发与线程

- `common.threadCoreCoefficient.highPriority/middlePriority/maxThreadsSize` 分别为 `40/60/320`，`master` 为 `10/5/16`。
- `proxy.ddlConcurrency` 为 64，`master` 为 16。
- `queryNode.delegatorPostLoadConcurrencyFactor`、`queryNode.scheduler.maxReadConcurrentRatio`、`queryNode.segcore.dynamicPoolSizeFactor` 都是 40，`master` 都是 1。
- 当前文件还显式增加了多项仅本分支存在的并发参数，包括 QueryView segment catch-up、TransformLog drain/live dispatch/catch-up（均为 320）、QueryCoord task parallelism（100）和 NodeScheduler ratio（100）。

这些值整体上提高了并发上限或线程池扩张倾向，也可能放大 CPU、内存、对象存储和下游服务压力。

### 3. 查询数据路径与存储策略

- 当前配置启用 `common.storage.enableGrowingSourceFlush` 和 `common.storage.useLoonFFI`；`master` 均关闭。
- 当前配置启用 scalar field、scalar index、vector index 的 mmap；`master` 均关闭。
- 当前配置启用 tiered-storage eviction 和 background eviction；`master` 均关闭。
- 当前配置将 scalar field/index/vector index warmup 全设为 `disable`；`master` 全为 `sync`。

因此当前文件更依赖 mmap/tiered storage，减少同步 warmup，但行为和资源曲线与 `master` 默认配置明显不同。

### 4. 外部依赖与安全连接

- etcd：当前 `auth.enabled=false`，`master=true`；请求超时为 120,000，`master` 为 10,000。
- MinIO/S3：当前使用 IAM、SSL 和 443 端口；`master` 为非 IAM、非 SSL 和 9000 端口。

这些差异属于部署环境绑定项，不能直接用 `master` 值覆盖当前环境。

### 5. 超时与 HTTP 兼容行为

- `queryCoord.loadTimeoutSeconds`：36,000 对 600，当前允许更长的 load 操作。
- `queryCoord.channelTaskTimeout`：60,000 对 120,000；`segmentTaskTimeout`：120,000 对 300,000，当前两类 task timeout 反而更短。
- `proxy.http.nativeJSONResponse`：当前为 `false`，`master` 为 `true`；直接同步配置可能改变 HTTP JSON 响应形式。
- `proxy.http.legacyArrayResponse` 仅 `master` 显式存在且为 `false`。

### 6. 配置演进

- 当前的 `proxy.maxBloomFilterSize=67108864` 对应 `master` 的 `proxy.maxMembershipFilterSize=67108864`。
- 当前的 `proxy.maxBloomFilterPlanSize=134217728` 对应 `master` 的 `proxy.maxMembershipFilterPlanSize=134217728`。
- `master` 源码对两个旧键保留 fallback，因此这是命名和适用范围扩展，不是数值变化。
- `master` 新增了 2 个 `common.storage.iops.*` 配置和 8 个 `rootCoord.clientTelemetry.*` 配置；当前 YAML 未显式设置。
- 当前 YAML 独有的 `common.metrics.collectionLevelMode="full"` 为启动时生效的进程级指标模式，保留真实 collection/VChannel 标签；官方 `master` YAML 当前无此键。

## 仅当前配置存在（26 项）

| # | 配置路径 | 当前值 | `master` 值 |
| ---: | --- | --- | --- |
| 1 | `common.arrow.ioThreadPoolCoefficient` | `40` | — |
| 2 | `common.arrow.ioThreadPoolMaxCapacity` | `320` | — |
| 3 | `common.metrics.collectionLevelMode` | `"full"` | — |
| 4 | `common.nodeScheduler.maxConcurrencyRatio` | `100` | — |
| 5 | `common.requery.searchPolicy` | `"Always"` | — |
| 6 | `dataCoord.jsonShreddingTriggerCount` | `10` | — |
| 7 | `dataCoord.jsonShreddingTriggerInterval` | `10` | — |
| 8 | `dataCoord.slot.backfillCompactionUsage` | `1` | — |
| 9 | `dataCoord.taskCheckInterval` | `1` | — |
| 10 | `dataNode.index.maxVecIndexBuildConcurrency` | `24` | — |
| 11 | `metastore.readConcurrency` | `120` | — |
| 12 | `minio.maxConnections` | `1000` | — |
| 13 | `proxy.maxBloomFilterPlanSize` | `134217728` | — |
| 14 | `proxy.maxBloomFilterSize` | `67108864` | — |
| 15 | `queryCoord.collectionBalanceSegmentBatchSize` | `100` | — |
| 16 | `queryCoord.enableSQNServeSegments` | `true` | — |
| 17 | `queryCoord.queryNodeTaskParallelismFactor` | `100` | — |
| 18 | `queryNode.idfOracle.lazyLoadSealedStats` | `false` | — |
| 19 | `queryNode.idfOracle.sealedStatsLoadConcurrencyRatio` | `40` | — |
| 20 | `queryNode.queryView.segmentCatchupConcurrency` | `320` | — |
| 21 | `queryNode.queryView.transformLogDrainConcurrency` | `320` | — |
| 22 | `queryNode.segcore.cgoPoolSizeRatio` | `2` | — |
| 23 | `queryNode.segcore.tieredStorage.lazyManifestReaderEnabled` | `false` | — |
| 24 | `streaming.flush.l1.commitConcurrency` | `4` | — |
| 25 | `streaming.queryView.liveEventDispatchConcurrencyPerPChannel` | `320` | — |
| 26 | `streaming.transformLog.catchupConcurrencyPerStream` | `320` | — |

## 仅 `master` 配置存在（13 项）

| # | 配置路径 | 当前值 | `master` 值 |
| ---: | --- | --- | --- |
| 27 | `common.storage.iops.initialRate` | — | `2000` |
| 28 | `common.storage.iops.maxRate` | — | `5000` |
| 29 | `proxy.http.legacyArrayResponse` | — | `false` |
| 30 | `proxy.maxMembershipFilterPlanSize` | — | `134217728` |
| 31 | `proxy.maxMembershipFilterSize` | — | `67108864` |
| 32 | `rootCoord.clientTelemetry.cleanupInterval` | — | `60` |
| 33 | `rootCoord.clientTelemetry.clientStatusThreshold` | — | `60` |
| 34 | `rootCoord.clientTelemetry.commandCleanupTimeout` | — | `10` |
| 35 | `rootCoord.clientTelemetry.inactiveClientThreshold` | — | `600` |
| 36 | `rootCoord.clientTelemetry.maxClientsInMemory` | — | `100000` |
| 37 | `rootCoord.clientTelemetry.maxMetricsPerClient` | — | `1048576` |
| 38 | `rootCoord.clientTelemetry.maxOperationTypesPerClient` | — | `100` |
| 39 | `rootCoord.clientTelemetry.retainedWindows` | — | `2` |

## 两边都有但值不同（31 项）

| # | 配置路径 | 当前值 | `master` 值 |
| ---: | --- | --- | --- |
| 40 | `common.storage.enableGrowingSourceFlush` | `true` | `false` |
| 41 | `common.storage.useLoonFFI` | `true` | `false` |
| 42 | `common.threadCoreCoefficient.highPriority` | `40` | `10` |
| 43 | `common.threadCoreCoefficient.maxThreadsSize` | `320` | `16` |
| 44 | `common.threadCoreCoefficient.middlePriority` | `60` | `5` |
| 45 | `etcd.auth.enabled` | `false` | `true` |
| 46 | `etcd.requestTimeout` | `120000` | `10000` |
| 47 | `indexNode.scheduler.buildParallel` | `2` | `1` |
| 48 | `minio.port` | `443` | `9000` |
| 49 | `minio.useIAM` | `true` | `false` |
| 50 | `minio.useSSL` | `true` | `false` |
| 51 | `proxy.ddlConcurrency` | `64` | `16` |
| 52 | `proxy.http.nativeJSONResponse` | `false` | `true` |
| 53 | `queryCoord.channelTaskTimeout` | `60000` | `120000` |
| 54 | `queryCoord.loadTimeoutSeconds` | `36000` | `600` |
| 55 | `queryCoord.segmentTaskTimeout` | `120000` | `300000` |
| 56 | `queryNode.delegatorPostLoadConcurrencyFactor` | `40` | `1` |
| 57 | `queryNode.mmap.scalarField` | `true` | `false` |
| 58 | `queryNode.mmap.scalarIndex` | `true` | `false` |
| 59 | `queryNode.mmap.vectorIndex` | `true` | `false` |
| 60 | `queryNode.scheduler.maxReadConcurrentRatio` | `40` | `1` |
| 61 | `queryNode.segcore.dynamicPoolSizeFactor` | `40` | `1` |
| 62 | `queryNode.segcore.tieredStorage.backgroundEvictionEnabled` | `true` | `false` |
| 63 | `queryNode.segcore.tieredStorage.evictionEnabled` | `true` | `false` |
| 64 | `queryNode.segcore.tieredStorage.warmup.scalarField` | `"disable"` | `"sync"` |
| 65 | `queryNode.segcore.tieredStorage.warmup.scalarIndex` | `"disable"` | `"sync"` |
| 66 | `queryNode.segcore.tieredStorage.warmup.vectorIndex` | `"disable"` | `"sync"` |
| 67 | `quotaAndLimits.limits.maxCollectionNum` | `10000000` | `65536` |
| 68 | `quotaAndLimits.limits.maxCollectionNumPerDB` | `10000000` | `65536` |
| 69 | `rootCoord.maxGeneralCapacity` | `2000002` | `65536` |
| 70 | `rootCoord.maxPartitionNum` | `100002` | `1024` |
