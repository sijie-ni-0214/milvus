# 本地 `milvus.yaml` 与 Milvus 官方 `master` 的完整差异

生成日期：2026-08-20

## 比较基线

- 本地：当前工作区的 `configs/milvus.yaml`（分支 `codex/load-1m-segments-pr-stack-rebased-qv-work`，HEAD `cfeb0c7b0fdd037460f29cbf242fc35273610677`）。
- 官方：[`milvus-io/milvus@ea48d3dd`](https://github.com/milvus-io/milvus/blob/ea48d3dd4b2b0f1177c7780538a647e85a9e6d45/configs/milvus.yaml)，即比较时官方 `master` 的最新提交。
- 比较方式：解析 YAML 后按叶子路径进行语义比较；注释、键顺序、空行和排版差异不计入下表。
- `—` 表示该 YAML 中不存在对应路径，不代表程序代码中一定没有默认值或兼容逻辑。

## 汇总

| 项目 | 数量 |
| --- | ---: |
| 本地叶子路径 | 854 |
| 官方叶子路径 | 851 |
| 仅本地存在 | 24 |
| 仅官方存在 | 21 |
| 两边都有但值不同 | 45 |
| 语义差异合计 | 90 |

## 安全说明

凭据、服务地址、存储位置、bucket、root path、region 等可能暴露环境信息的值已脱敏。差异判断仍基于原始值；`<空>` 表示原值为空，`<已脱敏：字符串>` 表示原值非空但未写入本文档。

## 仅本地存在（24 项）

| # | 配置路径 | 本地值 | 官方值 |
| ---: | --- | --- | --- |
| 1 | `common.requery.searchPolicy` | `"Always"` | — |
| 2 | `common.visibilityFilterEnabled` | `true` | — |
| 3 | `dataCoord.compaction.maxTaskNum` | `1000` | — |
| 4 | `dataCoord.compaction.maxTaskNumPerTrigger` | `500` | — |
| 5 | `dataCoord.jsonShreddingTriggerCount` | `10` | — |
| 6 | `dataCoord.jsonShreddingTriggerInterval` | `10` | — |
| 7 | `dataCoord.slot.backfillCompactionUsage` | `1` | — |
| 8 | `dataCoord.taskCheckInterval` | `1` | — |
| 9 | `dataNode.index.maxVecIndexBuildConcurrency` | `24` | — |
| 10 | `metastore.readConcurrency` | `120` | — |
| 11 | `minio.maxConnections` | `1000` | — |
| 12 | `proxy.maxBloomFilterPlanSize` | `134217728` | — |
| 13 | `proxy.maxBloomFilterSize` | `67108864` | — |
| 14 | `queryCoord.collectionBalanceSegmentBatchSize` | `100` | — |
| 15 | `queryCoord.enableSQNServeSegments` | `true` | — |
| 16 | `queryCoord.queryNodeTaskParallelismFactor` | `100` | — |
| 17 | `queryNode.idfOracle.lazyLoadSealedStats` | `false` | — |
| 18 | `queryNode.queryView.segmentCatchupConcurrency` | `4` | — |
| 19 | `queryNode.queryView.transformLogDrainConcurrency` | `4` | — |
| 20 | `queryNode.segcore.cgoPoolSizeRatio` | `2` | — |
| 21 | `queryNode.segcore.tieredStorage.lazyManifestReaderEnabled` | `false` | — |
| 22 | `streaming.flush.l1.commitConcurrency` | `4` | — |
| 23 | `streaming.queryView.liveEventDispatchConcurrencyPerPChannel` | `4` | — |
| 24 | `streaming.transformLog.catchupConcurrencyPerStream` | `4` | — |

## 仅官方存在（21 项）

| # | 配置路径 | 本地值 | 官方值 |
| ---: | --- | --- | --- |
| 25 | `common.fileResource.downloadTimeout` | — | `"5m"` |
| 26 | `common.fileResource.maxFileSize` | — | `0` |
| 27 | `common.fileResource.mode.proxy` | — | `"close"` |
| 28 | `dataCoord.compaction.enableTargetBasedCompaction` | — | `false` |
| 29 | `dataCoord.compaction.target.maxEventsPerReconcile` | — | `100` |
| 30 | `dataCoord.snapshot.crossBucketEndpointAllowlist` | — | <空> |
| 31 | `dataCoord.snapshot.exportCopyConcurrency` | — | `16` |
| 32 | `dataCoord.snapshot.exportJobRetention` | — | `10800` |
| 33 | `dataCoord.snapshot.exportJobTimeout` | — | `43200` |
| 34 | `dataCoord.snapshot.exportMaxConcurrentJobs` | — | `1` |
| 35 | `dataNode.import.copyObjectTimeout` | — | `3600` |
| 36 | `dataNode.import.writeRetryInitialInterval` | — | `1` |
| 37 | `dataNode.import.writeRetryMaxInterval` | — | `60` |
| 38 | `proxy.http.compatibilityMode` | — | `false` |
| 39 | `proxy.http.legacyArrayResponse` | — | `false` |
| 40 | `proxy.http.maxExprParamsDepth` | — | `100` |
| 41 | `proxy.http.nativeJSONResponse` | — | `true` |
| 42 | `proxy.maxMembershipFilterPlanSize` | — | `134217728` |
| 43 | `proxy.maxMembershipFilterSize` | — | `67108864` |
| 44 | `queryNode.fmindexCostRatio` | — | `0.001` |
| 45 | `queryNode.standaloneMigrateDataTimeout` | — | `"10s"` |

## 两边都有但值不同（45 项）

| # | 配置路径 | 本地值 | 官方值 |
| ---: | --- | --- | --- |
| 46 | `common.storage.enableGrowingSourceFlush` | `true` | `false` |
| 47 | `common.storage.useLoonFFI` | `true` | `false` |
| 48 | `common.threadCoreCoefficient.highPriority` | `100` | `10` |
| 49 | `common.threadCoreCoefficient.middlePriority` | `100` | `5` |
| 50 | `dataCoord.compaction.enableAutoCompaction` | `false` | `true` |
| 51 | `dataCoord.enableCompaction` | `false` | `true` |
| 52 | `dataCoord.segment.maxIdleTime` | `6000000` | `600` |
| 53 | `dataCoord.segment.minSizeFromIdleToSealed` | `1600` | `16` |
| 54 | `dataCoord.segment.sealProportion` | `0.99` | `0.12` |
| 55 | `dataCoord.slot.indexTaskSlotUsage` | `32` | `64` |
| 56 | `dataNode.dataSync.maxParallelSyncMgrTasksPerCPUCore` | `100` | `16` |
| 57 | `etcd.endpoints` | <已脱敏：字符串> | <已脱敏：字符串> |
| 58 | `etcd.requestTimeout` | `120000` | `10000` |
| 59 | `etcd.rootPath` | <已脱敏：字符串> | <已脱敏：字符串> |
| 60 | `indexNode.scheduler.buildParallel` | `2` | `1` |
| 61 | `localStorage.path` | <已脱敏：字符串> | <已脱敏：字符串> |
| 62 | `minio.accessKeyID` | <空> | <已脱敏：字符串> |
| 63 | `minio.address` | <已脱敏：字符串> | <已脱敏：字符串> |
| 64 | `minio.bucketName` | <已脱敏：字符串> | <已脱敏：字符串> |
| 65 | `minio.port` | `443` | `9000` |
| 66 | `minio.region` | <已脱敏：字符串> | <空> |
| 67 | `minio.rootPath` | <已脱敏：字符串> | <已脱敏：字符串> |
| 68 | `minio.secretAccessKey` | <空> | <已脱敏：字符串> |
| 69 | `minio.useIAM` | `true` | `false` |
| 70 | `minio.useSSL` | `true` | `false` |
| 71 | `proxy.ddlConcurrency` | `32` | `16` |
| 72 | `pulsar.address` | <已脱敏：字符串> | <已脱敏：字符串> |
| 73 | `queryCoord.channelTaskTimeout` | `60000` | `120000` |
| 74 | `queryCoord.loadTimeoutSeconds` | `36000` | `600` |
| 75 | `queryCoord.segmentTaskTimeout` | `120000` | `300000` |
| 76 | `queryNode.mmap.scalarField` | `true` | `false` |
| 77 | `queryNode.mmap.scalarIndex` | `true` | `false` |
| 78 | `queryNode.mmap.vectorIndex` | `true` | `false` |
| 79 | `queryNode.scheduler.maxReadConcurrentRatio` | `3` | `1` |
| 80 | `queryNode.segcore.storageV2.cellTargetSizeBytes` | `1048576` | `4194304` |
| 81 | `queryNode.segcore.tieredStorage.warmup.scalarField` | `"disable"` | `"sync"` |
| 82 | `queryNode.segcore.tieredStorage.warmup.scalarIndex` | `"disable"` | `"sync"` |
| 83 | `quotaAndLimits.enabled` | `false` | `true` |
| 84 | `quotaAndLimits.limits.maxCollectionNum` | `10000000` | `65536` |
| 85 | `quotaAndLimits.limits.maxCollectionNumPerDB` | `10000000` | `65536` |
| 86 | `rootCoord.maxGeneralCapacity` | `2000002` | `65536` |
| 87 | `rootCoord.maxPartitionNum` | `100002` | `1024` |
| 88 | `streaming.flush.growingSegmentBytesHwmThreshold` | `0.7` | `0.2` |
| 89 | `streaming.flush.growingSegmentBytesLwmThreshold` | `0.69` | `0.1` |
| 90 | `streaming.flush.memoryThreshold` | `0.99` | `0.6` |
