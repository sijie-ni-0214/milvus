# Collection / VChannel 级 Prometheus 指标审计

本文档记录对携带 `collection_id`、`collection_name` 或直接 VChannel 标签的
Prometheus 指标族所做的源码级审计。本文基于 2026-08-21 的当前工作树；审计时 HEAD 为
`03e3e16f3e45889ce5d4b94c18cc75980d2742f8`。结果以当时的工作树源码为准，并非
永久兼容性契约。指标声明、注册或写入点发生变化后，应重新执行审计。

基于本清单实现的 `full` / `aggregate` 降基数模式及全部 71 个指标的行为差异见
[Collection / VChannel 级 Prometheus 指标降基数模式](collection-level-metrics-mode.md)。

当前工作树已经从 `milvus_proxy_req_count` 中移除了 `db_name` 和
`collection_name`，因此该指标不计入下文清单。

## 审计范围与统计口径

审计覆盖已注册的 Go Prometheus collector 及其实际写入点，也覆盖经
`CRegistry` 合并到 `/metrics` 的 C++ Milvus core、Knowhere 和 jemalloc 导出链，
以及 milvus-storage 经 FFI 投影到 Go collector 的指标。如果一个指标族的可变
标签中明确包含 `collection_id` 或 `collection_name`，就将其计为 collection
级指标。VChannel 扩展口径则要求读取实际 writer，只有确认值为 VChannel 的
`channel_name` / `vchannel` / `shard` 才计入；同名 PChannel 标签不计入。

以下内容不计入：

- `*_collection_num` 等只统计 collection 数量、但不标识具体 collection 的指标。
- 日志字段、JSON metrics 接口、trace，以及没有直接作为 Prometheus label
  输出的 channel 值。
- C++ protobuf/config/business struct 中的 `collection_id` 或
  `collection_name` 字段；它们不是 Prometheus 标签。

Histogram 按一个指标族计数，尽管暴露时会展开为 `_bucket`、`_sum` 和
`_count` 多组时序。

## 汇总

| 组件 | 指标族数量 |
|---|---:|
| Proxy | 18 |
| RootCoord / MixCoord | 3 |
| DataCoord | 9 |
| DataNode | 7 |
| QueryCoord | 1 |
| QueryNode | 21 |
| QueryView（`qv`） | 1 |
| C++ 原生导出链（严格口径） | 0 |
| **合计** | **60** |

在这 60 个指标族中：

- 41 个包含 `collection_id`。
- 22 个包含 `collection_name`。
- 3 个同时包含两个标签。
- 59 个是 metric vector，1 个是 QueryView 自定义 collector。
- 包含 22 个 CounterVec、21 个 GaugeVec、16 个 HistogramVec；QueryView
  自定义 collector 输出 Gauge。
- 严格口径下 60 个指标族全部来自 Go collector；C++ 导出链没有显式
  `collection_id` 或 `collection_name` 标签。扩展到“通过 shard/channel 间接归因”
  时，C++ 侧另有 1 个 VChannel 指标族，不计入上述 60 个 collection 合计。

### VChannel 扩展口径

经声明和非测试 writer 逐项核对，共有 **15** 个指标族直接输出 VChannel；其中
4 个也在上述 60 个 collection 指标中，因此 `collection ∪ VChannel` 去重后共
**71** 个指标族。

| 组件 | VChannel 指标族数量 |
|---|---:|
| DataCoord | 2 |
| DataNode | 2 |
| QueryCoord | 3 |
| QueryNode | 6 |
| QueryView（`qv`） | 1 |
| C++ Milvus core caching layer | 1 |
| **合计** | **15** |

| 指标名 | VChannel 标签 | writer 语义 | 同时是 collection 指标 |
|---|---|---|---|
| `milvus_datacoord_channel_checkpoint_unix_seconds` | `channel_name` | DataCoord channel checkpoint map 的 key / `MsgPosition.ChannelName` | 否 |
| `milvus_datacoord_compaction_latency` | `channel_name` | clustering compaction task 的 `Channel` | 否 |
| `milvus_datanode_growing_source_sync_failure_count` | `channel_name` | write buffer 的 VChannel | 是 |
| `milvus_datanode_msg_dispatcher_tt_lag_ms` | `channel_name` | dispatcher target 的 `vchannel` | 否 |
| `milvus_querycoord_current_target_checkpoint_unix_seconds` | `channel_name` | current target 的 DML channel | 否 |
| `milvus_querycoord_current_target_all_replicas_checkpoint_unix_seconds` | `channel_name` | all-replica-ready target 的 DML channel | 否 |
| `milvus_querycoord_task_latency` | `channel_name` | scheduler task 的 `Shard()` | 是 |
| `milvus_querynode_growing_source_retained_bytes` | `channel_name` | delegator growing source 的 VChannel | 否 |
| `milvus_querynode_growing_source_retained_segments` | `channel_name` | delegator growing source 的 VChannel | 否 |
| `milvus_querynode_level_zero_size` | `channel_name` | shard delegator 的 VChannel | 是 |
| `milvus_querynode_msg_dispatcher_tt_lag_ms` | `channel_name` | dispatcher target 的 `vchannel` | 否 |
| `milvus_querynode_delete_buffer_size` | `channel_name` | shard delegator delete buffer 的 VChannel | 否 |
| `milvus_querynode_delete_buffer_row_num` | `channel_name` | shard delegator delete buffer 的 VChannel | 否 |
| `milvus_qv_view_state_max_age_seconds` | `vchannel` | QueryView `ShardID.VChannel` | 是 |
| `internal_cache_shard_disk_usage_bytes` | `shard` | segcore insert channel attribution | 否 |

`channel_name` 不能按名字统一处理：RootCoord、Proxy、Streaming Service / WAL 的
同名标签是 PChannel。`StreamingCoordVChannelTotal` 统计的是 VChannel 数量，但它的
标签仍是承载这些 VChannel 的 PChannel，也不属于直接 VChannel 标签。

## 完整清单

### Proxy

源码：[`pkg/metrics/proxy_metrics.go`](../../../pkg/metrics/proxy_metrics.go)

| 指标名 | 类型 | 完整可变标签 | 源码变量 |
|---|---|---|---|
| `milvus_proxy_received_nq` | Counter | `node_id`, `query_type`, `db_name`, `collection_name` | `ProxyReceivedNQ` |
| `milvus_proxy_search_vectors_count` | Counter | `node_id`, `db_name`, `collection_name` | `ProxySearchVectors` |
| `milvus_proxy_insert_vectors_count` | Counter | `node_id`, `db_name`, `collection_name` | `ProxyInsertVectors` |
| `milvus_proxy_upsert_vectors_count` | Counter | `node_id`, `db_name`, `collection_name` | `ProxyUpsertVectors` |
| `milvus_proxy_delete_vectors_count` | Counter | `node_id`, `db_name`, `collection_name` | `ProxyDeleteVectors` |
| `milvus_proxy_sq_latency` | Histogram | `node_id`, `query_type`, `db_name`, `collection_name` | `ProxySQLatency` |
| `milvus_proxy_collection_sq_latency` | Histogram | `node_id`, `query_type`, `db_name`, `collection_name` | `ProxyCollectionSQLatency`（已废弃） |
| `milvus_proxy_mutation_latency` | Histogram | `node_id`, `msg_type`, `db_name`, `collection_name` | `ProxyMutationLatency` |
| `milvus_proxy_collection_mutation_latency` | Histogram | `node_id`, `msg_type`, `db_name`, `collection_name` | `ProxyCollectionMutationLatency`（已废弃） |
| `milvus_proxy_receive_bytes_count` | Counter | `node_id`, `msg_type`, `db_name`, `collection_name` | `ProxyReceiveBytes` |
| `milvus_proxy_retry_search_cnt` | Counter | `node_id`, `query_type`, `db_name`, `collection_name` | `ProxyRetrySearchCount` |
| `milvus_proxy_retry_search_result_insufficient_cnt` | Counter | `node_id`, `query_type`, `db_name`, `collection_name` | `ProxyRetrySearchResultInsufficientCount` |
| `milvus_proxy_recall_search_cnt` | Counter | `node_id`, `query_type`, `db_name`, `collection_name` | `ProxyRecallSearchCount` |
| `milvus_proxy_search_sparse_num_non_zeros` | Histogram | `node_id`, `collection_name`, `query_type`, `field_id` | `ProxySearchSparseNumNonZeros` |
| `milvus_proxy_function_udf_call_latency` | Histogram | `node_id`, `collection_name`, `function_type_name`, `function_provider`, `function_name` | `ProxyFunctionlatency` |
| `milvus_proxy_scanned_remote_mb` | Counter | `node_id`, `msg_type`, `db_name`, `collection_name` | `ProxyScannedRemoteMB` |
| `milvus_proxy_scanned_total_mb` | Counter | `node_id`, `msg_type`, `db_name`, `collection_name` | `ProxyScannedTotalMB` |
| `milvus_proxy_limiter_rate` | Gauge | `node_id`, `collection_id`, `msg_type` | `ProxyLimiterRate`；`collection_id` 实际承载多层级 source ID |

### RootCoord / MixCoord

源码：
[`pkg/metrics/rootcoord_metrics.go`](../../../pkg/metrics/rootcoord_metrics.go)

| 指标名 | 类型 | 完整可变标签 | 源码变量 |
|---|---|---|---|
| `milvus_rootcoord_entity_num` | Gauge | `db_name`, `collection_name`, `status` | `RootCoordNumEntities` |
| `milvus_rootcoord_indexed_entity_num` | Gauge | `db_name`, `collection_name`, `index_name`, `is_vector_index` | `RootCoordIndexedNumEntities` |
| `milvus_rootcoord_rate_limit_ratio` | Gauge | `collection_id` | `RootCoordRateLimitRatio` |

### DataCoord

源码：
[`pkg/metrics/datacoord_metrics.go`](../../../pkg/metrics/datacoord_metrics.go)

| 指标名 | 类型 | 完整可变标签 | 源码变量 |
|---|---|---|---|
| `milvus_datacoord_store_level0_segment_size` | Histogram | `collection_id` | `DataCoordSizeStoredL0Segment` |
| `milvus_datacoord_l0_delete_entries_num` | Gauge | `db_name`, `collection_id` | `DataCoordL0DeleteEntriesNum` |
| `milvus_datacoord_stored_rows_num` | Gauge | `db_name`, `collection_id`, `collection_name`, `segment_state` | `DataCoordNumStoredRows` |
| `milvus_datacoord_bulk_insert_vectors_count` | Counter | `db_name`, `collection_id` | `DataCoordBulkVectors` |
| `milvus_datacoord_stored_binlog_size` | Gauge | `db_name`, `collection_id`, `segment_state` | `DataCoordStoredBinlogSize` |
| `milvus_datacoord_segment_binlog_file_count` | Gauge | `collection_id` | `DataCoordSegmentBinLogFileCount` |
| `milvus_datacoord_stored_index_files_size` | Gauge | `db_name`, `collection_name`, `collection_id` | `DataCoordStoredIndexFilesSize` |
| `milvus_datacoord_index_task_count` | Gauge | `collection_id`, `index_task_status` | `IndexTaskNum`（已废弃） |
| `milvus_datacoord_snapshot_active_pins` | Gauge | `collection_id`, `snapshot_name` | `DataCoordSnapshotActivePins` |

### DataNode

源码：
[`pkg/metrics/datanode_metrics.go`](../../../pkg/metrics/datanode_metrics.go)

| 指标名 | 类型 | 完整可变标签 | 源码变量 |
|---|---|---|---|
| `milvus_datanode_write_data_count` | Counter | `node_id`, `data_source`, `data_type`, `collection_id` | `DataNodeWriteDataCount` |
| `milvus_datanode_consume_tt_lag_ms` | Gauge | `node_id`, `msg_type`, `collection_id` | `DataNodeConsumeTimeTickLag` |
| `milvus_datanode_consume_msg_count` | Counter | `node_id`, `msg_type`, `collection_id` | `DataNodeConsumeMsgCount` |
| `milvus_datanode_growing_source_sync_failure_count` | Gauge | `node_id`, `collection_id`, `channel_name` | `DataNodeGrowingSourceSyncFailureCount` |
| `milvus_datanode_fg_buffer_size` | Gauge | `node_id`, `collection_id` | `DataNodeFlowGraphBufferDataSize` |
| `milvus_datanode_compaction_delete_count` | Counter | `collection_id` | `DataNodeCompactionDeleteCount` |
| `milvus_datanode_compaction_missing_delete_count` | Counter | `collection_id` | `DataNodeCompactionMissingDeleteCount` |

### QueryCoord

源码：
[`pkg/metrics/querycoord_metrics.go`](../../../pkg/metrics/querycoord_metrics.go)

| 指标名 | 类型 | 完整可变标签 | 源码变量 |
|---|---|---|---|
| `milvus_querycoord_task_latency` | Histogram | `collection_id`, `task_type`, `channel_name` | `QueryCoordTaskLatency` |

### QueryNode

源码：
[`pkg/metrics/querynode_metrics.go`](../../../pkg/metrics/querynode_metrics.go)

| 指标名 | 类型 | 完整可变标签 | 源码变量 |
|---|---|---|---|
| `milvus_querynode_consume_tt_lag_ms` | Gauge | `node_id`, `msg_type`, `collection_id` | `QueryNodeConsumeTimeTickLag` |
| `milvus_querynode_consume_msg_count` | Counter | `node_id`, `msg_type`, `collection_id` | `QueryNodeConsumerMsgCount` |
| `milvus_querynode_skipped_insert_field_count` | Counter | `node_id`, `collection_id` | `QueryNodeSkippedInsertFieldCount` |
| `milvus_querynode_segment_num` | Gauge | `node_id`, `collection_id`, `segment_state`, `segment_level` | `QueryNodeNumSegments` |
| `milvus_querynode_sq_req_count` | Counter | `node_id`, `query_type`, `status`, `scope`, `collection_id` | `QueryNodeSQCount` |
| `milvus_querynode_search_fts_num_tokens` | Histogram | `node_id`, `collection_id`, `field_id` | `QueryNodeSearchFTSNumTokens` |
| `milvus_querynode_search_hit_segment_num` | Histogram | `node_id`, `collection_id`, `query_type` | `QueryNodeSearchHitSegmentNum` |
| `milvus_querynode_segment_filter_hit_segment_num` | Histogram | `node_id`, `collection_id`, `query_type` | `QueryNodeSegmentFilterHitSegmentNum` |
| `milvus_querynode_segment_filter_skipped_segment_num` | Histogram | `node_id`, `collection_id`, `query_type` | `QueryNodeSegmentFilterSkippedSegmentNum` |
| `milvus_querynode_segment_filter_total_segment_num` | Histogram | `node_id`, `collection_id`, `query_type` | `QueryNodeSegmentFilterTotalSegmentNum` |
| `milvus_querynode_segment_prune_ratio` | Gauge | `node_id`, `collection_id`, `segment_prune_label` | `QueryNodeSegmentPruneRatio` |
| `milvus_querynode_segment_prune_bias` | Gauge | `node_id`, `collection_id`, `segment_prune_label` | `QueryNodeSegmentPruneBias` |
| `milvus_querynode_segment_prune_latency` | Histogram | `node_id`, `collection_id`, `segment_prune_label` | `QueryNodeSegmentPruneLatency` |
| `milvus_querynode_entity_num` | Gauge | `db_name`, `collection_name`, `node_id`, `collection_id`, `segment_state` | `QueryNodeNumEntities` |
| `milvus_querynode_entity_size` | Gauge | `node_id`, `collection_id`, `segment_state` | `QueryNodeEntitiesSize` |
| `milvus_querynode_level_zero_size` | Gauge | `node_id`, `collection_id`, `channel_name` | `QueryNodeLevelZeroSize` |
| `milvus_querynode_partial_result_count` | Counter | `node_id`, `query_type`, `collection_id` | `QueryNodePartialResultCount` |
| `milvus_querynode_two_stage_search_stage1_latency` | Histogram | `node_id`, `collection_id` | `QueryNodeTwoStageFilterLatency` |
| `milvus_querynode_two_stage_search_stage2_latency` | Histogram | `node_id`, `collection_id` | `QueryNodeTwoStageSearchLatency` |
| `milvus_querynode_two_stage_search_fallback_total` | Counter | `node_id`, `collection_id`, `reason` | `QueryNodeTwoStageSearchFallbackCount` |
| `milvus_querynode_global_refine_total` | Counter | `node_id`, `collection_id` | `QueryNodeGlobalRefineCount` |

### QueryView

源码：[`pkg/metrics/qv_metrics.go`](../../../pkg/metrics/qv_metrics.go)

| 指标名 | 类型 | 完整可变标签 | 源码变量 |
|---|---|---|---|
| `milvus_qv_view_state_max_age_seconds` | Gauge（自定义 collector） | `component`, `state`, `rank`, `collection_id`, `replica_id`, `vchannel`, `query_view_version`, `data_version` | `QVViewStateMaxAgeSeconds` |

该指标是 pull collector，而不是持久化 metric vector。其 provider 对每个
component 只输出状态持续时间最长的 5 个 view，因此尽管它会标识
collection，对外暴露的基数仍然有界。

### C++ 原生导出链

严格口径下，C++ 侧 collection 级指标族数量为 **0**。当前 `/metrics` 的原生
指标由 [`internal/util/metrics/c_registry.go`](../../../internal/util/metrics/c_registry.go)
合并：Knowhere registry 通过 `GetKnowhereMetrics()` 导出，Milvus core registry
通过 `GetCoreMetrics()` 导出，随后再加入 jemalloc 指标。审计结果如下：

| 导出源 | 最接近 collection 的指标或标签 | 类型/完整可变标签 | 判定 |
|---|---|---|---|
| Milvus core caching layer | `internal_cache_shard_disk_usage_bytes` | Gauge；`data_type`, `shard` | `shard` 来自 insert VChannel；计入 15 个 VChannel 扩展口径，不计入 60 个严格 collection 口径 |
| 其他 Milvus core 指标 | `type`, `status`, `pool`, `priority`, `module`, `location`, `data_type` 等 | Counter、Gauge、Histogram | 无 `collection_id` / `collection_name`，不计入 |
| Knowhere | `module`，部分 latency family 另有 `index_type` | Gauge、Histogram | 无 collection 标签，不计入 |
| milvus-storage | `milvus_storage_filesystem_*` | 8 个 Go Gauge；`fs` | C++/Rust FFI 只返回 filesystem 累计值，Go 侧按 filesystem key 发布，不计入 |
| jemalloc | `milvus_jemalloc_*` | 8 个 Gauge；无标签 | 进程级内存指标，不计入 |

`internal_cache_shard_disk_usage_bytes` 的完整链路是：

1. milvus-common caching layer 按 `{data_type, shard}` 动态创建 Gauge；声明可从
   构建依赖安装头 `internal/core/output/include/cachinglayer/Metrics.h` 核对。
2. [`CacheMetricAttribution.h`](../../../internal/core/src/segcore/CacheMetricAttribution.h)
   将 segcore 的 `shard` 直接作为 attribution；
   [`SegmentLoadInfo.cpp`](../../../internal/core/src/segcore/SegmentLoadInfo.cpp)
   将其设置为 `GetInsertChannel()`。
3. CacheSlot 在 cell 加载/卸载时按 `file_bytes` 增减该 Gauge。最后一个 slot handle
   消失后，过期 series 会在下一次收集时从 family 中移除。
4. [`monitor_c.cpp`](../../../internal/core/src/monitor/monitor_c.cpp) 的
   `GetCoreMetrics()` 在序列化 registry 前触发收集和过期 series 清理。

同一份 shard stats 还通过
[`cache_shard_disk_usage.go`](../../../internal/util/metrics/cache_shard_disk_usage.go)
进入 QueryNode distribution response，供 QueryCoord 的 shard disk balancer 使用；
这是 protobuf 控制面数据，不是第二个 Prometheus 指标族。

因此，这个指标可以按 VChannel/shard 定位缓存磁盘占用，但不能直接按
`collection_id` 归因。`aggregate` 模式在 `CRegistry` 解析 C++ Prometheus 文本后，
将该 family 的 `shard` 改为 `all` 并按 `data_type` 求和；供 QueryCoord shard disk
balancer 使用的逐 shard protobuf stats 不变。若产品希望把它纳入严格 collection
口径，应新增稳定的 `collection_id` attribution；不能仅把 `shard` 改名为
`collection_id`。

Knowhere 的 `collection_id` config key、C++ protobuf 的
`OperationMetrics.collection_metrics`，以及 storage/index metadata 中的
`collection_id` 都不是 Prometheus label，故没有误计入清单。milvus-storage 的
filesystem snapshot 则由
[`filesystem_metrics.go`](../../../internal/storagev2/filesystem_metrics.go) 读取，并由
[`persistent_store_metrics.go`](../../../pkg/metrics/persistent_store_metrics.go) 仅以
`fs` 标签发布。

## 定义、写入与清理入口

上述 59 个 metric vector 都已注册，并至少有一个非测试写入点；QueryView
自定义 collector 也已注册，且安装了生产 provider。主要生命周期入口如下：

| 组件 | 主要写入点 | Collection 清理入口 |
|---|---|---|
| Proxy | [`internal/proxy/impl.go`](../../../internal/proxy/impl.go)、[`internal/proxy/task_search.go`](../../../internal/proxy/task_search.go)、[`internal/proxy/task_delete.go`](../../../internal/proxy/task_delete.go)、[`internal/proxy/simple_rate_limiter.go`](../../../internal/proxy/simple_rate_limiter.go)、[`pkg/metrics/grpc_stats_handler.go`](../../../pkg/metrics/grpc_stats_handler.go)、[`pkg/metrics/restful_middleware.go`](../../../pkg/metrics/restful_middleware.go) | `CleanupProxyCollectionMetrics` |
| RootCoord / MixCoord | [`internal/rootcoord/quota_center.go`](../../../internal/rootcoord/quota_center.go) | `CleanupRootCoordCollectionMetrics`，但当前没有生产调用点 |
| DataCoord | [`internal/datacoord/meta.go`](../../../internal/datacoord/meta.go)、[`internal/datacoord/index_meta.go`](../../../internal/datacoord/index_meta.go)、[`internal/datacoord/index_size_tracker.go`](../../../internal/datacoord/index_size_tracker.go)、[`internal/datacoord/snapshot_manager.go`](../../../internal/datacoord/snapshot_manager.go) | `CleanupDataCoordWithCollectionID`；snapshot pin 另有精确删除 |
| DataNode | [`internal/util/flowgraph/input_node.go`](../../../internal/util/flowgraph/input_node.go)、[`internal/flushcommon/writebuffer`](../../../internal/flushcommon/writebuffer)、[`internal/flushcommon/syncmgr`](../../../internal/flushcommon/syncmgr)、[`internal/datanode/compactor`](../../../internal/datanode/compactor) | `CleanupDataNodeCollectionMetrics`；input-node cache 另有精确删除 |
| QueryCoord | [`internal/querycoordv2/task/scheduler.go`](../../../internal/querycoordv2/task/scheduler.go) | `CleanQueryCoordMetricsWithCollectionID` |
| QueryNode | [`internal/querynodev2/metrics_info.go`](../../../internal/querynodev2/metrics_info.go)、[`internal/querynodev2/handlers.go`](../../../internal/querynodev2/handlers.go)、[`internal/querynodev2/services.go`](../../../internal/querynodev2/services.go)、[`internal/querynodev2/delegator`](../../../internal/querynodev2/delegator)、[`internal/querynodev2/pipeline`](../../../internal/querynodev2/pipeline) | `CleanupQueryNodeCollectionMetrics` |
| QueryView | [`internal/views/qviews/observe/metrics_observer.go`](../../../internal/views/qviews/observe/metrics_observer.go) | pull collector 每次只输出 provider 的当前 Top 5，无持久化 vector series |

## 判定原则

当指标契约明确包含以下至少一种需求时，collection 维度是合理的：

- 指标值表示 collection 所属的状态、容量或配额。
- 运维人员需要 collection 身份来定位故障或被限流的资源。
- 某项功能或策略可以按 collection 独立配置。
- 明确支持按 collection 做用量归因、计费或 SLA 报告。

出现以下情况时，应移除或重新设计 collection 维度：

- 标签值实际上并不是 collection。
- 已经存在不带 collection 维度的替代指标。
- 同时用 name 和 ID 重复标识同一个 collection。
- 指标表达的是传输层或节点级聚合值，且没有受支持的消费者需要按
  collection 归因。
- 保留该标签的唯一原因只是方便清理指标。

`collection_name` 需要额外谨慎，因为它是用户可控的无界文本。不能仅仅为了
方便 Dashboard 或清理代码就保留它。

## 明确的整改项

### 1. 重新设计 `milvus_proxy_limiter_rate`

该指标把标签声明为 `collection_id`，但写入端会传入多个层级的 source ID：

- `root`
- `database.<id>`
- `collection.<id>`
- `partition.<id>`

参见
[`internal/proxy/simple_rate_limiter.go`](../../../internal/proxy/simple_rate_limiter.go)。
这并不是 collection 级契约，当前标签的语义是错误的。

仅将标签改名为 `source_id` 只能修正名称，无法解决基数问题。内置 Dashboard
最终按 node 和 message type 聚合出最小值、最大值和平均值，因此更适合在进程内
先计算有界聚合结果。如果产品明确要求精确查看每个资源，则应通过经过专门设计、
具备完整生命周期管理的诊断接口暴露。

### 2. 删除 `milvus_datacoord_index_task_count`

该指标已标记为废弃，并指向替代指标 `milvus_datacoord_task_count`。替代指标的
维度是 `task_type` 和 `task_state`，不再包含 collection ID。当前内置 Grafana
面板仍同时查询新旧两个指标，因此删除旧指标前需要先迁移面板。

相关源码：

- [`pkg/metrics/datacoord_metrics.go`](../../../pkg/metrics/datacoord_metrics.go)
- [`deployments/monitor/grafana/milvus-dashboard.json`](../../../deployments/monitor/grafana/milvus-dashboard.json)

### 3. 删除重复且已废弃的 Proxy Histogram

应完整删除以下两个指标族：

- `milvus_proxy_collection_sq_latency`
- `milvus_proxy_collection_mutation_latency`

它们与 `milvus_proxy_sq_latency`、`milvus_proxy_mutation_latency` 具有相同标签，
并接收相同的观测值。只删除旧指标的 collection 标签会制造另一份指标契约，
却不会消除重复采集。应先将 Grafana 面板迁移到仍支持 DB/collection 筛选的
非废弃指标，然后删除旧指标的声明、写入、注册、清理及测试。

### 4. 删除冗余的 `collection_name`

以下指标已经带有 `collection_id`，仍应保持 collection 粒度，但
`collection_name` 是冗余标签：

- `milvus_datacoord_stored_rows_num`
- `milvus_datacoord_stored_index_files_size`
- `milvus_querynode_entity_num`

应保留 `collection_id`，移除 `collection_name`。内置 Dashboard 不依赖这些
指标中的 `collection_name`。

## 需要产品决策的候选项

### QueryCoord task latency

`milvus_querycoord_task_latency` 同时带有 `collection_id` 和 `channel_name`。
QueryCoord 写入时始终将 `task.Shard()` 作为 channel 标签，因此该指标已经是
shard 粒度，collection 标签通常在功能上是冗余的。但是，移除它需要新的清理
方案，也会让按 collection 聚合变得不够直接。

应将其视为优先级较高的移除候选项，但需要先确认：按 collection 聚合是否属于
受支持的运维流程，以及 shard 为空的任务应该如何表示。

### Proxy 流量与吞吐量

内置 Dashboard 会聚合掉以下指标的 DB 和 collection 维度：

- `milvus_proxy_receive_bytes_count`
- `milvus_proxy_received_nq`
- `milvus_proxy_search_vectors_count`
- `milvus_proxy_insert_vectors_count`
- `milvus_proxy_upsert_vectors_count`
- `milvus_proxy_delete_vectors_count`

如果这些指标仅用于展示节点级流量和吞吐量，则应移除 DB 和 collection。如果
Milvus 明确通过它们支持按 collection 做用量、计费或 SLA 报告，则应保留这些
维度。仅凭仓库内现有 Dashboard 的使用方式，还不足以决定这项产品契约。

### 算法与功能诊断指标

segment pruning、two-stage search、sparse/FTS workload、partial result 和
compaction anomaly 等指标的 collection 标签成本较高，Histogram 尤其明显；
但这些标签也能帮助定位特定 collection 的 schema、index 或数据分布问题。
不能仅仅因为内置 Dashboard 暂未使用这些标签就将其删除，应先确认预期的诊断
流程。

## 天然属于 Collection 级别的指标

以下类别通常应保留 collection 身份：

- stored rows、binlog size、index size、entity count 和 entity memory size。
- collection 特定的配额因子和限流结果。
- 需要通过 collection 定位待处理资源的 snapshot pin 和 QueryView 状态。
- skipped insert field 等 schema 演进异常。
- 必须通过 collection 定位停滞或异常 flowgraph 的 pipeline 状态。

这些指标应该使用 ID 还是 name，是另一个独立决策。如果已经可以获得稳定 ID，
应优先使用 ID，而不是用户可控的 name。

## 清理问题

collection 级指标的清理逻辑并不完整，且部分实现明确不正确：

1. `CleanupRootCoordCollectionMetrics` 当前没有生产调用点；即使未来接入，它也连续
   删除了两次 `RootCoordNumEntities`，从未删除
   `RootCoordIndexedNumEntities`，也没有清理 `RootCoordRateLimitRatio`。
2. `CleanupDataNodeCollectionMetrics` 调用 `Delete` 清理
   `DataNodeWriteDataCount` 时只提供了 `collection_id`，但该 vector 有 4 个
   可变标签。Prometheus 无法匹配这组标签，因此删除不会生效。
3. 多个持久化 collection vector 没有直接调用 `Delete`、
   `DeletePartialMatch`、`DeleteLabelValues` 或 `Reset`，包括：
   - `milvus_proxy_limiter_rate`
   - `milvus_proxy_search_sparse_num_non_zeros`
   - `milvus_proxy_function_udf_call_latency`
   - `milvus_querynode_search_fts_num_tokens`
   - `milvus_datanode_growing_source_sync_failure_count`
   - `milvus_rootcoord_rate_limit_ratio`
4. Proxy 的精确标签清理遗漏了已经实际写入的部分枚举值，包括：废弃 search
   latency 和 scanned-byte 指标的 HybridSearch series，以及废弃 mutation
   latency 的 Upsert series。
5. `milvus_datacoord_snapshot_active_pins` 没有出现在 DataCoord 通用 collection
   清理函数中，但 snapshot manager 有独立的生命周期删除逻辑。在完整追踪该
   生命周期前，不应将其判断为泄漏。

清理逻辑不能替代低基数指标契约。移除 collection 标签时，应从声明和所有写入点
移除该维度，并删除已经不再需要的清理代码。不能仅为了支持
`DeletePartialMatch` 而保留无界标签。

## 建议实施顺序

1. 删除两个重复且已废弃的 Proxy Histogram，并迁移 Grafana。
2. 删除已废弃的 `milvus_datacoord_index_task_count`，将 Grafana 迁移到
   `milvus_datacoord_task_count`。
3. 按有界聚合语义重新设计 `milvus_proxy_limiter_rate`。
4. 从三个同时带 ID 和 name 的指标中移除冗余 `collection_name`。
5. 独立修复指标清理逻辑，不要与标签移除混为同一个问题。
6. 修改 Proxy 用量归因和 QueryCoord task latency 契约前，先做明确的产品决策。

每次修改指标契约，都必须同步更新声明、所有写入点、清理逻辑、Grafana 查询与
变量、测试及兼容性说明。验证时应检查最终暴露的 series 数量，不能只验证编译和
单元测试。
