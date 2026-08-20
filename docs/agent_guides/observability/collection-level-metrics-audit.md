# Collection 级 Prometheus 指标审计

本文档记录对携带 `collection_id` 或 `collection_name` 标签的 Prometheus
指标族所做的源码级审计。本文基于 2026-08-19 的工作树快照，并非永久兼容性
契约。指标声明或注册发生变化后，应重新执行审计。

当前工作树已经从 `milvus_proxy_req_count` 中移除了 `db_name` 和
`collection_name`，因此该指标不计入下文清单。

## 审计范围与统计口径

审计覆盖已注册的 Go Prometheus collector 及其实际写入点。如果一个指标族的
可变标签中明确包含 `collection_id` 或 `collection_name`，就将其计为
collection 级指标。

以下内容不计入：

- `*_collection_num` 等只统计 collection 数量、但不标识具体 collection 的指标。
- 日志字段、JSON metrics 接口、trace，以及只通过 channel 名间接编码
  collection 的标签。
- 未暴露显式 collection 标签的原生指标。

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
| **合计** | **60** |

在这 60 个指标族中：

- 41 个包含 `collection_id`。
- 22 个包含 `collection_name`。
- 3 个同时包含两个标签。
- 59 个是 metric vector，1 个是 QueryView 自定义 collector。
- 包含 22 个 CounterVec、21 个 GaugeVec、16 个 HistogramVec；QueryView
  自定义 collector 输出 Gauge。

## 完整清单

### Proxy

源码：[`pkg/metrics/proxy_metrics.go`](../../../pkg/metrics/proxy_metrics.go)

带 `collection_name`：

- `milvus_proxy_received_nq`
- `milvus_proxy_search_vectors_count`
- `milvus_proxy_insert_vectors_count`
- `milvus_proxy_upsert_vectors_count`
- `milvus_proxy_delete_vectors_count`
- `milvus_proxy_sq_latency`
- `milvus_proxy_collection_sq_latency`（已废弃）
- `milvus_proxy_mutation_latency`
- `milvus_proxy_collection_mutation_latency`（已废弃）
- `milvus_proxy_receive_bytes_count`
- `milvus_proxy_retry_search_cnt`
- `milvus_proxy_retry_search_result_insufficient_cnt`
- `milvus_proxy_recall_search_cnt`
- `milvus_proxy_search_sparse_num_non_zeros`
- `milvus_proxy_function_udf_call_latency`
- `milvus_proxy_scanned_remote_mb`
- `milvus_proxy_scanned_total_mb`

带 `collection_id`：

- `milvus_proxy_limiter_rate`

### RootCoord / MixCoord

源码：
[`pkg/metrics/rootcoord_metrics.go`](../../../pkg/metrics/rootcoord_metrics.go)

带 `collection_name`：

- `milvus_rootcoord_entity_num`
- `milvus_rootcoord_indexed_entity_num`

带 `collection_id`：

- `milvus_rootcoord_rate_limit_ratio`

### DataCoord

源码：
[`pkg/metrics/datacoord_metrics.go`](../../../pkg/metrics/datacoord_metrics.go)

同时带 `collection_id` 和 `collection_name`：

- `milvus_datacoord_stored_rows_num`
- `milvus_datacoord_stored_index_files_size`

仅带 `collection_id`：

- `milvus_datacoord_store_level0_segment_size`
- `milvus_datacoord_l0_delete_entries_num`
- `milvus_datacoord_bulk_insert_vectors_count`
- `milvus_datacoord_stored_binlog_size`
- `milvus_datacoord_segment_binlog_file_count`
- `milvus_datacoord_index_task_count`（已废弃）
- `milvus_datacoord_snapshot_active_pins`

### DataNode

源码：
[`pkg/metrics/datanode_metrics.go`](../../../pkg/metrics/datanode_metrics.go)

以下指标全部使用 `collection_id`：

- `milvus_datanode_write_data_count`
- `milvus_datanode_consume_tt_lag_ms`
- `milvus_datanode_consume_msg_count`
- `milvus_datanode_growing_source_sync_failure_count`
- `milvus_datanode_fg_buffer_size`
- `milvus_datanode_compaction_delete_count`
- `milvus_datanode_compaction_missing_delete_count`

### QueryCoord

源码：
[`pkg/metrics/querycoord_metrics.go`](../../../pkg/metrics/querycoord_metrics.go)

带 `collection_id`：

- `milvus_querycoord_task_latency`

### QueryNode

源码：
[`pkg/metrics/querynode_metrics.go`](../../../pkg/metrics/querynode_metrics.go)

同时带 `collection_id` 和 `collection_name`：

- `milvus_querynode_entity_num`

仅带 `collection_id`：

- `milvus_querynode_consume_tt_lag_ms`
- `milvus_querynode_consume_msg_count`
- `milvus_querynode_skipped_insert_field_count`
- `milvus_querynode_segment_num`
- `milvus_querynode_sq_req_count`
- `milvus_querynode_search_fts_num_tokens`
- `milvus_querynode_search_hit_segment_num`
- `milvus_querynode_segment_filter_hit_segment_num`
- `milvus_querynode_segment_filter_skipped_segment_num`
- `milvus_querynode_segment_filter_total_segment_num`
- `milvus_querynode_segment_prune_ratio`
- `milvus_querynode_segment_prune_bias`
- `milvus_querynode_segment_prune_latency`
- `milvus_querynode_entity_size`
- `milvus_querynode_level_zero_size`
- `milvus_querynode_partial_result_count`
- `milvus_querynode_two_stage_search_stage1_latency`
- `milvus_querynode_two_stage_search_stage2_latency`
- `milvus_querynode_two_stage_search_fallback_total`
- `milvus_querynode_global_refine_total`

### QueryView

源码：[`pkg/metrics/qv_metrics.go`](../../../pkg/metrics/qv_metrics.go)

带 `collection_id`：

- `milvus_qv_view_state_max_age_seconds`

该指标是 pull collector，而不是持久化 metric vector。其 provider 对每个
component 只输出状态持续时间最长的 5 个 view，因此尽管它会标识
collection，对外暴露的基数仍然有界。

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

1. `CleanupRootCoordCollectionMetrics` 连续删除了两次
   `RootCoordNumEntities`，从未删除 `RootCoordIndexedNumEntities`。
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
