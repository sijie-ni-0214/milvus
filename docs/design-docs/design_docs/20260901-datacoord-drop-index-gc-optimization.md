# DataCoord GC 可扩展性优化方案：DropIndex 与 dropped segment

- **创建日期：** 2026-09-01
- **状态：** DropIndex Phase 1/2/3、dropped-segment 统一分阶段 batch GC 已完成本地实现及 mock 验证；真实后端验证待完成
- **涉及组件：** DataCoord、Metastore、Object Storage
- **目标规模：** 1,000,000 个 field-index 定义和 10,000,000 条 SegmentIndex
- **关联 Issue：** 待补充

## 1. 摘要

本文给出 DataCoord DropIndex GC 在百万 Collection、千万 SegmentIndex
规模下的分阶段优化方案，并补充 dropped-segment GC 的独立基线和第一阶段优化。

当前运行期 GC 扫描常驻内存中的 `indexMeta`，逐条删除索引文件，再逐条删除
SegmentIndex KV。不同 SegmentIndex 之间串行处理。
`dataCoord.gc.removeConcurrent` 只会并发删除同一个 SegmentIndex 内的多个文件；
当每条 SegmentIndex 只有一个文件时，该参数几乎无法提升跨条目吞吐。

首期建议包含以下四项核心改造：

1. 从当前有效的 `indexMeta` 流式读取候选项并写入有界队列，避免再次构造完整
   SegmentIndex map。
2. 增加可选的对象存储批量删除能力，并返回逐对象结果。MinIO/S3 使用原生
   Multi-Delete，不支持的后端回退到现有逐文件删除。
3. 跨 SegmentIndex 组成有界文件批次；只有某个 buildID 的全部文件都确认删除
   后，才允许删除对应元数据。
4. 使用 etcd `MultiRemove` 批量删除成功的 SegmentIndex 和 field-index KV，
   单次事务不超过 `metastore.maxEtcdTxnNum`。

首期不保留 `recoveredViews` 等恢复阶段临时快照，也不增加另一个包含千万条记录的
常驻反向映射。是否增加反向索引必须由后续真实运行数据决定。

### 1.1 当前实现进度（2026-09-01）

已经实现并通过定向单元测试和目标规模 mock 测试：

- 可选的 `BulkObjectStorage` 和 `BatchRemoveChunkManager`，没有扩大原接口或生成
  mock 的强制实现面。
- `MinioObjectStorage.RemoveObjects` 使用 S3 Multi-Delete，完整返回逐 path 结果；
  缺失结果和无 path 的 batch error 均 fail closed。
- 不支持原生批删的 `RemoteChunkManager` 后端按
  `dataCoord.gc.removeConcurrent` 做有界逐文件回退。
- 优化路径直接 `Range` 常驻 `segmentBuildInfo`，不再通过 `GetAllSegIndexes`
  物化完整候选 map。
- 跨 SegmentIndex 组成有界文件批次；只有全部文件成功或不存在的 buildID 才进入
  元数据删除阶段。
- `DataCoordCatalog` 提供 `DropIndexes` 和 `DropSegmentIndexes`；KV catalog 构造精确 key，并按
  `metastore.maxEtcdTxnNum` 调用 `MetaKv.MultiRemove`。
- field-index 和 SegmentIndex 都在 KV 成功后才发布内存删除；SegmentIndex 会按
  buildID 加锁并重新校验状态和文件版本，已经变化的候选项留待下一轮处理。
- SegmentIndex 元数据发布批次限制为 `metastore.maxEtcdTxnNum`，避免一次持有 1000
  个 `KeyLock`，并保证一次发布批次只对应一个 etcd transaction。
- field-index 和 SegmentIndex batch 不维护额外汇总 stats；保留外层 stage 总耗时和失败 Warn。
- 所有 `ChunkManager` 都进入相同的有界 batch GC 流水线；实现
  `BatchRemoveChunkManager` 时使用逐 path 结果的后端批删，否则在流水线内部通过
  `dataCoord.gc.removeConcurrent` 有界调用 `Remove`。不会因 storage 能力不同切回另一套
  GC 遍历和提交顺序。
- field-index 和 SegmentIndex metadata 统一使用 catalog 批删，不保留运行时能力判断或
  逐条删除 fallback。

尚未实现：原生 batch 并发、Azure/GCP Native 原生批删、真实 MinIO/etcd 集成和
在线集群灰度。当前 mock 结果证明客户端请求数、CPU 和临时分配下降，但不能替代真实
后端的吞吐、限流和前台延迟验证。

### 1.2 dropped-segment GC 当前实现进度（2026-09-02）

dropped-segment GC 原来逐 segment 串行执行完整流水线；中间版本曾增加另一条
`recycleDroppedSegmentsConcurrently` 路径。当前实现已经删除该分叉，所有 storage 后端
都执行同一条分阶段 batch 流水线：

1. 在单次 pass 内按唯一 insert channel 缓存 channel checkpoint，并通过
   `LoadChannelExistence` 对精确 marker key 做批量读取。动态配置
   `dataCoord.gc.droppedSegment.channelStateBatchSize` 控制 GC pause barrier 之间提交给 catalog
   的 channel 数；catalog 还会按 `MetaKv.MaxTxnOps()` 拆分底层 `MultiLoad`。
2. marker 不存在或值为 `removed` 时返回 `false`；metadata storage 批量读取失败时，对该批
   未解析 channel fail-closed，本轮不回收其 segment，避免将读取错误解释成 marker 不存在。
3. candidate 按 segment 数和估算精确 path 数组成有界批次，严格分为“文件删除 ->
   SegmentIndex 元数据删除 -> segment 元数据删除”三个 stage。前一 stage 未确认成功的
   segment 不进入后一 stage。
4. 文件 stage 在内部选择能力：支持 `BatchRemoveChunkManager` 时使用后端逐 path 批删
   结果；否则通过现有 `removeObjectPool` 有界调用基础 `ChunkManager.Remove`。能力差异不
   再改变上层 GC 算法。
5. pause 只在 batch barrier 确认；已经进入的 storage 或 metadata batch 收敛后才 ACK，
   避免 ACK 后仍有该 Collection 的删除在飞。
当前 DataView 引用检查仍按 candidate 执行，V3 prefix 仍需逐 prefix list；真实集群启用
前仍需测量 etcd/对象存储限流和前台延迟。

### 1.3 dropped-segment GC 分阶段批删进度（2026-09-02）

所有 `ChunkManager` 都使用以下有界流水线；可选能力只决定文件 stage 使用原生批删还是
有界逐 path fallback：

1. V1/V2 的 binlog、stats、text/json stats 和索引文件跨 segment 合并为精确 key
   批次，使用 `MultiRemoveWithResult`；任一 key 失败或缺少结果时，只保留归属的
   segment。
2. V3 的 segment data 仍必须按 manifest base path 清理。`RemoteChunkManager` 对多个
   prefix 做有界并发 list，再把列出的精确 key 跨 prefix 合并为对象存储批删。它不是
   对象存储提供的 O(1)“批量 prefix delete”；list 请求数仍与 prefix 数量成正比。
3. 只有文件阶段完整成功的 segment 才进入 `SegmentIndex` metadata 批删；只有其全部
   `SegmentIndex` 已从当前内存状态消失，才进入最终 segment metadata 批删。
4. segment cache 中已有的持久化 revision 会直接转成 etcd `ModRevision` compare +
   delete，因此正常路径不再逐 segment GET。revision 冲突时拆成逐项条件删除：版本仍
   匹配的兄弟项可以推进，已经变化的 segment 保留到下一轮重新生成文件计划。TiKV
   恢复版本是 scan `StartTS` 水位而非单 key `CommitTS`，因此保留事务读，并拒绝
   `CommitTS` 晚于缓存水位的删除，不能直接套用 etcd 的等值比较。
5. batch-level error、缺失结果、prefix list 失败和 context cancel 均 fail closed；
   `ErrIoKeyNotFound` 仍按幂等成功处理。pause 只在 batch barrier ACK。

该实现没有保存 `recoveredViews`，也没有新增长期 DataView cache。当前仍调用
`IsSegmentReferenced` 对每个 candidate 读取最新 DataView，因此 1000 万 segment 会有
1000 万次 DataView 检查；解决它需要单独定义不会因过期引用造成误删的批量读取或一致性
边界，不能直接复用恢复阶段临时快照。

## 2. 范围

### 2.1 包含范围

- `recycleUnusedIndexes` 和 `recycleUnusedSegIndexes`。
- `recycleDroppedSegments` 的有界跨 segment 并发、channel 状态去重和分阶段批删。
- SegmentIndex 候选项遍历及其内存分配。
- 通过 `ChunkManager`、`ObjectStorage` 删除精确索引文件。
- 通过 `DataCoordCatalog` 删除 field-index 和 SegmentIndex KV。
- 有界并发、限流、重试、取消和 GC pause。
- stage/batch 级日志和指标。
- 大规模 mock 测试及小规模真实后端验证。

### 2.2 不包含范围

- DropIndex RPC、WAL broadcast 和 ACK 延迟。
- orphan-file、LOB、DataView 或 snapshot GC 的整体重构。
- 为每条 SegmentIndex 新增持久化 GC 任务记录。
- 首期修改对象路径布局或依赖 bucket lifecycle。
- 首期为所有云存储后端实现原生批删。

## 3. 当前逻辑

当前调用链如下：

```text
recycleUnusedIndexes
  -> GetDeletedIndexes
  -> RemoveIndex
  -> DataCoordCatalog.DropIndex
  -> MetaKv.Remove（单条 field-index KV）

recycleUnusedSegIndexes
  -> GetAllSegIndexes（把常驻元数据再次整理为完整 map）
  -> 串行遍历每条 SegmentIndex
       -> 根据 buildID 读取最新状态
       -> 检查 Collection pause
       -> 检查父 Index 或 Segment 是否存在
       -> 跳过非终态任务
       -> 检查 snapshot 保护
       -> removeObjectFiles
            -> 把当前 SegmentIndex 的文件提交给 removeObjectPool
            -> 等待这些文件全部完成
       -> RemoveSegmentIndex
            -> MetaKv.Remove（单条 SegmentIndex KV）
       -> 处理下一条 SegmentIndex
```

相关源码：

- [`recycleUnusedIndexes`](../../../internal/datacoord/garbage_collector.go#L1845)
- [`recycleUnusedSegIndexes`](../../../internal/datacoord/garbage_collector.go#L1879)
- [`removeObjectFiles`](../../../internal/datacoord/garbage_collector.go#L1464)
- [`GetAllSegIndexes`](../../../internal/datacoord/index_meta.go#L1304)
- [`RemoveSegmentIndex`](../../../internal/datacoord/index_meta.go#L1417)
- [`RemoteChunkManager.MultiRemove`](../../../internal/storage/remote_chunk_manager.go#L333)

运行期 `recycleUnusedSegIndexes` 扫描的是内存，不是 etcd。etcd 的开销来自后续
逐条删除，以及 DataCoord 重启恢复。内存扫描仍然昂贵，因为
`GetAllSegIndexes` 会物化完整快照，之后每条候选项还会按 buildID 刷新并 clone
文件路径。

当前 `RemoteChunkManager.MultiRemove` 也不是真正的后端批量操作，它只是循环调用
`Remove`。因此仅把调用点替换成现有 `MultiRemove`，不会减少对象存储请求数。

## 4. 已验证基线

可选性能测试 `TestDataCoordDropIndexGCLargeScalePerformance` 使用可配置的 mock KV
和文件后端，直接执行生产函数 `recycleUnusedIndexes` 和
`recycleUnusedSegIndexes`。

### 4.1 小规模延迟模型验证

| 项目 | 数值 |
|---|---:|
| field-index 定义 | 10 |
| SegmentIndex/文件 | 1,000 |
| mock KV 延迟 | 1 ms |
| mock 文件延迟 | 1 ms |
| 模型 I/O 时间 | 2.010 s |
| 实测 GC stage 总时间 | 约 2.160 s |
| 删除 field-index KV | 10 |
| 删除 SegmentIndex KV | 1,000 |
| 删除文件 | 1,000 |

该结果用于验证 mock 延迟模型在小规模下基本可靠。

### 4.2 大规模零延迟基线

| 项目 | 数值 |
|---|---:|
| field-index 定义 | 1,000,000 |
| SegmentIndex/文件 | 10,000,000 |
| 构造测试数据 | 11.17 s |
| field-index 清理 | 8.815 s |
| SegmentIndex/文件清理 | 2m45.36s |
| SegmentIndex 吞吐 | 约 60,476 entries/s |
| 峰值 RSS | 约 6.86 GiB |
| SegmentIndex stage 累计分配 | 约 28.25 GiB |
| 剩余 field-index / SegmentIndex | 0 / 0 |

测试使用 `-tags dynamic,test -gcflags="all=-N -l"`，关闭了热循环 Info 日志，
复用了轻量 SegmentInfo fixture，并且大规模运行没有注入存储延迟。因此 CPU 和
内存结果只代表当前 checkout 下的 mock 下界，不能当作生产对象存储实测结果。

### 4.3 串行 I/O 模型

假设每条 SegmentIndex 有一个文件，单次 KV 删除 1 ms，单次文件删除 5 ms：

```text
KV 时间   = (1,000,000 + 10,000,000) * 1 ms = 11,000 s
文件时间  = 10,000,000 * 5 ms                = 50,000 s
合计      = 61,000 s                         = 16h56m40s
```

这是 mock 推算，不是生产环境实测耗时。

## 5. 瓶颈分析

### 5.1 SegmentIndex 之间完全串行

当前 GC 必须等待一条 SegmentIndex 的全部文件和单条 KV 删除完成，才处理下一条。
本次测试使用的 `dataCoord.gc.removeConcurrent=16` 不能产生跨 SegmentIndex 并发。
其他配置值也不会改变该参数当前只作用于单条 SegmentIndex 内文件的事实。

### 5.2 每个文件一次对象存储请求

1000 万个文件会产生 1000 万次客户端请求。提高并发可以隐藏部分请求延迟，但不会
减少请求数，还可能触发对象存储限流。

### 5.3 每条元数据一次 etcd 请求

当前场景包含 100 万次 field-index remove 和 1000 万次 SegmentIndex remove。
`MetaKv` 已支持 `MultiRemove`，但 `DataCoordCatalog.DropIndex` 和
`DropSegmentIndex` 没有批量接口。

### 5.4 全量候选快照和 clone

`GetAllSegIndexes` 会为全部常驻 SegmentIndex 再构造一个 map。GC 随后逐条根据
buildID 获取最新数据并 clone 文件路径，产生大量累计分配和 Go GC 压力。

### 5.5 逐条日志和指标清理

热路径会输出多条 Info 日志并构造逐条字段。千万条记录下可能形成日志和分配风暴。
现有 stage 耗时日志有价值，但还不能区分扫描、命中、文件删除、KV 删除、重试和
批次数量。

### 5.6 父 tombstone 先于子项删除

`recycleUnusedIndexes` 先删除 field-index tombstone，
`recycleUnusedSegIndexes` 随后只能把对应 SegmentIndex 当成全局孤儿发现。

延长 tombstone 生命周期有助于表达所有权和重试状态，但需要先设计高效子项查找和
故障恢复语义，不是首期批删的必要前置条件。

## 6. 优化方案

### 6.1 目标执行流水线

```text
live indexMeta Range/分页
        |
        v
候选项校验
  - 最新 build 状态
  - Collection pause
  - 父 Index/Segment 是否存在
  - 是否终态
  - snapshot 保护
        |
        v
有界候选项/文件批次
        |
        v
对象存储删除
  - 支持时使用原生批删
  - 否则有界逐文件回退
  - 返回逐 key 成功/失败结果
        |
        v
筛选全部文件成功的 buildID
        |
        v
etcd MultiRemove，每事务不超过 maxEtcdTxnNum
        |
        v
KV 事务成功后才发布 indexMeta 内存删除
```

队列和批次状态只在一次 GC pass 内临时存在，不保存到长生命周期 field。

### 6.2 OPT-1：流式遍历候选项，限制内存

使用 `Range` callback 或有界分页迭代器替代 `GetAllSegIndexes` 返回完整 map。

要求如下：

- GC 最多保留一个有界 page/queue。
- 在判断资格和文件路径前，再根据 buildID 获取最新候选项。
- GC 是最终一致的；本轮未观察到的并发新增或更新可以在下轮处理。
- 扫描器不能修改候选对象。
- 队列必须提供 backpressure，不能为每条 SegmentIndex 创建 goroutine 或 Future。

该优化把临时候选内存从 O(总条目数) 降为 O(队列大小 + batch size)。如果所有条目
都需要回收，扫描复杂度仍然是 O(N)。

### 6.3 OPT-2：可选的对象存储原生批删能力

不直接给现有 `ObjectStorage` 接口增加强制方法，否则 Azure、GCP Native、测试和
生成 mock 都必须实现相同语义，即使后端没有对应 API。

增加可选能力：

```go
type RemoveResult struct {
    Path string
    Err  error // nil 表示已删除或原本不存在
}

type BulkObjectStorage interface {
    RemoveObjects(
        ctx context.Context,
        bucketName string,
        objectNames []string,
    ) []RemoveResult
}
```

同时增加可选的 ChunkManager 结果接口，不修改现有
`ChunkManager.MultiRemove(ctx, paths) error` 契约：

```go
type BatchRemoveChunkManager interface {
    MultiRemoveWithResult(
        ctx context.Context,
        paths []string,
    ) []RemoveResult
}
```

`RemoteChunkManager.MultiRemoveWithResult` 的行为：

1. 底层 client 实现 `BulkObjectStorage` 时调用批删。
2. 不支持时使用现有 `Remove` 做有界并发回退。
3. 将 `ErrIoKeyNotFound` 归一为成功，保证 GC 幂等。
4. 每个输入 path 必须有且只有一个结果。
5. 保留 `ErrIoTooManyRequests` 等 typed storage error；增加上下文时使用
   `merr.Wrap`/`merr.Wrapf`，不能覆盖原错误码。

原有 `MultiRemove` 保留逐文件行为，避免改变 DataCoord 之外调用者的后端请求形态；
GC 始终使用统一 batch 流水线，在内部优先调用 `MultiRemoveWithResult`，能力缺失时用
`ChunkManager.Remove` 生成一条输入 path 对应一条结果。

#### MinIO/S3 实现

对于支持 Multi-Delete 的 S3-compatible provider，
`MinioObjectStorage` 使用 MinIO SDK 的 `RemoveObjects` 或
`RemoveObjectsWithResult`。SDK 每个 S3 Multi-Delete 请求最多包含 1000 个对象。
如果某个 provider 虽然通过 MinIO client 接入但不支持该 API，则在实现内部回退到
逐文件删除。legacy `gcp` 的 XML API 支持
[S3-compatible Multi-Delete](https://docs.cloud.google.com/storage/docs/xml-api/post-bucket)；
GCP Native 和 Azure 因未实现可选批删接口，通过 `RemoteChunkManager` 的有界能力回退。

实现必须满足：

- 完整消费 SDK 返回 channel，避免 goroutine 阻塞。
- 将带 `ObjectName` 的失败映射到对应 key。
- 没有 `ObjectName` 的 batch-level error 表示整批结果未知，整批按失败处理。
- 对象不存在视为成功。
- SDK error 必须通过 `mapObjectStorageError` 转换。
- 上层传入的 batch 有界，不能一次缓存 1000 万个 key。

Azure 和 GCP Native 首期使用回退路径，后续可以独立增加原生实现，不影响
DataCoord。

### 6.4 OPT-3：跨 SegmentIndex 组成文件批次

增加只在当前 GC pass 内使用的候选描述：

```go
type indexGCCandidate struct {
    buildID      int64
    collectionID int64
    files        []string
}
```

GC 按累计文件数组成批次。正常情况下不把一条 SegmentIndex 拆到两个 GC 批次；
如果单条 SegmentIndex 的文件数超过后端请求上限，由 storage 层内部拆分，并向上
返回聚合后的逐 key 结果。

每个 buildID 的处理规则：

```text
全部文件成功或不存在
  -> buildID 可以进入元数据删除阶段

任一文件失败、取消或状态未知
  -> 保留该 buildID 的全部元数据
  -> 下一轮重试
```

一个文件成功、兄弟文件失败时，保留元数据仍然安全；下一轮重试会把已不存在的文件
视为成功。

批量 worker 与逐文件 fallback worker 必须使用独立的有界并发。不能让 worker
向同一个已经饱和的 pool 再提交任务并同步等待，否则可能形成嵌套线程池死锁。

### 6.5 OPT-4：批量删除 etcd KV

为 catalog 增加类似接口：

```go
DropIndexes(ctx context.Context, indexes []*model.Index) error
DropSegmentIndexes(ctx context.Context, indexes []*model.SegmentIndex) error
```

KV catalog 构造精确 key，并按不超过
`paramtable.Get().MetaStoreCfg.MaxEtcdTxnNum` 的批次调用
`MetaKv.MultiRemove`。当前默认上限为 64。

理论上，如果可以跨所有文件批次连续填满 KV transaction：

```text
当前 KV 操作数 = 1,000,000 + 10,000,000 = 11,000,000
batch size 64   = ceil(11,000,000 / 64)   = 171,875 个事务
```

当前实现不会跨 storage batch 暂存已经完成文件删除的候选项。`file batch=1000` 时，
每个 SegmentIndex storage batch 产生 `ceil(1000/64)=16` 个 KV transaction，因此
目标规模实测计数为：

```text
field-index KV     = ceil(1,000,000 / 64)       = 15,625
SegmentIndex KV    = 10,000 * ceil(1,000 / 64) = 160,000
合计               = 175,625 个事务
```

`indexMeta.RemoveSegmentIndexes` 必须保留现有锁和发布边界：

1. 对 buildID 排序，确定一致的加锁顺序。
2. 获取对应 per-build lock。
3. 重新读取当前常驻条目，剔除已经消失或发生变化的候选项。
4. 构造并执行一个有界 catalog transaction。
5. 事务成功后，才更新 size metric，并在 `fieldIndexLock` 下从
   `segmentIndexes` 和 `segmentBuildInfo` 删除。
6. 释放 per-build lock。

事务失败时，该批次不能从内存元数据删除，从而保持重试能力并避免内存与 etcd
不一致。

SegmentIndex 元数据批删沿用同一条文件批删 traversal；文件阶段完成后，field-index 和
SegmentIndex 都通过 `DataCoordCatalog` 的批量接口提交 metadata 删除。

### 6.6 OPT-5：聚合热路径日志和指标

保留现有 stage duration 日志，在每个 stage 结束时输出一条汇总；长时间任务可以
按固定时间间隔输出 rated progress 日志。

建议汇总字段：

- `scanned`
- `eligible`
- `nonterminal`
- `snapshotBlocked`
- `paused`
- `fileDeleteSuccess`
- `fileDeleteFailed`
- `kvDeleteSuccess`
- `kvDeleteFailed`
- `fileBatchCount`
- `kvBatchCount`
- `timeCost`

去掉逐条成功 Info 日志。重复失败使用限频 Warn；buildID、文件路径等明细放在
Debug 日志中。指标标签不能包含 collectionID、segmentID、indexID、buildID 或
文件路径。

建议的低基数指标：

| 指标 | 标签 | 含义 |
|---|---|---|
| GC stage duration histogram | `stage` | stage 端到端耗时 |
| GC item counter | `stage`, `result` | 扫描、跳过、成功和失败条目数 |
| GC batch-size histogram | `backend`, `operation` | 每请求包含的文件或 KV 数 |
| GC in-flight gauge | `backend`, `operation` | 当前有界并发数 |

`IndexTaskNum` 已废弃。Grafana 迁移完成后应删除该指标及其 collection-level 清理。
过渡期可以使用固定 label values 删除，避免每个 Collection 构造四个
`prometheus.Labels` map。不能针对每个 Collection 调用 `DeletePartialMatch`，
否则会扫描整个 metric vector，在百万 Collection 下可能退化为 O(N²)。aggregate
模式中的共享 `all` series 不能被某个 Collection 的清理删除。

### 6.7 OPT-6：保持 pause、取消和公平性

批处理会改变 pause 可以安全 ACK 的位置：

- Collection 被 pause 后停止接收其新候选项。
- 提交 storage batch 前重新检查 pause 状态。
- 每个批次之间处理 GC control signal。
- 只有已接收任务到达安全点后才能 ACK pause，不能 ACK 后仍删除该 Collection
  已排队的文件。
- 限制单批耗时，从而限制 pause 延迟。
- `ctx.Done()` 后停止生产候选项，并取消或收敛 worker。
- 为 etcd 和对象存储分别限制并发，避免 DropIndex GC 饿死其他 GC stage 或压满
  共享后端。

### 6.8 OPT-7：子项清理完成后再删除父 tombstone

这是第二阶段生命周期优化：

1. `MarkIndexAsDeleted` 持久化 field-index tombstone。
2. GC 删除该 tombstone 下处于终态且未被 snapshot 保护的 SegmentIndex 文件和 KV。
3. 非终态、受保护或删除失败的子项存在时，保留 tombstone。
4. 不再存在任何子 SegmentIndex 后，才删除 field-index tombstone。

该方案能改善所有权和重试可见性，但高效查找子项需要反向索引或扫描，因此不阻塞
首期批删。

### 6.9 OPT-8：由数据决定是否增加反向索引

`(collectionID, indexID) -> buildIDs` 可以在只 Drop 少量 index 时避免扫描全部
SegmentIndex。但在 1000 万条规模下，另一个 Go map 会带来明显常驻内存开销。

首期不实现该索引，先测量：

- 只有少量候选项时的扫描耗时。
- 流式遍历完成后的 allocation 和 RSS。
- `scanned / eligible` 比例。
- 同一 backlog 被重复全量扫描的频率。

只有生产数据证明扫描已经成为剩余主瓶颈时才增加反向索引。它必须通过正常当前元数据
恢复流程重建，并在每次 create、update、remove 时维护。仍需保留低频全量
reconciliation，以处理历史或不一致元数据。

## 7. 配置项

首个版本建议增加：

| 配置项 | 初始默认值 | 动态刷新 | 作用 |
|---|---:|---|---|
| `dataCoord.gc.indexFileBatchDelete.batchSize` | `1000` | 是 | 一个 GC storage batch 最多估算的文件引用数；无文件条目按一个候选计数 |
| `dataCoord.gc.droppedSegment.channelStateBatchSize` | `64` | 是 | 一个 catalog batch 最多读取的唯一 channel marker 数；范围 1--1000，底层仍受 `MetaKv.MaxTxnOps()` 限制 |
| `dataCoord.gc.droppedSegment.batchDelete.batchSize` | `1000` | 是 | 单批最多容纳的 candidate 数或估算精确对象 key 数；范围 1--1000 |

原生 batch request 并发配置尚未实现；当前批次串行提交。是否增加
`dataCoord.gc.indexFileBatchDelete.concurrency` 及其默认值，需要结合真实后端限流和
前台延迟测试决定，不能只按 mock 结果确定。

继续使用以下现有配置：

- `dataCoord.gc.removeConcurrent`：限制逐文件 fallback 并发。
- `metastore.maxEtcdTxnNum`：限制单次 KV transaction 的操作数。
- `dataCoord.gc.interval`：控制 meta GC pass 启动频率；调小它不会提高单个 pass
  的删除吞吐。

批删不设置布尔开关：实现可选能力接口的后端自动使用批量路径，不实现的后端自动回退。
`batchSize` 保留为动态调优参数，用于根据真实后端限流和在线业务影响收紧单批规模。

## 8. 正确性约束

实现必须保持以下不变量：

1. **先文件、后元数据。** 一条 SegmentIndex 的全部已知文件确认删除或不存在前，
   不能删除其 KV。
2. **Fail closed。** 未知或部分 storage 结果必须保留元数据。
3. **幂等。** 对象不存在、元数据已经删除都属于成功重试结果。
4. **使用最新状态。** 删除前按 buildID 重新校验候选项。
5. **只处理终态任务。** 不能删除 `Unissued`、`InProgress` 或 `Retry` 状态。
6. **保持 snapshot 保护。** 受保护 buildID 不能提交文件删除。
7. **保持 pause 语义。** pause ACK 后不能继续执行该 Collection 已排队的删除。
8. **保持竞争安全。** dropped-segment GC 和 DropIndex GC 可能同时处理同一 buildID，
   必须继续依赖 per-build lock 和幂等删除。
9. **持久化成功后发布。** 对应 KV transaction 成功后才能删除常驻元数据。
10. **资源有界。** 内存、goroutine、queue 和 in-flight request 数不能随
    SegmentIndex 总数无限增长。

### 8.1 崩溃与重试矩阵

| 故障位置 | 持久化状态 | 重试要求 |
|---|---|---|
| 文件请求前 | 文件和元数据都存在 | 正常重试 |
| 部分文件成功、一个失败 | 元数据存在 | 缺失文件视为成功，只重试失败文件 |
| 全部文件成功、KV 删除前 | 元数据存在 | 文件删除幂等，然后重试 KV |
| KV transaction 失败 | 该批元数据存在 | 重试 transaction |
| KV 成功、内存发布前 | KV 不存在，内存可能仍存在 | 幂等删除 KV，再发布内存删除 |
| DataCoord 重启 | etcd 为 source of truth | 只恢复仍然持久化的条目 |

## 9. 预期收益

### 9.1 请求数量

每条 SegmentIndex 一个文件时：

| 操作 | 当前请求数 | 优化后重试前上限 |
|---|---:|---:|
| 对象存储客户端请求 | 10,000,000 | batch size 1000 时约 10,000 |
| KV 请求/事务 | 11,000,000 | 当前实现和默认 batch size 下 175,625 |

这里只推算请求数。对象存储后端仍需物理回收 1000 万个对象，不能假设一个批次的延迟
等于一次单文件删除。最终完成时间取决于后端实现、限流、网络、对象版本和 bucket
策略，必须通过真实后端测试确认。

### 9.2 内存和 CPU

流式遍历把临时候选项内存从 O(总条目数) 降为 O(queue + batch size)。移除逐条成功
日志和减少 clone 应降低累计 allocation 与 Go GC 压力。

物理删除仍然是 O(文件数)。该方案减少的是客户端协调和请求次数，不会减少对象存储
实际需要回收的文件或对象元数据数量。

## 10. 备选方案

### 10.1 只提高 GC 频率

不作为主要方案。它只会更频繁地启动 pass，不会提升单次 pass 的删除吞吐；一个长
pass 完成后可能立刻进入下一轮全量扫描。

### 10.2 只提高 `dataCoord.gc.removeConcurrent`

不足。该参数当前只并发同一 SegmentIndex 的文件。即使扩展到跨条目并发，仍会发出
1000 万次独立请求，并可能放大后端限流。

### 10.3 直接使用当前 `ChunkManager.MultiRemove`

不足。当前 `RemoteChunkManager.MultiRemove` 仍然循环执行逐文件 `Remove`，只返回
聚合错误，既没有减少请求，也无法安全表达部分成功。

### 10.4 把 prefix 删除当成 O(1) 后端操作

不采用。S3-compatible 对象存储没有原子 O(1) prefix delete。V3 segment 必须按
manifest base path 清理，因此实现采用“多 prefix 有界并发 list -> 跨 prefix 合并精确
key 批删”，并为每个输入 prefix 保留独立成功边界。它减少 delete 请求，不能减少
prefix list 请求；也不会把多个 segment 放宽成一个更宽、可能误删的共同 prefix。

### 10.5 全部交给 bucket lifecycle

暂缓。lifecycle 可以把物理回收移出 DataCoord，但会改变完成语义，也不能立即提供
per-build 删除证明，需要单独设计对象布局、保留周期和运维流程。

### 10.6 每条 SegmentIndex 持久化一个 GC work record

首期不采用。它会新增 1000 万条 KV，并增加恢复及压缩开销。现有 index metadata
已经可以作为幂等、持久化的重试锚点。

### 10.7 将恢复快照用于运行期 GC

不采用。恢复快照是临时状态，后续可能过期；运行期 GC 必须读取当前有效
`indexMeta`，并按 buildID 重新校验。

## 11. 实施计划

### Phase 0：基线与可观测性（已完成本地实现）

- 保留可选的 100万/1000万性能测试。
- 保留 stage 总耗时和失败 Warn，不维护额外汇总 stats。
- 记录 scan、文件删除、KV 删除、allocation、RSS、请求数和限流基线。

### Phase 1：对象存储批删能力（已完成本地实现）

- 增加可选 `BulkObjectStorage` 和 `BatchRemoveChunkManager` 接口。
- 实现返回逐 key 结果的 MinIO/S3 Multi-Delete。
- 实现有界 fallback，并保持现有 `MultiRemove` 兼容。
- 增加 unit test 和后端 integration test。

### Phase 2：DataCoord 文件批处理（已完成本地实现）

- 将候选项流式校验并写入有界批次。
- 跨 SegmentIndex 批量删除文件。
- 只有全部文件成功的 buildID 才删除元数据。
- 暂时保留现有单条 `RemoveSegmentIndex` 发布路径。

### Phase 3：Metastore 批处理（已完成本地实现）

- 增加受 `MaxEtcdTxnNum` 限制的 catalog batch-drop 接口。
- 增加 batch-safe 的 `indexMeta` 加锁、重新校验和常驻状态发布。
- 覆盖与 dropped-segment GC 的竞争和 transaction failure。

### Phase 3A：dropped-segment 有界并发（已完成本地实现）

- 增加动态并发配置；`1` 在不支持批量能力的后端保留原串行路径。
- 对唯一 channel 状态做每 pass 去重并有界解析。
- 跨 segment 并发完整流水线，保持单 segment 删除顺序、fail-closed 和 pause barrier。
- 增加正常并发、pause barrier、文件删除失败和 opt-in 性能测试。
- 修复测试用 `OptimisticTxnMemoryPersist` 的并发事务保护，并增加 64 worker 回归测试，
  避免测试替身掩盖或误报并发持久化问题。
- 下一阶段根据真实后端结果决定是否增加跨 segment 文件/KV 批删，不能仅按 mock
  加大默认并发。

### Phase 3B：dropped-segment 分阶段批删（已完成本地实现）

- V1/V2 精确文件跨 segment 使用逐 key 结果批删；V3 多 prefix 有界 list 后合并精确
  key 批删。
- 文件、SegmentIndex metadata、segment metadata 保持严格阶段顺序和逐 segment
  fail-closed 边界。
- `SegmentIndex` 按 `metastore.maxEtcdTxnNum` 批删并复用 buildID 锁及最新版本校验。
- segment metadata 使用 cache revision 做条件批删，正常 etcd 路径不再逐 key GET；
  compare 冲突时逐项安全收敛。
- 增加 batch size、stage 汇总字段、partial result、V3、pause、持久化失败和
  revision conflict 测试。

### Phase 4：Tombstone 生命周期与可选索引

- 子项清空后再删除 field-index tombstone。
- 根据实测 `scanned / eligible` 决定是否值得增加反向索引。
- 保留低频全局 reconciliation。

### Phase 5：其他原生后端

- 在定义等价的逐 key 结果和限流语义后，再实现 Azure/GCP 原生批删。

## 12. 验证计划

### 12.1 Storage 测试

- 空批次、单 key、1000 key 和 1001 key。
- 对象不存在返回成功。
- 一批中只有一个带 key 的失败。
- batch-level response/transport failure。
- 向 SDK channel 写入或读取期间取消 context。
- permission、credentials、throttling、bucket-not-found 等 typed error。
- 不支持批删后端的 fallback。

### 12.2 DataCoord 正确性测试

- 一条 SegmentIndex 多个文件，其中一个失败。
- 一个 storage batch 中多条 SegmentIndex 部分成功。
- fake-finished/no-train index 的空文件列表。
- 非终态任务不会提交删除。
- snapshot 保护的 buildID 不会提交删除。
- admission 前 pause，以及 batch 执行期间 pause。
- dropped-segment GC 并发处理同一 buildID。
- 对象批删成功后 KV transaction 失败。
- 在第 8.1 节每个崩溃点后重启 DataCoord。

### 12.3 性能测试矩阵

使用相同 workload 覆盖：

- file batch size：1、100、500、1000；
- native batch concurrency：1、2、4、8；
- KV batch size：1、16、64；
- 不同请求延迟、限流和错误注入；
- 每条 SegmentIndex 一个或多个文件；
- 1000 万条中只有少量候选，以及全部条目都命中。

mock 后端必须分别记录 client request count 和 logical file/KV count。
使用大规模零延迟测试前，先用小规模真实 delay 验证模型。

### 12.4 当前 mock A/B 结果

以下结果使用相同生产 GC 调用链和轻量 mock backend，均使用
`-tags dynamic,test -gcflags="all=-N -l"`。它们不是生产 MinIO 或 etcd 的完成时间。

小规模延迟校验：10 个 field-index、1000 条 SegmentIndex、每条一个文件，KV 和文件
请求各注入 1 ms：

| 模式 | field-index stage | SegmentIndex stage | SegmentIndex 吞吐 | 文件请求 | KV 请求 |
|---|---:|---:|---:|---:|---:|
| legacy | 10.90 ms | 2.154 s | 464 entries/s | 1,000 | 1,010 |
| 仅文件 batch | - | 1.076 s | 929 entries/s | 1 | 1,010 |
| 文件 + metadata batch | 1.25 ms | 24.72 ms | 40,456 entries/s | 1 | 17 |

完整 batch 的模型 I/O 从 2.010 s 降为 18 ms，实测 stage 总时间约 26 ms。小规模
延迟注入用于校验模型；它不能代表大规模真实后端的并行吞吐。

中等规模零延迟：10,000 个 field-index、100,000 条 SegmentIndex、每条一个文件：

| 模式 | SegmentIndex stage | 吞吐 | stage 累计分配 | 文件请求 | SegmentIndex KV 请求 |
|---|---:|---:|---:|---:|---:|
| legacy | 1.597 s | 62,615 entries/s | 284.6 MiB | 100,000 | 100,000 |
| 仅文件 batch | 733.9 ms | 136,262 entries/s | 83.5 MiB | 100 | 100,000 |
| 文件 + metadata batch | 715.0 ms | 139,853 entries/s | 89.6 MiB | 100 | 1,600 |

该规模下完整 batch 路径 wall time 比 legacy 降低约 55%，累计分配降低约 69%。

完整目标规模零延迟：1,000,000 个 field-index、10,000,000 条 SegmentIndex、每条
一个文件：

| 项目 | legacy 基线 | 仅文件 batch | 文件 + metadata batch |
|---|---:|---:|---:|
| field-index stage | 8.815 s | 9.215 s | 4.199 s |
| field-index stage 累计分配 | - | - | 263.8 MiB |
| SegmentIndex stage | 2m45.36s | 1m17.55s | 1m18.57s |
| SegmentIndex 吞吐 | 60,476 entries/s | 128,944 entries/s | 127,275 entries/s |
| SegmentIndex stage 累计分配 | 28.25 GiB | 8.69 GiB | 9.29 GiB |
| 模拟对象存储请求 | 10,000,000 | 10,000 | 10,000 |
| field-index KV 请求 | 1,000,000 | 1,000,000 | 15,625 |
| SegmentIndex KV 请求 | 10,000,000 | 10,000,000 | 160,000 |
| 峰值 RSS | 约 6.86 GiB | 约 6.78 GiB | 约 6.44 GiB |

完整 batch 相比 legacy：两个删除 stage 合计 wall time 从约 174.2 s 降为 82.8 s，
降低约 52%；SegmentIndex stage 累计分配降低约 67%。峰值 RSS 基本不变，因为它
主要来自测试常驻的 100 万 field-index 和 1000 万 SegmentIndex fixture；流式遍历
降低的是 GC pass 的额外快照与累计临时分配。

按 KV 请求 1 ms、对象存储请求 5 ms 的串行 mock 模型，完整 batch 的 I/O 时间为：

```text
KV 时间   = 175,625 * 1 ms = 175.625 s
文件时间  = 10,000 * 5 ms  = 50 s
合计      = 225.625 s       = 3m45.625s
```

这是请求模型，不是“1000 万对象物理回收只需 3 分 45 秒”的生产结论。真实后端仍
可能在服务端逐对象处理、排队、限流或重试，必须完成下一节的集成与在线验证。

#### dropped-segment GC A/B（已删除并发原型的历史结果）

`TestDataCoordDroppedSegmentGCLargeScalePerformance` 直接调用生产
`recycleDroppedSegments`，分别统计 DataView、channel marker batch、文件、SegmentIndex KV
和 segment Get/Txn 的逻辑调用数。

以下“有界并发”数据来自已经删除的 `recycleDroppedSegmentsConcurrently` 原型，仅保留为
方案演进依据；当前代码不能再通过配置切回该原型。当前代码已经把唯一 channel 状态查询
替换为精确 key batch，所有 segment 都进入后续统一 batch pipeline。

100 万 dropped segment、100 万 Collection、每 segment 一个 SegmentIndex 和一个文件，
所有 mock 延迟为零。这是每个 channel 只有一个 segment 的保守场景，无法通过 channel
去重减少请求：

| 项目 | 串行（并发 1） | 有界并发 16 | 变化 |
|---|---:|---:|---:|
| candidate stage | 40.368525 s | 19.357875 s | 2.085x；耗时 -52.05% |
| 吞吐 | 24,772 segment/s | 51,659 segment/s | 2.085x |
| stage 累计分配 | 9,399.3 MiB | 10,073.3 MiB | +7.17% |
| `ChannelExists` 调用 | 1,000,000 | 1,000,000 | 不变 |
| 其余各类逻辑删除调用 | 1,000,000 | 1,000,000 | 不变 |
| `/usr/bin/time` 最大 RSS | 4,905,400 KiB | 4,911,636 KiB | +0.13% |

100 万 dropped segment、10 万 Collection、每 segment 一个 SegmentIndex 和一个文件，
所有 mock 延迟为零。该场景平均每个 channel 有 10 个 segment：

| 项目 | 串行（并发 1） | 有界并发 16 | 变化 |
|---|---:|---:|---:|
| candidate stage | 39.694408 s | 16.004038 s | 2.480x；耗时 -59.68% |
| 吞吐 | 25,192 segment/s | 62,484 segment/s | 2.480x |
| stage 累计分配 | 8,959.9 MiB | 9,400.0 MiB | +4.91% |
| `ChannelExists` 调用 | 1,000,000 | 100,000 | -90% |
| 其余各类逻辑删除调用 | 1,000,000 | 1,000,000 | 不变 |
| `/usr/bin/time` 最大 RSS | 3,655,112 KiB | 3,649,480 KiB | -0.15% |

10 万 dropped segment、1 万 Collection、每 segment 一个 SegmentIndex 和一个文件，
所有 mock 延迟为零：

| 项目 | 串行（并发 1） | 有界并发 16 | 变化 |
|---|---:|---:|---:|
| candidate stage | 3.660281 s | 1.462448 s | 2.503x；耗时 -60.05% |
| 吞吐 | 27,320 segment/s | 68,379 segment/s | 2.503x |
| stage 累计分配 | 890.6 MiB | 935.2 MiB | +5.01% |
| `ChannelExists` 调用 | 100,000 | 10,000 | -90% |
| 其余各类逻辑删除调用 | 100,000 | 100,000 | 不变 |
| `/usr/bin/time` 最大 RSS | 1,942,784 KiB | 1,954,536 KiB | +0.61% |

1000 dropped segment、100 Collection，注入 DataView 1 ms、`ChannelExists` 1 ms、文件
5 ms、SegmentIndex KV 1 ms、segment Get/Txn 各 1 ms 的真实 wall-clock sleep：

| 项目 | 串行（并发 1） | 有界并发 16 | 变化 |
|---|---:|---:|---:|
| candidate stage | 10.520106 s | 0.621549 s | 16.93x；耗时 -94.09% |
| 吞吐 | 95 segment/s | 1,609 segment/s | 约 17x |
| `ChannelExists` 调用 | 1,000 | 100 | -90% |
| 文件、SegmentIndex KV、segment Get/Txn | 各 1,000 | 各 1,000 | 不变 |

各组历史测试最终都删除了全部目标元数据，逻辑删除计数没有丢失。延迟注入结果只说明
当时的并发原型能隐藏 mock I/O 等待，不代表真实 etcd/MinIO 能线性扩展。

当前正确性测试仍覆盖 channel marker 批量读取失败：统一 batch 路径会保留未解析 channel 的
segment metadata，待下一轮重试，而不是把读取错误当作 marker 不存在。

#### dropped-segment 分阶段批删 A/B（统一前的历史结果，2026-09-02）

以下结果用于决定淘汰 legacy 路径：当时通过 mock 是否暴露批量能力接口切换
legacy/batch。当前实现已统一为 batch pipeline，不再存在该运行期开关。当前性能测试中的
`MILVUS_GC_PERF_DROPPED_NATIVE_FILE_BATCH` 只切换文件 stage 是原生批删还是有界逐 path
fallback；SegmentIndex 和 segment metadata 的 batch 算法保持相同。

100 个 segment 的延迟注入校验使用 DataView 1 ms、`ChannelExists` 1 ms、文件 5 ms、
SegmentIndex KV 1 ms、segment KV 1 ms：

| 模式 | 模型 I/O | 实测 stage | 对象请求 | SegmentIndex 事务 | segment GET / 事务 |
|---|---:|---:|---:|---:|---:|
| legacy | 1.000 s | 1.062 s | 100 | 100 | 100 / 100 |
| batch 1000 | 119 ms | 139 ms | 1 | 2 | 0 / 2 |

100 万 segment / 10 万 Collection 的零延迟 mock：

| 模式 | GC stage | 吞吐 | stage 累计分配 | 最大 RSS | 对象请求 | SegmentIndex 事务 | segment GET / 事务 |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 38.814 s | 25,764/s | 8,968.3 MiB | 3,651,628 KiB | 1,000,000 | 1,000,000 | 1,000,000 / 1,000,000 |
| batch 1000 | 28.477 s | 35,116/s | 5,960.3 MiB | 3,550,732 KiB | 1,000 | 16,000 | 0 / 16,000 |

1000 万 segment / 100 万 Collection 的零延迟 mock：

| 模式 | GC stage | 吞吐 | stage 累计分配 | 最大 RSS | 对象请求 | SegmentIndex 事务 | segment GET / 事务 |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 6m58.793s | 23,878/s | 89,737.4 MiB | 33,149,644 KiB | 10,000,000 | 10,000,000 | 10,000,000 / 10,000,000 |
| batch 1000 | 5m22.019s | 31,054/s | 59,677.6 MiB | 32,516,068 KiB | 10,000 | 160,000 | 0 / 160,000 |

在千万规模下，batch 的 stage wall time 降低 23.11%，吞吐提高 30.05%，累计分配降低
33.50%；最大 RSS 只降低 1.91%，因为 1000 万 SegmentInfo/SegmentIndex fixture 才是
live heap 主体。测试期间同机 MixCoord/Proxy 保持运行且 MixCoord 有持续 CPU 负载，
因此 wall time 可能包含环境噪声；请求计数和最终状态断言不受该噪声影响。

按相同的 1 ms/5 ms 串行请求延迟模型，千万规模从 legacy 的 27h46m40s 降到 batch 的
3h09m30s。batch 模型中 1000 万次 DataView 检查本身占 2h46m40s，已经成为主项；这再次
说明不能把 mock 模型写成生产完成时间，也不能用一个可能过期的长期 `recoveredViews`
缓存直接消掉该检查。

#### 统一 batch pipeline 的 storage capability A/B（2026-09-02）

统一路径后，用 100 个 dropped segment、10 个唯一 channel、每 segment 一个索引文件做
5 ms 文件延迟注入；DataView、channel 和 metadata 延迟设为零，
`dataCoord.gc.removeConcurrent=16`：

| 文件实现 | GC stage | 吞吐 | 逻辑文件数 | 后端文件请求 | SegmentIndex 事务 | segment GET / 事务 |
|---|---:|---:|---:|---:|---:|---:|
| 基础 `ChunkManager.Remove` fallback | 39.698 ms | 2,519 segment/s | 100 | 100 | 2 | 0 / 2 |
| 原生 `MultiRemoveWithResult` | 8.486 ms | 11,784 segment/s | 100 | 1 | 2 | 0 / 2 |

两组都使用相同的 candidate batching 和 metadata batching，只切换 storage 能力。该小型
mock 验证了普通 `ChunkManager` 不再切回旧 GC 算法，并验证请求计数和 stage 顺序；结果
不是生产对象存储吞吐结论。

### 12.5 真实后端验证

使用 embedded etcd 和真实本地 MinIO/S3-compatible 后端，在较小但有代表性的规模
记录：

- 实际回收对象数和 KV 数。
- 客户端请求数。
- 有效 objects/s 和 KVs/s。
- batch latency P50/P95/P99。
- throttling 和 retry 数量。
- DataCoord CPU、RSS、heap allocation、goroutine 和 Go GC pause。
- GC 期间前台 DDL/search/query 延迟。
- pause 和 shutdown 延迟。

不能把 mock 推算写成生产完成耗时。

### 12.6 Go 测试命令要求

相关 Go 测试必须使用仓库要求的 flags：

```bash
source scripts/setenv.sh
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 ./internal/storage/... ./internal/datacoord/...
```

## 13. 验收标准

前三个实施阶段满足以下条件后才算完成：

1. 当 GC file batch 不超过 S3 的 1000-object 上限时，MinIO/S3 在无重试情况下，
   Multi-Delete 请求数不超过 `ceil(fileCount / 1000)`；更大单条候选由 SDK 拆分。
2. etcd 在无重试情况下，transaction 数不超过
   `ceil(keyCount / maxEtcdTxnNum)`。
3. logical deletion count 完全准确，所有符合条件的文件和 KV 均被回收。
4. 任一文件失败时保留对应 SegmentIndex 元数据；成功的兄弟 buildID 可以独立推进。
5. 所有正确性约束和故障注入用例通过。
6. SegmentIndex 总数增长时，候选 buffer、worker 和 in-flight request 仍然有界。
7. 优化后的 benchmark 分别报告实测 CPU、allocation、RSS、wall time 和模型延迟。
8. 真实后端测试确定可支持的 concurrency/batch size 默认值，不造成不可接受的
   前台延迟或持续限流。
9. 不暴露可选批量能力接口的后端仍走旧删除路径，不需要迁移元数据。

## 14. 待确认问题

1. 首个验收环境使用哪种对象存储，以及 bucket 是否开启 versioning/retention？
2. 清理 1000 万对象 backlog 时，可以接受多大的前台 P99 延迟回退？
3. 完成流式遍历后，生产数据是否证明反向
   `(collectionID, indexID) -> buildIDs` 索引值得其常驻内存开销？
4. field-index tombstone 延迟删除应该与 KV batch 同时修改，还是拆成独立生命周期
   变更？
