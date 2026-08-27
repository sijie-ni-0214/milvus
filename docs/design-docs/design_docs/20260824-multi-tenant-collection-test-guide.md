# 多租模式改动及测试指南

> 状态：草案 日期：2026-08-24 读者：测试、开发、SRE 范围：目标产品的 Collection-per-Tenant 多租模式，不代表通用 Milvus 的全部能力。

## 1. 产品介绍

多租场景是指同一个 Milvus 集群同时服务大量相互独立的用户或业务租户。租户共享集群基础设施，但各自拥有独立的数据空间和生命周期；一个租户的写入、查询、预热、驱逐、限流、故障或删除不能影响其他租户。典型场景包括 SaaS 平台为不同客户提供独立的向量检索服务。

本产品不再使用 Partition Key 或 Partition 隔离租户，而是让每个租户独占一个 Collection：

```Plain
1 Tenant = 1 Collection = 1 Business Shard = 1 Fixed Partition
```

Collection 是租户的数据、权限、查询资源、限流和生命周期边界。创建租户对应创建 Collection，删除租户对应删除整个 Collection。

产品目标是支持百万级 Collection，也就是百万级租户元数据。数据规模暂按记录条数口径定义为：单租户最多 500M，集群总数据量达到 5B。主要压力不再来自单个 Collection 内的大量 Partition 或 Shard，而是来自百万级 Collection、VChannel，以及由活跃租户产生的 Segment、QueryView 和监控时间序列。

## 2. 产品约束

1. 一个租户只能对应一个 Collection，一个 Collection 也只能属于一个租户。
2. 每个 Collection 固定只有一个业务 Shard 和一个 Partition。
3. 不支持 Partition Key。
4. Shard 和 Partition 由系统创建和维护，用户不能创建、删除或修改。
5. 创建 Collection 时不能指定 `shards_num > 1` 或 `num_partitions > 1`。
6. 不支持 CreatePartition、DropPartition 和 AlterPartition。
7. 不支持 LoadCollection、ReleaseCollection、LoadPartitions 和 ReleasePartitions，查询资源使用 Prewarm 和 Evict 管理。
8. Prewarm、Evict、Drop、Index 和 Compaction 均以 Collection 为管理单位。
9. 单租户数据量目标暂定为 500M 条记录。
10. 集群总数据量目标暂定为 5B 条记录。

禁止操作应明确报错，不建议静默转换成合法值。非法请求被拒绝后，不能残留 Collection、Partition、Channel、WAL、Prewarm 任务、查询资源或监控数据。

以下行为需要产品在测试前确认：

- 唯一 Partition 是完全隐藏，还是允许查看但只读；
- `shards_num=0`、`num_partitions=0`、负数和未传参数时的行为；
- Search/Query 指定 `partition_names` 时是拒绝、忽略，还是只允许唯一 Partition；
- Evict 返回成功时的完成边界，以及 Evict 后首次查询的预期行为；
- 历史多 Shard、多 Partition Collection 的升级和恢复策略。



## 3. 主要改动/优化


|                  |                                                             |                                          |
| ---------------- | ----------------------------------------------------------- | ---------------------------------------- |
| **方向**           | **主要改动**                                                    | **测试价值**                                 |
| 租户模型             | 租户隔离从 Partition/Partition Key 改为 Collection                 | 测试重点转为 Collection 隔离和大量 Collection 场景    |
| 元数据恢复            | RootCoord 批量扫描元数据；etcd 使用 RangeStream，并保留分页回退；DataView 分批恢复 | 降低大量 Collection/Segment 启动时的 RPC、内存和恢复耗时 |
| 查询资源管理           | 不提供 Load/Release；使用 Collection 级 Prewarm/Evict 管理查询资源       | 验证预热、驱逐及其与 QueryView、查询并发时的行为            |
| DDL 并发           | 缩短 RootCoord 全局元数据锁范围                                       | 不同 Collection 的 DDL 不再被无关慢操作长期串行化        |
| Drop/ACK         | ACK callback 使用非阻塞资源锁并合并唤醒；Tombstone 注册不阻塞                  | 降低 Drop 风暴下的锁冲突、日志风暴和调度开销                |
| 限流               | SetRates 更新不再通过全局读写锁阻塞请求检查；租户限流落在 Collection 级别             | 高频配额更新时减少全局延迟，并保证租户隔离                    |
| Index/Compaction | 减少重复元数据扫描，批量筛选候选；提交阶段执行每轮任务上限                               | 控制大量 Segment 下的 CPU、队列和任务风暴              |
| StreamingNode 调度 | 普通任务使用单 worker 唤醒；Collection/Partition 数量采用增量计数             | 降低大量 Collection 下的空载 CPU 和无效遍历           |
| Prometheus 指标    | 减少部分高基数标签；提供 full/aggregate Collection/VChannel 指标模式        | 控制大量租户带来的 series 数和抓取开销                  |




### 3.1 关联的独立改造：QueryView

QueryView 不是多租资源模型或多租优化的一部分，而是一项独立的查询链路改造。本文保留 QueryView 测试项，是为了验证它与百万级 Collection、单租户 500M 和集群 5B 场景叠加后的行为。相关设计文档：

- [QueryView 总体设计](https://github.com/milvus-io/milvus/blob/03e3e16f3e45889ce5d4b94c18cc75980d2742f8/docs/design-docs/design_docs/qviews/query_view_handler.md)
- [QueryView 状态机](https://github.com/milvus-io/milvus/blob/03e3e16f3e45889ce5d4b94c18cc75980d2742f8/docs/design-docs/design_docs/qviews/query_view_state_machine.md)
- [Query Client 与查询链路](https://github.com/milvus-io/milvus/blob/03e3e16f3e45889ce5d4b94c18cc75980d2742f8/docs/design-docs/design_docs/qviews/query/query_client.md)
- [QueryNode QueryView 资源准备](https://github.com/milvus-io/milvus/blob/03e3e16f3e45889ce5d4b94c18cc75980d2742f8/docs/design-docs/design_docs/qviews/qnview/querynode_queryview_resource_preparation.md)



## 4. 测试数据量

产品规模目标为：

```Plain
Tenant 数 = Collection 数 = Partition 数 = 业务 VChannel 数 = 1M 级别
单租户数据量 <= 500M
集群总数据量 = 5B 级别
```

测试数据需要区分 `C_total`（总 Collection 数）、`C_prewarmed`（已预热数）、`C_hot`（活跃读写数）、`D_tenant`（单租户记录数）、`D_total`（集群总记录数）和 `S_total`（Segment 总数）。

上述限制共同定义一套目标集群数据：Collection 总数达到 1M 级别，集群总数据量达到 5B 级别，任一 Collection 不超过 500M。大部分 Collection 可以为空或处于冷状态，5B 数据由部分活跃租户承载，并覆盖均匀和倾斜两种分布。

重点覆盖以下数据场景：


|                           |                                                                                 |                                     |
| ------------------------- | ------------------------------------------------------------------------------- | ----------------------------------- |
| **测试场景**                  | **数据形态**                                                                        | **主要验证内容**                          |
| 目标容量组合                    | Collection 总数达到 1M 级别，集群总量达到 5B 级别，任一 Collection 不超过 500M；大部分 Collection 为空或冷数据 | 组合容量下的元数据、存储、调度、查询、后台任务和故障恢复能力      |
| 均匀数据分布                    | 5B 数据较均匀地分布到活跃租户                                                                | 多租并行吞吐、调度公平性和逐租户开销                  |
| 倾斜数据分布                    | 少数租户接近 500M，其余租户为小数据量或空 Collection                                              | 热点隔离、长尾延迟和大租户对小租户的影响                |
| 冷热租户混合                    | 大量冷 Collection 中只有部分租户持续读写                                                      | 热租户时延、冷租户开销和 Prewarm/Evict 行为       |
| 高 Segment 密度              | 少量活跃 Collection 内包含大量 Segment                                                   | DataCoord、Index、Compaction 和查询视图压力  |
| 索引构建与更新                   | 空 Collection、已有大量 Segment，以及持续 Flush/Import 产生新 Segment                         | 任务发现、构建时延、状态流转、增量覆盖、查询正确性和资源公平性     |
| Compaction 积压与执行          | 大量小 Segment、Delete/L0 积压，多个 Collection 同时满足触发条件                                 | 候选发现、每轮任务上限、执行公平性、数据正确性和资源影响        |
| GC 积压与回收                  | 目标容量组合下批量 Drop，并由 Compaction、Index 失败或重建产生大量废弃对象和孤儿文件                         | 回收安全性、积压清理速度、对象存储开销和在线业务影响            |
| Collection 与 Segment 组合压力 | Collection 总数和 Segment 总数同时较高                                                   | 元数据与数据面叠加后的资源和恢复能力                  |
| 查询性能与稳定性                  | 目标容量组合下覆盖冷热查询及均匀/倾斜访问                                                           | 延迟、吞吐、成功率、结果正确性和长时间稳定性              |
| DDL 性能与并发                 | 在大规模存量下并发执行不同 Collection 的 DDL，并混合冲突 DDL                                        | 吞吐、尾延迟、顺序、锁竞争以及 WAL/ACK callback 耗时 |
| 生命周期变更                    | 在大规模存量上持续 Create、Prewarm、Evict、Drop，包括删除大租户                                     | 幂等性、大量数据清理、资源回收和长期泄漏                |
| 启动与故障恢复时间                 | 在上述数据形态下冷启动、重启 Coord/Node 或注入存储故障                                               | 各阶段耗时、恢复完整性和业务恢复时间                  |
| 容量边界                      | 在同一集群中逼近 1M Collection、单 Collection 500M 和总量 5B 的容量限制                           | 边界前后行为、错误语义及恢复后计数一致性                |


并发度、同时执行 Prewarm 的 Collection 数和 Segment 数不在本文中指定固定值，应由产品验收目标和测试集群规格确定。测试报告还需要固定并记录向量维度、Schema、索引类型、副本数和 Segment 生成方式，否则同样的记录数无法直接比较资源和性能结果。

## 5. 测试关注点



### 5.1 产品约束

- 默认创建和显式传 1，最终都只能得到一个 Shard、一个 Partition；
- `shards_num > 1`、`num_partitions > 1` 和 Partition Key 必须被拒绝；
- Partition 创建、删除和修改接口必须不可用；
- LoadCollection、ReleaseCollection、LoadPartitions 和 ReleasePartitions 必须被拒绝；
- Prewarm 和 Evict 只能作用于整个 Collection，不能指定 Partition；
- REST、gRPC、当前 SDK 和旧 SDK 的行为一致；
- 非法请求失败后没有任何元数据、WAL、Channel 或指标副作用。



### 5.2 Prewarm、Evict 与查询

- 分别在未 Prewarm、Prewarm 执行中、Prewarm 完成和 Evict 后发起 Search/Query，验证行为、结果及延迟；
- 验证 Prewarm 任务的状态、进度和失败原因准确，完成后对应查询资源已准备；
- 同一 Collection 重复 Prewarm 或重复 Evict 应保持幂等；Prewarm 与 Evict 并发时，执行顺序和最终状态必须明确；
- 对目标容量组合中接近 500M 的大租户，Prewarm、Search、Query、Delete 和 Evict 的结果仍需完整、稳定；
- Evict、Balance、Compaction、QueryNode/StreamingNode 重启期间，查询失败、重试和恢复语义应符合定义；成功请求不能漏查、重复或脏读；
- QueryView 进入 Unrecoverable 后能够生成替代视图并恢复服务。



### 5.3 启动与故障恢复时间

- 记录集群冷启动、单组件重启和整集群重启到服务可用的完整时间；
- 记录故障发生、被检测、开始恢复到读写重新可用的完整时间；
- 在 1M 级别 Collection、单 Collection 不超过 500M、集群总量 5B 级别的目标容量组合下验证；
- 拆分元数据读取、DataView/QueryView 恢复、Channel 恢复、Index/Compaction 任务恢复等阶段耗时；
- 大量空 Collection 和大量 Segment 两种模型分别验证，恢复后的 Collection/VChannel 数必须精确一致；
- 注入 etcd 超时、流中断、对象存储异常及 Coord/Node 重启；
- 恢复后 Collection、Segment、Index、Prewarm/Evict 和 QueryView 状态符合恢复语义，无孤儿或重复任务。



### 5.4 DDL 性能与并发

- 覆盖 Create/Alter/Drop Collection、Create/Drop Index 等 Collection 级 DDL；
- 大规模存量下并发操作不同 Collection，验证吞吐和尾延迟不会因全局锁出现非线性退化；
- 同一 Collection 的冲突 DDL 必须保持顺序，不同 Collection 的 DDL 应能并行；
- 覆盖 Create/Drop 混合、Drop 后同名重建、DDL 与读写并行以及超时重试；
- 记录端到端延迟、DDL queue、资源锁等待、WAL broadcast、ACK wait 和 ACK callback 各阶段耗时；
- WAL append、ACK、callback 任一阶段超时或重启后，DDL 任务可恢复且最终效果幂等；
- 百万 Collection 下，单个 Collection 的 DDL 不应触发全量扫描；
- 在目标容量组合中删除接近 500M 的大租户时，不能长期阻塞其他 Collection 的 DDL，后台数据和索引需要最终清理；
- Drop RPC 成功后确认名称、元数据、VChannel、QueryView、Limiter 和指标已清理，Segment 和 Index 已进入预期 GC 流程；不能只以 ACK 数量或 Collection 不可见作为完成依据，物理文件回收时间单独记录。



### 5.5 查询性能与稳定性

- 在目标容量组合下验证 Search、Query 及带过滤条件查询，并覆盖接近 500M 的大租户；
- 区分未 Prewarm 首次查询、Prewarm 完成后首次查询、稳定热查询和 Evict 后首次查询，分别记录性能；
- 覆盖均匀访问、热点倾斜和冷热租户混合，观察吞吐、P50/P95/P99、成功率、超时和重试；
- 查询期间并行执行写入、Flush、Index、Compaction、Prewarm/Evict 和 Collection DDL，确认性能可控且结果正确；
- QueryView 切换、Balance、Proxy/QueryNode/StreamingNode 重启及网络异常期间，查询应可恢复且不能漏查或重复；
- 长时间持续查询时，延迟和错误率不能持续恶化，队列、内存、goroutine 和缓存不能泄漏；
- 除性能指标外，还需验证结果正确性；近似索引场景应同时记录 Recall。



### 5.6 索引测试

- 覆盖先写入后建索引、先建索引后持续 Flush/Import 两种顺序，所有可索引 Segment 最终都应完成索引构建；分别记录 CreateIndex 返回时间和索引全部 Ready 的时间；
- 覆盖产品支持的向量和标量索引，验证参数校验、状态流转、查询结果；近似索引同时验证 Recall；
- 在目标容量组合下对多个 Collection 同时建索引，验证任务排队、构建时间、资源占用和 Collection 间公平性；
- 百万级 Collection 和高 Segment 密度下，只新增少量待建索引 Segment，确认单次增量事件不会触发全量 Collection/Segment 扫描；
- Compaction 生成新 Segment 后应继续构建索引并正确切换，不能出现漏建、重复构建或查询结果异常；



### 5.7 Compaction 测试

- 在目标容量组合下覆盖产品启用的 Compaction 类型，以及大量小 Segment、Delete/L0 积压和接近 500M 的大租户；
- 高 Segment 密度下持续产生少量新候选，验证候选批量筛选和触发开销，单次事件不能反复扫描全部 Collection/Segment；
- 多 Collection 同时触发时，每轮任务提交上限必须生效，只统计成功提交的任务，并验证 Collection 间公平性；
- Compaction 前、中、后验证行数、Insert/Delete、MVCC、查询结果、Recall 和索引可用性，不能漏数据或重复数据；
- 覆盖 Compaction 与写入、Flush、Index、Prewarm/Evict、Search 及 DropCollection 并发；
- 注入 DataCoord/DataNode 重启、任务失败和对象存储异常，验证任务幂等恢复，无重复输出 Segment、孤儿元数据或文件；
- DropCollection 时，验证排队中和执行中任务的取消或完成语义，相关 Segment、索引和任务元数据最终清理；
- 记录 Compaction 对查询和写入尾延迟、CPU、内存、I/O 及任务队列的影响。



### 5.8 GC 测试

- 在目标容量组合下持续执行 Drop、Compaction 和 Index 重建，并注入孤儿文件，形成 GC 积压；
- 覆盖废弃 Segment 及日志、Index/Analyze/Text/JSON/LOB 文件、DataView/快照和 Channel checkpoint 等产品启用的回收对象；
- 未达到安全窗口或仍被 Segment、DataView、快照引用的对象不能删除；引用解除并超过安全窗口后应最终回收；
- 覆盖全局及 Collection 级暂停、恢复和自动过期；暂停的 Collection 不被回收，其他 Collection 的 GC 可继续执行；
- 注入 DataCoord 重启、对象存储 List/Delete 超时和部分删除失败，验证重试幂等，无误删、重复元数据或永久积压；
- GC 与写入、查询、Prewarm/Evict、Index、Compaction 和 DropCollection 并发时，成功请求结果正确，存活对象不能被误删；
- 区分 Drop 逻辑完成时间和物理文件回收时间，记录待回收对象数与字节数、扫描/删除吞吐、对象存储请求量及 GC 对 CPU、内存、I/O 和在线延迟的影响。



### 5.9 数据面性能与公平性

- 对比 5B 数据均匀分布和热点倾斜分布，确认大租户不会使小租户长期饥饿；
- 高频 SetRates 期间不能出现所有租户共同的延迟尖峰；
- 大量 Index/Compaction 任务下，每轮提交上限生效，失败提交不计入成功数；
- 热租户不能让冷租户长期饥饿；
- 大量空闲 Collection 时，StreamingNode 空载 CPU 不应异常增长。



### 5.10 长稳与可观测性

- 持续创建、Prewarm、查询、Evict 和删除，检查内存、goroutine、FD、元数据和指标是否泄漏；
- 对 CreateCollection、首次查询、Create/DropCollection 和恢复建立 P50/P95/P99 基线；
- 百万 Collection 规模重点验证 `aggregate` 模式的 series 数、抓取耗时和聚合语义；`full` 模式先在受控规模验证明细准确性；
