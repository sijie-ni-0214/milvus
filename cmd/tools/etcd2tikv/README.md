# etcd2tikv 迁移演练

工具把 Milvus 默认布局中的 `<root>/kv/` 和 `<root>/meta/` 复制到专用 TiKV，排除 `<root>/meta/session/`。Session、选主、配置覆盖层和 Woodpecker 继续使用 etcd。源端需要支持 RangeStream 的 etcd 3.7，下面的命令适用于集群内的明文端点和默认的 `kv`、`meta` 子路径。

## 准备环境

在仓库根目录构建 Linux amd64 的迁移工具，并为独立 namespace 生成 TiKV 清单。这里的 200 GiB 是每个 TiKV store 的演练起始容量。三副本会在每个 store 保留一份数据，还需为 MVCC、Raft 和 compaction 留出空间。三个 PD 和三个 TiKV 分别分布在至少三台可调度节点上。

```bash
set -euo pipefail
mkdir -p bin
GIT_DIR="$(git rev-parse --absolute-git-dir)" GIT_WORK_TREE="$PWD" \
  CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath \
    -o bin/etcd2tikv-linux-amd64 ./cmd/tools/etcd2tikv
go version -m bin/etcd2tikv-linux-amd64 | grep vcs.revision
RELEASE_NAME=tikv-rehearsal NAMESPACE=tikv-rehearsal DRY_RUN=1 \
  PD_IMAGE=pingcap/pd:v8.5.8 TIKV_IMAGE=pingcap/tikv:v8.5.8 \
  TIKV_REPLICAS=3 TIKV_STORAGE_SIZE=200Gi PD_STORAGE_SIZE=20Gi \
  bash tests/_helm/scripts/deploy-nightly-tikv.sh > bin/tikv-rehearsal.yaml
```

清单包含 Service、PD 和 TiKV StatefulSet。演练时在 `tikv-rehearsal` namespace 应用清单，核对 StorageClass、六个 PVC 和 Pod 放置，确认三个 PD 成员健康、三个 TiKV store 均为 Up。记录节点规格、Pod 的 CPU 和内存限制，供解释性能结果使用。迁移进程需要能访问 PD 返回的 TiKV Pod 地址。

直接读取现有 Milvus 集群的 etcd。迁移工具对源端只执行 Get 和 RangeStream，写入演练专用的 TiKV。启动演练 Milvus 时，服务发现、配置、对象存储和 WAL 的隔离要求见下文。

## 从现有 etcd 切分、拷贝与复核

在能访问源 etcd 和演练 TiKV 的 Linux amd64 工具容器里运行，使用 Bash、etcdctl 3.7 和 jq。先扫描 key 来确定分片边界。分片文件覆盖整个迁移前缀，源端增删 key 后仍然可用。切分完成后，读取一个分配器 key，将响应头里的当前 revision 记在演练目录中。拷贝和随后的自动逐键校验都读这个版本。

```bash
set -euo pipefail
: "${ETCD_ROOT:?set the source Milvus etcd.rootPath}"
: "${SOURCE_ETCD:?set the source etcd endpoint}"
root="$ETCD_ROOT"
: "${REHEARSAL_PD:?set the dedicated PD endpoint}"
: "${SOURCE_READ_RATE:?set the aggregate source read limit in keys per second}"
tool=./etcd2tikv-linux-amd64
source_args=(-etcd "$SOURCE_ETCD" -etcdRoot "$root" -checkAllocators
             -rangesFile "$PWD/ranges.json" -ranges 8 -rate "$SOURCE_READ_RATE")
target_args=(-pd "$REHEARSAL_PD" -tikvRoot "$root")

"$tool" -mode split "${source_args[@]}"
revision=$(etcdctl --endpoints="$SOURCE_ETCD" get "$root/kv/gid/idTimestamp" -w json |
           jq -er 'select(.count == 1) | .header.revision | select(. > 0)')
printf '%s\n' "$revision" > source.revision
"$tool" -mode copy "${source_args[@]}" "${target_args[@]}" -revision "$revision" -writers 8 -preSplit
```

`-checkAllocators` 检查该版本的 ID 和 TSO 两条 8 字节记录。`-rate` 按每秒 key 数限制切分、拷贝和校验的合计并发读取速率。`-preSplit` 在采样边界预切目标 region 并等待 scatter。工具在目标迁移范围为空时才开始拷贝，按 key 数和字节数分批写入，允许单 region 事务使用 1PC。拷贝和校验期间保持目标独占。

源端可以继续写入。高于记录 revision 的写入属于后续状态。连接中断时，工具在相同 revision 从最后处理的 key 之后继续。如果 etcd 已清理该版本，工具报错退出。若源端按 revision 自动压缩，实际可读时长取决于保留的 revision 数、全集群写入量和压缩调度。对于大批量数据，需要在该版本仍可读时完成拷贝和校验。如果版本提前失效，已提交的目标批次可能仍在。重跑应选空目标，或确认目标独占后使用 `-overwrite` 清空目标迁移范围，再从新的 revision 全量开始。该参数在 TiKV 执行物理 DeleteRange。

如需另行复核，可在该版本仍可读时执行以下命令。

```bash
"$tool" -mode verify "${source_args[@]}" "${target_args[@]}" -revision "$(cat source.revision)"
```

成功条件为进程退出码 0，且 missing、extra、different 均为 0。记录 revision、拷贝和校验耗时、key 数、value 字节数、各前缀计数，以及 etcd、PD、TiKV 的 CPU、RSS、磁盘和事务延迟。若预计完整演练长于历史版本保留窗口，可在已安排的元数据停写窗口使用 `-revision 0`，保持迁移前缀从拷贝开始到校验结束都静止。这个模式允许各个范围和断流重连分别读取最新 revision。

## 后续实测

- 用合成数据和带本分支修复的 Milvus 镜像做空集群启动、新 pchannel 首个 checkpoint、DDL/DML、flush、compaction 和完整重启；在有副本的环境分别观察 TiKV store 故障与 PD leader 切换后的重试行为。
- 以实际元数据写入速率测试 p99，持续写入 24 小时观察 GC safepoint 和磁盘增长，验证备份及恢复。拷贝期间目标保持独占；启动 MixCoord 会启动它的 TiKV GC worker。
- 在演练 TiKV 中用复制的真实元数据记录拷贝、独立复核和元数据加载的耗时与峰值内存。需要测 MixCoord 加载时，先用合成数据验证隔离配置，使用专用服务发现和只读对象存储，关闭可能写外部资源的任务，保持 StreamingNode 停止。
- 正式切换需另定停写窗口。放行 StreamingNode 后，checkpoint 推进可触发 WAL 截断，新写入也只存在 TiKV。简单切回旧 etcd 的回滚边界在 StreamingNode 启动前，且要求尚无新的元数据变更。
