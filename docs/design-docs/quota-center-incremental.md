# Incremental QuotaCenter with full-snapshot recovery

This change targets large, mostly idle catalogs in MixCoord. DataCoord caches absolute statistics by collection; RootCoord reuses a small metadata projection and recomputes affected policies. Proxy applies versioned full snapshots or the latest delta. There is no historical replay queue and no per-Proxy copy of policy state.

## Activation

All four options default to false:

| Option under `quotaAndLimits` | Effect |
| --- | --- |
| `collectionStatsCacheEnabled` | Use per-collection segment statistics instead of two periodic global segment scans. |
| `localQuotaCalculationEnabled` | Enable the in-process statistics interface, metadata projection, affected collection calculation and incremental display metrics. Requires statistics caching and a local MixCoord source. |
| `versionedRatesEnabled` | Startup option for coordinator and proxies: negotiate versioned snapshots and enforce ordering. |
| `incrementalRatesEnabled` | Send a delta when the receiver is at its base version. Turning this off preserves versioned FULL and stale-message rejection. |

Roll out statistics caching, then local calculation, then versioned FULL, then deltas. Before enabling deltas, measure FULL construction, transport and application at the maximum catalog size and relevant deny/default configurations. All-default catalogs, finite rates on every collection, partition limits, and mass deny states have different sizes. A full snapshot that cannot fit the transport limit or finish within the timeout is not a usable recovery mechanism.

A Proxy started with `versionedRatesEnabled` rejects rate-checked requests with a retriable `ServiceNotReady` until it installs its first baseline, either a versioned or a legacy FULL. Versioned Proxy sessions do not accept a later unversioned update. Rolling back to a controller without this protocol requires quiescing the old publisher and restarting/resetting Proxy sessions in a controlled rollout. Network isolation preserves the last valid policy; this change does not add a lease or fail-closed deadline.

## Statistics publication

DataCoord segment writes stay lock-free on the segment cache. A writer holds the per-segment writer locks, persists through the optimistic segment store and publishes the new values with a versioned compare-and-swap (`PublishSegment`). Only an accepted publication counts: a stale version, a tombstone or an aborted update invalidates nothing.

After a segment is published, it invalidates its old and new collections. The dirty table is guarded by a small leaf mutex. While no statistics reader exists, an atomic tracking flag skips that bookkeeping, so the flags-off write path takes no additional lock. A refresh reads a collection's dirty generation before copying its segment pointers. It therefore attaches a generation only to a snapshot at least as new as the writes that generation covers. A write it may have missed leaves a newer generation for the next refresh.

A committed compaction or other multi-segment mutation is installed by `CachedSegmentsInfo.PublishSegments` under the cache's `pubMu`. The pointer copy of a refresh runs under `pubMu.RLock`, the same consistent cut used by the loadable DataView projection. A refresh therefore never sees half of a compaction. Persistence and RPC work stay outside `pubMu`.

Statistics refresh copies only the changed collection's segment pointers, reduces them outside every lock and publishes an absolute replacement. The cache owns a separate refresh/publication mutex; writers never take it. The nested lock order is statistics mutex, then `pubMu`, then the dirty-table mutex. Collection snapshots, the global byte sum and pending absolute changes are visible together under the statistics mutex.

A later metadata write does not prevent publishing a consistent older snapshot. Its newer dirty generation remains for the next cycle. Concurrent refreshes serialize, and older generations cannot overwrite newer results. This avoids starving continuously written collections.

Each refresh takes ownership of the current nonempty dirty map and gives writers a fresh work set. Acknowledged pending maps are compacted after large deletions, and returned maps are sized by their live entries. This avoids retaining a million-entry hash table as the backing storage for otherwise empty or sparse periodic work.

Quota bytes, per-partition bytes and L0 deletes count healthy, non-importing segments. Total rows count healthy segments including importing segments. Missing collection metadata suppresses display labels but does not remove healthy segment contributions from admission statistics.

Cached segment objects are read-only after publication. Local row updates (`SetRowCount`) republish a clone through `Cache.Transform` and invalidate their collection. Other local updates change no statistics input and do not invalidate. These include positions, allocations, expiry, flush/written time and the compaction flag. Import/copy callbacks, storage updates and state changes go through `UpdateSegmentsInfo` operators and `publishSegments`; stale persistence revisions do not invalidate statistics. Tests compare cached and full-scan results through repeated state, row, import, deletion and stale-version updates. One of these tests goes through the production meta writers: `AddSegment`, multi-segment `UpdateSegmentsInfo`, `SetRowCount` and `DropSegment`.

Metadata dirty work and unacknowledged statistics changes are separate. The local consumer copies input and marks affected policy scopes before acknowledging its generation. A diagnostic full read does not acknowledge an active local consumer. Disabled statistics tracking and a disabled local consumer discard unnecessary pending state; enabling reconstructs a full baseline.

Baseline generations distinguish that reconstruction from ordinary updates. The receiver replaces its local statistics on a full baseline, removing entries no longer present. An acknowledgement from an older baseline cannot clear the new baseline requirement. MetaTable similarly records a reset while local tracking is disabled, including a disable/re-enable between calculation cycles.

## Local calculation

MetaTable records changed collection IDs only at successful cache publication boundaries. Collection writes go through `putCollectionLocked`, including restore completion, when a restoring collection becomes available. Partition edits call the same invalidation. A snapshot copies IDs, names, DB ownership, properties and partitions, without cloning schemas or other collection metadata. The available partition projection used by node membership and the complete partition list used by baseline limits preserve the existing distinction.

RootCoord retains this projection and absolute DataCoord inputs. It recomputes changed collections, current and previous node effects, L0 effects, and previously restricted scopes. Cluster/DB policies are recomputed each round. DB byte totals are updated by replacing each collection's contribution. Configuration or Proxy-count changes expand calculation to the full catalog.

The original rule order remains: defaults, KMS, write protections, read protections and DB DDL rules. Disk exhaustion still exempts Delete. TimeTick checks continue every round. A collection leaving a pressure set is recomputed to clear its previous restriction. Collection deletion does not prematurely deduct remaining healthy segments from global usage.

Pressure effects are recorded where a limiter is actually changed, including streaming TimeTick-derived collections not enumerated in the sampled node membership. The next calculation resets those scopes before applying current protections. Deleting a database enumerates its controlled descendants before detaching its tree, so later full snapshots cannot resurrect orphan policies.

Configuration, catalog generations and Proxy count are checked after calculation and immediately before publication. The calculation captures the Proxy count used to split all published rates; a count change rejects that candidate and forces recalculation. A failed candidate never changes the published version or immutable wire policy. Input dirty work remains pending. Synchronization can still repair a Proxy using the last complete publication even when fresh sampling/calculation failed.

Calculation and wire publication are serial in the first implementation. FULL may therefore delay the next calculation; there is no unbounded asynchronous transmission queue. Node reports still cost O(reported membership), and a continuously changing large collection still requires scanning its own segments.

## Wire protocol

`SetRatesRequest` adds protocol/mode, `(epoch, seq)` target/base versions, complete scope-policy upserts, explicit deletions, scope defaults and a per-process Proxy session token. The coordinator obtains its epoch from the initialized TSO allocator. Sequence numbers advance only on semantic policy changes, independent of map iteration order.

The controller stores current policies, the current version and one latest delta. It builds a full protobuf tree only when needed; clients share its immutable payload and clone routing headers. A defaults change uses FULL. The unchanged-publication path keeps the existing latest delta.

For current version v12 and latest delta v11 -> v12:

| Proxy state | Action |
| --- | --- |
| v12 | No policy payload; continue periodic reconciliation. |
| v11 | Send the latest delta. |
| Older, unknown or newly started | Send the current FULL. |
| Newer than publisher | Report stale publisher; never roll the Proxy back. |

Capability/version discovery uses a read-only `quota_sync` metrics request. For an unknown client, successful normal metrics without explicit capability selects legacy FULL. A failed query does not prove legacy capability. Previously negotiated capability is retained for the client session; an unknown applied version then uses versioned FULL. Never probe an old SetRates endpoint using an empty snapshot: it would reset its limits.

The receiver serializes control updates, verifies its process token, rejects stale epochs/versions, deduplicates equal versions, and checks a delta's exact base version. A new epoch requires FULL. It validates rates, hierarchy and state/code arrays before modifying buckets. Upserts replace complete policies over pinned defaults, clearing omitted old overrides and empty deny sets. Explicit parent deletion removes its controlled subtree. Lazy request-created buckets do not establish controller-owned scope membership.

FULL restores missing policies and applies defaults to existing runtime buckets; it does not recreate all buckets. DELTA only updates named scopes. Equal versions do not call `SetLimit`; existing buckets and Cancel/refund behavior are retained. Replies use `common.Status.ExtraInfo` for applied/already_applied/need_full/stale, version and process identity, without changing merr code projection.

Control updates serialize with each other, while admission checks remain concurrent. Prevalidation prevents a rejected request from partially changing live buckets; a successful multi-scope update is applied progressively, as on the existing path. This does not introduce a request-wide atomic admission snapshot.

Lost final requests are found by periodic reconciliation. Lost replies are resolved by the applied version. Missing intermediate changes trigger FULL. A restart between sampling and applying returns the new process identity and requires a new full baseline. There is no historical per-Proxy change ledger.

## Observability and verification

New metrics have bounded labels only:

- `milvus_datacoord_quota_cache_work_total{kind}`: recomputed collections and scanned segments.
- `milvus_datacoord_quota_cache_pending{kind}`: metadata work and consumer acknowledgements pending.
- `milvus_rootcoord_quota_calculated_scopes_total`: locally recomputed scopes.
- `milvus_rootcoord_quota_sync_total{mode,result}` and `quota_unsynced_proxies`: per-Proxy reconciliation outcomes aggregated without Proxy/version IDs as labels.
- `milvus_rootcoord_quota_snapshot_bytes{mode}`: encoded constructed FULL payload size.

Display metrics update only changed contributions on the local path. Full diagnostics and aggregate/full mode changes rebuild display state. Statistics revisions prevent an older diagnostic snapshot from replacing newer display data. Drop and rename clean up collection labels; aggregate cleanup removes a series only after its final contribution disappears. RootCoord records only active quota-state nodes but retains the existing per-successful-round force-deny counter behavior. The source baseline constructs an empty `DataCoordCollectionInfo.IndexInfo`; this change preserves that behavior and does not add index statistics.

Focused tests cover reference equality, importing eligibility, row updates, stale revisions, acknowledgement races, compaction publication, continuously changing collections, baseline resets, local/full rule equality, DB/property/Proxy-count changes (including between calculation and publication), pressure release, deletion, lost request/reply/final update, full fallback, receiver restart, stale full snapshots, complete prevalidation, token-bucket identity, concurrent admission/refunds, and preservation of new fields in the hand-written RPC envelope clone.

## Local validation, 2026-09-26

Implementation baseline: `392ac7b98c728dc17f3160c276f05b96f2e10086`. Tests use `-tags dynamic,test -gcflags="all=-N -l"`; focused behavior tests also use `-race -count=1`. Native libraries are reused from the local Milvus build, so this is not a clean native rebuild.

Passed checks:

- Focused tests across DataCoord, RootCoord, Proxy, ProxyClientManager, quota protocol utilities and the distributed Proxy client, with the race detector.
- Metrics registration/cleanup, quota configuration and merr contract/guard checks, with the race detector; the full merr package also passed separately.
- Changed-line lint for the touched main-module and pkg packages.
- The C++ formatting script and `git diff --check`.

Broader checks did not pass. Repository-wide `make lint-fix` encounters an existing missing `mocks` import in `internal/metastore/kv/querycoord/kv_catalog_test.go`. Expanded tests encounter existing metrics label-cardinality and configured-default mismatches, DataCoord mock/setup failures, and a testify/context inspection race; those test failures were reproduced from an untouched archive of the baseline commit. `make test-go` was attempted using the existing native build and stopped after native-loader, external-service and script-path failures. These checks remain outstanding; focused success is not a claim that the full repository suite passes.

Local benchmarks ran on Apple M5, darwin/arm64, with optimization/inlining disabled and without the race detector. The idle cases ran for 100 iterations:

| Measured operation | Collections | Time/op | Allocation/op |
| --- | ---: | ---: | ---: |
| DataCoord idle statistics read | 1,000 | 0.512 microseconds | 96 B |
| DataCoord idle statistics read | 100,000 | 1.186 microseconds | 96 B |
| RootCoord idle prepare/calculate/publish | 1,000 | 0.545 milliseconds | 72,310 B |
| RootCoord idle prepare/calculate/publish | 100,000 | 0.529 milliseconds | 72,311 B |

Idle statistics tests assert zero scanned segments. RootCoord figures exclude node polling, statistics-source reading and RPC. They demonstrate scale-independent idle work for these fixtures, not a complete cycle latency or an online improvement ratio.

The large FULL cases use 1,050,104 collection IDs, no partition nodes, and one measured iteration. Zero overrides and six finite overrides per collection are different fixtures, not two samples from production. Payload and allocation units below are decimal MB/GB.

| Fixture | Build + encode + decode | Encoded payload | Build/encode/decode allocation | First Proxy apply | Apply allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero overrides | 1.593 s | 16.802 MB | 0.487 GB | 2.921 s | 1.926 GB |
| Six finite overrides/collection | 3.601 s | 98.710 MB | 1.099 GB | 4.195 s | 1.928 GB |

The encoding microbenchmark excludes the four small defaults; the receiver fixture includes pinned unlimited defaults and uses aggregate metrics. Receiver timing excludes transport and encoding. Allocated bytes are cumulative allocations per operation, not retained heap. Separate benchmark times cannot establish that an actual FULL RPC meets the existing 10-second timeout.

## Port to the QV stack, 2026-10-15

The implementation was ported from `392ac7b98c` onto `c982b33e56`, the tip of the rebased QV stack. It was ported onto `e1dff0faba` first and then rebased over the CAS segment-store encoding fix (`b137d2f74c`) and the schema lookup bypass (`c982b33e56`), which needed no further change. The stack re-implements DataCoord segment persistence on the optimistic CAS store and the lock-free segment cache. RootCoord `quota_center.go`, the Proxy limiter, the Proxy client envelope and `proxy.proto` were identical to the original base. Those parts carry over unchanged, and `proxypb` was regenerated from the proto. The adaptations are:

- DataCoord has no global publication gate on every cache write. Invalidation follows the accepted `Swap`/`Erase`/`Transform` result of the write (see Statistics publication). `Cache.UpdateWithResult` is not needed.
- `pubMu` moved from `meta` into `CachedSegmentsInfo`. `meta.publishSegments` installs through `PublishSegments`. `selectSegmentsConsistent`, the external-refresh snapshot and the statistics refresh use `ReadConsistent`.
- `SetRowCount` is the only local update that invalidates. The original invalidated on every local update, including per-insert allocation updates.
- `MetaTable.CompleteRestoreCollection` was added after the original base, and the original hooks missed it. Published collection writes now go through `putCollectionLocked`, which invalidates the projection. Startup reload loads the metadata in bulk and requests one full read rather than marking every collection.
- The YAML comments moved into the parameter `Doc` strings, matching the format that the configuration generator writes.

Checks on Apple M5, darwin/arm64, against a native build of the stack tip. They use `-tags dynamic,test -gcflags="all=-N -l"`, and the four flags were switched with environment variables:

- `go vet` passes on the touched packages, and `gofmt`, `gofumpt` and `gci` find nothing on the changed files. `gofumpt` also reports blank lines in `meta.go`, which exist at the base and were not changed here.
- The full DataCoord package was run with the flags off and with all four on. Both runs fail only `TestGetExpectedSegmentSize/all_DISKANN` and `TestValidateImportFilePaths_SymlinkedInternalDir`, which are existing macOS failures. Two mockey tests that abort on darwin/arm64 were skipped.
- The full RootCoord package passes with the flags off. With all flags on, only `TestQuotaCenter/test_setRates` fails, because it asserts the legacy `SetRates` call that `versionedRatesEnabled` replaces.
- The Proxy package was run with `-skip TestProxy`, which hits an AWS CRT abort in this environment. It fails only `TestCachedProxyServiceProvider_DescribeCollection_IDOnlyOldRootCoordUsesOneDescribeAndKeepsUnknownDB`, which also fails at the base. ProxyClientManager, quota protocol utilities and the distributed Proxy client pass.
- The focused DataCoord, RootCoord, Proxy, ProxyClientManager, quotasync and Proxy client tests pass with `-race`.
- New tests cover these cases:
  - cache/full-scan equality through the production meta writers;
  - a concurrent `PublishSegments` compaction against statistics refresh;
  - restore completion invalidating the projection;
  - an unversioned Proxy behind a versioned coordinator, which installs the legacy FULL and refuses versioned requests.
- Mutation checks:
  - DataCoord: dropping the refresh read lock or the `PublishSegments` write lock, each publication/drop/row invalidation, the enabling pass, or the collection-label invalidation fails a test.
  - Proxy: accepting older versions, skipping the delta base check, letting a legacy request downgrade a versioned session, or letting an unversioned Proxy accept the protocol fails a test.

Idle benchmarks on the port, 100 iterations each:

| Measured operation | Collections | Time/op | Allocation/op |
| --- | ---: | ---: | ---: |
| DataCoord idle statistics read | 1,000 | 0.497 microseconds | 96 B |
| DataCoord idle statistics read | 100,000 | 0.527 microseconds | 96 B |
| RootCoord idle prepare/calculate/publish | 1,000 | 1.42 milliseconds | 144,356 B |
| RootCoord idle prepare/calculate/publish | 100,000 | 1.45 milliseconds | 144,356 B |

The RootCoord idle cost is higher than at the original base. The largest allocation is the configuration snapshot (`GetComponentConfigurations`), and the stack has more configuration keys. The cost does not grow with catalog size. The large FULL measurements above were not repeated.

Before enabling the flags, validate real full-snapshot construction/transport/application at maximum catalog and partition sizes, defaults and mass-deny states against message limits and timeout. Also measure the shared publication gate under sustained metadata writes (including flags off), retained heap, QuotaCenter cycle time, CPU/GC and request latency on the deployed workload. No online A/B or CPU/GC improvement is claimed by this implementation.
