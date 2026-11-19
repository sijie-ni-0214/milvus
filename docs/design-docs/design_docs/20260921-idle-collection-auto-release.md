# Automatic Release of Idle Collections

## Goal and assumptions

In streaming mode, QueryCoord releases collections after their time to idle (TTI) or when Worker QueryNode memory pressure calls for evicting the least recently accessed collections. Subsequent queries use the existing Load on Search path to reload them.

- The feature is effective only when both `queryCoord.autoRelease.enabled` and `proxy.enableAutoLoad` are enabled. It depends on the existing complete DQL retry: a collection-not-loaded, view-not-found, or view-invalidated error during loading or execution sends the request back through automatic loading before another query attempt.
- The TTI should be substantially longer than normal query latency. It measures time since the latest query attempt. Request completion does not refresh it; there are no in-flight request counts or completion notifications.
- Loading, a long-running query, or an iterator waiting for its next page does not prevent expiration. Retries remain bounded by the original request deadline and retry limit, so repeated releases or slow loading can still cause the request to fail.
- Collections loaded manually or automatically are eligible. The collection property `collection.autorelease.tti.seconds=-1` exempts a collection from both release triggers.

## Component responsibilities and flow

**Proxy** sends one `EnsureCollectionReady` RPC per complete DQL attempt of Search, HybridSearch, and Query, and per read attempt of a Delete by expression, partial Upsert or AutoID Upsert, and retries the whole request or read on a recoverable release or view error. QueryCoord shares load submission and the readiness wait among all callers of a collection.

**QueryCoord** keeps an in-memory `collectionID -> lastActiveAt` map. Existing readiness RPCs refresh activity, and a periodic scan of load configs selects candidates for release. Requests from different Proxies update the same collection timestamp using QueryCoord's local clock.

**SQN and Worker QN** handle release through their existing QueryView and resource-reference lifecycles. Automatic release submits the existing `DropLoadConfig` broadcast.

```text
Proxy receives DQL
  -> EnsureCollectionReady refreshes existing activity on entry
  -> QueryCoord loads and waits for readiness if needed
  -> A successful shared load/wait refreshes existing activity again
  -> Execute the query
  -> Repeat the flow on a recoverable release/view error

QueryCoord periodically scans load configs
  -> No activity record: initialize it to the current time; skip release this round
  -> Queue TTI expirations and the oldest ready collections in pressured groups
  -> Return without waiting; queued releases start whenever fewer than releaseConcurrency run
  -> Recheck eligibility, then run the existing release path
```

### Activity timestamps

- `EnsureCollectionReady` refreshes an existing record on entry.
- A caller that joined a shared load or wait refreshes the record again after readiness is confirmed, immediately before returning success. It does not refresh activity while waiting or when returning an error.
- Readiness RPCs only refresh records already initialized by the scanner. Request completion and internal requery stages do not refresh activity separately. A complete DQL retry refreshes it through the next initial check.
- DML other than those reads, load-progress requests, monitoring, and background operations do not refresh activity.

### Scanning and release

The TTI starts when a scan first sees a load config. Later scans keep the existing timestamp. A scan reads the collection's TTI override. Loading time counts toward TTI expiration.

The primary QueryCoord samples balanceable Worker QueryNodes by resource group. Each group's memory ratio is the sum of process resident memory excluding shared pages divided by the sum of memory limits. A group enters memory pressure at the high watermark and leaves it at the low watermark. A group with incomplete readings is skipped for the current scan. In pressured groups, the scanner selects the oldest ready collections, up to `lruMaxEvictionsPerScan` across all groups. TTI expirations do not count toward that limit.

The scanner queues candidates and returns. A collection can appear in the queue or in a running release once at a time. When a release finishes, QueryCoord fills the available slots from the queue. Failed releases are eligible for a later scan if their load config still exists.

A successful release log means the `DropLoadConfig` broadcast call returned successfully. Physical resource cleanup on every node may still be in progress.

## Concurrency semantics

When a candidate reaches the head of the queue, QueryCoord checks the latest collection TTI and activity. A memory-pressure candidate also needs unchanged activity, a ready collection, and pressure in its current resource group according to the latest scan. The release uses the existing collection DDL lock and load-config existence check. Once release starts, subsequent queries do not cancel it.

Existing resource references protect segment resources already acquired by a query. If a later query stage encounters `CollectionNotLoaded`, `ViewNotFound`, or `ViewInvalidated`, the existing complete DQL retry handles recovery. The TTI does not provide a query or iterator view lease.

The activity mutex protects timestamps, pressure state, and the release queue. Reading load configs, submitting broadcasts, and waiting for releases happen outside that mutex. The load-config removal observer only removes activity records and does not call back into the store.

## Recovery and cleanup

- **QueryCoord restart or leader change:** activity is not persisted. After load configs are recovered, the first scan starts a fresh TTI for each collection. Proxies do not need to report previous activity.
- **Manual release, drop, or automatic release:** a synchronous load-config removal observer deletes the activity record. After a reload, a later scan initializes a new timestamp.
- **Proxy exit:** there are no per-request registrations to clean up. Collections continue to expire according to their latest query attempt.
- **Either feature disabled:** clear activity, stop refreshing timestamps, and stop the scan timer. Queued releases that have not started are skipped; started ones may still finish. When both features are enabled again, scanning initializes fresh TTIs.

## Configuration

```yaml
proxy:
  enableAutoLoad: true
queryCoord:
  autoRelease:
    enabled: false
    ttiSeconds: 3600
    checkIntervalSeconds: 10
    releaseConcurrency: 64
    lruHighWatermarkPercentage: 85
    lruLowWatermarkPercentage: 80
    lruMaxEvictionsPerScan: 64
```

All autoRelease parameters are marked as version `3.0.0` and support runtime updates:

| Parameter | Effect of an update |
|---|---|
| `enabled` | Requires autoLoad as well; disabling clears activity, and re-enabling starts fresh TTIs through scanning |
| `ttiSeconds` | Scans use the latest idle threshold without resetting timestamps |
| `checkIntervalSeconds` | The event loop resets the timer to the new interval |
| `releaseConcurrency` | The next dispatch uses the new concurrency limit |
| `lruHighWatermarkPercentage` | A group at or above this percentage enters memory pressure |
| `lruLowWatermarkPercentage` | A group at or below this percentage leaves memory pressure |
| `lruMaxEvictionsPerScan` | Maximum memory-pressure candidates selected per scan |

The TTI, scan interval, and concurrency must be positive. The TTI and scan interval must also fit in `time.Duration`. If autoRelease is enabled while autoLoad is disabled, QueryCoord logs a warning and leaves automatic release disabled.

## Implementation entry points

- `internal/querycoordv2/collection_usage.go`: activity records, configuration watchers, scanning, and concurrent release.
- `internal/querycoordv2/collection_readiness.go`: activity refresh from readiness requests.
- `internal/querycoordv2/server.go`: manager lifecycle.
- `pkg/util/paramtable/component_param.go` and `configs/milvus.yaml`: configuration definitions.

[Query client](qviews/query/query_client.md) describes `EnsureCollectionReady`. `internal/proxy/search_auto_load.go` sends it and retries complete queries; `internal/querycoordv2/collection_readiness.go` performs automatic loading.
