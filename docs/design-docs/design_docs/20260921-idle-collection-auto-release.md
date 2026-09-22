# Automatic Release of Idle Collections

## Goal and assumptions

In streaming mode, QueryCoord releases collections that have received no DQL attempts for an idle TTL. Subsequent queries use the existing Load on Search path to reload them.

- The feature is effective only when both `queryCoord.autoRelease.enabled` and `proxy.enableAutoLoad` are enabled. It depends on the existing complete DQL retry: a collection-not-loaded, view-not-found, or view-invalidated error during loading or execution sends the request back through automatic loading before another query attempt.
- The TTL should be substantially longer than normal query latency. It measures time since the latest query attempt. Request completion does not refresh it; there are no in-flight request counts or completion notifications.
- Loading, a long-running query, or an iterator waiting for its next page does not prevent expiration. Retries remain bounded by the original request deadline and retry limit, so repeated releases or slow loading can still cause the request to fail.
- All collections with a load config are eligible, including collections loaded manually.

## Component responsibilities and flow

**Proxy** sends one `EnsureCollectionReady` RPC per complete DQL attempt of Search, HybridSearch, and Query, and retries the whole request on a recoverable release or view error. QueryCoord shares load submission and the readiness wait among all callers of a collection.

**QueryCoord** keeps an in-memory `collectionID -> lastActiveAt` map. Existing readiness RPCs refresh activity, and a periodic scan of load configs selects expired collections. Requests from different Proxies update the same collection timestamp using QueryCoord's local clock.

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
  -> Existing activity has expired: add the collection to the release batch
  -> Run the existing release path with bounded concurrency; wait for the batch
```

### Activity timestamps

- `EnsureCollectionReady` refreshes an existing record on entry.
- A caller that joined a shared load or wait refreshes the record again after readiness is confirmed, immediately before returning success. It does not refresh activity while waiting or when returning an error.
- Readiness RPCs only refresh records already initialized by the scanner. Request completion and internal requery stages do not refresh activity separately. A complete DQL retry refreshes it through the next initial check.
- DML, load-progress requests, monitoring, and background operations do not refresh activity.

### Scanning and release

The TTL starts when a scan first sees a load config. Later scans do not refresh existing timestamps. The scanner reads only load configs and activity timestamps; it does not call `DescribeCollection` or check collection readiness. Loading time can therefore count toward the TTL, and a slow load can be released before it first becomes ready.

After collecting candidates, the scanner limits parallel releases with `releaseConcurrency`. Each release uses the existing collection DDL lock and load-config existence check. A failed release is logged without canceling other releases in the batch. If its load config remains, a later scan can select it again. The scanner waits for all calls in the current batch before processing another scan, so batches do not overlap.

A successful release log means the `DropLoadConfig` broadcast call returned successfully. Physical resource cleanup on every node may still be in progress.

## Concurrency semantics

If a query refreshes activity before the scanner reads the timestamp, the scanner uses the new timestamp. Once a collection has been selected, a later query does not cancel its release. Activity, readiness, and load-config versions are not rechecked before release.

Existing resource references protect segment resources already acquired by a query. If a later query stage encounters `CollectionNotLoaded`, `ViewNotFound`, or `ViewInvalidated`, the existing complete DQL retry handles recovery. The TTL does not provide a query or iterator view lease.

The activity mutex protects only in-memory records. Reading load configs, submitting broadcasts, and waiting for releases happen outside that mutex. The load-config removal observer only removes activity records and does not call back into the store.

## Recovery and cleanup

- **QueryCoord restart or leader change:** activity is not persisted. After load configs are recovered, the first scan starts a fresh TTL for each collection. Proxies do not need to report previous activity.
- **Manual release, drop, or automatic release:** a synchronous load-config removal observer deletes the activity record. After a reload, a later scan initializes a new timestamp.
- **Proxy exit:** there are no per-request registrations to clean up. Collections continue to expire according to their latest query attempt.
- **Either feature disabled:** clear activity, stop refreshing timestamps, and stop the scan timer. Already-selected releases may still finish. When both features are enabled again, scanning initializes fresh TTLs.

## Configuration

```yaml
proxy:
  enableAutoLoad: true
queryCoord:
  autoRelease:
    enabled: false
    idleTTLSeconds: 600
    checkIntervalSeconds: 30
    releaseConcurrency: 16
```

All four autoRelease parameters are marked as version `3.0.0` and support runtime updates:

| Parameter | Effect of an update |
|---|---|
| `enabled` | Requires autoLoad as well; disabling clears activity, and re-enabling starts fresh TTLs through scanning |
| `idleTTLSeconds` | Scans use the latest TTL without resetting existing timestamps |
| `checkIntervalSeconds` | The event loop resets the timer to the new interval after any running release batch finishes |
| `releaseConcurrency` | The next batch uses the new limit; the current batch keeps its original limit |

The TTL, scan interval, and concurrency must be positive. The TTL and scan interval must also fit in `time.Duration`. If autoRelease is enabled while autoLoad is disabled, QueryCoord logs a warning and leaves automatic release disabled.

## Implementation entry points

- `internal/querycoordv2/collection_usage.go`: activity records, configuration watchers, scanning, and concurrent release.
- `internal/querycoordv2/collection_readiness.go`: activity refresh from readiness requests.
- `internal/querycoordv2/server.go`: manager lifecycle.
- `pkg/util/paramtable/component_param.go` and `configs/milvus.yaml`: configuration definitions.

[Query client](qviews/query/query_client.md) describes `EnsureCollectionReady`. `internal/proxy/search_auto_load.go` sends it and retries complete queries; `internal/querycoordv2/collection_readiness.go` performs automatic loading.
