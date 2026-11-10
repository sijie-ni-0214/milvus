# Copying a Milvus catalog from etcd to TiKV

A Milvus cluster using TiKV for persistent metadata still uses etcd for discovery and leader election. The migration tool copies the source root's `kv/` allocator records and `meta/` catalog records into a dedicated TiKV key range. It skips `meta/session/`. The destination defaults to the existing root. The operator can supply a separate target root for testing. The tool leaves the choice of when to start Milvus on the new metastore to the operator.

## Consistency and verification

For a rehearsal against a running etcd cluster, the operator selects a retained revision. The tool uses that revision for source preflight, copy and verification, including reads after reconnects. A keys-only pass samples split points and produces disjoint ranges that cover the selected prefixes. The range plan can be prepared before choosing the revision because later inserts and deletes only affect the distribution of keys among those ranges.

Each reader commits sorted batches to TiKV. Batch size and writer concurrency bound in-flight writes. The destination must be exclusive for the duration of copy and verification. An independent pass scans each source and target range in key order, reports missing and extra keys, and compares the logical value of each matching key. Both the copier and the catalog use the same TiKV encoding for empty values.

A transport interruption resumes after the last key accepted by the reader. An expired source revision fails the run. Committed batches survive failure, so a full retry needs an empty target or an explicit overwrite of the selected ranges. For a cutover, all catalog writers must stop before selecting the final source view and stay stopped until verification finishes. Writes after a rehearsal revision require a fresh migration before cutover.

## Catalog behavior after migration

The streaming catalog creates the first checkpoint with a compare-and-swap at version zero. TiKV implements this as an atomic absence check and insert. Transactions guarded by a predicate on a separate key lock the guard and check its value before committing the dependent writes. These operations support checkpoint creation and fence stale writers.

On a dedicated TiKV cluster, the active MixCoord can advance the cluster-wide GC safe point so that old versions become eligible for reclamation. It takes the retention window from configuration and respects the minimum PD service safe point, including pins held by backups. Operators who already have a GC worker managing the same TiKV cluster can disable the MixCoord worker.

Build and rehearsal commands are in the [migration runbook](../../../cmd/tools/etcd2tikv/README.md).
